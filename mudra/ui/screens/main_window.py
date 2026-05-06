from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from database.db import DatabaseManager
from inference.camera.camera_worker import CameraService
from inference.engines.predictor import GesturePredictor
from inference.overlay.draw import draw_overlay
from ui.state.session import SessionState
from utils.common.security import create_access_token, verify_password
from utils.environment_check import check_environment
from utils.gesture_media_mapper import get_media_path, get_gesture_reference, get_gesture_description, get_reference_image_path
from utils.io.config_loader import load_config, save_config


class ReferenceVideoThread(QThread):
    frame_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.media_path: Optional[str] = None
        self._media_changed = False

    def set_media(self, media_path: Optional[str]):
        self.media_path = media_path
        self._media_changed = True

    def run(self):
        while self._run_flag:
            if not self.media_path:
                if self._media_changed:
                    self.status_signal.emit("Text reference shown (no video)")
                    self._media_changed = False
                time.sleep(0.5)
                continue

            cap = cv2.VideoCapture(self.media_path)
            if not cap.isOpened():
                self.status_signal.emit("Could not open reference video")
                self._media_changed = False
                time.sleep(1.0)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 1:
                fps = 24
            sleep_s = 1.0 / min(max(fps, 10.0), 30.0)

            self.status_signal.emit("▶ Playing reference")
            current_path = self.media_path
            while self._run_flag and self.media_path == current_path:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.frame_signal.emit(qimg)
                time.sleep(sleep_s)
            cap.release()

    def stop(self):
        self._run_flag = False
        self.wait(1000)


class InferenceThread(QThread):
    frame_signal = pyqtSignal(QImage)
    result_signal = pyqtSignal(dict)

    def __init__(self, predictor: GesturePredictor, env_status: Dict[str, bool]):
        super().__init__()
        self._run_flag = True
        self.predictor = predictor
        self.camera = CameraService(max_process_fps=18.0)
        self.target_name = ""
        self.target_mode = "static"
        self.env_status = env_status
        self.last_result: Dict[str, object] = {
            "status": "uncertain",
            "label": "-",
            "confidence": 0.0,
            "model_used": "-",
            "latency_ms": 0,
            "stable": False,
        }

    def run(self):
        fail_count = 0
        if not self.camera.open():
            self.result_signal.emit({"status": "camera_error", "label": "CAMERA_ERROR", "confidence": 0.0, "latency_ms": 0})
            return

        while self._run_flag:
            packet = self.camera.read()
            if packet is None:
                fail_count += 1
                if fail_count > 20:
                    self.result_signal.emit({"status": "camera_error", "label": "CAMERA_DISCONNECTED", "confidence": 0.0, "latency_ms": 0})
                    break
                continue

            fail_count = 0
            frame = packet.frame
            if packet.should_process:
                self.last_result = self.predictor.predict(frame, target_mode=self.target_mode)

            display_result = dict(self.last_result)
            display_result["env"] = dict(self.env_status)
            display_result["perf_warning"] = "Low Performance Warning" if packet.fps < 15 else ""

            self.predictor.tracker.draw(frame, self.last_result.get("extraction", {}))
            draw_overlay(frame, display_result, packet.fps, target=self.target_name)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_signal.emit(qimg)

            emit_result = {
                "status": display_result.get("status"),
                "label": display_result.get("label"),
                "confidence": float(display_result.get("confidence", 0.0)),
                "latency_ms": int(display_result.get("latency_ms", 0)),
                "fps": float(packet.fps),
                "model_used": display_result.get("model_used", "-"),
                "stable": bool(display_result.get("stable", False)),
                "perf_warning": display_result.get("perf_warning", ""),
            }
            self.result_signal.emit(emit_result)

        self.camera.release()

    def set_target(self, target_name: str) -> None:
        self.target_name = target_name

    def set_target_mode(self, target_mode: str) -> None:
        self.target_mode = target_mode if target_mode in {"static", "dynamic"} else "static"

    def stop(self):
        self._run_flag = False
        self.wait(1500)


class MudraMainWindow(QMainWindow):
    IDX_LOGIN = 0
    IDX_DASH = 1
    IDX_STUDY = 2
    IDX_PRACTICE = 3
    IDX_QUIZ = 4
    IDX_ANALYTICS = 5
    IDX_ADMIN = 6

    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.db = DatabaseManager(self.config.get("database", {}).get("sqlite_path", "database/mudra.db"))
        self.db.seed_core_data()
        self.session = SessionState()
        self.env_status = check_environment(self.config)
        self.predictor = GesturePredictor(self.config)

        self.gesture_rows: List[Dict[str, object]] = []
        self.selected_gesture: Optional[Dict[str, object]] = None
        self.current_result: Optional[Dict[str, object]] = None
        self.quiz_queue: List[Dict[str, object]] = []
        self.quiz_index = 0
        self.quiz_score = 0

        self._study_gesture_id: Optional[str] = None
        self._study_started_at: Optional[float] = None

        self._current_theme_mode = "dark"
        self.setWindowTitle(self.config.get("window_title", "MUDRA"))
        self.resize(1280, 840)
        self.setMinimumSize(980, 680)
        import platform
        self.sys_font = "Segoe UI" if platform.system() == "Windows" else "Helvetica Neue"
        self.setStyleSheet(self._theme(self._current_theme_mode))

        self.inference_thread: Optional[InferenceThread] = None
        self.study_ref_thread: Optional[ReferenceVideoThread] = None
        self.practice_ref_thread: Optional[ReferenceVideoThread] = None
        self._last_qimage: Optional[QImage] = None
        self._last_study_ref: Optional[QImage] = None
        self._last_practice_ref: Optional[QImage] = None

        self._build_ui()
        self._start_reference_threads()
        self._apply_environment_status()

    def _theme(self, mode: str = "dark") -> str:
        f = self.sys_font
        if mode == "light":
            return f"""
            QMainWindow, QWidget {{ background-color:#F0F4F7; color:#1A3040; font-family:"{f}",sans-serif; }}
            QLabel {{ color:#1A3040; font-size:13px; background:transparent; }}
            QPushButton {{
                background-color:#58CC02; color:#FFFFFF; border:none; border-radius:14px;
                border-bottom:4px solid #46A302; padding:12px 20px; font-size:14px;
                font-weight:700; font-family:"{f}",sans-serif;
            }}
            QPushButton:hover {{ background-color:#67E002; }}
            QPushButton:pressed {{ border-bottom:1px solid #46A302; padding-top:15px; }}
            QPushButton:disabled {{ background-color:#D4E4EB; color:#8CAAB5; border-bottom:4px solid #B8CDD6; }}
            QPushButton[secondary="true"] {{ background-color:transparent; color:#58CC02; border:2px solid #58CC02; border-bottom:4px solid #46A302; }}
            QPushButton[secondary="true"]:hover {{ background-color:rgba(88,204,2,0.1); }}
            QPushButton[danger="true"] {{ background-color:#FF4B4B; border-bottom-color:#CC3333; }}
            QPushButton[danger="true"]:hover {{ background-color:#FF6B6B; }}
            QPushButton[accent="true"] {{ background-color:#1CB0F6; border-bottom-color:#0A8FCC; }}
            QPushButton[accent="true"]:hover {{ background-color:#35BEFF; }}
            QPushButton[nav="true"] {{
                background-color:transparent; color:#4A6070; border:none; border-radius:12px;
                border-bottom:none; text-align:left; padding:10px 14px; font-size:14px; font-weight:600;
            }}
            QPushButton[nav="true"]:hover {{ background-color:#E0EBF0; color:#1A3040; }}
            QPushButton[nav="true"][active="true"] {{ background-color:#E0EBF0; color:#388A00; border-left:4px solid #58CC02; }}
            QLineEdit, QTextEdit {{
                background-color:#FFFFFF; color:#1A3040; border:2px solid #C8D8DF;
                border-radius:12px; padding:10px 14px; font-size:14px; selection-background-color:#58CC02;
            }}
            QLineEdit:focus, QTextEdit:focus {{ border-color:#58CC02; }}
            QComboBox {{ background-color:#FFFFFF; color:#1A3040; border:2px solid #C8D8DF; border-radius:10px; padding:8px 12px; font-size:13px; }}
            QComboBox::drop-down {{ border:none; }}
            QComboBox QAbstractItemView {{ background-color:#FFFFFF; color:#1A3040; selection-background-color:#58CC02; }}
            QListWidget {{ background-color:#FFFFFF; color:#1A3040; border:2px solid #C8D8DF; border-radius:12px; font-size:13px; outline:none; }}
            QListWidget::item {{ padding:10px 14px; border-bottom:1px solid #E8EFF3; }}
            QListWidget::item:selected {{ background-color:#58CC02; color:#FFFFFF; border-radius:8px; }}
            QListWidget::item:hover:!selected {{ background-color:#EEF5F8; }}
            QTableWidget {{ background-color:#FFFFFF; color:#1A3040; border:2px solid #C8D8DF; border-radius:12px; gridline-color:#E8EFF3; font-size:12px; outline:none; }}
            QTableWidget::item {{ padding:8px; }}
            QTableWidget::item:selected {{ background-color:rgba(88,204,2,0.18); }}
            QHeaderView::section {{ background-color:#E4EDF1; color:#4A7080; padding:8px; border:none; font-weight:700; font-size:11px; }}
            QScrollBar:vertical {{ background:transparent; width:6px; }}
            QScrollBar::handle:vertical {{ background:#C8D8DF; border-radius:3px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
            QProgressBar {{ background-color:#DDE7EC; border-radius:7px; border:none; color:transparent; }}
            QProgressBar::chunk {{ background-color:#58CC02; border-radius:7px; }}
            QCheckBox {{ color:#1A3040; spacing:8px; font-size:13px; }}
            QCheckBox::indicator {{ width:20px; height:20px; border:2px solid #C8D8DF; border-radius:6px; background:#FFFFFF; }}
            QCheckBox::indicator:checked {{ background:#58CC02; border-color:#58CC02; }}
            QScrollArea {{ border:none; background:transparent; }}
            """
        return f"""
        QMainWindow, QWidget {{
            background-color: #131F24;
            color: #FFFFFF;
            font-family: "{f}", sans-serif;
        }}
        QLabel {{
            color: #FFFFFF;
            font-size: 13px;
            background: transparent;
        }}
        QPushButton {{
            background-color: #58CC02;
            color: #FFFFFF;
            border: none;
            border-radius: 14px;
            border-bottom: 4px solid #46A302;
            padding: 12px 20px;
            font-size: 14px;
            font-weight: 700;
            font-family: "{f}", sans-serif;
        }}
        QPushButton:hover {{ background-color: #67E002; }}
        QPushButton:pressed {{ border-bottom: 1px solid #46A302; padding-top: 15px; }}
        QPushButton:disabled {{
            background-color: #1A2D35;
            color: #3A5562;
            border-bottom: 4px solid #152530;
        }}
        QPushButton[secondary="true"] {{
            background-color: transparent;
            color: #58CC02;
            border: 2px solid #58CC02;
            border-bottom: 4px solid #46A302;
        }}
        QPushButton[secondary="true"]:hover {{ background-color: rgba(88,204,2,0.12); }}
        QPushButton[danger="true"] {{
            background-color: #FF4B4B;
            border-bottom-color: #CC3333;
        }}
        QPushButton[danger="true"]:hover {{ background-color: #FF6B6B; }}
        QPushButton[accent="true"] {{
            background-color: #1CB0F6;
            border-bottom-color: #0A8FCC;
        }}
        QPushButton[accent="true"]:hover {{ background-color: #35BEFF; }}
        QPushButton[nav="true"] {{
            background-color: transparent;
            color: #7D9EA8;
            border: none;
            border-radius: 12px;
            border-bottom: none;
            text-align: left;
            padding: 10px 14px;
            font-size: 14px;
            font-weight: 600;
        }}
        QPushButton[nav="true"]:hover {{
            background-color: #1A2D35;
            color: #FFFFFF;
        }}
        QPushButton[nav="true"][active="true"] {{
            background-color: #1A2D35;
            color: #58CC02;
            border-left: 4px solid #58CC02;
        }}
        QLineEdit, QTextEdit {{
            background-color: #1A2D35;
            color: #FFFFFF;
            border: 2px solid #2A3F4A;
            border-radius: 12px;
            padding: 10px 14px;
            font-size: 14px;
            selection-background-color: #58CC02;
        }}
        QLineEdit:focus, QTextEdit:focus {{ border-color: #58CC02; }}
        QComboBox {{
            background-color: #1A2D35;
            color: #FFFFFF;
            border: 2px solid #2A3F4A;
            border-radius: 10px;
            padding: 8px 12px;
            font-size: 13px;
        }}
        QComboBox::drop-down {{ border: none; }}
        QComboBox QAbstractItemView {{
            background-color: #1A2D35;
            color: #FFFFFF;
            selection-background-color: #58CC02;
        }}
        QListWidget {{
            background-color: #1A2D35;
            color: #FFFFFF;
            border: 2px solid #2A3F4A;
            border-radius: 12px;
            font-size: 13px;
            outline: none;
        }}
        QListWidget::item {{
            padding: 10px 14px;
            border-bottom: 1px solid #243B47;
        }}
        QListWidget::item:selected {{
            background-color: #58CC02;
            color: #FFFFFF;
            border-radius: 8px;
        }}
        QListWidget::item:hover:!selected {{ background-color: #243B47; }}
        QTableWidget {{
            background-color: #1A2D35;
            color: #FFFFFF;
            border: 2px solid #2A3F4A;
            border-radius: 12px;
            gridline-color: #243B47;
            font-size: 12px;
            outline: none;
        }}
        QTableWidget::item {{ padding: 8px; }}
        QTableWidget::item:selected {{ background-color: rgba(88,204,2,0.25); }}
        QHeaderView::section {{
            background-color: #243B47;
            color: #7D9EA8;
            padding: 8px;
            border: none;
            font-weight: 700;
            font-size: 11px;
        }}
        QScrollBar:vertical {{
            background: transparent;
            width: 6px;
        }}
        QScrollBar::handle:vertical {{
            background: #2A3F4A;
            border-radius: 3px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QProgressBar {{
            background-color: #243B47;
            border-radius: 7px;
            border: none;
            text-align: center;
            color: transparent;
        }}
        QProgressBar::chunk {{
            background-color: #58CC02;
            border-radius: 7px;
        }}
        QCheckBox {{
            color: #FFFFFF;
            spacing: 8px;
            font-size: 13px;
        }}
        QCheckBox::indicator {{
            width: 20px;
            height: 20px;
            border: 2px solid #2A3F4A;
            border-radius: 6px;
            background: #1A2D35;
        }}
        QCheckBox::indicator:checked {{
            background: #58CC02;
            border-color: #58CC02;
        }}
        QScrollArea {{ border: none; background: transparent; }}
        """


    def _palette(self) -> dict:
        """Theme-appropriate color palette for all inline setStyleSheet calls."""
        if self._current_theme_mode == "light":
            return dict(
                sidebar="#FFFFFF",        sidebar_border="#D0DCE4",
                header="#F0F4F7",         header_border="#D0DCE4",
                card="#FFFFFF",           card_border="#C8D8DF",
                card_hover="#F0F7FA",     deep_bg="#EBF1F5",
                streak_bg="#E4EDF2",      chip_bg="#E4EDF2",
                muted_text="#4A6070",     bright_text="#2A4050",
                ref_bg="#EBF1F5",
            )
        return dict(
            sidebar="#0D1820",        sidebar_border="#1A2D35",
            header="#0D1820",         header_border="#1A2D35",
            card="#1A2D35",           card_border="#2A3F4A",
            card_hover="#1F3340",     deep_bg="#0D1820",
            streak_bg="#1A2D35",      chip_bg="#1A2D35",
            muted_text="#A8C4CF",     bright_text="#C4D8E0",
            ref_bg="#0D1820",
        )

    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setCentralWidget(central)

        # ── Sidebar ──────────────────────────────────────────
        p = self._palette()
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(230)
        self.sidebar.setStyleSheet(
            f"QFrame {{ background-color: {p['sidebar']}; border-right: 1px solid {p['sidebar_border']}; }}"
        )
        sbl = QVBoxLayout(self.sidebar)
        sbl.setContentsMargins(12, 20, 12, 20)
        sbl.setSpacing(4)

        logo_row = QHBoxLayout()
        logo_icon = QLabel("🦉")
        logo_icon.setFont(QFont(self.sys_font, 22))
        logo_icon.setFixedWidth(36)
        logo_text = QLabel("MUDRA")
        logo_text.setFont(QFont(self.sys_font, 20, QFont.Bold))
        logo_text.setStyleSheet("color:#58CC02; background:transparent; letter-spacing:2px;")
        logo_row.addWidget(logo_icon)
        logo_row.addWidget(logo_text)
        logo_row.addStretch()
        sbl.addLayout(logo_row)

        tagline = QLabel("Learn Indian Sign Language")
        tagline.setStyleSheet(
            "color:#A8C4CF; font-size:12px; background:transparent; margin-left:4px; margin-bottom:12px;"
        )
        sbl.addWidget(tagline)
        sbl.addSpacing(12)

        nav_items = [
            ("🏠", "Home",      self.IDX_DASH),
            ("📖", "Study",     self.IDX_STUDY),
            ("🎯", "Practice",  self.IDX_PRACTICE),
            ("⚡", "Quiz",      self.IDX_QUIZ),
            ("📊", "Analytics", self.IDX_ANALYTICS),
            ("⚙️", "Admin",     self.IDX_ADMIN),
        ]
        self.nav_buttons = {}
        for icon, title, idx in nav_items:
            b = QPushButton(f"  {icon}   {title}")
            b.setProperty("nav", "true")
            b.setFixedHeight(44)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, i=idx: self.navigate_to(i))
            sbl.addWidget(b)
            self.nav_buttons[title] = b

        sbl.addStretch()

        streak_frame = QFrame()
        streak_frame.setStyleSheet(
            f"QFrame {{ background:{p['streak_bg']}; border-radius:12px; border:1px solid {p['card_border']}; }}"
        )
        sf_l = QHBoxLayout(streak_frame)
        sf_l.setContentsMargins(12, 8, 12, 8)
        streak_lbl = QLabel("🔥  Keep practicing!")
        streak_lbl.setStyleSheet("color:#FFC800; font-size:11px; font-weight:600; background:transparent;")
        sf_l.addWidget(streak_lbl)
        sbl.addWidget(streak_frame)
        sbl.addSpacing(4)

        # Theme switcher
        theme_row = QHBoxLayout()
        theme_row.setSpacing(6)
        self._theme_btns = {}
        for t_mode, t_icon, t_tip in [("dark", "🌙", "Dark"), ("light", "☀️", "Light")]:
            tb = QPushButton(f"{t_icon}  {t_tip}")
            tb.setFixedHeight(34)
            tb.setCursor(Qt.PointingHandCursor)
            tb.setToolTip(f"{t_tip} theme")
            tb.setStyleSheet(
                "QPushButton{background:#1A2D35;color:#A8C4CF;border:1px solid #2A3F4A;"
                "border-radius:10px;border-bottom:2px solid #2A3F4A;font-size:12px;font-weight:600;padding:4px 8px;}"
                "QPushButton:hover{color:#FFFFFF;background:#243B47;}"
                "QPushButton[active_theme=\"true\"]{background:#58CC02;color:#FFFFFF;border-color:#46A302;}"
            )
            tb.setProperty("active_theme", "true" if t_mode == self._current_theme_mode else "false")
            tb.clicked.connect(lambda _=False, m=t_mode: self._switch_theme(m))
            self._theme_btns[t_mode] = tb
            theme_row.addWidget(tb)
        sbl.addLayout(theme_row)
        sbl.addSpacing(4)

        self.btn_logout = QPushButton("Logout")
        self.btn_logout.setProperty("secondary", "true")
        self.btn_logout.setFixedHeight(40)
        self.btn_logout.setCursor(Qt.PointingHandCursor)
        self.btn_logout.clicked.connect(self.logout)
        sbl.addWidget(self.btn_logout)

        # ── Content ──────────────────────────────────────────
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        self.status_bar_widget = self._build_env_header()
        content_layout.addWidget(self.status_bar_widget)

        self.stack = QStackedWidget()
        self.stack.currentChanged.connect(self._on_stack_changed)
        self.stack.addWidget(self._build_login_page())
        self.stack.addWidget(self._build_dashboard_page())
        self.stack.addWidget(self._build_study_page())
        self.stack.addWidget(self._build_practice_page())
        self.stack.addWidget(self._build_quiz_page())
        self.stack.addWidget(self._build_analytics_page())
        self.stack.addWidget(self._build_admin_page())
        content_layout.addWidget(self.stack)

        root.addWidget(self.sidebar)
        root.addWidget(content_widget)
        self.sidebar.setVisible(False)



    def _build_env_header(self) -> QWidget:
        p = self._palette()
        bar = QFrame()
        bar.setFixedHeight(48)
        bar.setStyleSheet(f"background:{p['header']}; border-bottom:1px solid {p['header_border']};")
        l = QHBoxLayout(bar)
        l.setContentsMargins(16, 0, 16, 0)
        self.env_title = QLabel("Environment")
        self.env_title.setStyleSheet(
            "color:#7D9EA8; font-size:10px; font-weight:700; background:transparent; text-transform:uppercase;"
        )
        l.addWidget(self.env_title)
        l.addSpacing(10)

        self.env_mp     = QLabel()
        self.env_torch  = QLabel()
        self.env_cam    = QLabel()
        self.env_static = QLabel()
        self.env_dynamic= QLabel()
        chip_style = (
            f"QLabel {{ background:{p['chip_bg']}; border-radius:11px; "
            "padding:2px 10px; font-size:11px; font-weight:600; }"
        )
        for w in [self.env_mp, self.env_torch, self.env_cam, self.env_static, self.env_dynamic]:
            w.setFixedHeight(24)
            w.setStyleSheet(chip_style)
            l.addWidget(w)
            l.addSpacing(4)

        l.addStretch()
        btn_r = QPushButton("↻ Refresh")
        btn_r.setFixedSize(88, 30)
        btn_r.setCursor(Qt.PointingHandCursor)
        btn_r.setStyleSheet(
            "QPushButton { background:#1A2D35; color:#7D9EA8; border:1px solid #2A3F4A; "
            "border-radius:8px; border-bottom:2px solid #2A3F4A; font-size:12px; font-weight:600; padding:4px 8px; }"
            "QPushButton:hover { color:#FFFFFF; background:#243B47; }"
        )
        btn_r.clicked.connect(self.refresh_environment_status)
        l.addWidget(btn_r)
        return bar



    def _build_login_page(self) -> QWidget:
        p = self._palette()
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setAlignment(Qt.AlignCenter)

        card = QFrame()
        card.setFixedWidth(420)
        card.setStyleSheet(
            f"QFrame {{ background:{p['card']}; border-radius:24px; border:1px solid {p['card_border']}; }}"
        )
        fl = QVBoxLayout(card)
        fl.setContentsMargins(40, 36, 40, 36)
        fl.setSpacing(14)

        hero = QLabel("🦉")
        hero.setAlignment(Qt.AlignCenter)
        hero.setFont(QFont(self.sys_font, 48))
        hero.setStyleSheet("background:transparent;")
        fl.addWidget(hero)

        app_name = QLabel("MUDRA")
        app_name.setAlignment(Qt.AlignCenter)
        app_name.setFont(QFont(self.sys_font, 26, QFont.Bold))
        app_name.setStyleSheet("color:#58CC02; background:transparent; letter-spacing:4px;")
        fl.addWidget(app_name)

        tagline = QLabel("The fun way to learn Indian Sign Language")
        tagline.setAlignment(Qt.AlignCenter)
        tagline.setWordWrap(True)
        tagline.setStyleSheet("color:#7D9EA8; font-size:13px; background:transparent;")
        fl.addWidget(tagline)
        fl.addSpacing(8)

        self.login_email = QLineEdit()
        self.login_email.setPlaceholderText("Email address")
        self.login_email.setFixedHeight(48)
        fl.addWidget(self.login_email)

        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("Password")
        self.login_password.setEchoMode(QLineEdit.Password)
        self.login_password.setFixedHeight(48)
        fl.addWidget(self.login_password)
        fl.addSpacing(4)

        btn_login = QPushButton("LOG IN")
        btn_login.setFixedHeight(52)
        btn_login.setFont(QFont(self.sys_font, 15, QFont.Bold))
        btn_login.setCursor(Qt.PointingHandCursor)
        btn_login.clicked.connect(self.handle_login)
        fl.addWidget(btn_login)

        div = QLabel("— or —")
        div.setAlignment(Qt.AlignCenter)
        div.setStyleSheet("color:#3A5562; font-size:12px; background:transparent;")
        fl.addWidget(div)

        btn_demo = QPushButton("Use Demo Account")
        btn_demo.setFixedHeight(46)
        btn_demo.setCursor(Qt.PointingHandCursor)
        btn_demo.setProperty("secondary", "true")
        btn_demo.clicked.connect(
            lambda: (self.login_email.setText("demo@mudra.local"),
                     self.login_password.setText("demo123"))
        )
        fl.addWidget(btn_demo)

        outer.addWidget(card)
        return page



    def _build_dashboard_page(self) -> QWidget:
        p = self._palette()
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(20)

        # Header row
        header = QHBoxLayout()
        self.welcome = QLabel("Welcome back!")
        self.welcome.setFont(QFont(self.sys_font, 20, QFont.Bold))
        header.addWidget(self.welcome)
        header.addStretch()
        for icon, attr, style in [("🔥", "streak_lbl", "color:#FFC800;"), ("⭐", "xp_lbl", "color:#FFC800;")]:
            pill = QFrame()
            pill.setStyleSheet(f"QFrame{{background:{p['card']};border-radius:12px;border:1px solid {p['card_border']};}}")
            pl = QHBoxLayout(pill)
            pl.setContentsMargins(10, 4, 10, 4)
            pl.setSpacing(4)
            ic = QLabel(icon); ic.setStyleSheet("background:transparent;font-size:14px;")
            vl = QLabel("—");  vl.setStyleSheet(f"background:transparent;font-size:12px;font-weight:700;{style}")
            setattr(self, attr, vl)
            pl.addWidget(ic); pl.addWidget(vl)
            header.addWidget(pill); header.addSpacing(6)
        layout.addLayout(header)

        self.lesson_summary = QLabel("Loading…")
        self.lesson_summary.setStyleSheet("color:#C4D8E0; font-size:13px;")
        self.progress_summary = QLabel("")
        self.progress_summary.setStyleSheet("color:#A8C4CF; font-size:12px;")
        layout.addWidget(self.lesson_summary)
        layout.addWidget(self.progress_summary)

        # Unit cards
        cards_row = QHBoxLayout(); cards_row.setSpacing(14)
        units = [
            ("🤟", "ISL Alphabets",    "26 letters · Beginner",      "#58CC02", "#46A302"),
            ("📝", "Core Words",        "50–100 signs · Intermediate", "#1CB0F6", "#0A8FCC"),
            ("💬", "Conversation",      "Daily phrases · Advanced",    "#9B59B6", "#7D3C98"),
            ("📈", "Your Progress",     "View analytics",              "#FFC800", "#CC9900"),
        ]
        for icon, ttl, sub, col, dark in units:
            c = QFrame()
            c.setMinimumHeight(155)
            c.setStyleSheet(f"""
                QFrame{{background:{p['card']};border-radius:20px;border:2px solid {p['card_border']};border-bottom:4px solid {dark};}}
                QFrame:hover{{border-color:{col};background:{p['card_hover']};}}
            """)
            cl = QVBoxLayout(c); cl.setContentsMargins(18,18,18,18); cl.setSpacing(6)
            ie = QLabel(icon); ie.setFont(QFont(self.sys_font,28)); ie.setStyleSheet("background:transparent;")
            hl = QLabel(ttl);  hl.setFont(QFont(self.sys_font,14,QFont.Bold)); hl.setStyleSheet("background:transparent;")
            sl = QLabel(sub);  sl.setStyleSheet(f"background:transparent;color:{col};font-size:13px;font-weight:600;")
            cl.addWidget(ie); cl.addWidget(hl); cl.addWidget(sl); cl.addStretch()
            cards_row.addWidget(c)
        layout.addLayout(cards_row)

        # Progress bars card
        prog = QFrame()
        prog.setStyleSheet(f"QFrame{{background:{p['card']};border-radius:16px;border:1px solid {p['card_border']};}}")
        pfl = QVBoxLayout(prog); pfl.setContentsMargins(20,14,20,14); pfl.setSpacing(8)
        ph = QLabel("📊  Lesson Progress")
        ph.setFont(QFont(self.sys_font,12,QFont.Bold)); ph.setStyleSheet("background:transparent;")
        pfl.addWidget(ph)
        self.progress_bars: Dict[str, QProgressBar] = {}
        for lbl, key in [("Alphabets","alphabet"),("Core Words","word"),("Conversation","conversation")]:
            row = QHBoxLayout()
            lb = QLabel(lbl); lb.setFixedWidth(110); lb.setStyleSheet("background:transparent;color:#A8C4CF;font-size:13px;font-weight:600;")
            pb = QProgressBar(); pb.setRange(0,100); pb.setValue(0); pb.setFixedHeight(10)
            self.progress_bars[key] = pb
            row.addWidget(lb); row.addWidget(pb)
            pfl.addLayout(row)
        layout.addWidget(prog)
        layout.addStretch()
        return page



    def _build_study_page(self) -> QWidget:
        p = self._palette()
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 20, 28, 20)
        layout.setSpacing(14)

        ttl = QLabel("📖  Study Gestures")
        ttl.setFont(QFont(self.sys_font, 17, QFont.Bold))
        layout.addWidget(ttl)

        split = QHBoxLayout(); split.setSpacing(16)

        # Left column: gesture list
        left = QVBoxLayout(); left.setSpacing(6)
        ll = QLabel("Choose a gesture:")
        ll.setStyleSheet("color:#C4D8E0; font-size:13px; font-weight:700;")
        self.study_gesture_list = QListWidget()
        self.study_gesture_list.setMinimumWidth(230)
        self.study_gesture_list.itemSelectionChanged.connect(self.on_select_study_gesture)
        left.addWidget(ll); left.addWidget(self.study_gesture_list)
        split.addLayout(left, 2)

        # Right column: info card + reference
        right = QVBoxLayout(); right.setSpacing(10)

        info = QFrame()
        info.setFixedHeight(120)
        info.setStyleSheet(f"QFrame{{background:{p['card']};border-radius:16px;border:1px solid {p['card_border']};}}")
        il = QVBoxLayout(info); il.setContentsMargins(18,14,18,14); il.setSpacing(6)
        self.study_name = QLabel("Select a gesture to begin")
        self.study_name.setFont(QFont(self.sys_font, 15, QFont.Bold))
        self.study_name.setStyleSheet("background:transparent;")
        badges = QHBoxLayout()
        self.study_type = QLabel("—")
        self.study_type.setStyleSheet(
            "QLabel{background:#243B47;border-radius:8px;padding:3px 10px;color:#1CB0F6;font-size:11px;font-weight:700;}"
        )
        self.study_diff = QLabel("—")
        self.study_diff.setStyleSheet(
            "QLabel{background:#243B47;border-radius:8px;padding:3px 10px;color:#58CC02;font-size:11px;font-weight:700;}"
        )
        badges.addWidget(self.study_type); badges.addWidget(self.study_diff); badges.addStretch()
        il.addWidget(self.study_name); il.addLayout(badges)
        right.addWidget(info)

        self.study_desc = QLabel("Select a gesture to see its description.")
        self.study_desc.setWordWrap(True)
        self.study_desc.setStyleSheet("color:#7D9EA8; font-size:13px; background:transparent;")
        right.addWidget(self.study_desc)

        self.study_ref_label = QLabel("Select a gesture to see its reference")
        self.study_ref_label.setAlignment(Qt.AlignCenter)
        self.study_ref_label.setMinimumHeight(300)
        self.study_ref_label.setWordWrap(True)
        self.study_ref_label.setStyleSheet(
            f"QLabel{{background:{p['ref_bg']};border:2px solid {p['card_border']};border-radius:16px;padding:20px;color:{p['muted_text']};font-size:14px;}}"
        )
        right.addWidget(self.study_ref_label)

        bot = QHBoxLayout()
        self.study_ref_status = QLabel("Reference not available")
        self.study_ref_status.setStyleSheet("color:#3A5562; font-size:11px; background:transparent;")
        bot.addWidget(self.study_ref_status); bot.addStretch()
        self.btn_start_practice_from_study = QPushButton("🎯  Start Practice")
        self.btn_start_practice_from_study.setFixedHeight(44)
        self.btn_start_practice_from_study.setCursor(Qt.PointingHandCursor)
        self.btn_start_practice_from_study.clicked.connect(self.start_practice_from_study)
        bot.addWidget(self.btn_start_practice_from_study)
        right.addLayout(bot)

        split.addLayout(right, 3)
        layout.addLayout(split)
        return page



    def _build_practice_page(self) -> QWidget:
        p = self._palette()
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 20, 28, 20)
        layout.setSpacing(14)

        ttl = QLabel("🎯  Live Practice")
        ttl.setFont(QFont(self.sys_font, 17, QFont.Bold))
        layout.addWidget(ttl)

        sel_lbl = QLabel("Choose your target gesture:")
        sel_lbl.setStyleSheet("color:#C4D8E0; font-size:13px; font-weight:700;")
        layout.addWidget(sel_lbl)

        self.practice_target_list = QListWidget()
        self.practice_target_list.setMaximumHeight(110)
        self.practice_target_list.itemSelectionChanged.connect(self.on_select_practice_gesture)
        layout.addWidget(self.practice_target_list)

        split = QHBoxLayout(); split.setSpacing(16)

        # Left: reference panel
        lv = QVBoxLayout(); lv.setSpacing(6)
        rl = QLabel("Reference"); rl.setStyleSheet("color:#C4D8E0;font-size:13px;font-weight:700;")
        lv.addWidget(rl)
        self.practice_ref_label = QLabel("Reference not available")
        self.practice_ref_label.setAlignment(Qt.AlignCenter)
        self.practice_ref_label.setMinimumSize(300, 230)
        self.practice_ref_label.setStyleSheet(
            f"QLabel{{background:{p['ref_bg']};border:2px solid {p['card_border']};border-radius:16px;padding:12px;}}"
        )
        self.practice_ref_status = QLabel("")
        self.practice_ref_status.setStyleSheet("color:#3A5562;font-size:10px;background:transparent;")
        lv.addWidget(self.practice_ref_label); lv.addWidget(self.practice_ref_status)
        split.addLayout(lv, 4)

        # Right: camera + controls
        rv = QVBoxLayout(); rv.setSpacing(8)
        cl = QLabel("Camera Feed"); cl.setStyleSheet("color:#C4D8E0;font-size:13px;font-weight:700;")
        rv.addWidget(cl)
        self.camera_view = QLabel("Camera preview")
        self.camera_view.setMinimumSize(430, 290)
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet(
            f"QLabel{{background:{p['ref_bg']};border:2px solid {p['card_border']};border-radius:16px;padding:14px;}}"
        )
        rv.addWidget(self.camera_view)

        self.live_prediction = QLabel("Prediction: —")
        self.live_prediction.setStyleSheet(
            "QLabel{background:#1A2D35;border-radius:10px;padding:6px 14px;color:#FFFFFF;font-size:12px;font-weight:600;}"
        )
        rv.addWidget(self.live_prediction)

        self.practice_feedback = QLabel("Select a target and press Start to begin")
        self.practice_feedback.setStyleSheet("color:#7D9EA8;font-size:13px;font-weight:600;background:transparent;")
        self.practice_feedback.setWordWrap(True)
        rv.addWidget(self.practice_feedback)

        br = QHBoxLayout(); br.setSpacing(10)
        self.btn_start_camera = QPushButton("▶  Start Practice")
        self.btn_start_camera.setFixedHeight(46); self.btn_start_camera.setCursor(Qt.PointingHandCursor)
        self.btn_start_camera.clicked.connect(self.start_camera)

        self.btn_stop_camera = QPushButton("■  Stop Camera")
        self.btn_stop_camera.setFixedHeight(46); self.btn_stop_camera.setCursor(Qt.PointingHandCursor)
        self.btn_stop_camera.setProperty("danger", "true")
        self.btn_stop_camera.clicked.connect(self.stop_camera)

        self.btn_mark_attempt = QPushButton("✓  Record Attempt")
        self.btn_mark_attempt.setFixedHeight(46); self.btn_mark_attempt.setCursor(Qt.PointingHandCursor)
        self.btn_mark_attempt.setProperty("accent", "true")
        self.btn_mark_attempt.clicked.connect(self.record_current_attempt)

        br.addWidget(self.btn_start_camera, 2)
        br.addWidget(self.btn_stop_camera, 1)
        br.addWidget(self.btn_mark_attempt, 2)
        rv.addLayout(br)

        split.addLayout(rv, 5)
        layout.addLayout(split)
        return page



    def _build_quiz_page(self) -> QWidget:
        p = self._palette()
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 28, 40, 28)
        layout.setSpacing(18)

        ttl = QLabel("⚡  Quick Quiz")
        ttl.setFont(QFont(self.sys_font, 17, QFont.Bold))
        layout.addWidget(ttl)

        self.quiz_progress = QProgressBar()
        self.quiz_progress.setRange(0, 10)
        self.quiz_progress.setValue(0)
        self.quiz_progress.setFixedHeight(14)
        layout.addWidget(self.quiz_progress)

        self.quiz_state = QLabel("Score: 0 / 0")
        self.quiz_state.setStyleSheet("color:#FFC800; font-size:15px; font-weight:700; background:transparent;")
        layout.addWidget(self.quiz_state)

        # Gesture card
        g_card = QFrame()
        g_card.setFixedHeight(200)
        g_card.setStyleSheet(
            f"QFrame{{background:{p['card']};border-radius:24px;border:2px solid {p['card_border']};border-bottom:6px solid {p['deep_bg']};}}"
        )
        gcl = QVBoxLayout(g_card); gcl.setAlignment(Qt.AlignCenter)
        hint = QLabel("✋  SIGN THIS GESTURE")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color:#A8C4CF;font-size:13px;font-weight:700;letter-spacing:2px;background:transparent;")
        self.quiz_target = QLabel("Press Start Quiz")
        self.quiz_target.setAlignment(Qt.AlignCenter)
        self.quiz_target.setFont(QFont(self.sys_font, 34, QFont.Bold))
        self.quiz_target.setStyleSheet("color:#58CC02; background:transparent;")
        gcl.addWidget(hint); gcl.addWidget(self.quiz_target)
        layout.addWidget(g_card)

        btn_row = QHBoxLayout(); btn_row.setSpacing(14)
        btn_start = QPushButton("🎲  Start Quiz (10 Questions)")
        btn_start.setFixedHeight(50); btn_start.setCursor(Qt.PointingHandCursor)
        btn_start.setFont(QFont(self.sys_font, 13, QFont.Bold))
        btn_start.clicked.connect(self.start_quiz)

        btn_submit = QPushButton("✓  Submit Answer")
        btn_submit.setFixedHeight(50); btn_submit.setCursor(Qt.PointingHandCursor)
        btn_submit.setFont(QFont(self.sys_font, 13, QFont.Bold))
        btn_submit.setProperty("accent", "true")
        btn_submit.clicked.connect(self.submit_quiz_answer)

        btn_row.addWidget(btn_start); btn_row.addWidget(btn_submit)
        layout.addLayout(btn_row)
        layout.addStretch()
        return page



    def _build_analytics_page(self) -> QWidget:
        p = self._palette()
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 20, 28, 20)
        layout.setSpacing(16)

        ttl = QLabel("📊  Analytics")
        ttl.setFont(QFont(self.sys_font, 17, QFont.Bold))
        layout.addWidget(ttl)

        # Stat tiles
        tiles = QHBoxLayout(); tiles.setSpacing(12)
        tile_defs = [
            ("🎯", "Total Attempts", "total_attempts_lbl", "#58CC02"),
            ("✅", "Accuracy",       "accuracy_lbl",       "#1CB0F6"),
            ("💡", "Avg Confidence", "conf_lbl",           "#FFC800"),
            ("⚡", "Avg Latency",    "latency_lbl",        "#9B59B6"),
        ]
        for icon, label, attr, color in tile_defs:
            t = QFrame()
            t.setMinimumHeight(95)
            t.setStyleSheet(f"QFrame{{background:{p['card']};border-radius:16px;border:1px solid {p['card_border']};border-top:3px solid {color};}}")
            tl = QVBoxLayout(t); tl.setContentsMargins(16,12,16,12)
            hd = QLabel(f"{icon}  {label}"); hd.setStyleSheet("color:#7D9EA8;font-size:10px;font-weight:700;background:transparent;")
            vl = QLabel("—"); vl.setFont(QFont(self.sys_font,20,QFont.Bold))
            vl.setStyleSheet(f"color:{color};background:transparent;")
            setattr(self, attr, vl)
            tl.addWidget(hd); tl.addWidget(vl)
            tiles.addWidget(t)
        layout.addLayout(tiles)

        # Hidden compat label (updated by load_analytics)
        self.analytics_summary = QLabel(""); self.analytics_summary.hide()
        layout.addWidget(self.analytics_summary)

        tbl_lbl = QLabel("Recent Attempts")
        tbl_lbl.setStyleSheet("color:#C4D8E0;font-size:13px;font-weight:700;background:transparent;")
        layout.addWidget(tbl_lbl)

        self.analytics_table = QTableWidget(0, 6)
        self.analytics_table.setHorizontalHeaderLabels(["Time","Target","Predicted","Conf","Correct","Mode"])
        self.analytics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.analytics_table.setAlternatingRowColors(True)
        layout.addWidget(self.analytics_table)

        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        br = QPushButton("↻  Refresh"); br.setFixedHeight(40); br.setCursor(Qt.PointingHandCursor)
        br.clicked.connect(self.load_analytics)
        bc = QPushButton("🔲  Confusion Matrix"); bc.setFixedHeight(40); bc.setCursor(Qt.PointingHandCursor)
        bc.setProperty("accent","true"); bc.clicked.connect(self.load_confusion_matrix_view)
        btn_row.addWidget(br); btn_row.addWidget(bc); btn_row.addStretch()
        layout.addLayout(btn_row)

        self.confusion_note = QLabel("")
        self.confusion_note.setStyleSheet("color:#7D9EA8;font-size:11px;background:transparent;")
        layout.addWidget(self.confusion_note)

        self.confusion_table = QTableWidget(0, 0)
        self.confusion_table.setMaximumHeight(300)
        self.confusion_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.confusion_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.confusion_table)
        return page



    def _build_admin_page(self) -> QWidget:
        p = self._palette()
        outer = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        page_wrap = QVBoxLayout(outer)
        page_wrap.setContentsMargins(0,0,0,0)
        page_wrap.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(28, 20, 28, 24)
        layout.setSpacing(14)

        ttl = QLabel("⚙️  Admin Panel")
        ttl.setFont(QFont(self.sys_font, 17, QFont.Bold))
        layout.addWidget(ttl)

        self.admin_label = QLabel("")
        self.admin_label.setStyleSheet("color:#58CC02;font-size:12px;font-weight:600;background:transparent;")
        layout.addWidget(self.admin_label)

        ctrl = QHBoxLayout(); ctrl.setSpacing(10)
        for txt, slot, prop in [
            ("🌱 Reseed Data",        self.reseed,                        None),
            ("↻ Refresh Registry",    self.load_model_versions,           None),
            ("✓ Activate Selected",   self.activate_selected_model,       "accent"),
            ("🔄 Reload Predictor",   self.reload_predictor_from_registry,None),
        ]:
            b = QPushButton(txt); b.setFixedHeight(40); b.setCursor(Qt.PointingHandCursor)
            if prop: b.setProperty(prop, "true")
            b.clicked.connect(slot); ctrl.addWidget(b)
        layout.addLayout(ctrl)

        rl = QLabel("Model Registry")
        rl.setStyleSheet("color:#7D9EA8;font-size:11px;font-weight:700;background:transparent;")
        layout.addWidget(rl)
        self.model_table = QTableWidget(0, 6)
        self.model_table.setHorizontalHeaderLabels(["Model","Version","Framework","Artifact","Active","Trained At"])
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.model_table.setFixedHeight(170)
        layout.addWidget(self.model_table)

        # Register card
        reg_box = QFrame()
        reg_box.setStyleSheet(f"QFrame{{background:{p['card']};border-radius:16px;border:1px solid {p['card_border']};}}")
        rl2 = QVBoxLayout(reg_box); rl2.setContentsMargins(20,16,20,16); rl2.setSpacing(8)
        rl2.addWidget(QLabel("Register New Model Version"))

        self.reg_model_name = QComboBox(); self.reg_model_name.addItems(["static_mlp","dynamic_bigru"])
        self.reg_framework = QLineEdit("pytorch")
        self.reg_version_tag = QLineEdit(); self.reg_version_tag.setPlaceholderText("Version tag (optional)")
        self.reg_artifact_path = QLineEdit(); self.reg_artifact_path.setPlaceholderText("Artifact path")
        self.reg_label_map_path = QLineEdit("models/registry/label_map.json")
        self.reg_norm_stats_path = QLineEdit("models/registry/norm_stats.json")
        self.reg_activate = QCheckBox("Activate immediately"); self.reg_activate.setChecked(True)
        self.reg_metrics = QTextEdit('{"accuracy":0.0,"precision":0.0,"recall":0.0,"f1":0.0}')
        self.reg_metrics.setFixedHeight(68)

        for widget, label in [
            (self.reg_model_name,"Model Name"),(self.reg_framework,"Framework"),
            (self.reg_version_tag,"Version Tag"),(self.reg_artifact_path,"Artifact Path"),
            (self.reg_label_map_path,"Label Map Path"),(self.reg_norm_stats_path,"Norm Stats Path"),
        ]:
            rl2.addWidget(QLabel(label)); rl2.addWidget(widget)

        browse_row = QHBoxLayout(); browse_row.setSpacing(8)
        for txt, le in [("Browse Artifact",self.reg_artifact_path),
                        ("Browse Label Map",self.reg_label_map_path),
                        ("Browse Norm Stats",self.reg_norm_stats_path)]:
            b = QPushButton(txt); b.setFixedHeight(34); b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet("QPushButton{font-size:12px;padding:6px 10px;border-radius:10px;border-bottom:3px solid #46A302;}")
            b.clicked.connect(lambda _=False, l=le: self._pick_file_into(l))
            browse_row.addWidget(b)
        rl2.addLayout(browse_row)

        rl2.addWidget(QLabel("Metrics JSON")); rl2.addWidget(self.reg_metrics)
        rl2.addWidget(self.reg_activate)

        action_row = QHBoxLayout(); action_row.setSpacing(8)
        for txt, slot, prop in [
            ("Prefill From Active",           self.prefill_model_paths,             None),
            ("Register Version",              self.register_model_version_from_ui,  None),
            ("Rollback Family",               self.rollback_model_family_from_ui,   "danger"),
        ]:
            b = QPushButton(txt); b.setFixedHeight(40); b.setCursor(Qt.PointingHandCursor)
            if prop: b.setProperty(prop,"true")
            b.clicked.connect(slot); action_row.addWidget(b)
        rl2.addLayout(action_row)
        layout.addWidget(reg_box)
        layout.addStretch()
        return outer



    def _start_reference_threads(self):
        self.study_ref_thread = ReferenceVideoThread()
        self.study_ref_thread.frame_signal.connect(self._update_study_ref_frame)
        self.study_ref_thread.status_signal.connect(self.study_ref_status.setText)
        self.study_ref_thread.start()

        self.practice_ref_thread = ReferenceVideoThread()
        self.practice_ref_thread.frame_signal.connect(self._update_practice_ref_frame)
        self.practice_ref_thread.status_signal.connect(self.practice_ref_status.setText)
        self.practice_ref_thread.start()

    def _on_stack_changed(self, idx: int):
        if idx != self.IDX_STUDY:
            self._flush_study_timer()

    def navigate_to(self, idx: int):
        self.stack.setCurrentIndex(idx)
        # Update active nav button highlight
        nav_map = {
            self.IDX_DASH: "Home",
            self.IDX_STUDY: "Study",
            self.IDX_PRACTICE: "Practice",
            self.IDX_QUIZ: "Quiz",
            self.IDX_ANALYTICS: "Analytics",
            self.IDX_ADMIN: "Admin",
        }
        for name, btn in self.nav_buttons.items():
            is_active = (nav_map.get(idx) == name)
            btn.setProperty("active", "true" if is_active else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _apply_environment_status(self):
        self._set_indicator(self.env_mp,     "MediaPipe",   self.env_status.get("mediapipe", False))
        self._set_indicator(self.env_torch,  "Torch",       self.env_status.get("torch", False))
        self._set_indicator(self.env_cam,    "Camera",      self.env_status.get("camera", False))
        self._set_indicator(self.env_static, "StaticModel", self.env_status.get("static_model_loaded", False))
        self._set_indicator(self.env_dynamic,"DynamicModel",self.env_status.get("dynamic_model_loaded", False))

        can_practice = self.env_status.get("mediapipe", False) and self.env_status.get("camera", False)
        if hasattr(self, "btn_start_camera"):
            self.btn_start_camera.setEnabled(can_practice)
            if not can_practice:
                self.btn_start_camera.setText("▶  Start Practice (Unavailable)")
                self.practice_feedback.setText("Practice disabled: MediaPipe and camera are required.")
            else:
                self.btn_start_camera.setText("▶  Start Practice")

    @staticmethod
    def _set_indicator(label: QLabel, name: str, ok: bool):
        dot = "●"
        text_color = "#34d399" if ok else "#f87171"
        bg = "rgba(52,211,153,0.12)" if ok else "rgba(248,113,113,0.12)"
        label.setText(f"{dot}  {name}")
        label.setStyleSheet(
            f"QLabel {{ background:{bg}; border-radius:11px; padding:2px 10px; "
            f"font-size:11px; font-weight:600; color:{text_color}; }}"
        )



    def refresh_environment_status(self):
        self.env_status = check_environment(self.config)
        self._apply_environment_status()

    def _switch_theme(self, mode: str) -> None:
        """Switch between dark and light themes, rebuilding the UI to apply inline styles."""
        if mode == self._current_theme_mode:
            return
        self._current_theme_mode = mode
        # Preserve session state across rebuild
        saved_session = self.session
        self.stop_camera()
        if self.study_ref_thread:
            self.study_ref_thread.stop()
            self.study_ref_thread = None
        if self.practice_ref_thread:
            self.practice_ref_thread.stop()
            self.practice_ref_thread = None

        # Tear down old central widget
        old = self.centralWidget()
        if old:
            old.setParent(None)

        # Rebuild with new theme
        self._last_qimage = None
        self._last_study_ref = None
        self._last_practice_ref = None
        self.setStyleSheet(self._theme(mode))
        self._build_ui()
        self._start_reference_threads()
        self._apply_environment_status()

        # Restore session state
        self.session = saved_session
        if self.session.is_authenticated():
            self.sidebar.setVisible(True)
            self.welcome.setText(f"Welcome, {self.session.full_name}")
            self.refresh_after_login()
            self.navigate_to(self.IDX_DASH)
        else:
            self.navigate_to(self.IDX_LOGIN)



    def handle_login(self):
        email = self.login_email.text().strip().lower()
        password = self.login_password.text()
        row = self.db.get_user_by_email(email)
        if not row or not verify_password(password, row["password_hash"]):
            QMessageBox.warning(self, "Login Failed", "Invalid credentials")
            return
        self.session.user_id = row["user_id"]
        self.session.email = row["email"]
        self.session.full_name = row["full_name"]
        self.session.role = row["role"]
        self.session.token = create_access_token(row["user_id"])

        self.sidebar.setVisible(True)
        self.navigate_to(self.IDX_DASH)
        self.welcome.setText(f"Welcome, {self.session.full_name}")
        self.refresh_after_login()

    def refresh_after_login(self):
        gestures = [dict(g) for g in self.db.get_gestures()]
        self.gesture_rows = gestures
        self.lesson_summary.setText(f"Lessons ready: 3 | Gestures loaded: {len(gestures)}")

        self.study_gesture_list.clear()
        self.practice_target_list.clear()
        for g in gestures:
            text = f"{g['display_name']}  [{g['gesture_mode']}]"
            self.study_gesture_list.addItem(text)
            self.practice_target_list.addItem(text)

        self.load_analytics()
        self.load_model_versions()

    def logout(self):
        self.stop_camera()
        self._flush_study_timer()
        self.session = SessionState()
        self.sidebar.setVisible(False)
        self.navigate_to(self.IDX_LOGIN)

    def _difficulty_for(self, gesture: Dict[str, object]) -> str:
        mode = str(gesture.get("gesture_mode", "static"))
        category = str(gesture.get("category", ""))
        if mode == "dynamic":
            return "Intermediate"
        if category in {"emergency", "conversation"}:
            return "Intermediate"
        return "Beginner"

    def _show_text_reference(self, label: QLabel, gesture_name: str, ref_info: Dict[str, str]) -> None:
        """Show a reference image + text description when no video is available."""
        # Try to show a reference image
        img_path = get_reference_image_path(gesture_name)
        if img_path:
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet(
                    "background:#111827; border:1px solid #334155; border-radius:14px; padding:8px;"
                )
                return

        # Fallback: styled HTML text
        desc = ref_info.get("description", "")
        tips = ref_info.get("tips", "")
        difficulty = ref_info.get("difficulty", "")
        hands = ref_info.get("hands", ref_info.get("hand", ""))

        parts = [f"<b style='color:#10b981; font-size:18px;'>{gesture_name}</b>"]
        if difficulty:
            parts.append(f"<span style='color:#94a3b8; font-size:12px;'>Difficulty: {difficulty.capitalize()}</span>")
        if hands:
            parts.append(f"<span style='color:#94a3b8; font-size:12px;'>Hand(s): {hands}</span>")
        parts.append("")
        if desc:
            parts.append(f"<p style='color:#e2e8f0; font-size:14px; line-height:1.6;'>{desc}</p>")
        if tips:
            parts.append(f"<p style='color:#fbbf24; font-size:13px;'>💡 {tips}</p>")
        if not desc and not tips:
            parts.append("<p style='color:#94a3b8;'>No reference available for this gesture.</p>")

        label.setText("<br>".join(parts))
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        label.setWordWrap(True)
        label.setStyleSheet(
            "background:#111827; border:1px solid #334155; border-radius:14px; "
            "padding:18px; font-family:sans-serif;"
        )

    def on_select_study_gesture(self):
        idx = self.study_gesture_list.currentRow()
        if idx < 0 or idx >= len(self.gesture_rows):
            return
        self._flush_study_timer()
        self.selected_gesture = dict(self.gesture_rows[idx])
        g = self.selected_gesture
        ref_info = get_gesture_reference(str(g["display_name"]))
        self.study_name.setText(f"Gesture: {g['display_name']}")
        self.study_type.setText(f"Type: {g['gesture_mode']}")
        diff = ref_info.get("difficulty") or self._difficulty_for(g)
        self.study_diff.setText(f"Difficulty: {diff.capitalize()}")
        desc = ref_info.get("description") or str(g.get("description") or "No description available.")
        tips = ref_info.get("tips", "")
        full_desc = desc
        if tips:
            full_desc += f"\n\n💡 Tips: {tips}"
        self.study_desc.setText(full_desc)

        media_path = get_media_path(str(g["display_name"]))
        self.study_ref_thread.set_media(media_path)
        if media_path is None:
            self._show_text_reference(self.study_ref_label, str(g["display_name"]), ref_info)
        self._study_gesture_id = str(g["gesture_id"])
        self._study_started_at = time.time()

    def start_practice_from_study(self):
        if not self.selected_gesture:
            QMessageBox.information(self, "Study", "Select a gesture first")
            return
        target_id = self.selected_gesture["gesture_id"]
        for i, row in enumerate(self.gesture_rows):
            if row["gesture_id"] == target_id:
                self.practice_target_list.setCurrentRow(i)
                break
        self.navigate_to(self.IDX_PRACTICE)

    def _flush_study_timer(self):
        if not self.session.is_authenticated() or not self._study_gesture_id or self._study_started_at is None:
            self._study_started_at = None
            return
        elapsed = int(max(time.time() - self._study_started_at, 0))
        if elapsed > 0:
            self.db.record_study_session(self.session.user_id, self._study_gesture_id, elapsed)
        self._study_started_at = None

    def on_select_practice_gesture(self):
        idx = self.practice_target_list.currentRow()
        if idx < 0 or idx >= len(self.gesture_rows):
            return
        row = dict(self.gesture_rows[idx])
        self.selected_gesture = row

        media_path = get_media_path(str(row["display_name"]))
        self.practice_ref_thread.set_media(media_path)
        if media_path is None:
            ref_info = get_gesture_reference(str(row["display_name"]))
            self._show_text_reference(self.practice_ref_label, str(row["display_name"]), ref_info)

        self.practice_feedback.setText(f"Target selected: {row['display_name']} [{row['gesture_mode']}]")
        if self.inference_thread:
            self.inference_thread.set_target(str(row["display_name"]))
            self.inference_thread.set_target_mode(str(row["gesture_mode"]))

    def start_camera(self):
        if not (self.env_status.get("mediapipe") and self.env_status.get("camera")):
            QMessageBox.warning(self, "Unavailable", "Cannot start practice. MediaPipe or camera is unavailable.")
            return
        if self.inference_thread and self.inference_thread.isRunning():
            return
        if not self.selected_gesture:
            QMessageBox.information(self, "Select Target", "Choose a gesture first.")
            return

        self.inference_thread = InferenceThread(self.predictor, self.env_status)
        self.inference_thread.set_target(str(self.selected_gesture["display_name"]))
        self.inference_thread.set_target_mode(str(self.selected_gesture["gesture_mode"]))
        self.inference_thread.frame_signal.connect(self.update_camera_view)
        self.inference_thread.result_signal.connect(self.update_result)
        self.inference_thread.start()

    def stop_camera(self):
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread = None

    def update_camera_view(self, qimg: QImage):
        self._last_qimage = qimg
        pix = QPixmap.fromImage(qimg).scaled(self.camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_view.setPixmap(pix)

    def _update_study_ref_frame(self, qimg: QImage):
        self._last_study_ref = qimg
        pix = QPixmap.fromImage(qimg).scaled(self.study_ref_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.study_ref_label.setPixmap(pix)

    def _update_practice_ref_frame(self, qimg: QImage):
        self._last_practice_ref = qimg
        pix = QPixmap.fromImage(qimg).scaled(self.practice_ref_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.practice_ref_label.setPixmap(pix)

    def update_result(self, result: dict):
        self.current_result = result
        status = result.get("status", "-")
        label = result.get("label", "-")
        model_used = result.get("model_used", "-")
        conf = float(result.get("confidence", 0.0))
        fps = float(result.get("fps", 0.0))
        stable = bool(result.get("stable", False))
        warn = result.get("perf_warning", "")

        self.live_prediction.setText(
            f"Target: {self.selected_gesture['display_name'] if self.selected_gesture else '-'} | "
            f"Model: {model_used} | Confidence: {conf:.2f} | FPS: {fps:.1f}"
        )

        if warn:
            self.practice_feedback.setText(warn)
            self.practice_feedback.setStyleSheet("color:#fbbf24; font-weight:600;")
            return

        if self.selected_gesture and status in {"ok", "uncertain"}:
            target = self.selected_gesture["display_name"]
            if stable and label == target:
                self.practice_feedback.setText("Correct and stable")
                self.practice_feedback.setStyleSheet("color:#34d399; font-weight:700;")
            elif status == "uncertain":
                self.practice_feedback.setText("Hold steady. Confidence not stable yet.")
                self.practice_feedback.setStyleSheet("color:#fbbf24; font-weight:600;")
            else:
                self.practice_feedback.setText(f"Incorrect. Predicted {label}, target {target}")
                self.practice_feedback.setStyleSheet("color:#f87171; font-weight:700;")
        elif status in {"mediapipe_unavailable", "dynamic_model_unavailable"}:
            env_text = (
                f"Environment: MediaPipe: {'❌' if not self.env_status.get('mediapipe') else '✅'} | "
                f"Torch: {'✅' if self.env_status.get('torch') else '❌'} | "
                f"Camera: {'✅' if self.env_status.get('camera') else '❌'}"
            )
            self.practice_feedback.setText(env_text)
            self.practice_feedback.setStyleSheet("color:#f87171; font-weight:700;")
        elif status == "no_hand":
            self.practice_feedback.setText("No hand detected. Place hand in frame.")
            self.practice_feedback.setStyleSheet("color:#fbbf24; font-weight:600;")
        elif status == "camera_error":
            self.practice_feedback.setText("Camera error. Restart camera.")
            self.practice_feedback.setStyleSheet("color:#f87171; font-weight:700;")

    def record_current_attempt(self):
        if not self.session.is_authenticated() or not self.selected_gesture or not self.current_result:
            return
        predicted = str(self.current_result.get("label", "UNKNOWN"))
        conf = float(self.current_result.get("confidence", 0.0))
        latency = int(self.current_result.get("latency_ms", 0))
        fps = float(self.current_result.get("fps", 0.0))
        stable = bool(self.current_result.get("stable", False))
        is_correct = stable and predicted == self.selected_gesture["display_name"]

        self.db.record_attempt(
            user_id=self.session.user_id,
            gesture_id=self.selected_gesture["gesture_id"],
            target_gesture_id=self.selected_gesture["gesture_id"],
            predicted_label=predicted,
            confidence=conf,
            is_correct=is_correct,
            latency_ms=latency,
            fps=fps,
            attempt_mode="practice",
        )
        self.practice_feedback.setText(f"Attempt recorded | correct={is_correct}")
        self.load_analytics()

    def start_quiz(self):
        self.quiz_queue = [dict(r) for r in self.db.get_random_gestures(limit=10)]
        self.quiz_index = 0
        self.quiz_score = 0
        self._set_quiz_target()

    def _set_quiz_target(self):
        if self.quiz_index >= len(self.quiz_queue):
            self.quiz_target.setText("Quiz Complete! 🎉")
            self.quiz_state.setText(f"Final score: {self.quiz_score} / {len(self.quiz_queue)}")
            if hasattr(self, "quiz_progress"):
                self.quiz_progress.setValue(10)
            return
        target = self.quiz_queue[self.quiz_index]
        self.quiz_target.setText(str(target["display_name"]))
        self.quiz_state.setText(f"Score: {self.quiz_score} / {self.quiz_index}")
        if hasattr(self, "quiz_progress"):
            self.quiz_progress.setValue(self.quiz_index)



    def submit_quiz_answer(self):
        if self.quiz_index >= len(self.quiz_queue) or not self.current_result:
            return
        target = self.quiz_queue[self.quiz_index]
        pred = str(self.current_result.get("label", "UNKNOWN"))
        conf = float(self.current_result.get("confidence", 0.0))
        stable = bool(self.current_result.get("stable", False))
        is_correct = stable and pred == target["display_name"] and conf >= 0.65
        self.quiz_score += int(is_correct)

        self.db.record_attempt(
            user_id=self.session.user_id,
            gesture_id=target["gesture_id"],
            target_gesture_id=target["gesture_id"],
            predicted_label=pred,
            confidence=conf,
            is_correct=is_correct,
            latency_ms=int(self.current_result.get("latency_ms", 0)),
            fps=float(self.current_result.get("fps", 0.0)),
            attempt_mode="quiz",
        )

        self.quiz_index += 1
        self._set_quiz_target()
        self.load_analytics()

    def load_analytics(self):
        if not self.session.is_authenticated():
            return
        summary = self.db.get_analytics_summary(self.session.user_id)
        # Update hidden compat label
        self.analytics_summary.setText(
            f"Attempts: {int(summary['total_attempts'])} | Accuracy: {summary['accuracy']:.2f} | "
            f"Avg Conf: {summary['avg_confidence']:.2f} | Avg Latency: {summary['avg_latency_ms']:.1f}ms"
        )
        # Update stat tiles
        if hasattr(self, "total_attempts_lbl"):
            self.total_attempts_lbl.setText(str(int(summary["total_attempts"])))
        if hasattr(self, "accuracy_lbl"):
            self.accuracy_lbl.setText(f"{summary['accuracy']:.1%}")
        if hasattr(self, "conf_lbl"):
            self.conf_lbl.setText(f"{summary['avg_confidence']:.2f}")
        if hasattr(self, "latency_lbl"):
            self.latency_lbl.setText(f"{summary['avg_latency_ms']:.0f} ms")

        rows = self.db.get_user_attempts(self.session.user_id, limit=120)
        self.analytics_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [
                r["created_at"][:19],
                r["target_name"],
                r["predicted_label"],
                f"{r['confidence']:.2f}",
                "Yes" if r["is_correct"] else "No",
                r["attempt_mode"],
            ]
            for c, v in enumerate(vals):
                self.analytics_table.setItem(i, c, QTableWidgetItem(str(v)))

        progress_rows = self.db.get_user_progress(self.session.user_id)
        if progress_rows:
            txt = " | ".join([f"{r['title']}: {r['accuracy']:.2f} ({r['attempts_count']} attempts)" for r in progress_rows])
            self.progress_summary.setText(f"Progress: {txt}")
            # Update progress bars on dashboard
            if hasattr(self, "progress_bars"):
                for r in progress_rows:
                    key = r["lesson_type"]
                    if key in self.progress_bars:
                        self.progress_bars[key].setValue(int(r["accuracy"] * 100))



    def load_confusion_matrix_view(self):
        if not self.session.is_authenticated():
            return
        attempts = self.db.get_user_attempts(self.session.user_id, limit=5000)
        if not attempts:
            self.confusion_note.setText("No attempts available to build confusion matrix.")
            self.confusion_table.setRowCount(0)
            self.confusion_table.setColumnCount(0)
            return

        labels = sorted({str(a["target_name"]) for a in attempts} | {str(a["predicted_label"]) for a in attempts})
        max_labels = 25
        if len(labels) > max_labels:
            freq = {}
            for a in attempts:
                freq[a["target_name"]] = freq.get(a["target_name"], 0) + 1
            labels = [x for x, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:max_labels]]
            self.confusion_note.setText("Showing top 25 target classes by frequency.")
        else:
            self.confusion_note.setText(f"Showing {len(labels)} classes.")

        idx = {name: i for i, name in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=np.int32)
        for a in attempts:
            t = str(a["target_name"])
            p = str(a["predicted_label"])
            if t in idx and p in idx:
                cm[idx[t], idx[p]] += 1

        self.confusion_table.setRowCount(len(labels))
        self.confusion_table.setColumnCount(len(labels))
        self.confusion_table.setVerticalHeaderLabels(labels)
        self.confusion_table.setHorizontalHeaderLabels(labels)
        row_totals = cm.sum(axis=1, keepdims=True).astype(np.float32)
        row_totals[row_totals == 0] = 1.0
        cm_norm = cm / row_totals

        for r in range(len(labels)):
            for c in range(len(labels)):
                val = int(cm[r, c])
                item = QTableWidgetItem(str(val))
                v = float(cm_norm[r, c])
                green = int(60 + 150 * v)
                red = int(30 + 120 * (1.0 - v))
                item.setBackground(QColor(red, green, 70))
                self.confusion_table.setItem(r, c, item)

    def reseed(self):
        if self.session.role != "admin":
            QMessageBox.warning(self, "Forbidden", "Admin role required")
            return
        self.db.seed_core_data()
        self.refresh_after_login()
        self.admin_label.setText("Database reseeded successfully")

    def load_model_versions(self):
        rows = self.db.list_model_versions()
        self.model_table.setRowCount(len(rows))
        self._model_rows = [dict(r) for r in rows]
        for i, row in enumerate(rows):
            vals = [
                row["model_name"],
                row["version_tag"],
                row["framework"],
                row["artifact_path"],
                "Yes" if row["is_active"] else "No",
                row["trained_at"][:19],
            ]
            for c, v in enumerate(vals):
                self.model_table.setItem(i, c, QTableWidgetItem(str(v)))

    def activate_selected_model(self):
        if self.session.role != "admin":
            QMessageBox.warning(self, "Forbidden", "Admin role required")
            return
        row = self.model_table.currentRow()
        if row < 0 or row >= len(getattr(self, "_model_rows", [])):
            QMessageBox.information(self, "Select Model", "Choose a model row first.")
            return
        model_row = self._model_rows[row]
        ok = self.db.activate_model_version(model_row["model_version_id"])
        if not ok:
            QMessageBox.warning(self, "Activation Failed", "Could not activate model version.")
            return
        self.load_model_versions()

    def reload_predictor_from_registry(self):
        active = self.db.get_active_model_paths()
        model_cfg = self.config.setdefault("model", {})
        changed = []
        for key in ("static_model_path", "dynamic_model_path", "label_map_path", "norm_stats_path"):
            if key in active and model_cfg.get(key) != active[key]:
                model_cfg[key] = active[key]
                changed.append(key)
        if changed:
            save_config(self.config, "config/app.yaml")

        was_running = self.inference_thread is not None and self.inference_thread.isRunning()
        self.stop_camera()
        self.predictor = GesturePredictor(self.config)
        self.refresh_environment_status()
        if was_running:
            self.start_camera()
        self.admin_label.setText("Predictor reloaded" + (f" | updated: {', '.join(changed)}" if changed else ""))

    def _pick_file_into(self, line_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if path:
            line_edit.setText(path)

    def prefill_model_paths(self):
        active = self.db.get_active_model_paths()
        model_name = self.reg_model_name.currentText()
        if model_name == "static_mlp":
            self.reg_artifact_path.setText(active.get("static_model_path", self.reg_artifact_path.text()))
            self.reg_norm_stats_path.setText(active.get("norm_stats_path", self.reg_norm_stats_path.text()))
            self.reg_label_map_path.setText(active.get("label_map_path", self.reg_label_map_path.text()))
        elif model_name == "dynamic_bigru":
            self.reg_artifact_path.setText(active.get("dynamic_model_path", self.reg_artifact_path.text()))
            self.reg_norm_stats_path.setText("models/registry/dynamic_norm_stats.json")
            self.reg_label_map_path.setText(active.get("label_map_path", self.reg_label_map_path.text()))

    def register_model_version_from_ui(self):
        if self.session.role != "admin":
            QMessageBox.warning(self, "Forbidden", "Admin role required")
            return
        model_name = self.reg_model_name.currentText().strip()
        framework = self.reg_framework.text().strip() or "pytorch"
        version_tag = self.reg_version_tag.text().strip() or None
        artifact_path = self.reg_artifact_path.text().strip()
        label_map_path = self.reg_label_map_path.text().strip()
        norm_stats_path = self.reg_norm_stats_path.text().strip()

        if not artifact_path:
            QMessageBox.warning(self, "Invalid Input", "Artifact path is required")
            return
        try:
            metrics = json.loads(self.reg_metrics.toPlainText().strip() or "{}")
        except Exception:
            QMessageBox.warning(self, "Invalid Input", "Metrics JSON is invalid")
            return

        self.db.register_model_version(
            model_name=model_name,
            framework=framework,
            artifact_path=artifact_path,
            label_map_path=label_map_path or "models/registry/label_map.json",
            norm_stats_path=norm_stats_path or "models/registry/norm_stats.json",
            metrics=metrics if isinstance(metrics, dict) else {},
            activate=bool(self.reg_activate.isChecked()),
            version_tag=version_tag,
        )
        self.load_model_versions()
        if self.reg_activate.isChecked():
            self.reload_predictor_from_registry()

    def rollback_model_family_from_ui(self):
        if self.session.role != "admin":
            QMessageBox.warning(self, "Forbidden", "Admin role required")
            return
        model_name = self.reg_model_name.currentText().strip()
        ok = self.db.rollback_model_family(model_name)
        if not ok:
            QMessageBox.information(self, "Rollback", "Rollback not possible (need at least two versions).")
            return
        self.load_model_versions()
        self.reload_predictor_from_registry()

    def closeEvent(self, event):
        self.stop_camera()
        self._flush_study_timer()
        if self.study_ref_thread:
            self.study_ref_thread.stop()
        if self.practice_ref_thread:
            self.practice_ref_thread.stop()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_qimage is not None:
            self.update_camera_view(self._last_qimage)
        if self._last_study_ref is not None:
            self._update_study_ref_frame(self._last_study_ref)
        if self._last_practice_ref is not None:
            self._update_practice_ref_frame(self._last_practice_ref)
