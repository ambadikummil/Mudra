"""Backward-compatible launcher for MUDRA.

This keeps the original entrypoint name while running the modular v2 app.
"""

import os
from pathlib import Path

# Always run with CWD = this file's directory so all relative paths resolve correctly.
os.chdir(Path(__file__).parent)

import torch
from ui.app import main


if __name__ == "__main__":
    raise SystemExit(main())
