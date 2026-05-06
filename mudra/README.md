# MUDRA - Interactive ISL Learning with Realtime Gesture Feedback

## What is implemented
- Modular architecture (UI, inference, training, backend API, database, tests)
- 26 alphabets + 100-word catalog seed-ready
- Realtime camera loop with MediaPipe landmarks and prediction smoothing
- Login/auth (local), practice, quiz, analytics, and admin reseed control
- Reproducible training/evaluation scripts with metrics and confusion matrix export
- Admin model registry table with active-version switching and predictor hot-reload
- Analytics confusion matrix heatmap view (per-user attempt data)
- Admin model upload/register workflow with version metadata, optional immediate activation, and rollback by model family

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m database.seed.seed_database
python mudra_app.py
```

## Demo credentials
- `demo@mudra.local / demo123`
- `admin@mudra.local / admin123`

## Run backend API
```bash
python -m backend.run_api
```

## Train static model
```bash
python -m training.features.extract_landmarks --input data/raw
python -m training.datasets.build_dataset
python -m training.trainers.train_static
python -m training.evaluation.evaluate
python -m training.trainers.train_dynamic
python -m training.evaluation.evaluate_dynamic
```

## Notes
- If MediaPipe/Torch model files are unavailable, app falls back to stable rule-based predictions for demo continuity.
- Dynamic model is used live for motion-based signs when selected target is `gesture_mode=dynamic`.
- Analytics page now includes a confusion-matrix view from recorded attempts.

## Phase 5 Deployment Ops
```bash
# preflight checks
python scripts/preflight.py

# start API + UI together
bash scripts/start_local.sh
```

### New API endpoints
- `GET /health`
- `GET /models`
- `POST /models/register` (admin)
- `POST /models/{model_version_id}/activate` (admin)
- `POST /models/{model_name}/rollback` (admin)

## Phase 6 Reproducible Env + CI/CD
```bash
# create .venv, install runtime/dev deps, seed DB, run preflight
bash scripts/bootstrap.sh

# deterministic local CI gate
make ci
```

CI workflow:
- GitHub Actions: [.github/workflows/ci.yml](/Users/joakimmanoj/Downloads/mudra/.github/workflows/ci.yml)
- Checks: `ruff`, DB seed, `scripts/ci_checks.py`, pytest suite

## Phase 7 Packaging + Release Engineering
```bash
# 1) migrate schema + release SQL patches
make migrate

# 2) backup current DB before release
make backup

# 3) build desktop package (requires pyinstaller in dev env)
make package

# 4) create signed-hash release bundle (artifacts + deploy configs)
make release TAG=v0.1.0
```

Deployment manifests:
- Docker API image: [Dockerfile.api](/Users/joakimmanoj/Downloads/mudra/deploy/Dockerfile.api)
- Docker Compose: [docker-compose.yml](/Users/joakimmanoj/Downloads/mudra/deploy/docker-compose.yml)
- systemd service: [mudra-api.service](/Users/joakimmanoj/Downloads/mudra/deploy/systemd/mudra-api.service)
- env template: [env.example](/Users/joakimmanoj/Downloads/mudra/deploy/env.example)

Database release safety:
- Migrations runner: [migrate.py](/Users/joakimmanoj/Downloads/mudra/scripts/migrate.py)
- Backup script: [backup_db.py](/Users/joakimmanoj/Downloads/mudra/scripts/backup_db.py)
- Restore script: [restore_db.py](/Users/joakimmanoj/Downloads/mudra/scripts/restore_db.py)

## Phase 8 Production Readiness + Launch
```bash
# security hardening checks
make security

# API load test (API should be running)
make loadtest

# health monitor loop (12 checks by default)
make monitor
```

Runbooks:
- Launch checklist: [launch_checklist.md](/Users/joakimmanoj/Downloads/mudra/docs/runbooks/launch_checklist.md)
- UAT test plan: [uat_test_plan.md](/Users/joakimmanoj/Downloads/mudra/docs/runbooks/uat_test_plan.md)

New API readiness endpoint:
- `GET /ready`

## Learning Runtime Upgrades
- Environment health checks: [environment_check.py](/Users/joakimmanoj/Downloads/mudra/utils/environment_check.py)
  - checks `mediapipe`, `torch`, camera access, and model artifact availability
  - practice start is automatically disabled if MediaPipe/camera is unavailable
- Gesture reference media mapping: [gesture_media_mapper.py](/Users/joakimmanoj/Downloads/mudra/utils/gesture_media_mapper.py)
  - asset root: `data/assets/gestures/`
  - expected examples:
    - `data/assets/gestures/alphabets/A.mp4`
    - `data/assets/gestures/words/hello.mp4`
- Dedicated Study Mode (separate from Practice)
  - left: gesture metadata (name/type/description/difficulty)
  - right: looping reference media
  - `Start Practice` button switches to Practice without auto-starting camera
- Side-by-side Practice layout
  - left: reference loop panel
  - right: live camera with target/model/fps/confidence overlay
- Stability filters
  - static: must remain same for 1.5s
  - dynamic: confidence must stay above threshold for consecutive frames
- FPS guard
  - inference processing capped to ~18 FPS
  - warning shown when measured FPS < 15
