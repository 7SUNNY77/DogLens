# DogLens — release checklist

Statuses: **PASS** · **FAIL** · **NOT_RUN** · **NEEDS_OWNER**. A passing `AppTest` run, a healthy
server endpoint or an automated browser run is *not* the owner's manual acceptance.

Last update: 2026-10-13. Environment: Windows 11 x86_64, Intel Core Ultra 7 155H, Python 3.11.15,
uv 0.11.7, torch 2.14.1+cpu.

## 1. Automated checks

| # | Check | Status | Evidence |
|---|---|---|---|
| A1 | `uv run ruff check .` | PASS | All checks passed |
| A2 | `uv run ruff format --check .` | PASS | 66 files already formatted |
| A3 | `uv run pytest` (no dataset/bundle/network) | PASS | 301 passed, 14 deselected |
| A4 | `uv run pytest -m integration` (real dataset + bundle) | PASS | 14 passed |
| A5 | Bundle packaging and offline verification | PASS | see §5 below |
| A6 | Clean copy launch (see §6) | PASS | fresh venv from `uv.lock`, bundle from the archive, offline inference, tests, Streamlit start |
| A7 | Local Markdown links | PASS | 30 local links in 16 files, 0 broken. External links: only the two uv docs URLs (HTTP 200); no other external URLs checked |
| A8 | Git hygiene | PASS | 97 files to commit, 7.5 MB, largest 2.1 MB (manifest); no dataset images, weights, archives, logits, SQLite, logs or absolute local paths |
| A9 | Automated real-browser UI run (headless Chrome via DevTools protocol, clean copy) | PASS | 30/30 checks at 1366×900 and 390×844 (scratch script, not part of the repo). One earlier attempt failed only because of a race in the script (file input queried before render) |
| A10 | GitHub Actions workflow on GitHub | PASS | [run 38462373068](https://github.com/7SUNNY77/DogLens/actions/runs/38462373068) on `ed06bd8`: ubuntu-latest and windows-latest — install from lockfile, ruff, format, tests all green |
| A11 | Dependency installation from the internet | PASS | CI runners installed everything from `uv.lock` without a cache (Ubuntu and Windows) |
| A12 | Platforms other than Windows | PASS for Linux (CI: install + default tests); app launch on Linux and anything on macOS NOT_RUN | |

## 2. Manual checks by the owner

| # | Check | Status |
|---|---|---|
| M1 | Open the UI in your own browser: desktop and narrow window | NEEDS_OWNER — partial: the owner ran the app in a desktop browser (4 photos, whole-image mode, top-3 shown; screenshots supplied). Narrow window, ROI sliders, feedback and clear by the owner not yet evidenced |
| M2 | Full scenario with your own photos (checklist in `docs/EXTERNAL_VALIDATION.md`) | NEEDS_OWNER |
| M3 | External validation on own photos with known breeds (`scripts/check_photos.py`) | NEEDS_OWNER — status `not_performed` |
| M4 | Demo screenshots (`docs/DEMO_GUIDE.md`) | PASS — 3 owner-supplied screenshots in `docs/assets/`; a 4th with a photographer's watermark was left out |

## 3. Presentation

| # | Item | Status |
|---|---|---|
| P1 | README.md (English) + README.ru.md, consistent, language links | PASS |
| P2 | MODEL_CARD, EXTERNAL_VALIDATION, DEMO_GUIDE, DEVELOPMENT, REPOSITORY_SETUP | PASS |
| P3 | CI workflow `.github/workflows/ci.yml` (Ubuntu + Windows) | PASS (green on GitHub) |
| P4 | Demo section in README | PASS |

## 4. Actions that need the owner's permission or decision

| # | Action | Status |
|---|---|---|
| R1 | Source-code license | PASS — MIT (`LICENSE`), owner's decision |
| R2 | Terms for the trained model weights | PASS — `docs/MODEL_TERMS.md` (non-commercial research/education, attribution), delegated by the owner |
| R3 | GitHub repository | PASS — `7SUNNY77/DogLens` |
| R4 | Commits, tag, push | PASS — commits authored by the owner only, pushed to `main`, tag `v0.1.0` |
| R5 | GitHub Release v0.1.0 with the bundle archive | PASS — [https://github.com/7SUNNY77/DogLens/releases/tag/v0.1.0](https://github.com/7SUNNY77/DogLens/releases/tag/v0.1.0); public download + SHA-256 + `verify` checked |
| R6 | Verify the ImageNet / Stanford Dogs terms before any public use of dataset photos | NEEDS_OWNER |
| R7 | Public deploy | not wanted (owner's decision) |

## 5. Bundle archive (local, not published)

- File: `dist/doglens-effb0-effb0-seed42-exp1-de6c9770.zip` (15,632,893 bytes) and
  `dist/doglens-effb0-effb0-seed42-exp1-de6c9770.zip.sha256`.
- SHA-256: `13a1612fd6fda27759a239dc8f03f19d32b4c7365d7bbffb3af5551c9849713d`.
- Contents: exactly `model_bundle/{weights.pt, metadata.json, class_mapping.json, metrics.json}`.
- Checked:
  - `pack` validated the source bundle;
  - the deterministic zip was unpacked to a temp directory, loaded, and inference ran with
    outgoing connections disabled;
  - `verify --extract-to artifacts` was run in the clean copy.
- Published as an asset of [release v0.1.0](https://github.com/7SUNNY77/DogLens/releases/tag/v0.1.0). Downloaded anonymously from the public URL:
  `sha256sum -c` OK, `package_bundle.py verify` OK.
- Fixed during publishing: the `.zip.sha256` file was written with CRLF on Windows, which breaks
  `sha256sum -c` on Linux/macOS. It is now written with LF (test added) and the asset was replaced;
  the archive bytes and hash did not change.

## 6. Clean copy check (not a "fresh clone": the repository has no commits)

- Copy:
  - created in a temporary directory from the explicit list `git ls-files --others --exclude-standard`
    (92 files at the time of the check);
  - this list excludes `.venv`, `data/images`, `data/annotations`, `.mat`, checkpoints, `artifacts/`,
    `dist/`, SQLite, caches and secrets;
  - files added afterwards were only docs and the CI workflow; `.streamlit/config.toml` and
    `tests/test_dataset_real.py` were re-synced after fixes.
- `uv sync --frozen --offline` installed from the **local uv cache** in 19 s (environment ≈ 1.1 GB).
- Bundle installed from the archive: `scripts/package_bundle.py verify <archive> --extract-to artifacts`.
- Offline smoke: an explicitly passed fixture photo, with `socket.connect` disabled; real
  predictor on the whole image and on a region; top-1 `chihuahua`. This is a pipeline check, not
  an external evaluation.
- `uv run pytest`: 301 passed. `uv run pytest -m integration`: 8 passed, 6 skipped (they need the
  dataset, which is correctly absent).
- `uv run streamlit run app.py`: health `ok`; the real-browser run (A9) used this server.
- Found and fixed: `test_official_membership_is_preserved` failed instead of skipping when
  `data/splits/` exists without the `.mat` files.

## 7. Known limitations (documented, not blockers)

- Always answers with one of 120 Stanford Dogs breeds: no reliable "not a dog" class, no
  mixed-breed composition, no automatic dog detector.
- ROI is chosen with two sliders, not by dragging a box.
- Breed names are shown in English (no verified Russian translations).
- Test metrics are optimistic (Stanford Dogs comes from ImageNet; the backbone is
  ImageNet-pretrained). External validation not performed. Scores are not calibrated probabilities.
- On Linux, `uv.lock` resolves the default CUDA build of torch (large download), even though the app
  uses CPU.
