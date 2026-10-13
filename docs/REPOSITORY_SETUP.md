# Repository setup (done)

Decisions (2026-10-13, owner): repository `7SUNNY77/DogLens`, public; code license MIT; model
weights under [MODEL_TERMS.md](MODEL_TERMS.md); commit, push and release approved; no public deploy.
Commits are authored by the owner only; no AI co-author trailers.

## Repository

- **Name:** `DogLens` (`https://github.com/7SUNNY77/DogLens`)
- **Description:** Local Streamlit app that suggests the top-3 visually similar dog breeds
  (120 Stanford Dogs classes) for a user-selected region of a photo. EfficientNet-B0, CPU inference.
- **Topics:** `computer-vision`, `image-classification`, `dog-breeds`, `stanford-dogs`,
  `efficientnet`, `pytorch`, `torchvision`, `streamlit`, `python`, `uv`

## Before the first commit (owner)

1. ~~Choose a license~~ — done: MIT (`LICENSE`), model terms in `docs/MODEL_TERMS.md`.
2. Check `git status`: only source, configs, docs, reports and small metadata are listed. The
   following must **not** appear: `data/images`, `data/annotations`, `*.mat`, `*.pt`,
   `artifacts/`, `dist/`, `*.sqlite`, `reports/evaluation/**/logits.npz`, `reports/external_check/`.
   These are excluded by `.gitignore`.
3. ~~Demo assets~~ — done: three screenshots in `docs/assets/`.

## Suggested first commit message

```
Initial DogLens MVP: data pipeline, EfficientNet-B0 training, evaluation, Streamlit app

- Stanford Dogs preparation with the official split, audit and exact-duplicate handling
- Two-stage EfficientNet-B0 training with epoch-boundary resume; selected stage A, epoch 3
- Evaluation on the official test split (Top-1 87.25%, Top-3 98.02%) with diagnostic slices
- Verified model bundle, predictor with a single ROI + context policy, inference benchmark
- Russian-language Streamlit UI with slider ROI, top-3, feedback (no photos stored), catalog
- Tests (unit, AppTest, integration markers), ruff, CI workflow
- MIT license for the code; separate non-commercial terms for the model weights
```

## Release proposal (after the owner approves publishing)

- **Tag:** `v0.1.0`
- **Title:** `DogLens v0.1.0 — local MVP with trained EfficientNet-B0 bundle`
- **Asset:** `dist/doglens-effb0-effb0-seed42-exp1-de6c9770.zip` (≈ 15.6 MB) and its `.zip.sha256`,
  created by `uv run python scripts/package_bundle.py pack`.

Release notes draft:

```markdown
DogLens suggests the top-3 visually similar breeds (120 Stanford Dogs classes) for a selected
region of a dog photo. Local Streamlit app, CPU inference.

Model bundle: `doglens-effb0-effb0-seed42-exp1-de6c9770.zip`
SHA-256: 13a1612fd6fda27759a239dc8f03f19d32b4c7365d7bbffb3af5551c9849713d

Install: `uv sync`, then
`uv run python scripts/package_bundle.py verify doglens-effb0-effb0-seed42-exp1-de6c9770.zip --extract-to artifacts`,
then `uv run streamlit run app.py`.

Results on the official Stanford Dogs test split (8,580 images): Top-1 87.25%, Top-3 98.02%,
macro F1 0.8665. These are optimistic (Stanford Dogs comes from ImageNet, the backbone is
ImageNet-pretrained). Not externally validated; scores are not calibrated probabilities.

Limitations: always answers with one of 120 breeds (no "not a dog" class, no mixed-breed
composition), no automatic dog detector, region chosen with sliders, English breed names.

Code: MIT. Model weights: non-commercial research/education with attribution, see
docs/MODEL_TERMS.md.
```

Confirm the SHA-256 against the file you actually upload: rebuilding the archive from the same
bundle gives identical bytes, but a different bundle gives a different hash.
