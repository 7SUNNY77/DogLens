# Demo guide (screenshots and video)

Status: **three screenshots added** (`docs/assets/demo_cardigan.png`, `demo_collie.png`,
`demo_german_shepherd.png`), supplied by the repository owner and shown in the README "Demo"
section. A fourth supplied screenshot was **not** added: the dog photo in it carries a
photographer's watermark, which indicates third-party copyright. The owner is responsible for the
rights to the photos visible in the added screenshots.

## Rights first

- Use **your own photos** of dogs (or photos whose owner allowed it).
- Do **not** use Stanford Dogs / ImageNet images in public screenshots or videos: their terms were
  not verified for public redistribution.
- Do not download "free" pictures for the demo without checking their license and attribution.
- Remove EXIF/GPS from published screenshots. The app itself never stores them, but your original
  files may contain them.

## Screenshot plan (`docs/assets/`)

Desktop window around 1366×900, browser zoom 100%. A narrow screenshot is optional.

| File | What it shows |
|---|---|
| `01_empty.png` | Start screen with the mandatory notice |
| `02_whole_image.png` | Uploaded photo, "use whole image" on, crop preview |
| `03_roi_sliders.png` | Whole-image toggle off, two sliders, red box (selection) and green box (+10% context), crop preview |
| `04_result.png` | Top-3 with scores and the note under the result |
| `05_feedback.png` | Feedback form after a successful submission |
| `06_catalog.png` | "Supported breeds" tab with a search query |
| `07_error.png` | Unsupported file → Russian error message |

## Short video (30–60 s)

1. Upload your own dog photo → whole-image prediction.
2. Switch to the region, move the sliders onto one dog and point at the crop preview.
3. Show the top-3 and the note under the result.
4. Send feedback, then press "Очистить" (clear).

Keep the video small (MP4/WebM, ≤ 10 MB) or link it from a release instead of committing it.

## After the assets exist

Add a "Demo" section to both README.md and README.ru.md that references `docs/assets/...`
with relative paths, and check the links (see `docs/RELEASE_CHECKLIST.md`).
