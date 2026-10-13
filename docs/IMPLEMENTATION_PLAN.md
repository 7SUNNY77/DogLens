# DogLens — план реализации

Основа требований: [CLAUDE.md](../CLAUDE.md). SPEC.md отсутствует.

## Исходное состояние (2026-10-13)

- Репозиторий: только `CLAUDE.md` и `data/`; не был git-репозиторием (выполнен `git init`, коммитов нет).
- Данные: `data/images/Images/<wnid>-<breed>/*.jpg` — 20 580 файлов, 120 папок;
  `data/annotations/Annotation/<wnid>-<breed>/<image_id>` — 20 580 XML (PASCAL VOC-подобные).
- Официальных split-файлов изначально не было; 2026-10-13 `lists.tar` получен с официального
  источника (с разрешения пользователя) и распакован в `data/splits/` — см. `data/splits/SOURCE.json`.
- Окружение: Windows 11 x86_64, Intel Core Ultra 7 155H (16 потоков), 16 GB RAM, Intel Arc iGPU
  (нет CUDA/MPS) → обучение и inference на CPU. Python 3.11.15 (uv-managed), uv 0.11.7.
- Зависимости (uv.lock): torch 2.14.1+cpu, torchvision 0.29.1+cpu, Pillow 12.3.0, Streamlit 1.65.0.

## Допущения

1. Данные используются из фактического расположения `data/images/Images` + `data/annotations/Annotation`;
   пользовательские файлы не перемещаются. `prepare_dataset.py` принимает явные `--images-dir`,
   `--annotations-dir`, `--splits-dir`, `--output-dir` (вместо `--raw-dir` из CLAUDE.md; README синхронизирован).
2. Git-репозиторий инициализирован локально, чтобы фиксировать commit в metadata; коммиты создаются
   только по запросу пользователя.
3. Устройство по умолчанию — auto (CUDA → MPS → CPU); на этой машине фактически CPU.
4. JPEG с MPF-индексом (Pillow определяет как MPO; типично для фото с телефонов) принимается как
   JPEG, используется только основное изображение (кадр 0) — это не анимация. Анимированные PNG/WebP
   отклоняются.
5. Изображение с нечитаемыми EXIF-метаданными ориентации отклоняется (а не показывается в
   потенциально неверной ориентации).
6. Небольшая корректировка структуры: `src/doglens/errors.py` — общие исключения с русскими
   сообщениями для UI; `data_prep.py` — подготовка Stanford Dogs (без torch); `dataset.py` зарезервирован
   под torch Dataset (этап 4); `fileutil.py` — атомарная запись/хеши; `visual_check.py` — панели проверки.
7. Валидация пользовательского ввода (`image_io`, `roi`) и исходного датасета (`data_prep`) разделены:
   лимиты загрузки к датасету не применяются; общая операция — `image_io.to_rgb`.
8. Validation-разбиение держит точные дубликаты внутри официального train по одну сторону train/val
   (CLAUDE.md: производные одного фото не попадают в разные split); поэтому val = 2 396, а не 2 400.
   Официальные train/test не меняются; межсплитовые дубликаты с test только документируются.
9. Кропы не создаются заранее: manifest хранит исходную рамку, контекст 10% применяется на лету.
10. `manifest.csv`, `class_mapping.json`, `preparation_report.json`, `data/splits/SOURCE.json` —
   производные метаданные без изображений — хранятся в Git (исключения в `.gitignore`), чтобы split и
   mapping были зафиксированы; изображения, архивы, `.mat` и подборка — нет.
11. Русские названия пород: проверенных переводов нет → `name_ru = null`, UI показывает английское
   название; переводы добавляются в `assets/breed_names_ru.json` только с источником.
12. Шаг 4.0 (решение пользователя): из train/val исключены записи с копией в официальном test и
   записи с конфликтом меток; test не изменён, помечен флагами. Внесено в CLAUDE.md §7.
13. Preprocessing содержит только детерминированные шаги; train-аугментация (горизонтальное
   отражение) применяется к PIL-кропу до него. Параметры сохраняются в `assets/preprocessing.json`
   и сверяются с метаданными весов перед обучением.
14. Замороженные блоки backbone целиком в eval (BatchNorm и stochastic depth), `model.train()` их не
   переключает. Политика BN на этапе B — `frozen` (из config).
15. `scripts/train.py`: `--smoke-test`, `--benchmark`, полный запуск, `--resume` (с границы
   завершённой эпохи), `--max-minutes` (остановка между batch, код 3). Полный цикл — в
   `src/doglens/experiment.py` (отдельно от примитивов `training.py`).
16. Bundle экспортируется отдельной командой `scripts/export_bundle.py` из `best.pt` (лучший по
   validation Top-1 за весь эксперимент, никогда не `last.pt` неявно); predictor и UI — следующий
   этап. Политика кропа для ROI приложения фиксируется на этапе predictor.
17. Оценка: основной отчёт — полный test (8 580); диагностические срезы clean / multi_box /
   single_box; macro F1 по фиксированным 120 labels и по присутствующим (`zero_division=0`).

## Этапы и критерии проверки

| # | Этап | Критерий готовности |
|---|------|---------------------|
| 1 ✅ | Inspect, plan, dependencies, config/schemas, tests scaffold | `uv sync` ок; импорт torch/torchvision; `pytest`, `ruff check`, `ruff format --check` зелёные |
| 2 ✅ | Image validation и ROI | Unit-тесты: JPEG/PNG/WebP, alpha/grayscale/CMYK/16-bit, EXIF, corrupt/oversized/animated/bomb/unsupported, масштаб ROI preview, округление, границы, минимум 64 px |
| 3 | Dataset parser, manifest, mapping | ✅ Fixture-тесты парсера MAT/XML, координат, bbox (+10% контекста, clamp), stratified 80/20 seed=42, mapping 0..119; реальная подготовка; `preparation_report.json`; визуальная подборка |
| 4 ✅ | Точные межсплитовые дубликаты SHA-256 (4.0), weights, preprocessing, model, loader, smoke | 4.0: из train/val исключены копии test и конфликты меток (67), флаги в manifest, проверка пересечения по SHA-256; веса проверены по SHA-256; preprocessing = `weights.transforms()` (тест); freeze + BN-режимы и статистики (тесты); детерминированный DataLoader; `train.py --smoke-test` проходит |
| 5 ✅ | Bundle и predictor | Bundle и `predictor.py`: `predict(image, roi, top_k)`, контекст 10% один раз (`roi.model_region`), ties по class_id, lock, кеш модели, без сети; эквивалентность с evaluation на реальном bundle |
| 6 ✅ | Streamlit UI | `app.py` + `ui_state.py`: состояния EMPTY…ERROR, ROI слайдерами с пересчётом через roi.py, превью фактического кропа, сбросы, изоляция сессий; 11 AppTest-тестов + integration с реальным bundle |
| 7 ✅ | Feedback и каталог | SQLite (parameterized, идемпотентный submission_id, без фото), каталог с поиском, сверка с bundle |
| 8 | Реальное обучение и оценка | Полный цикл, resume (побитово проверен тестами), benchmark, один полный CPU-эксперимент с разрешения пользователя, `evaluate.py` со срезами. Статус — в PROGRESS.md |
| 9 ◐ | Benchmark, документация, clean-start | inference benchmark ✅ (`scripts/benchmark.py`); внешняя проверка — инструмент и инструкция, статус not_performed; clean-start check (свежий clone + uv sync) — не выполнен |

## Риски и блокеры

- ~~Нет official split~~ — снят 2026-10-13 (`lists.tar`, SHA-256 сверен с TFDS).
- **Дубликаты и шум разметки**: 37 пар test↔train/val, 37 межклассовых пар, «крупнейшая рамка»
  иногда выбирает собаку другой породы → на этапе оценки отдельно приводить test без дубликатов.
- **Только CPU**: полное обучение EfficientNet-B0 на ~9 600 train-изображениях займёт часы; запуск
  только после smoke-оценки времени и согласования.
- **ImageNet-пересечение**: Stanford Dogs взят из ImageNet, как и pretrained-веса — test не является
  независимой внешней оценкой (будет отражено в MODEL_CARD).
- Компонент кадрирования для Streamlit может быть нестабилен — fallback на числовые контролы
  с предпросмотром кропа.
- Лицензия: Stanford Dogs — для некоммерческого исследования; изображения не публикуются в репозитории.

## Решения этапа UI/predictor
18. ROI приложения: рамка пользователя проверяется (≥ 64 px), затем +10% контекста с каждой стороны
    с обрезкой; whole-image — без расширения. Единственная реализация — `roi.model_region`; её
    используют и превью UI, и predictor. Метаданные bundle не изменены.
19. Компонент кадрирования (streamlit-cropper и т. п.) не используется: он требует новой
    зависимости и не тестируется headless. Fallback — два слайдера в координатах превью с
    пересчётом через `roi.display_box_to_roi` и предпросмотром фактического кропа.
20. Тексты интерфейса вынесены в `doglens.app_texts`, логика состояний — в `doglens.ui_state`
    (тестируется без Streamlit).
