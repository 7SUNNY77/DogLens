# DogLens — прогресс

Последнее обновление: 2026-10-13. План: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

Статусы (раздельно):

| Направление | Статус |
|---|---|
| MVP (код, модель, UI, feedback, каталог, тесты) | **готов** |
| Модель обучена и оценена на Stanford Dogs test | **да** (Top-1 87.25%, Top-3 98.02%) |
| Локальный запуск из чистой копии (офлайн, bundle из архива) | **проверен** (кеш uv, не fresh clone) |
| UI в реальном браузере (автоматизация headless Chrome) | **проверен автоматически**, 30/30 |
| Ручная приёмка владельцем | **частично**: владелец запустил UI в браузере (4 фото, всё изображение); ROI, отзыв, очистка и узкий экран владельцем ещё не подтверждены |
| Внешняя проверка на своих фото | **not_performed** |
| Оформление репозитория (README EN/RU, docs, CI) | **готово**; CI зелёный на GitHub (Ubuntu + Windows) |
| Публикация | **выполнена**: [github.com/7SUNNY77/DogLens](https://github.com/7SUNNY77/DogLens), [релиз v0.1.0](https://github.com/7SUNNY77/DogLens/releases/tag/v0.1.0); deploy не нужен |

Подробный чеклист — [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).

## Completed

### Этап 1. Inspect, plan, dependencies, config/schemas, tests scaffold
- Исследованы репозиторий, данные и окружение (см. «Исходное состояние» в плане).
- `git init` (без коммитов), `.gitignore` (данные, веса, SQLite, кеши, venv, secrets).
- `pyproject.toml` (runtime/dev разделены, uv), `.python-version` = 3.11, `uv.lock`.
- Зафиксированные версии: torch 2.14.1+cpu, torchvision 0.29.1+cpu, Pillow 12.3.0,
  Streamlit 1.65.0, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, scikit-learn 1.9.1, pytest 9.1.1, ruff 0.17.0.
- Импорт проверен; `EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()` в этой версии:
  resize 256, crop 224, BICUBIC, antialias=True, mean/std ImageNet (будет сохранено в metadata явно).
- `src/doglens/config.py` — лимиты загрузки, строгая загрузка/валидация `configs/training.yaml`
  (неизвестные ключи и значения вне диапазона → `ConfigError`).
- `src/doglens/schemas.py` — `Roi`, колонки manifest, `Verdict`, `Prediction`, `PredictionResult`,
  список файлов bundle. `src/doglens/errors.py` — исключения с кодом и русским сообщением.

### Этап 2. Image validation и ROI
- `src/doglens/image_io.py::load_image(bytes)`: размер → magic bytes → заголовок (размеры, кадры)
  → полное декодирование → EXIF transpose → RGB (alpha на белый фон, 16-bit масштабируется) →
  удаление метаданных. Decompression-bomb warning/error → отказ; собственный лимит 20 MP отдельно;
  `LOAD_TRUNCATED_IMAGES=True` глобально → отказ. Логи без содержимого и имени файла.
- `src/doglens/roi.py`: `validate_roi` (отклоняет, не обрезает), `full_image_roi`, `make_preview` /
  `preview_geometry` (реальные размеры превью), `display_box_to_roi` (floor min / ceil max,
  допуск 1 px отображения), `roi_to_display`, `crop_roi`, `expand_box` (для bbox +10%).
- Тесты: `tests/test_image_io.py`, `tests/test_roi.py`, `tests/test_config_schemas.py`
  (синтетические изображения генерируются в памяти, `tests/_imaging.py`).

### Этап 3. Stanford Dogs: официальный split, аудит, mapping, каталог
- `lists.tar` получен 2026-10-13 с официального адреса (по разрешению пользователя):
  - SHA-256 `34b47cac…4d6c` совпадает с TFDS;
  - безопасно распакован в `data/splits/`;
  - происхождение записано в `data/splits/SOURCE.json`.
  - Фото, аннотации и веса не скачивались.
- Код:
  - `src/doglens/data_prep.py` — MAT/XML, координаты, аудит, mapping, split, manifest;
  - `src/doglens/catalog.py` — mapping и каталог из единого источника, `mapping_sha256`;
  - `src/doglens/fileutil.py` — атомарная запись и хеши;
  - `src/doglens/visual_check.py`, `scripts/prepare_dataset.py`;
  - `assets/breed_names_ru.json` — пустой оверлей проверенных переводов;
  - `image_io._to_rgb` → публичный `to_rgb`, поведение не изменено.
- Артефакты:
  - `data/prepared/{manifest.csv, class_mapping.json, preparation_report.json}`;
  - `assets/breeds.json`;
  - подборка `data/prepared/visual_check/` (26 панелей, вне Git).
- Отчёты: `reports/dataset_preparation.md`, `reports/dataset_visual_check.md`.
- Тесты:
  - `tests/test_data_prep.py`, `tests/test_catalog.py` (фикстуры `tests/_stanford_fixture.py`);
  - `tests/test_dataset_real.py` (маркер `integration`, по умолчанию исключён).
- Результат: 120 классов; **train 9 604 / val 2 396 / test 8 580**; исключено 0.
  - Val ≠ 2 400, потому что дубликаты внутри официального train держатся по одну сторону.
- Соглашение XML: 0-based, `xmax`/`ymax` включающие → внутренний формат `(xmin, ymin, xmax+1, ymax+1)`.
  Установлено статистикой по 22 126 рамкам: `xmax == W` встречается 0 раз, `xmax == W−1` — 1 867 раз.
- Проблемы, все зафиксированы:
  - 2 обрезанные рамки;
  - 1 PNG-файл под расширением .jpg;
  - 1 невалидное значение EXIF orientation (0);
  - 624 шаблона `%s` в `<filename>`;
  - 89 пар точных дубликатов: test↔train 29, test↔val 8, train↔val 0; 37 пар — разные породы.
- Визуальная проверка: все 26 панелей просмотрены ассистентом (не человеком), грубых ошибок нет.
  Найдено ограничение политики «крупнейшая рамка»: в n02109525_10743 выбрана собака другой породы.

### Этап 4. Точные межсплитовые дубликаты (4.0), веса, preprocessing, модель, загрузка данных, smoke-test
- 4.0 Устранены обнаруженные точные межсплитовые дубликаты по SHA-256 (near-duplicates и
  пересечение с ImageNet не проверялись):
  - из train/val исключено 67 записей: `dup_with_test` 23, `dup_with_test+label_conflict` 14,
    `label_conflict` 30; из train 53, из val 14; затронуто 40 классов;
  - test не изменён;
  - итог: **train 9 551 / val 2 382 / test 8 580**;
  - в manifest добавлены `dup_with_test`, `label_conflict`, `n_boxes`;
  - исключения записаны в `data/prepared/excluded.csv`;
  - проверка пересечения по SHA-256 — фатальная в `prepare` и интеграционный тест;
  - CLAUDE.md обновлён: политика утечек (§7) и аргументы CLI (§5, §13).
- 4.1 Веса `EfficientNet_B0_Weights.IMAGENET1K_V1` с `download.pytorch.org`:
  - 21 444 401 байт, SHA-256 `7f5810bc96de…4934`;
  - проверено: префикс хеша из имени файла, размер по метаданным torchvision, strict load,
    5 288 548 параметров;
  - происхождение в `data/weights/SOURCE.json`; файл весов вне Git.
- 4.2 `src/doglens/preprocessing.py` + `assets/preprocessing.json`:
  - resize 256 bicubic с antialias → center crop 224 → float → normalize ImageNet;
  - побайтово совпадает с `weights.transforms()` (тест).
- 4.3 `src/doglens/model.py`:
  - `BreedClassifier` — EfficientNet-B0 с новым classifier на 120 классов;
  - режимы head/finetune/full, BN-политики frozen/train_unfrozen;
  - обучаемых параметров: head 153 720, finetune(6,7,8) 3 309 460.
- 4.4 `src/doglens/dataset.py`:
  - `ManifestDataset` — кроп с контекстом на лету, PNG на белом;
  - `HorizontalFlip` (только train);
  - `make_loader` с seeded generator.
- 4.5 `src/doglens/training.py` и `scripts/train.py --smoke-test`. Последний запуск: CPU, 24.6 с.
  - Loss этапа A на 15 батчах: 4.868 → 4.625.
  - Переобучение одного батча: 4.008 → 1.1e-7.
  - Шаги fine-tune: loss конечный.
  - BN-статистики не изменились.
  - Логиты после перезагрузки чекпоинта совпали точно (max diff 0.0).
  - Отчёт: `reports/train_smoke_test.json`.
- Грубая оценка времени полного обучения на CPU по smoke-test (без проходов validation):
  - этап A ≈ 5.3 мин/эпоха;
  - этап B ≈ 8.0 мин/эпоха;
  - конфигурация 3 + 5 эпох — порядка 1–1.5 ч вместе с validation.
- Тесты: `test_preprocessing.py`, `test_model.py`, `test_dataset_loader.py`, `test_training.py`;
  тесты подготовки данных расширены.

### Этап 5/8. Полный цикл обучения, resume, оценка, bundle
- Сверка данных:
  - 120 классов; split 9 551 / 2 382 / 8 580; минимум на класс 76 / 19 / 48, пустых классов нет;
  - `excluded.csv` (67) не пересекается с manifest, вместе 20 580;
  - флаги стоят только в test: `dup_with_test` 37, `label_conflict` 30, оба — 14, хотя бы один — 53;
  - дубликаты: 89 групп = 89 пар = 178 уникальных записей.
- Происхождение 9 604 / 2 396: 40 пар внутри официального train → 11 960 представителей;
  0.2 × 11 960 = 2 392 без округления; плюс 4 копии → 2 396.
- Формулировки уточнены: «устранены обнаруженные точные межсплитовые дубликаты по SHA-256»;
  near-duplicates и пересечение с ImageNet не проверялись.
- `src/doglens/experiment.py`:
  - этапы A/B, validation-loss с учётом размера batch, Top-1/Top-3;
  - early stopping, stage/global best, `best_stage_{A,B}.pt`, `best.pt`, `last.pt`;
  - resume с границы эпохи: RNG + generator + optimizer;
  - проверка совместимости mapping/manifest/preprocessing/pretrained/device/config.
- `scripts/train.py`: полный режим, `--resume`, `--max-minutes` (код 3), `--benchmark`, каталог запуска
  с `resolved_config.yaml`, `run_metadata.json`, `history.json`, `train.log`, `final_report.json`.
- `src/doglens/bundle.py` + `scripts/export_bundle.py`: экспорт `best.pt`, CPU-загрузка с
  `weights=None` и проверками, сравнение logits.
- `src/doglens/evaluation.py` + `scripts/evaluate.py`:
  - срезы full/clean/multi_box/single_box;
  - macro F1 по 120 и по присутствующим labels;
  - per-class, confusion matrix, top-5 predictions, `logits.npz` для пересчёта без forward.
- Resume проверен тестами: 4 точки прерывания (до 1-го batch, середина A, граница A→B, середина B).
  Порядок данных, счётчики, история метрик, веса, optimizer и `best.pt` совпали **побитово**; различаются
  только длительности эпох.
- Benchmark на реальных данных: прогноз 58 мин (≤ 2 ч) → запущен один полный эксперимент.
- Найденная ошибка: `TorchVersion` в identity ломал `weights_only`-загрузку на переходе A→B. Исправлено,
  добавлены регрессионные тесты, эксперимент продолжен штатным `--resume`.
  Подробности — `reports/training_exp1.md`.
- Результат:
  - выбрана голова (этап A, эпоха 3): val Top-1 0.8745 / Top-3 0.9702;
  - fine-tuning хуже (лучший 0.8497, val loss растёт);
  - test full (8 580): Top-1 0.8725, Top-3 0.9802, macro F1 0.8665; срезы — в MODEL_CARD.
- Bundle `artifacts/model_bundle` (`doglens-effb0-effb0-seed42-exp1-de6c9770`):
  - logits совпали с checkpoint (diff 0);
  - val на bundle воспроизведён (0.8745);
  - `metrics.json` — status `evaluated`.

### Этап 6–7 + inference benchmark. Predictor, Streamlit UI, feedback, каталог
- **ROI-политика приложения:**
  - рамка пользователя → `validate_roi` (≥ 64 px) → +10% контекста с обрезкой;
  - whole-image — без расширения;
  - единая функция `roi.model_region` / `crop_model_region` для превью и predictor;
  - метаданные bundle не изменены.
- **`src/doglens/predictor.py`** — `Predictor.predict(image, roi=None, top_k=3)`:
  - bundle (`weights=None`, строгая загрузка, проверки) и каталог со сверкой mapping;
  - lock на forward, `inference_mode`, softmax по 120 классам, ties по `class_id`, без
    перенормировки;
  - warnings; `user_roi` и `crop_box` в результате; `get_predictor` — кеш на процесс.
  - Исключение для `TorchVersion` в путь приложения не попадает: импорт `experiment` в
    `bundle.py` ленивый, только для экспорта.
- **`app.py` + `src/doglens/ui_state.py` + `src/doglens/app_texts.py`:**
  - русский UI, состояния EMPTY / VALIDATING / READY / PREDICTING / RESULT / ERROR;
  - ROI слайдерами в координатах превью (fallback вместо компонента кадрирования) и превью
    фактического кропа;
  - top-3, обязательные тексты, технические сведения, отзыв, «Очистить», каталог с поиском;
  - в `st.cache_resource` кешируется только модель.
- **`src/doglens/feedback.py`** — SQLite:
  - parameterized SQL, отдельное соединение на операцию;
  - идемпотентный `submission_id`, CHECK на verdict, без фото.
- **`scripts/benchmark.py`:**
  - фото 333×500 с рамкой: `predict` 23.2 / 33.2 мс (медиана / p95);
  - 4032×3024 целиком: декодирование + `predict` 214 / 249 мс;
  - холодно: импорт 6.5 с, bundle 0.27 с.
- **`scripts/check_photos.py`** + `docs/EXTERNAL_VALIDATION.md` — инструмент и сценарии; внешняя
  проверка **not_performed**.
- **Тесты:**
  - `test_model_region`, `test_predictor`, `test_feedback`, `test_ui_state`, `test_app`
    (11 AppTest), `test_scripts_inference`;
  - integration `test_predictor_real` (эквивалентность с evaluation на 22 test-изображениях,
    приложение с реальным bundle).
  - Общая фикстура `tiny_bundle` — маленький bundle со случайными весами, экспортированный
    релизным путём.

### Выполненные проверки (фактические результаты)
| Команда | Результат |
|---|---|
| smoke реального predictor (9 test-изображений, рамка из manifest) | 9/9 совпали с `predictions.csv` (top-1 и score до 1e-5) |
| `uv run python scripts/benchmark.py --image data/images/Images/n02085620-Chihuahua/n02085620_10074.jpg --roi 25,10,277,499 --report reports/benchmark/inference_stanford_333x500_roi.json` | exit 0; `predict` 23.2 / 33.2 мс |
| `uv run python scripts/benchmark.py --image <scratchpad>/synthetic_4032x3024.jpg --repeats 20 --report reports/benchmark/inference_synthetic_4032x3024_whole.json` | exit 0; декодирование + `predict` 214 / 249 мс |
| `uv run streamlit run app.py --server.port 8599` + `curl /_stcore/health` | сервер стартовал, health=ok; в браузере UI **не просматривался** |
| `uv run pytest -q` | 299 passed, 14 deselected |
| `uv run pytest -m integration -q` | 14 passed (датасет, реальный bundle, predictor-эквивалентность, приложение с реальной моделью) |
| `uv run ruff check .` / `ruff format --check .` | All checks passed / 59 files already formatted |

### Подготовка к релизу
- `docs/RELEASE_CHECKLIST.md` — статусы PASS / FAIL / NOT_RUN / NEEDS_OWNER.
- `scripts/package_bundle.py` (pack/verify) + тесты:
  - архив `dist/doglens-effb0-effb0-seed42-exp1-de6c9770.zip`;
  - SHA-256 `13a1612f…713d`;
  - распаковка, загрузка и inference без сети проверены.
- `artifacts/model_bundle/` и `dist/` исключены из Git: bundle доставляется только архивом.
- Чистая копия по явному списку файлов:
  - `uv sync --frozen --offline` из кеша;
  - bundle из архива, офлайн-smoke, pytest 301 passed, integration 8 passed / 6 skipped;
  - Streamlit стартует.
- Реальный браузер: headless Chrome через DevTools protocol, 30/30 проверок на 1366 и 390 px.
- Найдено и исправлено:
  - integration-тест падал вместо skip, когда в `data/splits/` есть только `SOURCE.json`;
  - абсолютный локальный путь (с именем пользователя) в отчёте benchmark — `benchmark.py` теперь
    пишет только относительные пути;
  - меню Streamlit «Deploy» скрыто (`client.toolbarMode = "viewer"`).
- Документация:
  - README.md (русский, основной) и README.en.md (английский);
  - `docs/DEVELOPMENT.md` — бывший подробный README;
  - `docs/DEMO_GUIDE.md`, `docs/REPOSITORY_SETUP.md`, уточнены EXTERNAL_VALIDATION и MODEL_CARD.
- `.github/workflows/ci.yml`: Ubuntu + Windows, uv 0.11.7, `uv sync --frozen`, ruff, pytest.
- Лицензии:
  - на странице Stanford Dogs явной лицензии нет;
  - условия ImageNet не перепроверены;
  - лицензия кода не выбрана.

| Команда | Результат |
|---|---|
| `uv run python scripts/package_bundle.py pack` | exit 0, архив 15 632 893 байт, самопроверка ok |
| чистая копия: `uv sync --frozen --offline` | ok, 19 с (кеш) |
| чистая копия: `package_bundle.py verify <archive> --extract-to artifacts` | ok |
| чистая копия: офлайн-smoke predictor | top-1 chihuahua (whole и ROI) |
| чистая копия: `uv run pytest` / `-m integration` | 301 passed / 8 passed, 6 skipped |
| чистая копия: `streamlit run` + браузерная автоматизация | health ok, 30/30 |
| основной репозиторий: ruff check / format / pytest / integration | PASS / PASS / 301 passed / 14 passed |
| проверка локальных Markdown-ссылок | 30 ссылок, 0 битых |

## In progress
- Нет.

### Решения владельца и публикация
- Репозиторий `7SUNNY77/DogLens`, лицензия кода MIT (`LICENSE`).
- Условия для весов: `docs/MODEL_TERMS.md` — выбор делегирован владельцем: некоммерческое
  исследовательское и учебное использование, атрибуция, без гарантий.
- Демо: 3 скриншота владельца в `docs/assets/`. Скриншот с водяным знаком фотографа не добавлен.
- Коммиты — только от имени владельца, без AI co-author.
- Публичный deploy не нужен.
- Публикация:
  - создан публичный репозиторий `7SUNNY77/DogLens` с topics;
  - push `main`: коммиты `437c862` и `ed06bd8` (`.gitattributes`, LF для файлов с контрольными
    суммами), автор — только владелец;
  - [CI](https://github.com/7SUNNY77/DogLens/actions/runs/38462373068) зелёный на Ubuntu и Windows;
  - [релиз v0.1.0](https://github.com/7SUNNY77/DogLens/releases/tag/v0.1.0) с архивом bundle и `.zip.sha256`; публичное скачивание проверено
    (`sha256sum -c` OK, `verify` OK).
  - Исправлено: CRLF в `.zip.sha256` (`package_bundle.py` пишет LF, тест, asset заменён).

## Blocked / нужны решения пользователя
- Полная ручная приёмка UI и систематическая внешняя проверка на своих фото с известной породой.

## Next
1. Владелец проходит чеклист ручной приёмки и, по возможности, проверку на своих фото.
2. Решения по лицензии и репозиторию → первый коммит → push → первый прогон CI.
3. GitHub Release с архивом bundle; после этого — ссылка в README.
