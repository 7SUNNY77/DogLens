# CLAUDE.md — DogLens

## 1. Роль и задача
Ты — ведущий разработчик DogLens. Реализуй законченный MVP веб-приложения: пользователь загружает фотографию, выбирает область с одной собакой и получает три наиболее похожие породы из Stanford Dogs.

Не ограничивайся планом или псевдокодом. Создай код, тесты, конфигурации и документацию. Работай итеративно и проверяй каждый этап. Если нет данных, вычислительных ресурсов или разрешения на скачивание, заверши доступные части и явно сообщи блокер. Никогда не выдавай недоступный эксперимент за выполненный.

Если существует SPEC.md, прочитай его до начала работы. Этот файл задаёт инструкции агенту, SPEC.md — продуктовые требования. При существенном противоречии задай один конкретный вопрос, не меняй требования молча. Если SPEC.md отсутствует, реализуй MVP по этому документу.

## 2. Продукт и границы
Обязательный сценарий:
1. Загрузить JPEG, PNG или WebP.
2. Проверить файл и показать корректно ориентированное изображение.
3. Выбрать прямоугольную область с одной собакой или всё изображение.
4. Запустить реальную обученную модель.
5. Показать top-3, русские/английские названия и scores.
6. Дать возможность оценить результат и очистить сессию.

Обязательный текст интерфейса:
«Приложение сравнивает внешность с поддерживаемыми породами. Результат не подтверждает происхождение собаки».
Под результатом:
«Оценки отражают выбор модели среди поддерживаемых пород. Они не являются генетическими долями или гарантией правильного ответа».

Не включай в MVP: автоматический detector, несколько собак за запрос, видео, авторизацию, платежи, REST API, React, генетические доли, автоматический класс «метис»/«не собака», автоматическое дообучение.
Не обещай распознавать породы вне 120 классов. Не называй softmax вероятностью правильности ответа без отдельной проверки калибровки.

## 3. Режим работы
Перед изменениями:
- изучи существующие файлы, git status, окружение и доступные инструменты;
- составь короткий план в docs/IMPLEMENTATION_PLAN.md;
- сохраняй пользовательские изменения; не выполняй destructive git-команды;
- не перезаписывай действующий проект ради удобства;
- не устанавливай глобальные пакеты и не используй sudo.

Далее выполняй этапы самостоятельно. Не запрашивай подтверждение для каждого обычного локального изменения. Запрашивай разрешение на платные ресурсы, публикацию, удаление пользовательских данных и действия с внешними аккаунтами. Соблюдай ограничения среды и доступные разрешения.
Не запускай без согласования многогигабайтные загрузки или длительное полное обучение. Сначала smoke test и оценка времени. Уже существующие подготовленные данные используй после проверки.

В конце этапа фиксируй: изменённые файлы, команды проверки, фактические результаты и блокеры. Веди docs/PROGRESS.md: completed / in progress / blocked / next.
Не останавливай всю работу из-за отсутствия датасета: тестируй инфраструктуру синтетическими изображениями и явными test doubles, не подменяя ими релизную модель.

## 4. Стек и окружение
- Python 3.11.
- PyTorch + torchvision: совместимая пара стабильных релизов, доступных в среде.
- Streamlit, Pillow, numpy, pandas, scipy для чтения MAT split-файлов.
- scikit-learn для split и метрик; PyYAML для конфигурации.
- SQLite через sqlite3 стандартной библиотеки.
- pytest, ruff.
- Один package manager: uv; если недоступен, venv + pip.

Проверь Python, ОС, архитектуру и установленную пару torch/torchvision до установки. Не угадывай версии и API. Выбери совместимые зависимости, проверь импорт и зафиксируй lockfile. Не обновляй всё окружение без необходимости.
Разделяй runtime- и dev-зависимости. Не внедряй Docker, MLflow, Redis, Celery или облачную инфраструктуру в MVP.

## 5. Структура
app.py
pyproject.toml
README.md
CLAUDE.md
.gitignore
configs/training.yaml
src/doglens/{config,image_io,roi,preprocessing,model,predictor,schemas,catalog,feedback}.py
src/doglens/{dataset,training,evaluation}.py
src/doglens/{data_prep,experiment,bundle,fileutil,errors,visual_check,ui_state,app_texts}.py  # фактическое уточнение структуры
scripts/{prepare_dataset,train,export_bundle,evaluate,benchmark,check_photos}.py
tests/
assets/breeds.json
docs/{IMPLEMENTATION_PLAN,PROGRESS,MODEL_CARD}.md
reports/
data/{images/Images,annotations/Annotation,splits,prepared,weights}/
artifacts/model_bundle/

Можно минимально скорректировать структуру для ясности, но не смешивать UI, обучение и inference.
Gitignore: datasets, checkpoints, SQLite, пользовательские файлы, кеши, venv, secrets. Сохраняй в Git конфигурации, schemas, mapping/catalog и код; большие веса распространяй отдельно.
Скрипты обязаны иметь --help, проверку аргументов и ненулевой exit code при ошибке.

## 6. Обработка фото
Проектные лимиты:
- не более 10 MiB;
- не более 20 000 000 пикселей;
- только JPEG/PNG/WebP, один кадр;
- минимум 64 пикселя по каждой стороне ROI;
- HEIC, SVG, PDF и анимация не поддерживаются.

Порядок: размер bytes → формат по содержимому → dimensions/frames → полное декодирование → EXIF transpose → RGB → ROI → model transforms.
Не полагайся на расширение. Обрабатывай Pillow decompression-bomb warnings/errors как отказ. Проверяй собственный pixel limit отдельно. Для PNG с alpha композить на белый фон. Не сохраняй исходное имя файла как путь.

ROI относится к изображению после EXIF transpose, до resize. Координаты (xmin,ymin,xmax,ymax), xmax/ymax исключающие. Preview использует уменьшенную копию: преобразуй координаты по реальным displayed dimensions, floor для min, ceil для max. Проверяй конечные числовые значения, границы и площадь; недопустимые входы отклоняй. Учитывай погрешность отображения, но не исправляй произвольные неверные координаты молча.

Для UI сначала проверь поддерживаемый компонент кадрирования. Если его интеграция нестабильна, реализуй рабочий fallback стандартными контролами координат и предпросмотром кропа. Не трать весь срок на компонент. Пользователь должен реально выбирать ROI, а не видеть декоративную рамку.

Ошибки показывай понятным русским сообщением, без traceback. Логи — технические, без фото, EXIF, GPS и пользовательского имени файла.

## 7. Данные
Stanford Dogs: 120 классов, ожидаемые официальные split 12 000 train / 8 580 test. Проверяй фактические архивы и структуру, не выдумывай файлы. Источник и условия использования документируй; наличие копии на Kaggle не доказывает право публично публиковать изображения.

Подготовь manifest:
image_id, relative_path, class_id, breed_key, split, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax.

Прочитай официальные split-файлы; official test не используй для настройки. Из official train выдели 20% stratified validation, seed=42; сохрани split и не пересоздавай при каждом запуске.
Mapping: индексы 0..119, стабильные breed_key, одинаковый порядок на всех этапах. mapping хранится в bundle, не создаётся заново при inference.

Проверь decode, classes, bounds, split overlap, точные дубликаты, class counts. Ошибки и исключения — preparation_report.json. Не исключай данные молча. Для координат аннотаций отдельно проверь исходное соглашение на визуальных примерах; не допускай off-by-one.
Выбирай крупнейший valid bbox, добавляй 10% контекста с каждой стороны, ограничивай изображением. Производные одного фото не могут попасть в разные split. Если обнаружены межсплитовые дубликаты, документируй; официальный бенчмарк не модифицируй незаметно.
Политика утечек (решение пользователя, этап 4.0): из train/validation исключаются все записи, точная копия которых (SHA-256 файла) есть в официальном test, и все записи, одинаковое содержимое которых размечено разными породами (если такая пара целиком внутри train/validation — обе копии). Official test не изменяется; его записи помечаются флагами dup_with_test / label_conflict в manifest. Исключения пишутся в data/prepared/excluded.csv с причиной. Manifest дополнительно содержит n_boxes. Train/validation/test не должны пересекаться по содержимому (проверяется тестом).
Не генерируй фиктивные реальные фотографии для заполнения недоступных данных.

Каталог: английские названия из mapping, русский перевод — отдельное поле. При отсутствии проверенного перевода используй английский fallback, не меняй индекс. Не придумывай медицинские/поведенческие описания пород.

## 8. Модель и preprocessing
EfficientNet-B0, ImageNet pretrained, новый classifier на 120 классов. Используй torchvision weights enum и документированный API установленной версии.

Один модуль deterministic preprocessing для validation/test/app. Получи параметры transforms выбранных весов; сохрани resize/crop/interpolation/mean/std/antialias и порядок операций в metadata. Не рассчитывай на defaults, способные поменяться после обновления пакета.
Train аугментации умеренные; eval/app — без случайности.

Стартовая конфигурация:
seed=42, batch_size=16, num_workers=0, precision=float32,
head_epochs=3, head_lr=0.001,
fine_tune_epochs=5, fine_tune_lr=0.0001,
optimizer=AdamW, weight_decay=0.0001, early_stopping_patience=3.

Этап A: замороженный backbone, только classifier обучается. Backbone BatchNorm остаётся eval; model.train() не должен незаметно переактивировать эти слои. Проверяй режимы тестом.
Этап B: точный перечень размороженных блоков в config; новый optimizer; согласованная политика BatchNorm. Лучший checkpoint — validation Top-1, при равенстве меньший validation loss.

Device auto: CUDA → MPS → CPU. Явный --device обязан либо работать, либо дать ошибку; auto не означает скрытый fallback посреди обучения. Запиши фактическое устройство. Начни float32, AMP необязателен. OOM — понятный отказ с советом снизить batch 16→8→4.

До full run: forward [B,120], finite loss, backward, optimizer step, save/load, inference. Tiny overfit test — отдельная диагностическая команда. Synthetic smoke проверяет pipeline, но не доказывает качество.
Сохраняй last и best. Для --resume сохраняй optimizer, epoch, config, random states и scheduler, если он используется. Запрети resume с несовместимым mapping/config либо требуй явный новый run.

Не обещай побитовую воспроизводимость между устройствами. Фиксируй seed, версии, manifest hash, git commit и конфигурацию.

## 9. Bundle и inference
artifacts/model_bundle/{weights.pt,metadata.json,class_mapping.json,metrics.json}.
metadata: schema_version, model_version, architecture, num_classes, preprocessing, pretrained provenance, crop policy, seed, manifest hash, weights hash, package versions, commit, timestamp UTC.

Приложение создаёт architecture с weights=None, загружает собственный state_dict из доверенного bundle, проверяет schema/checksum/mapping/shapes, eval и smoke forward. Не скачивай pretrained weights при старте приложения. Не принимай checkpoint от пользователя.
Используй безопасную загрузку state_dict, доступную в выбранной версии PyTorch; не сериализуй весь Python model object.

Контракт:
predict(image: PIL.Image.Image, roi: tuple[int,int,int,int] | None, top_k: int = 3) -> PredictionResult.
Result: request_id, model_version, predictions[class_id,breed_key,name_ru,name_en,score], timings_ms, warnings.
Inference: torch.inference_mode(), eval(), tensor/device consistency. Softmax по 120 logits; проверка finite. top_k 1..120; сортировка score descending, class_id ascending при равенстве. Top-3 не перенормировать до суммы 1.
Не порогуй max softmax для утверждения «это не собака» или «порода неизвестна».

Модель кешируется один раз на процесс; пользовательские фото не кешируются глобально. Сериализуй доступ к общей модели lock. Отсутствие bundle — понятное сообщение с инструкцией, не случайные ответы.

## 10. UI и feedback
Streamlit UI, русский язык. Состояния EMPTY / VALIDATING / READY / PREDICTING / RESULT / ERROR.
Новый файл сбрасывает ROI/result. Новая ROI сбрасывает результат или явно делает устаревшим. Повторный запуск во время inference блокируется. Clear очищает session references. Сессии изолированы.
Feedback: SQLite, parameterized SQL, короткие транзакции. Поля: id, unique submission_id, created_at UTC, model_version, top_predictions_json, user_verdict, optional user_breed_key.
Verdict enum: looks_correct, looks_wrong, unknown, mixed_breed, breed_not_supported.
submission_id генерируется для результата; duplicate submit идемпотентен. Фото/EXIF/GPS не записываются. SQL connections не разделяй небезопасно между потоками. Ошибка feedback не должна ломать показ prediction.
Каталог пород searchable. Иллюстрации optional: только с проверенными правами и атрибуцией.

## 11. Оценка и benchmark
Метрики validation/test раздельно: Top-1, Top-3, macro F1 с явным списком 120 labels, per-class recall, confusion matrix, часто путаемые пары. Проверяй непустую выборку.
Test не используется для выбора checkpoint. Отчёт связывай с checksum модели, manifest и preprocessing.
Внешние фото с подтверждённой породой — отдельная оценка; неизвестные породы/метисы/не-собаки — анализ ограничений, не смешивать в одну accuracy.
Stanford Dogs происходит из ImageNet, как и pretrained weights: явно опиши возможное пересечение происхождения данных. Не представляй встроенный test как гарантированную внешнюю независимую оценку.
Никаких выдуманных метрик и обязательного accuracy до baseline. Без выполненного обучения metrics.json содержит статус not_evaluated, а не fake numbers.

Benchmark: warmup, несколько повторов, median/p95, отдельно preprocessing/model/total. Синхронизируй CUDA/MPS перед остановкой соответствующего таймера. Укажи устройство, размеры входа, версии, число повторов. Цель warm inference ≤3 секунд — ориентир, не подтверждённый факт.

## 12. Проверки
Unit:
- valid JPEG/PNG/WebP; alpha/grayscale; EXIF orientation;
- corrupt/oversized/animated/bomb/unsupported images;
- ROI preview scaling, rounding, bounds, minimum size;
- mapping unique continuous 120 labels;
- top-k order/ties/finiteness; sum all probabilities ≈1;
- shared eval preprocessing;
- freeze backbone and BN modes;
- feedback enums, uniqueness, no image persistence.
Integration:
- bundle save/load, CPU inference, missing/bad bundle;
- full image→ROI→prediction;
- session reset/isolation;
- app startup without network after dependencies and bundle are installed.

Default tests должны проходить без скачивания данных и весов. Реальную модель проверяй отдельным integration marker; отсутствие bundle — явный skip, не «всё проверено». Test doubles разрешены только в тестах. Реальный inference smoke — отдельная обязательная команда перед релизом.
CI по возможности: ruff check, ruff format --check, pytest CPU. Не блокируй базовые тесты наличием GPU.

## 13. CLI и команды
Реализуй и проверь команды (при необходимости адаптируй флаги, синхронно исправляя README):
uv sync
uv run python scripts/prepare_dataset.py --images-dir data/images/Images --annotations-dir data/annotations/Annotation --splits-dir data/splits --output-dir data/prepared
uv run python scripts/train.py --config configs/training.yaml --smoke-test
uv run python scripts/train.py --config configs/training.yaml
uv run python scripts/train.py --config configs/training.yaml --benchmark
uv run python scripts/train.py --config configs/training.yaml --resume artifacts/runs/<run_id>/last.pt
uv run python scripts/export_bundle.py --run-dir artifacts/runs/<run_id>
uv run python scripts/evaluate.py --bundle artifacts/model_bundle --manifest data/prepared/manifest.csv --split test
uv run python scripts/benchmark.py --bundle artifacts/model_bundle --image <path>
uv run python scripts/check_photos.py --input-dir <folder> [--labels <labels.csv>]
uv run python scripts/package_bundle.py pack
uv run python scripts/package_bundle.py verify dist/<model_version>.zip --extract-to artifacts
uv run streamlit run app.py
uv run pytest
uv run ruff check .
uv run ruff format --check .

При использовании pip документируй эквиваленты. Не выводи команды как рабочие, пока CLI не реализован.

## 14. Порядок реализации
1. Inspect, plan, dependencies, config/schemas, tests scaffold.
2. Image validation и ROI с tests.
3. Dataset parser/manifests/mapping и fixture-based tests.
4. Model/training/preprocessing и smoke.
5. Bundle/predictor и tests.
6. UI: полный сценарий, состояния, ошибки.
7. Feedback/catalog.
8. Реальное обучение/оценка, если разрешены и доступны.
9. Benchmark, документация, clean-start check.

Сокращай сначала иллюстрации, историю и оформление. Не сокращай mapping, input validation, test isolation, honest reporting.
Публичный deploy — optional и только с разрешением пользователя. Проверь память, persistent storage, доставку weights, ограничения и лицензии. Не обещай постоянную SQLite на ephemeral filesystem.

## 15. Документация и приёмка
README: цель/ограничения, установка, bundle, запуск, данные, train/eval/test, troubleshooting.
MODEL_CARD: architecture, provenance, split, preprocessing, реальные метрики или not_evaluated, внешняя проверка, limitations.
Reports: фактические команды, результаты, устройство, checkpoint, ошибки.

Definition of Done:
- рабочий UI с ROI и реальным top-3;
- consistent mapping и preprocessing;
- защищённая обработка файлов;
- отсутствие фото в feedback/logs;
- scripts подготовки/train/eval;
- tests/linters;
- чистый запуск без датасета и повторной загрузки pretrained weights;
- честные метрики и ограничения;
- README + MODEL_CARD.

Разделяй статусы:
A. code complete — код и тесты готовы;
B. model trained — реальные веса созданы;
C. validated — реальная оценка и пользовательский сценарий проверены;
D. deployed — опубликовано с разрешением.
Если B/C не выполнены, не объявляй проект полностью завершённым.

## 16. Финальная самопроверка агента
Перед итоговым ответом:
1. Просмотри git diff и список изменённых файлов.
2. Запусти доступные lint/tests; приведи реальные outcomes.
3. Проверь README-команды и пути.
4. Проверь отсутствие secrets, данных, checkpoints и SQLite в tracked files.
5. Проверь mapping/preprocessing/bundle consistency.
6. Проверь новый файл/ROI/session reset.
7. Проверь отсутствие fake inference/метрик в release path.
8. Запиши невыполненные проверки и причины.
9. Обнови PROGRESS.

Финальный ответ: что сделано; что реально проверено; как запустить; где находится bundle; что блокирует завершение; какие известны ограничения. Не скрывай skips и недоступное обучение. Не утверждай, что запускал команды, если не запускал.
