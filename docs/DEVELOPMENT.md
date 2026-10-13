# DogLens — руководство разработчика

Подробное техническое руководство: данные, обучение, оценка, bundle, приложение, проверки.
Краткое описание продукта — в [README.md](../README.md) / [README.ru.md](../README.ru.md);
статус работ — в [PROGRESS.md](PROGRESS.md); инструкции для агента-разработчика — в
[CLAUDE.md](../CLAUDE.md).

## Установка

Требуется [uv](https://docs.astral.sh/uv/) и Python 3.11 (uv установит его сам).

```bash
uv sync
```

## Запуск приложения

Нужен готовый bundle в `artifacts/model_bundle/`: `weights.pt`, `metadata.json`,
`class_mapping.json`, `metrics.json`. Веса в Git не хранятся, их получают через
обучение и экспорт (см. ниже) или передают отдельно. Другой путь задаётся переменной
`DOGLENS_BUNDLE_DIR`.

```bash
uv run streamlit run app.py        # затем откройте http://localhost:8501
```

Приложение работает локально на CPU и ничего не скачивает при запуске. Если bundle отсутствует
или повреждён, показывается сообщение с инструкцией, а кнопка распознавания недоступна.

Сценарий:
1. Загрузите фото (JPEG/PNG/WebP, до 10 МиБ и 20 Мпикс). Формат проверяется по содержимому,
   EXIF-ориентация учитывается.
2. Оставьте «Использовать всё изображение» или выключите переключатель и задайте рамку **одной**
   собаки двумя слайдерами в координатах превью. Координаты пересчитываются в пиксели исходного
   фото: floor для min, ceil для max. Рамка должна быть не меньше 64×64 px исходного фото.
3. Справа показан **фактический кроп**, который увидит модель:
   - к рамке добавляется по 10% ширины слева и справа и по 10% высоты сверху и снизу, с
     обрезкой по границам фото; так же формировался вход при обучении;
   - в режиме «всё изображение» расширения нет;
   - контекст добавляется ровно один раз, в общей функции `roi.model_region`; её используют и
     превью, и predictor.
4. «Показать похожие породы» — top-3 с оценками модели.
   - Оценки — softmax среди 120 поддерживаемых пород, а не вероятность правильного ответа.
   - Проверенных русских названий пока нет, показываются английские.
5. Отзыв: вариант ответа и, по желанию, порода. «Очистить» удаляет фото и результат из сессии.

Вкладка «Поддерживаемые породы» — каталог 120 пород с поиском. Он сверяется с mapping bundle.

Ограничения:
- модель всегда выбирает одну из 120 пород, даже для метиса, другой породы или фото без собаки;
- автоматического детектора собак нет;
- точность на ваших фото не измерена (см. `docs/MODEL_CARD.md`).

**Хранение отзывов.**
- SQLite-файл `data/feedback/feedback.sqlite` (вне Git); путь меняется через
  `DOGLENS_FEEDBACK_DB`.
- Сохраняются `submission_id`, время UTC, версия модели, top-3 (`class_id`, `breed_key`, score),
  ответ и необязательная порода.
- Фото, EXIF/GPS, имена файлов и персональные данные не сохраняются.
- Отзывы — не проверенные метки, дообучение по ним не выполняется.
- Ошибка записи отзыва не скрывает результат распознавания.

Фото, область и результат живут только в сессии браузера. Между процессами кешируется только
модель.

### Скорость (CPU, `scripts/benchmark.py`)

```bash
uv run python scripts/benchmark.py --bundle artifacts/model_bundle --image <путь к фото> [--roi xmin,ymin,xmax,ymax]
```

Замер на Intel Core Ultra 7 155H, Windows 11, torch 2.14.1+cpu, 16 потоков, batch 1, 5 прогревов,
медиана / p95 (`reports/benchmark/`):

| Вход | Декодирование | Preprocessing | Forward | `predict()` | Декодирование + `predict()` |
|---|---|---|---|---|---|
| фото 333×500, рамка (кроп 303×500), 30 повторов | 2.0 / 2.3 мс | 4.7 / 6.4 мс | 19.9 / 24.1 мс | 24.9 / 31.0 мс | 26.5 / 35.0 мс |
| синтетический JPEG 4032×3024, всё изображение, 20 повторов | 98.5 / 147 мс | 77.1 / 81.2 мс | 21.4 / 39.1 мс | 133 / 148 мс | 191 / 222 мс |

- `predict()` включает подготовку кропа, preprocessing, forward и top-k, но не декодирование
  файла.
- Холодный старт отдельно: импорт библиотек ≈ 6.9–7.6 с, загрузка bundle ≈ 0.2–0.3 с.
- Цель ≤ 3 с для прогретого inference на этой машине выполняется; на других машинах это
  ориентир, а не гарантия.
- Для больших фото основное время занимают декодирование и уменьшение изображения.

### Проверка на своих фото

```bash
uv run python scripts/check_photos.py --input-dir my_photos [--labels my_photos/labels.csv]
```

Подробности и сценарии ручной проверки — [EXTERNAL_VALIDATION.md](EXTERNAL_VALIDATION.md). Статус: **not_performed**.

## Данные (Stanford Dogs)

Ожидаемая структура (исходные данные только читаются, скрипт их не изменяет):

```
data/images/Images/<wnid>-<Breed>/<image_id>.jpg       # images.tar
data/annotations/Annotation/<wnid>-<Breed>/<image_id>  # annotation.tar (XML без расширения)
data/splits/{train_list,test_list,file_list}.mat       # lists.tar (официальное разделение)
```

Источник: <http://vision.stanford.edu/aditya86/ImageNetDogs/>. Изображения взяты из ImageNet и
разрешены только для некоммерческого исследовательского использования; в репозиторий они не
добавляются. Происхождение и SHA-256 `lists.tar` записаны в `data/splits/SOURCE.json`.

Подготовка выполняет аудит, строит mapping классов, официальный train/test, stratified validation
20% (seed 42) из официального train и каталог пород. Из train/validation исключаются копии
тест-фото и фото с конфликтом меток (одинаковое содержимое под разными породами); официальный test
не изменяется:

```bash
uv run python scripts/prepare_dataset.py \
  --images-dir data/images/Images \
  --annotations-dir data/annotations/Annotation \
  --splits-dir data/splits \
  --output-dir data/prepared
```

Показанные значения совпадают с умолчаниями. Остальные параметры:
- `--catalog-path` (`assets/breeds.json`);
- `--ru-names` (`assets/breed_names_ru.json`);
- `--config` (`configs/training.yaml`: seed, `val_fraction`, `bbox_context`);
- `--workers`;
- `--visual-sample` (26, `0` отключает подборку);
- `--visual-dir`;
- `--allow-changes`.

Подробнее: `--help`. Ненулевой код выхода означает ошибку: 1 — ошибка данных, 2 — неверные аргументы.

Результаты:

| Файл | Содержимое |
|---|---|
| `data/prepared/manifest.csv` | `image_id, relative_path, class_id, breed_key, split, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, dup_with_test, label_conflict, n_boxes`. `relative_path` — относительно `--images-dir`, POSIX-разделители. Рамка — крупнейшая исходная рамка без контекста, `xmax`/`ymax` исключающие. Флаги: `dup_with_test` — test-фото, копия которого была в train; `label_conflict` — такое же содержимое есть под другой породой; `n_boxes` — число рамок в XML |
| `data/prepared/excluded.csv` | все исключённые записи с причиной (`dup_with_test`, `label_conflict`, …), SHA-256 и копией-партнёром |
| `data/prepared/class_mapping.json` | единый mapping: `class_id = официальная метка − 1` |
| `data/prepared/preparation_report.json` | источники, SHA-256, количества, проблемы, дубликаты, соглашения |
| `assets/breeds.json` | каталог, сгенерированный из mapping (тот же `mapping_sha256`) |
| `data/prepared/visual_check/` | панели для визуальной проверки (в Git не добавляются) |

Повторный запуск даёт тот же mapping и manifest. Если новый manifest или mapping отличается от
существующего, скрипт завершается с ошибкой и ничего не перезаписывает. Перезапись разрешается
только явным флагом `--allow-changes`. Все файлы пишутся атомарно.

Отчёт с соглашением координат и найденными проблемами: [reports/dataset_preparation.md](../reports/dataset_preparation.md).

## Pretrained-веса (только для обучения)

Обучение инициализирует backbone официальными весами torchvision
`EfficientNet_B0_Weights.IMAGENET1K_V1`. Файл читается только локально и проверяется по SHA-256
(путь и хеш заданы в `configs/training.yaml`, происхождение — в `data/weights/SOURCE.json`).
Веса в Git не добавляются, неявно не скачиваются и приложением не используются.

```text
URL:     https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth
файл:    data/weights/efficientnet_b0_rwightman-7f5810bc.pth  (21 444 401 байт)
SHA-256: 7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934
```

## Preprocessing

Единый детерминированный pipeline для train, validation, test и приложения хранится явно в
`assets/preprocessing.json`:
- resize меньшей стороны до 256 (bicubic, antialias);
- center crop 224;
- приведение к float;
- нормализация ImageNet mean/std.

Параметры взяты из метаданных весов torchvision; тест проверяет совпадение с `weights.transforms()`.
Вход модели — кроп рамки из manifest с контекстом 10%, вычисляемый на лету. Случайная аугментация
(горизонтальное отражение) применяется только при обучении и только до этого pipeline.

## Обучение

Все режимы используют `configs/training.yaml`:
- EfficientNet-B0, seed 42, batch 16, float32;
- этап A — 3 эпохи головы при замороженном backbone, lr 1e-3;
- этап B — до 5 эпох fine-tuning `features.6–8`, lr 1e-4, BatchNorm `frozen`;
- AdamW, weight decay 1e-4, early stopping patience 3.

Устройство выбирается автоматически (CUDA → MPS → CPU) или задаётся `--device`; явно указанное
недоступное устройство даёт ошибку.

```bash
uv run python scripts/train.py --smoke-test     # проверка пайплайна на 240 изображениях (~30 с)
uv run python scripts/train.py --benchmark      # замер реальных batch и прогноз длительности
uv run python scripts/train.py --run-id <id> [--max-minutes 120]   # полный эксперимент
uv run python scripts/train.py --resume artifacts/runs/<id>/last.pt [--max-minutes N]
```

Каталог запуска `artifacts/runs/<run_id>/` (вне Git):

| Файл | Содержимое |
|---|---|
| `resolved_config.yaml` | итоговая конфигурация запуска |
| `run_metadata.json` | хеши, версии, устройство, git, события start/resume |
| `train.log` | лог обучения |
| `history.json` | метрики по эпохам |
| `last.pt` | состояние после последней завершённой эпохи — точка resume |
| `best_stage_A.pt`, `best_stage_B.pt` | лучшие эпохи каждого этапа |
| `best.pt` | лучший результат всего эксперимента |
| `final_report.json` | итоговый отчёт |

- **Выбор лучшего:** максимум validation Top-1, при равенстве — меньший validation loss. Test не
  используется.
- **Переход A → B:** восстанавливаются веса `best_stage_A.pt`, размораживаются блоки, создаётся
  новый optimizer, early stopping сбрасывается.
- **Resume** возможен только с границы завершённой эпохи. Прерванная эпоха (лимит
  `--max-minutes`, Ctrl+C, сбой) при `--resume` повторяется с начала; продолжение посреди batch
  не поддерживается.
  - При совместимых данных и конфигурации resume воспроизводит непрерывный запуск побитово
    (проверено тестами на CPU).
  - Resume отклоняется, если отличаются mapping, manifest, preprocessing, pretrained-веса,
    устройство или ключевые разделы конфигурации.
- **Лимит времени:** при превышении `--max-minutes` процесс останавливается между batch,
  сохраняет статус `interrupted` и выходит с кодом 3.
- Существующий каталог запуска не перезаписывается.

## Экспорт модели и оценка

```bash
uv run python scripts/export_bundle.py --run-dir artifacts/runs/<id>      # best.pt -> bundle
uv run python scripts/evaluate.py --bundle artifacts/model_bundle --manifest data/prepared/manifest.csv --split test --update-bundle-metrics
```

**Bundle** `artifacts/model_bundle/{weights.pt, metadata.json, class_mapping.json, metrics.json}`
(веса вне Git):
- `weights.pt` — обученная DogLens-модель, а не ImageNet-инициализация;
- при загрузке архитектура создаётся с `weights=None`, ничего не скачивается;
- проверяются схема, SHA-256 весов, mapping и формы тензоров, затем `state_dict` загружается
  строго и выполняется проверочный forward;
- экспорт сравнивает logits bundle и checkpoint на CPU.

**Оценка** пишет в `reports/evaluation/<split>/`:
- `metrics.json` — срезы, per-class, confusion matrix, путаемые пары;
- `summary.md`;
- `predictions.csv` — top-5 для каждого `image_id`;
- `confusion_matrix.csv`;
- `logits.npz` — для пересчёта метрик без forward: `--predictions .../logits.npz`.

Срезы:

| Срез | Правило |
|---|---|
| `full` | все записи split |
| `clean` | `dup_with_test == 0 and label_conflict == 0` |
| `multi_box` | `n_boxes > 1` |
| `single_box` | `n_boxes == 1` |

Macro F1 приводится по фиксированным 120 labels и отдельно по классам, присутствующим в срезе
(`zero_division=0`).

## Доставка model bundle (архив для GitHub Release)

Веса не хранятся в Git. Bundle упаковывается в архив и публикуется как asset GitHub Release:
[v0.1.0](https://github.com/7SUNNY77/DogLens/releases/tag/v0.1.0). Условия использования весов — [MODEL_TERMS.md](MODEL_TERMS.md).

```bash
uv run python scripts/package_bundle.py pack     # -> dist/<model_version>.zip + .zip.sha256
```

`pack`:
- проверяет bundle — схему, SHA-256 весов, mapping, строгую загрузку и inference без сети;
- создаёт детерминированный zip, в котором ровно один каталог `model_bundle/` с четырьмя файлами;
- пишет внешний SHA-256;
- проверяет архив: распаковывает во временный каталог и загружает.
- Каталог `dist/` исключён из Git.

Установка из архива (после того как он будет опубликован и скачан):

```bash
uv run python scripts/package_bundle.py verify <archive>.zip --extract-to artifacts
# или: проверить SHA-256 вручную и распаковать так, чтобы появился artifacts/model_bundle/
```

Без `--sha256` ожидаемый хеш читается из `<archive>.zip.sha256` рядом с архивом. Другой путь к
bundle задаётся переменной `DOGLENS_BUNDLE_DIR`.

## Проверки

```bash
uv run pytest                    # unit-тесты; не требуют датасета и сети
uv run pytest -m integration     # реальные датасет, bundle, predictor и приложение (skip при отсутствии)
uv run ruff check .
uv run ruff format --check .
```

## Troubleshooting

- `official split files not found` — распакуйте `lists.tar` в `data/splits/`. Неофициальное
  разделение автоматически не подставляется.
- `existing manifest ... differs` — подготовка дала другой split или mapping, чем сохранённый.
  Выясните причину; `--allow-changes` используйте только для явного нового запуска.
- Если `vision.stanford.edu` не резолвится локальным DNS, проверьте резолвинг через другой DNS.
  Подлинность архива сверяйте по SHA-256 из `data/splits/SOURCE.json`.
- «Модель не найдена или повреждена» — проверьте, что в `artifacts/model_bundle/` лежат все 4 файла
  и `weights.pt` совпадает с `weights_sha256` из `metadata.json`. Для другого пути задайте
  `DOGLENS_BUNDLE_DIR`. Bundle экспортируется командой `scripts/export_bundle.py`.
- «Область слишком маленькая» — рамка меньше 64×64 px исходного фото; увеличьте её или
  используйте всё изображение.
- «Не удалось сохранить отзыв» — каталог `data/feedback/` недоступен для записи. Результат
  распознавания при этом сохраняется на экране.
- Файл больше 11 МБ отклоняет сам Streamlit (`.streamlit/config.toml`); файлы от 10 до 11 МиБ
  отклоняет проверка DogLens с сообщением на русском.
