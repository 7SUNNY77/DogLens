# Эксперимент `effb0-seed42-exp1`

Единственный полный CPU-эксперимент: 3 эпохи головы + до 5 эпох
fine-tuning с early stopping. Подбор гиперпараметров не проводился.

## Условия

- Устройство: CPU (Intel Core Ultra 7 155H, 16 потоков torch по умолчанию, `num_workers=0`).
- Окружение: Windows 11; torch 2.14.1+cpu, torchvision 0.29.1+cpu; seed 42.
- Конфигурация: `configs/training.yaml` без изменений; копия — `artifacts/runs/effb0-seed42-exp1/resolved_config.yaml`.
- Данные: train 9 551 / val 2 382 (manifest SHA-256 `396b1f63…`), mapping `d9779c77…`.
- Инициализация backbone: `EfficientNet_B0_Weights.IMAGENET1K_V1`, SHA-256 `7f5810bc…4934`.
- Git: эксперимент запущен до первого коммита → `commit: null`, `dirty: true`.

## Предварительный benchmark (`reports/train_benchmark.json`)

`uv run python scripts/train.py --benchmark --benchmark-batches 8 --benchmark-warmup 2`
(2 batch прогрева + 8 замеряемых, медианы, batch 16):

| Операция | с/batch |
|---|---|
| загрузка train (decode + crop + preprocessing) | 0.294 |
| шаг этапа A | 0.212 |
| шаг этапа B | 0.354 |
| загрузка val | 0.356 |
| forward val | 0.193 |

- Batch на эпоху: 597 в train, 149 в val.
- Прогноз: этап A — 6.4 мин/эпоха, этап B — 7.8 мин/эпоха; весь эксперимент без early stopping —
  **58 мин** (в пределах бюджета ≤ 2 ч).
- Настройки производительности (число потоков, workers) не менялись.

## Ход запуска

1. `uv run python scripts/train.py --run-id effb0-seed42-exp1 --max-minutes 120`
   - Этап A прошёл все 3 эпохи.
   - На переходе A→B загрузка `best_stage_A.pt` упала: `weights_only` отклонил объект
     `torch.torch_version.TorchVersion`. В identity checkpoint попадало значение
     `torch.__version__`, а это подкласс `str`, не обычная строка. В тестах версии задавались
     обычными строками, поэтому ошибка не проявилась.
   - Статус запуска — `failed`. `last.pt` уже был сохранён после завершённого этапа A.
2. Исправление кода:
   - версии записываются как `str(...)`, identity и config нормализуются через JSON;
   - загрузчик разрешает `TorchVersion` только для уже записанных файлов этого запуска;
   - добавлены регрессионные тесты.
   - Артефакты запуска не переписывались.
3. `uv run python scripts/train.py --resume artifacts/runs/effb0-seed42-exp1/last.pt --max-minutes 100`
   - Продолжен тот же эксперимент: переход на этап B, восстановление `best_stage_A.pt`, 5 эпох
     этапа B.
   - Статус `completed`, события в `run_metadata.json`: start → failed → resume → completed.

Это не два эксперимента: этап A выполнен один раз, этап B — один раз, с штатной границы эпохи.

## История

| Этап | Эпоха | train loss | val loss | val Top-1 | val Top-3 | train, с | val, с | patience |
|---|---|---|---|---|---|---|---|---|
| A | 1 | 1.2024 | 0.5135 | 0.8573 | 0.9740 | 436 | 70 | 0 |
| A | 2 | 0.3606 | 0.4468 | 0.8724 | 0.9719 | 269 | 70 | 0 |
| A | 3 | 0.2432 | 0.4336 | **0.8745** | 0.9702 | 291 | 67 | 0 |
| B | 1 | 0.2482 | 0.5428 | 0.8476 | 0.9601 | 487 | 72 | 0 |
| B | 2 | 0.1532 | 0.5514 | 0.8497 | 0.9639 | 435 | 46 | 0 |
| B | 3 | 0.1073 | 0.6209 | 0.8321 | 0.9626 | 300 | 67 | 1 |
| B | 4 | 0.0944 | 0.6048 | 0.8392 | 0.9597 | 270 | 66 | 2 |
| B | 5 | 0.0826 | 0.6520 | 0.8338 | 0.9572 | 402 | 66 | 3 |

- Суммарное время эпох — 3 413 с (≈ 57 мин), что согласуется с прогнозом benchmark.
- Эпохи A1 и B1–B2 медленнее, потому что параллельно шли тесты и сборка оценки на том же CPU.
- Этап B прошёл 5 эпох: patience достиг 3/3 на последней эпохе, причина остановки записана как
  `max_epochs`.
- На этапе B train loss падает, а val loss растёт: fine-tuning `features.6–8` с lr 1e-4
  переобучается при этой конфигурации.

## Выбор модели

- Правило: максимум validation Top-1, при равенстве — меньший validation loss. Test не
  использовался.
- **Выбран этап A, эпоха 3** (`best.pt` = `best_stage_A.pt`, SHA-256 `9498b5d4…`): val Top-1
  0.8745, Top-3 0.9702, loss 0.4336. Лучшая эпоха этапа B (B2, 0.8497) хуже.
- Экспорт: `uv run python scripts/export_bundle.py --run-dir artifacts/runs/effb0-seed42-exp1`
  → `artifacts/model_bundle`, model_version `doglens-effb0-effb0-seed42-exp1-de6c9770`,
  weights SHA-256 `de6c9770…7393`.
- Проверка bundle: строгая загрузка на CPU, logits совпали с checkpoint (max |diff| = 0).
  `evaluate.py --split val` на bundle воспроизвёл val Top-1 0.8745 / Top-3 0.9702.

## Оценка на test

`uv run python scripts/evaluate.py --bundle artifacts/model_bundle --manifest data/prepared/manifest.csv --split test --update-bundle-metrics`
— выполнена один раз после выбора модели. Результаты: `reports/evaluation/test/summary.md`,
`metrics.json`, `predictions.csv`. Сводка — в `docs/MODEL_CARD.md`.
