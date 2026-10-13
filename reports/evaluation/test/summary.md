# Evaluation: split `test`

- Model: `doglens-effb0-effb0-seed42-exp1-de6c9770` (weights sha256 `de6c9770242e2047…`)
- Manifest sha256 `396b1f630f414f18…`; generated 2026-10-13T18:54:59Z
- Scores are softmax over 120 supported classes, not calibrated probabilities.
- Macro F1: `f1_score(average='macro', zero_division=0)`;
  *all labels* = fixed 120 labels,
  *present* = classes with support > 0 in the slice. Slices with fewer classes are not
  directly comparable on macro metrics.

| Slice | Rule | Images | Classes | Top-1 | Top-3 | Macro F1 (120) | Macro F1 (present) |
|---|---|---|---|---|---|---|---|
| full | `all records of the split` | 8580 | 120 | 0.8725 | 0.9802 | 0.8665 | 0.8665 |
| clean | `dup_with_test == 0 and label_conflict == 0` | 8527 | 120 | 0.8732 | 0.9803 | 0.8670 | 0.8670 |
| multi_box | `n_boxes > 1` | 595 | 111 | 0.8202 | 0.9513 | 0.7128 | 0.7706 |
| single_box | `n_boxes == 1` | 7985 | 120 | 0.8764 | 0.9823 | 0.8710 | 0.8710 |

## Most confused pairs (full slice, true → predicted)

| True | Predicted | Count | Share of true support |
|---|---|---|---|
| siberian_husky | eskimo_dog | 41 | 0.45 |
| shih-tzu | lhasa | 20 | 0.18 |
| silky_terrier | yorkshire_terrier | 20 | 0.24 |
| pembroke | cardigan | 20 | 0.25 |
| scottish_deerhound | irish_wolfhound | 16 | 0.12 |
| irish_wolfhound | scottish_deerhound | 15 | 0.13 |
| staffordshire_bullterrier | american_staffordshire_terrier | 15 | 0.27 |
| border_collie | collie | 14 | 0.28 |
| malamute | eskimo_dog | 14 | 0.18 |
| miniature_pinscher | toy_terrier | 13 | 0.15 |
| toy_poodle | miniature_poodle | 13 | 0.25 |
| american_staffordshire_terrier | staffordshire_bullterrier | 12 | 0.19 |
| english_foxhound | walker_hound | 11 | 0.19 |
| entlebucher | appenzeller | 11 | 0.11 |
| eskimo_dog | siberian_husky | 11 | 0.22 |

## Lowest per-class recall (full slice)

| Breed | Support | Recall |
|---|---|---|
| siberian_husky | 92 | 0.457 |
| border_collie | 50 | 0.620 |
| eskimo_dog | 50 | 0.620 |
| toy_poodle | 51 | 0.647 |
| silky_terrier | 83 | 0.651 |
| american_staffordshire_terrier | 64 | 0.672 |
| staffordshire_bullterrier | 55 | 0.673 |
| miniature_poodle | 55 | 0.673 |
| malamute | 78 | 0.692 |
| miniature_pinscher | 84 | 0.702 |
