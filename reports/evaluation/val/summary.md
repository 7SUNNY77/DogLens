# Evaluation: split `val`

- Model: `doglens-effb0-effb0-seed42-exp1-de6c9770` (weights sha256 `de6c9770242e2047…`)
- Manifest sha256 `396b1f630f414f18…`; generated 2026-10-13T19:13:29Z
- Scores are softmax over 120 supported classes, not calibrated probabilities.
- Macro F1: `f1_score(average='macro', zero_division=0)`;
  *all labels* = fixed 120 labels,
  *present* = classes with support > 0 in the slice. Slices with fewer classes are not
  directly comparable on macro metrics.

| Slice | Rule | Images | Classes | Top-1 | Top-3 | Macro F1 (120) | Macro F1 (present) |
|---|---|---|---|---|---|---|---|
| full | `all records of the split` | 2382 | 120 | 0.8745 | 0.9702 | 0.8737 | 0.8737 |
| clean | `dup_with_test == 0 and label_conflict == 0` | 2382 | 120 | 0.8745 | 0.9702 | 0.8737 | 0.8737 |
| multi_box | `n_boxes > 1` | 143 | 75 | 0.8252 | 0.9091 | 0.4976 | 0.7962 |
| single_box | `n_boxes == 1` | 2239 | 120 | 0.8776 | 0.9741 | 0.8771 | 0.8771 |

## Most confused pairs (full slice, true → predicted)

| True | Predicted | Count | Share of true support |
|---|---|---|---|
| siberian_husky | eskimo_dog | 9 | 0.45 |
| american_staffordshire_terrier | staffordshire_bullterrier | 6 | 0.30 |
| standard_schnauzer | miniature_schnauzer | 6 | 0.30 |
| english_foxhound | walker_hound | 5 | 0.25 |
| border_collie | collie | 5 | 0.26 |
| eskimo_dog | siberian_husky | 5 | 0.25 |
| collie | shetland_sheepdog | 4 | 0.21 |
| toy_poodle | miniature_poodle | 4 | 0.21 |
| shih-tzu | lhasa | 3 | 0.15 |
| staffordshire_bullterrier | american_staffordshire_terrier | 3 | 0.15 |
| lakeland_terrier | airedale | 3 | 0.16 |
| miniature_schnauzer | standard_schnauzer | 3 | 0.16 |
| silky_terrier | yorkshire_terrier | 3 | 0.15 |
| lhasa | shih-tzu | 3 | 0.15 |
| german_shepherd | malinois | 3 | 0.15 |

## Lowest per-class recall (full slice)

| Breed | Support | Recall |
|---|---|---|
| siberian_husky | 20 | 0.450 |
| tibetan_mastiff | 20 | 0.550 |
| eskimo_dog | 20 | 0.600 |
| border_collie | 19 | 0.632 |
| standard_schnauzer | 20 | 0.650 |
| appenzeller | 20 | 0.650 |
| collie | 19 | 0.684 |
| toy_poodle | 19 | 0.684 |
| shih-tzu | 20 | 0.700 |
| english_foxhound | 20 | 0.700 |
