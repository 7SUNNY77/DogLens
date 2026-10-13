# DogLens model weights — terms of use

These terms apply to the trained model bundle (`weights.pt`, `metadata.json`,
`class_mapping.json`, `metrics.json`) distributed as a release asset
(`doglens-effb0-effb0-seed42-exp1-de6c9770.zip`). They do **not** change the MIT license of the
source code (see [LICENSE](../LICENSE)), and the MIT license does **not** apply to the weights.

## Why separate terms

The weights were trained on **Stanford Dogs**, which is built from **ImageNet** images. The backbone
was initialized from torchvision's `EfficientNet_B0_Weights.IMAGENET1K_V1`. Stanford Dogs gives no
explicit license on its official page, and ImageNet's terms of access are oriented to
non-commercial research and education (not re-verified in this project). To stay on the safe side,
the weights are offered under conservative terms.

## Terms

1. **Non-commercial research and educational use only.** You may download, run, evaluate and study
   the weights for personal, research or teaching purposes.
2. **No commercial use** of the weights or of outputs produced at scale with them, unless you have
   obtained the necessary rights to the underlying data yourself.
3. **Attribution.** If you share results or derived work, credit DogLens and cite the data and
   model sources:
   - A. Khosla, N. Jayadevaprakash, B. Yao, L. Fei-Fei. *Novel dataset for Fine-Grained Image
     Categorization.* FGVC Workshop, CVPR 2011.
   - J. Deng et al. *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR 2009.
   - torchvision `EfficientNet_B0_Weights.IMAGENET1K_V1` (EfficientNet: M. Tan, Q. V. Le, ICML 2019).
4. **Redistribution** of the unmodified archive is allowed for the same non-commercial purposes, with
   these terms and the SHA-256 checksum kept alongside it.
5. **No warranty.** The weights are provided "as is". They are not suitable for decisions about a
   dog's breed or ancestry, breeding, insurance, veterinary or legal matters
   (see [MODEL_CARD.md](MODEL_CARD.md)).
6. These terms do not grant any rights to the Stanford Dogs or ImageNet images themselves, which are
   not distributed by this project.

If you are a rights holder and believe these terms are inappropriate, please open an issue in the
repository.
