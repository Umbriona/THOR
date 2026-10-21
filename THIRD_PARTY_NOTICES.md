# Third-party notices

THOR is licensed under CC BY-NC 4.0 (see `LICENSE`). Parts of this repository
are derived from third-party works under their own licenses, listed here.

---

## TensorFlow Addons

**Where:** `src/utils/layers_new.py` — the `SpectralNormalization` class
(and the `SelfAttentionSN` / `ResModSN` layers that build on it).

**Source:** https://github.com/tensorflow/addons — `v0.14.0`,
`tensorflow_addons/layers/spectral_normalization.py`

**License:** Apache License 2.0

```
Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

**Modifications:** the class was vendored into this repository so the layer
could be used without depending on `tensorflow-addons` at import time, and
adapted for 1-D convolutions as used by THOR's generator and discriminator.

---

## Models and data used at runtime

These are **not** redistributed in this repository. They are downloaded when
you run the code, and your use of them is governed by their own licenses.

| | |
|---|---|
| **ESM-1v** (`facebook/esm1v_t33_650M_UR90S_1`) | Downloaded from Hugging Face at runtime by `generate_thor.py`, `compute_MLM_features.py` and `finetune_ESM.py`. See the model card for its license. |
| **NGC TensorFlow base image** | `nvcr.io/nvidia/tensorflow:23.09-tf2-py3`, pulled when building `env/singularity/thor.recipe`. Subject to the NVIDIA Deep Learning Container license. |
| **Python dependencies** | Installed from PyPI at build time; each retains its own license. |

Training data derives from IMG/JGI and UniProt; see the manuscript for
accession details and the terms attached to those sources.
