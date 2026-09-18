# Third-party software

## TimesFM

- Source: https://github.com/google-research/timesfm
- Pinned revision: `bf88c5dc88f6275b7071ebcd9051406c11435c0c`
- Copyright 2025 Google LLC.
- License: Apache License, Version 2.0; included in `licenses/TimesFM.txt`.

Adaptation: `spefm/model.py` adds spatial prompts and prompt-aware masking to the TimesFM forward interface. Model weights are supplied locally.

## Forecasting baselines

ARIMA uses [statsmodels](https://github.com/statsmodels/statsmodels), licensed under BSD-3-Clause. Installed dependencies retain their respective licenses.

## Informer

- Source: https://github.com/zhouhaoyi/Informer2020
- Authors: Haoyi Zhou and the Informer2020 contributors.
- Pinned revision: `29f2a739226a509202a092b464163da81fa74960`.
- License: Apache License, Version 2.0; included in `licenses/Informer.txt`.
- Included modules: `models/model.py`, `models/attn.py`, `models/encoder.py`, `models/decoder.py`, `models/embed.py` and `utils/masking.py`, located under `baselines/vendor/informer/`.

Modified files: `baselines/vendor/informer/UPSTREAM.json`. Adapter: `baselines/temporal.py`.

Reference: Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting* (https://arxiv.org/abs/2012.07436).

## PatchTST

- Source: https://github.com/yuqinie98/PatchTST
- Authors: Yuqi Nie and the PatchTST contributors.
- Pinned revision: `204c21efe0b39603ad6e2ca640ef5896646ab1a9`.
- License: Apache License, Version 2.0; included in `licenses/PatchTST.txt`.
- Included modules: `PatchTST_supervised/layers/PatchTST_backbone.py`, `PatchTST_supervised/layers/PatchTST_layers.py` and `PatchTST_supervised/layers/RevIN.py`, located under `baselines/vendor/patchtst/`.

Modified files: `baselines/vendor/patchtst/UPSTREAM.json`. Adapter: `baselines/temporal.py`.

The included RevIN module attributes https://github.com/ts-kim/RevIN. Its MIT license, Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI), is included in `licenses/RevIN.txt`.

Reference: Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* (https://arxiv.org/abs/2211.14730).

## A3TGCN

- Source: https://github.com/benedekrozemberczki/pytorch_geometric_temporal
- Copyright (c) 2020 Benedek Rozemberczki.
- Pinned revision: `ea40a6a396b6688a94d7482d9d5fd288eaa2cb3b`.
- License: MIT; included in `licenses/PyTorchGeometricTemporal.txt`.
- Included modules: `torch_geometric_temporal/nn/recurrent/attentiontemporalgcn.py` and `temporalgcn.py`, located under `baselines/vendor/pgt/`.

Modified files: `baselines/vendor/pgt/UPSTREAM.json`. Adapter: `baselines/graph.py`, using `A3TGCN2` and PyTorch Geometric `GCNConv`.

References: [A3T-GCN](https://arxiv.org/abs/2006.11583), [T-GCN](https://arxiv.org/abs/1811.05320).
