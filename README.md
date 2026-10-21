# MAPS

MAPS is a model-agnostic score for the predictability of a multivariable time-series forecasting task. It measures whether similar input windows have directionally coherent future targets. Scores range from `0` (low coherence) to `1` (high coherence).

This repository contains only the MAPS implementation used by the accompanying paper, *Model-agnostic predictability quantification of multivariable time series*.

## Install

```bash
pip install .
```

## Use

```python
import numpy as np
from maps import maps_rolling_windows

X = np.random.default_rng(0).normal(size=(500, 4))
y = X[:, 0]

result = maps_rolling_windows(X, y, lookback=24, horizon=6)
print(result["predictability"])
```

`X` must have shape `(time, features)` and `y` shape `(time,)` or `(time, targets)`. By default, MAPS uses `floor(log2(N))` neighbours and at most 50 principal components, where `N` is the number of rolling windows. Pass `pca_components=None` to disable PCA. For large, full-rank state matrices, `MAPS_PCA_SOLVER=randomized` can reduce PCA runtime; the default exact solver is deterministic and is recommended for rank-deficient data.

The returned dictionary contains the overall `predictability` score, per-window `local_results`, run `metadata`, and the generated `windows`.

## License

MIT

