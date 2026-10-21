# MAPS

This repository contains the official implementation of the method presented in our paper, **Model-agnostic Predictability Quantification of Multivariable Time Series**.

MAPS quantifies the predictability of a multivariable time-series forecasting task without fitting or evaluating a forecasting model. It measures whether similar input windows lead to directionally coherent future targets and returns a score between `0` (low predictability) and `1` (high predictability).

## Highlights

- **Model-agnostic:** evaluates a forecasting task independently of any particular model architecture.
- **Multivariable:** supports multiple input variables and one or more target variables.
- **Horizon-aware:** measures predictability for a specified lookback window and forecasting horizon.
- **Scalable:** supports PCA-based state compression and chunked coherence computation.

## Setting Up the Environment

We recommend using Conda to manage dependencies. MAPS requires Python 3.9 or later.

### 1. Create and activate a Conda environment

```bash
conda create --name maps python=3.9
conda activate maps
```

### 2. Install MAPS

From the repository root, run:

```bash
pip install -e .
```

## Running MAPS

```python
import numpy as np

from maps import maps_rolling_windows

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 4))
y = X[:, 0]

result = maps_rolling_windows(
    X,
    y,
    lookback=24,
    horizon=6,
)

print(result["predictability"])
```

`X` must have shape `(time, features)`. The target `y` can have shape `(time,)` for a single target or `(time, targets)` for multiple targets. Both arrays must contain the same number of time steps and only finite values.

### Main parameters

- `lookback`: number of past time steps in each input window.
- `horizon`: number of future time steps in each target window.
- `k`: number of nearest neighbours. By default, MAPS uses `floor(log2(N))`, where `N` is the number of rolling windows.
- `pca_components`: maximum number of principal components used to represent input windows. The default is `50`; set it to `None` to disable PCA.
- `stride`: step between consecutive rolling windows. The default is `1`.
- `verbose`: prints a short summary when set to `True`.

### Output

`maps_rolling_windows` returns a dictionary containing:

- `predictability`: the overall MAPS score.
- `local_results`: per-window unpredictability ratios and directional alignments.
- `metadata`: the lookback, horizon, neighbourhood, PCA, and window settings used for the run.
- `windows`: the generated window indices, input states, and future targets.

## Advanced Configuration

MAPS uses the deterministic exact PCA solver by default. For large, full-rank state matrices, set `MAPS_PCA_SOLVER=randomized` before running the code to reduce PCA runtime:

```bash
export MAPS_PCA_SOLVER=randomized
```

For rank-deficient data, the default exact solver is recommended.

