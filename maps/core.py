"""Core MAPS implementation."""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

_EPS = 1e-8
_PCA_SOLVER = os.environ.get("MAPS_PCA_SOLVER", "full")


def _as_2d(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _neighbourhood(
    states: np.ndarray,
    futures: np.ndarray,
    k: int,
    pca_components: Optional[int],
    eps: float = _EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the neighbour graph and normalised future directions."""
    states = _as_2d("states", states)
    futures = _as_2d("futures", futures)
    n_windows = len(states)

    if len(futures) != n_windows:
        raise ValueError("states and futures must contain the same number of rows.")
    if n_windows < 2:
        raise ValueError("At least two windows are required.")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    if pca_components is not None and (
        not isinstance(pca_components, (int, np.integer)) or pca_components < 1
    ):
        raise ValueError("pca_components must be a positive integer or None.")

    states_norm = (states - states.mean(axis=0)) / (states.std(axis=0) + eps)
    futures_norm = (futures - futures.mean(axis=0)) / (futures.std(axis=0) + eps)

    states_for_knn = states_norm
    if pca_components is not None and states_norm.shape[1] > pca_components:
        n_components = min(pca_components, n_windows - 1, states_norm.shape[1])
        states_for_knn = PCA(
            n_components=n_components, svd_solver=_PCA_SOLVER
        ).fit_transform(states_norm)

    model = NearestNeighbors(
        n_neighbors=min(k + 1, n_windows), metric="euclidean"
    ).fit(states_for_knn)
    neighbors = model.kneighbors(states_for_knn, return_distance=False)[:, 1:]

    future_norms = np.linalg.norm(futures_norm, axis=1, keepdims=True)
    futures_unit = futures_norm / np.maximum(future_norms, eps)
    degenerate = future_norms[:, 0] < eps
    return neighbors, futures_unit, degenerate


def _coherence(
    neighbors: np.ndarray,
    futures_unit: np.ndarray,
    degenerate: np.ndarray,
    chunk: Optional[int] = None,
) -> dict[str, Any]:
    """Compute per-window directional coherence for a neighbour graph."""
    n_windows = len(neighbors)
    signed = np.zeros(n_windows)

    if chunk is None:
        elements_per_row = max(1, neighbors.shape[1] * futures_unit.shape[1])
        chunk = int(np.clip(20_000_000 // elements_per_row, 1, 2048))

    for start in range(0, n_windows, chunk):
        stop = min(start + chunk, n_windows)
        neighbor_ids = neighbors[start:stop]
        neighbor_degenerate = degenerate[neighbor_ids]
        query_degenerate = degenerate[start:stop]

        similarities = np.einsum(
            "nkd,nd->nk", futures_unit[neighbor_ids], futures_unit[start:stop]
        )

        valid = ~neighbor_degenerate
        n_valid = valid.sum(axis=1)
        mean_normal = np.where(
            n_valid > 0,
            (similarities * valid).sum(axis=1) / np.maximum(n_valid, 1),
            0.0,
        )
        mean_degenerate = neighbor_degenerate.mean(axis=1)
        signed[start:stop] = np.where(
            query_degenerate, mean_degenerate, mean_normal
        )

    alignments = np.abs(signed)
    return {
        "unpredictability_ratios": 1.0 - alignments,
        "direction_alignments": alignments,
        "signed_alignments": signed,
        "global_mean_alignment": float(np.mean(alignments)),
        "n_degenerate": int(degenerate.sum()),
    }


def local_direction_coherence(
    states: np.ndarray,
    futures: np.ndarray,
    k: int = 20,
    pca_components: Optional[int] = 50,
    eps: float = _EPS,
) -> dict[str, Any]:
    """Compute MAPS directional coherence for preconstructed windows.

    ``states`` and ``futures`` must have shape ``(n_windows, n_dimensions)``.
    The result contains per-window unpredictability and signed/absolute
    alignments, their global mean, and the number of degenerate future windows.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    neighbors, futures_unit, degenerate = _neighbourhood(
        states, futures, k, pca_components, eps
    )
    return _coherence(neighbors, futures_unit, degenerate)


def compute_predictability_functional(u_ratios: np.ndarray) -> float:
    """Aggregate per-window unpredictability into a score in ``[0, 1]``."""
    values = np.asarray(u_ratios, dtype=float)
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return 0.0
    return float(1.0 - np.mean(np.clip(valid, 0.0, 1.0)))


def maps_rolling_windows(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
    horizon: int,
    k: Optional[int] = None,
    pca_components: Optional[int] = 50,
    stride: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute the MAPS score from aligned input and target time series.

    ``X`` has shape ``(time, features)`` and ``y`` has shape ``(time,)`` or
    ``(time, targets)``. The score is the mean absolute directional coherence
    of future targets among nearest-neighbour input windows.
    """
    X = _as_2d("X", X)
    y = _as_2d("y", y)
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of time steps.")
    for name, value in (
        ("lookback", lookback),
        ("horizon", horizon),
        ("stride", stride),
    ):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    n_possible = len(y) - lookback - horizon + 1
    if n_possible < 2:
        raise ValueError(
            "The series is too short; lookback and horizon must produce at "
            "least two windows."
        )

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + _EPS)
    y = (y - y.mean(axis=0)) / (y.std(axis=0) + _EPS)

    indices = list(range(0, n_possible, stride))
    if len(indices) < 2:
        raise ValueError("stride leaves fewer than two windows.")
    states = np.asarray([X[t : t + lookback].ravel() for t in indices])
    futures = np.asarray(
        [y[t + lookback : t + lookback + horizon].ravel() for t in indices]
    )

    if k is None:
        k = int(np.floor(np.log2(len(states))))

    local_results = local_direction_coherence(
        states, futures, k=k, pca_components=pca_components
    )
    predictability = compute_predictability_functional(
        local_results["unpredictability_ratios"]
    )

    if verbose:
        negative = float(np.mean(local_results["signed_alignments"] < 0))
        print(
            f"Created {len(states)} windows "
            f"(L={lookback}, H={horizon}, k={k})\n"
            f"Predictability: {predictability:.4f}, "
            f"mean |alignment|: {local_results['global_mean_alignment']:.4f}, "
            f"negative mean similarity: {negative:.1%}, "
            f"degenerate windows: {local_results['n_degenerate']}"
        )

    return {
        "predictability": predictability,
        "local_results": local_results,
        "metadata": {
            "lookback": lookback,
            "horizon": horizon,
            "k": k,
            "pca_components": pca_components,
            "n_windows": len(states),
            "method": "maps_v10",
        },
        "windows": {"indices": indices, "states": states, "futures": futures},
    }

