
"""PCA-score Quantile-Guided K-Means initialization.

Algorithm:
1. Fit deterministic one-component PCA to the supplied feature matrix.
2. Orient PC1 so its largest-absolute loading is positive.
3. Rank scores ascending; break exact score ties by original row index.
4. Select floor(p * (n - 1)) at equally spaced probabilities from 0 to 1.
5. Return complete observed rows as initial centroids.

No point-to-centroid distance evaluations are performed here.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def _groups_by_key(keys):
    """Return groups of positions sharing an identical key."""
    groups = {}
    for position, key in enumerate(keys):
        groups.setdefault(key, []).append(position)
    return [positions for positions in groups.values() if len(positions) > 1]


def quantile_guided_init(X: np.ndarray, k: int, seed: int | None = None):
    """Select K observed rows using quantiles of deterministic PC1 scores.

    Parameters
    ----------
    X : np.ndarray
        Finite, two-dimensional feature matrix. The caller is responsible
        for applying the experiment's chosen preprocessing consistently.
    k : int
        Number of initial centroids; must be at least 2.
    seed : int | None
        Accepted for compatibility with the shared runner. The full PCA
        solver is deterministic; this seed is not used to randomize PCA.

    Returns
    -------
    centers : np.ndarray
        K complete rows selected from X.
    info : dict
        Reproducibility and duplicate/tie diagnostics.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    n, n_features = X.shape
    if n_features < 1:
        raise ValueError("X must contain at least one feature.")
    if not np.isfinite(X).all():
        raise ValueError("X contains missing or non-finite values.")
    if k < 2:
        raise ValueError("Quantile-Guided initialization requires K >= 2.")
    if n < k:
        raise ValueError("Number of observations must be at least K.")

    # Full SVD avoids randomized PCA behavior.
    pca = PCA(n_components=1, svd_solver="full")
    scores = pca.fit_transform(X).ravel()
    loadings = pca.components_[0].copy()

    # Make the sign convention deterministic:
    # the first maximum-absolute loading must be positive.
    pivot = int(np.argmax(np.abs(loadings)))
    if loadings[pivot] < 0:
        scores = -scores
        loadings = -loadings

    # Primary key: score ascending. Secondary key: original row index.
    original_indices = np.arange(n, dtype=int)
    sorted_indices = np.lexsort((original_indices, scores))

    probabilities = np.linspace(0.0, 1.0, num=k)
    rank_positions = np.floor(probabilities * (n - 1)).astype(int)
    selected_indices = sorted_indices[rank_positions]
    centers = X[selected_indices].copy()

    # Exact score ties are resolved by original row index.
    score_groups = _groups_by_key(scores.tolist())
    tied_score_groups = [
        [int(original_index) for original_index in group]
        for group in score_groups
    ]

    # Detect exact duplicate selected feature vectors. These are retained
    # and reported rather than silently replaced or shifted.
    vector_keys = [tuple(row.tolist()) for row in centers]
    duplicate_position_groups = _groups_by_key(vector_keys)
    duplicate_centroid_groups = [
        {
            "selected_centroid_positions_zero_based": group,
            "original_row_indices_zero_based": [
                int(selected_indices[pos]) for pos in group
            ],
        }
        for group in duplicate_position_groups
    ]

    info = {
        "algorithm": "PCA-score Quantile-Guided initialization",
        "algorithm_version": "1.0",
        "pca_solver": "full",
        "pca_components": 1,
        "pc1_loadings": loadings.tolist(),
        "pc1_explained_variance_ratio": float(
            pca.explained_variance_ratio_[0]
        ),
        "sign_convention": (
            "largest-absolute PC1 loading positive; first feature wins ties"
        ),
        "ranking_rule": (
            "PC1 score ascending; exact score ties broken by original row index"
        ),
        "quantile_probabilities": probabilities.tolist(),
        "rank_position_rule": "floor(p * (n - 1))",
        "sorted_rank_positions_zero_based": rank_positions.tolist(),
        "selected_row_indices_zero_based": selected_indices.tolist(),
        "selected_pc1_scores": scores[selected_indices].tolist(),
        "score_tie_group_count": len(tied_score_groups),
        "score_tie_groups_original_indices_zero_based": tied_score_groups,
        "duplicate_centroid_group_count": len(duplicate_centroid_groups),
        "duplicate_centroid_groups": duplicate_centroid_groups,
        "duplicate_policy": "retain and report; no silent replacement",
        "initialization_distance_evaluations": 0,
        "seed_argument": seed,
        "seed_used": False,
    }
    return centers, info
