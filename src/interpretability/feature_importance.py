"""Feature importance via linear classifier coefficients (Option A).

Pour un modèle linéaire ``y = sigmoid(x @ w + b)``, les coefficients ``w``
sont **directement** l'importance signée des features (leur magnitude
indique combien chaque feature pèse, le signe indique la direction).
C'est mathématiquement équivalent à SHAP pour un modèle linéaire — on
n'a donc pas besoin de SHAP/LIME pour cette catégorie de modèles.

L'intérêt fairness : croiser les top features avec les attributs
sensibles (gender, region) pour comprendre **mécaniquement** pourquoi
le modèle introduit du biais. Si une feature en top-10 est fortement
corrélée à `gender`, on a une explication concrète : le modèle utilise
cette feature, et cette feature porte du signal gender.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def rank_features_by_coef(
    coefs: np.ndarray,
    feature_names: list[str],
    top_k: int = 10,
) -> pl.DataFrame:
    """Return the ``top_k`` features by absolute coefficient magnitude.

    Args:
        coefs: 1-D array of model coefficients, shape ``(n_features,)``.
            For multi-class LR pass the class-1 row (binary case) or a
            mean across classes (multi-class).
        feature_names: list of names aligned with ``coefs``.
        top_k: number of top features to return.

    Returns:
        polars DataFrame columns: ``feature``, ``coef`` (signed),
        ``abs_coef``, ``rank`` (1 = most important).
    """
    coefs = np.asarray(coefs, dtype=np.float64).ravel()
    if coefs.shape[0] != len(feature_names):
        raise ValueError(
            f"coefs has {coefs.shape[0]} entries but feature_names has {len(feature_names)}"
        )
    abs_coefs = np.abs(coefs)
    rank = np.argsort(abs_coefs)[::-1][:top_k]
    return pl.DataFrame(
        {
            "rank": np.arange(1, len(rank) + 1, dtype=np.int64),
            "feature": [feature_names[int(i)] for i in rank],
            "coef": coefs[rank].astype(np.float32),
            "abs_coef": abs_coefs[rank].astype(np.float32),
        }
    )


def correlation_with_sensitive(
    x: np.ndarray,
    feature_names: list[str],
    target_features: list[str],
    sensitive: np.ndarray,
    sensitive_name: str = "sensitive",
) -> pl.DataFrame:
    """Pearson correlation between each target feature and a sensitive attr.

    Vectorized over target_features. Used to cross-check whether top
    important features carry sensitive-attribute information.

    Args:
        x: feature matrix, shape ``(n, d)``.
        feature_names: column names aligned with ``x``.
        target_features: subset of names to score.
        sensitive: 1-D array, shape ``(n,)``, treated as continuous for
            the correlation (binary 0/1 works fine).
        sensitive_name: label used in the output column header.

    Returns:
        polars DataFrame columns: ``feature``,
        ``corr_<sensitive_name>``, ``abs_corr``.
    """
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(sensitive, dtype=np.float64).ravel()
    if x.shape[0] != s.shape[0]:
        raise ValueError(f"x has {x.shape[0]} rows, sensitive has {s.shape[0]}")

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    indices = np.array([name_to_idx[f] for f in target_features], dtype=np.int64)

    # Vectorized Pearson via covariance / std product.
    s_centered = s - s.mean()
    x_subset = x[:, indices]  # (n, k)
    x_centered = x_subset - x_subset.mean(axis=0, keepdims=True)

    s_std = float(s.std())
    x_std = x_subset.std(axis=0)

    cov = (x_centered.T @ s_centered) / x.shape[0]  # (k,)
    denom = x_std * s_std
    corrs = np.where(denom > 1e-12, cov / np.maximum(denom, 1e-12), 0.0)

    return pl.DataFrame(
        {
            "feature": target_features,
            f"corr_{sensitive_name}": corrs.astype(np.float32),
            "abs_corr": np.abs(corrs).astype(np.float32),
        }
    )
