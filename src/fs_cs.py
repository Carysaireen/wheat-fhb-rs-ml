"""FS-CS feature selection — Feature Selection with Compactness and Separability.

Reference
---------
Sun H., Mai H., Mao Y., Li Q., Guo M., Liu Y., Feng Z., Feng H., Guo W.,
Yang G., Deng X., Song X. (2025). *Multi-variety monitoring of potato late
blight severity using UAV data with improved SMOTE-CS for small sample
modeling and deep feature learning.* European Journal of Agronomy.
(ScienceDirect PII: S1161030125001984)

NOTE: an earlier draft of docs/模型改进方向_文献检索报告.md attributed FS-CS to
"Zhou et al. 2023". That was a mis-citation and has been corrected — the
documented application source is the Sun et al. 2025 paper above.

Principle
---------
A discriminative feature should be COMPACT within each class and SEPARABLE
between classes. This is the classical class-separability (Fisher) criterion.
For a label vector y with classes c:

    S_B(j) = sum_c n_c * (mu_cj - mu_j)^2            # between-class scatter
    S_W(j) = sum_c sum_{i in c} (x_ij - mu_cj)^2     # within-class scatter
    J(j)   = S_B(j) / S_W(j)                         # higher = more discriminative

Properties
----------
* Scale-invariant: J is a ratio, so per-feature rescaling (e.g. z-scoring)
  does NOT change the ranking. We therefore score on the raw feature values.
* Fully deterministic: no RNG anywhere -> bit-reproducible across runs.
* Univariate filter: no hyper-parameter besides the number of features K.

The `select_fs_cs_greedy` variant extends the criterion to feature *subsets*
by maximising the multivariate trace ratio trace(S_B)/trace(S_W) of the
currently selected columns. Because a redundant (collinear) feature barely
increases that ratio, greedy selection implicitly suppresses redundancy.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def fs_cs_score(X, y) -> np.ndarray:
    """Univariate FS-CS (Fisher) score for every column of `X`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,) with integer class labels

    Returns
    -------
    ndarray of shape (n_features,) — J(j) = S_B(j) / S_W(j)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    mu_all = X.mean(axis=0)
    s_b = np.zeros(X.shape[1], dtype=float)
    s_w = np.zeros(X.shape[1], dtype=float)
    for c in np.unique(y):
        mask = y == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        mu_c = X[mask].mean(axis=0)
        s_b += n_c * (mu_c - mu_all) ** 2
        s_w += ((X[mask] - mu_c) ** 2).sum(axis=0)
    return s_b / (s_w + _EPS)


def rank_fs_cs(X, y) -> np.ndarray:
    """Feature indices sorted by descending FS-CS score (ties -> lower index)."""
    return np.argsort(-fs_cs_score(X, y), kind="stable")


def select_fs_cs(X, y, k: int) -> np.ndarray:
    """Top-k univariate FS-CS features, returned in descending-score order."""
    return rank_fs_cs(X, y)[: int(k)]


def _fisher_criterion(idx, Z, y) -> float:
    """Multivariate Fisher criterion trace(S_W^-1 S_B) on columns `idx` of Z.

    Uses the FULL within-class scatter matrix S_W, so off-diagonal (covariance)
    terms matter -> collinear / redundant features are penalised. `Z` must be
    standardised by the caller so the criterion is scale-free.
    """
    z = np.asarray(Z, dtype=float)[:, idx]
    d = z.shape[1]
    mu_all = z.mean(axis=0)
    s_w = np.zeros((d, d))
    s_b = np.zeros((d, d))
    for c in np.unique(y):
        mask = y == c
        zc = z[mask]
        mu_c = zc.mean(axis=0)
        diff = zc - mu_c
        s_w += diff.T @ diff
        dd = (mu_c - mu_all).reshape(-1, 1)
        s_b += int(mask.sum()) * (dd @ dd.T)
    try:
        return float(np.trace(np.linalg.solve(s_w + 1e-6 * np.eye(d), s_b)))
    except np.linalg.LinAlgError:
        return float(np.trace(np.linalg.pinv(s_w) @ s_b))


def select_fs_cs_greedy(X, y, k: int) -> np.ndarray:
    """Redundancy-aware FS-CS: greedy forward selection on the Fisher criterion.

    Features are first z-scored with the *provided* data's own statistics (the
    caller passes the training fold, so this stays leakage-free). Then, starting
    from the best univariate feature, the algorithm repeatedly appends the
    feature that most improves trace(S_W^-1 S_B). Because S_W carries the
    covariance structure, adding a redundant (collinear) feature barely helps.
    Returns the selected indices in selection order; ties -> lower index.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    n_features = X.shape[1]
    k = int(min(k, n_features))
    score = fs_cs_score(X, y)
    selected = [int(np.argmax(score))]
    while len(selected) < k:
        best_j, best_val = -1, -np.inf
        for j in range(n_features):
            if j in selected:
                continue
            val = _fisher_criterion(selected + [j], Z, y)
            if val > best_val:
                best_val, best_j = val, j
        selected.append(best_j)
    return np.asarray(selected, dtype=int)


if __name__ == "__main__":  # tiny self-check
    rng = np.random.default_rng(0)
    x0 = rng.normal(0.0, 1.0, (40, 3))
    x1 = rng.normal(3.0, 1.0, (40, 3))
    X = np.vstack([x0, x1])
    X[:, 2] = rng.normal(0.0, 5.0, 80)          # pure noise column
    y = np.array([0] * 40 + [1] * 40)
    print("J      :", np.round(fs_cs_score(X, y), 4), "-> expect col 2 lowest")
    print("rank   :", rank_fs_cs(X, y))
    print("top2   :", select_fs_cs(X, y, 2))
    print("greedy2:", select_fs_cs_greedy(X, y, 2))
