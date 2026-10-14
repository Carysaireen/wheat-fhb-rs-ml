"""Selector zoo — a unified, leakage-safe library of feature selectors.

Every selector has the SAME signature::

    fn(X, y, k, ctx) -> np.ndarray           # sorted column indices, len == k

* `X`, `y` are the TRAINING-fold rows only (the caller guarantees this, so no
  leakage can occur: nothing here ever sees the validation fold).
* `ctx` is a dict carrying: {"task": "clf"|"reg", "seed": int,
  "mi_matrix": ndarray|None, ...}. `ctx["mi_matrix"]` lets mRMR reuse a
  redundancy matrix that the driver computed ONCE per training fold (it is
  target-independent, so clf/reg can share it).

Selector families
-----------------
* filter (model-agnostic): ReliefF, mRMR, SPA
* filter (spectral, PLSR-based): CARS            (regression only)
* embedded (model-bound): LASSO, ElasticNet, Permutation importance
* wrapper (model-bound): RFE (linear SVM/SVR)
* meta: Stability selection (subsample + fast base selector + frequency)
* control: Random feature set (noise floor)

Native models used by embedded/wrapper selectors are recorded in
`NATIVE_MODEL` so the paper can disclose "selection model vs evaluation model".
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

try:  # sibling import when run as a script / driver
    from fs_cs import select_fs_cs
except Exception:  # pragma: no cover
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fs_cs import select_fs_cs

_EPS = 1e-12

# Deterministic L1 solver.
#
# WHY NOT liblinear: sklearn's liblinear solver solves L1 logistic regression
# with a RANDOM coordinate permutation and exposes NO seed. The same data can
# therefore converge to numerically equivalent but differently-ORDERED
# solutions, and since we keep the top-k by |coef|, the selected feature SET
# can change between runs. We measured it: 4 back-to-back fits gave 3 different
# top-10 orderings on this dataset. That is fatal for reproducibility, so we
# use `saga`, which is deterministic once `random_state` is fixed (verified:
# 4 back-to-back fits gave bit-identical solutions).
#
# MAX_ITER / TOL were checked empirically on this dataset: with tol=1e-3 the
# solver converges in ~370-430 iterations (< 0.05 s) for both l1 and
# elasticnet, so max_iter=3000 is a comfortable ceiling, not a limiter.
L1_SOLVER = "saga"
L1_SEED = 0
L1_MAX_ITER = 3000
L1_TOL = 1e-3


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _z(X):
    """Standardise on the *provided* data only (caller passes training fold)."""
    return StandardScaler().fit_transform(np.asarray(X, dtype=float))


def _topk(scores, k):
    """Indices of the k largest scores (ties -> lower index), returned sorted."""
    scores = np.asarray(scores, dtype=float)
    k = int(min(k, scores.size))
    order = np.argsort(-scores, kind="stable")
    return np.sort(order[:k].astype(int))


def compute_mi_matrix(X):
    """Full MI redundancy matrix (target-independent) — ~9 s for 117 features."""
    X = np.asarray(X, dtype=float)
    n = X.shape[1]
    mat = np.zeros((n, n))
    for j in range(n):
        mat[j] = mutual_info_regression(X, X[:, j], random_state=0)
    return mat


# --------------------------------------------------------------------------- #
# filter selectors
# --------------------------------------------------------------------------- #
def select_relieff(X, y, k, ctx):
    """ReliefF (skrebate). Nearest-neighbour based, catches local structure."""
    from skrebate import ReliefF
    X = np.asarray(X, dtype=float)
    r = ReliefF(n_features_to_select=int(min(k, X.shape[1])), n_neighbors=10)
    r.fit(X, np.asarray(y))
    return _topk(r.feature_importances_, k)


def select_mrmr(X, y, k, ctx):
    """mRMR (Peng et al. 2005): maximise relevance, minimise MI redundancy.

    Greedy forward: first the most relevant feature, then repeatedly the
    feature maximising relevance - mean(redundancy to already-selected).
    Redundancy uses mutual information (faithful to the original paper); the
    MI matrix is cached in ctx because it does not depend on the labels.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    mm = ctx.get("mi_matrix")
    if mm is None:
        mm = compute_mi_matrix(X)
    if ctx.get("task") == "clf":
        rel = mutual_info_classif(X, y.astype(int), random_state=0)
    else:
        rel = mutual_info_regression(X, y.astype(float), random_state=0)
    n = X.shape[1]
    k = int(min(k, n))
    selected = [int(np.argmax(rel))]
    while len(selected) < k:
        best_j, best_v = -1, -np.inf
        for j in range(n):
            if j in selected:
                continue
            red = float(np.mean(mm[j, selected]))
            v = rel[j] - red
            if v > best_v:
                best_v, best_j = v, j
        selected.append(best_j)
    return np.sort(np.asarray(selected, dtype=int))


def select_spa(X, y, k, ctx):
    """Successive Projections Algorithm — collinearity reduction (y-agnostic).

    Deterministic: starts from the highest-variance (standardised) column, then
    repeatedly adds the column whose residual after projecting out the span of
    the already-selected columns is largest (i.e. most orthogonal to them).
    It ignores y entirely, so it is a *diversity* selector, not a relevance one.
    """
    Z = _z(X)
    n = Z.shape[1]
    k = int(min(k, n))
    start = int(np.argmax(np.linalg.norm(Z, axis=0)))
    selected = [start]
    while len(selected) < k:
        Q, _ = np.linalg.qr(Z[:, selected])
        best_j, best_v = -1, -np.inf
        for j in range(n):
            if j in selected:
                continue
            v = Z[:, j]
            resid = v - Q @ (Q.T @ v)
            nv = float(np.linalg.norm(resid))
            if nv > best_v:
                best_v, best_j = nv, j
        selected.append(best_j)
    return np.sort(np.asarray(selected, dtype=int))


def select_cars(X, y, k, ctx):
    """CARS — Competitive Adaptive Reweighted Sampling (regression only).

    Faithful-in-spirit simplification (deterministic given ctx["seed"]):
      * each of `cars_mc` Monte-Carlo runs fits PLSR on a random 80% sample;
      * inside a run, `cars_iters` passes shrink the candidate set by an
        exponentially decreasing factor, keeping the largest PLSR |coef| each
        time (EDF + hard reweighting);
      * the final per-run subset votes into a selection-frequency vector.
    Features are ranked by that frequency, so the result is stable.
    """
    Z = _z(X)
    y = np.asarray(y, dtype=float)
    n, p = Z.shape
    seed = int(ctx.get("seed", 42))
    rng = np.random.default_rng(seed)
    n_mc = int(ctx.get("cars_mc", 20))
    iters = int(ctx.get("cars_iters", 30))
    n_comp = int(ctx.get("cars_ncomp", 5))
    cache = ctx.setdefault("_cache", {})
    key = ("cars", n_mc, iters, n_comp, seed)
    if key in cache:
        return _topk(cache[key], k)
    decay = (2.0 / p) ** (1.0 / max(iters - 1, 1))
    freq = np.zeros(p)
    sub = max(2, int(round(0.8 * n)))
    for _ in range(n_mc):
        idx = rng.choice(n, size=sub, replace=False)
        Zs, ys = Z[idx], y[idx]
        keep = np.arange(p)
        for _t in range(iters):
            if keep.size <= 2:
                break
            n_c = max(2, min(n_comp, keep.size, len(idx) - 1))
            pls = PLSRegression(n_components=n_c)
            try:
                pls.fit(Zs[:, keep], ys)
            except Exception:
                break
            w = np.abs(np.asarray(pls.coef_).ravel())
            if w.size != keep.size:
                w = w[:keep.size]
            n_keep = max(2, int(round(keep.size * decay)))
            keep = keep[np.argsort(-w)[:n_keep]]
        freq[keep] += 1.0
    cache[key] = freq
    return _topk(freq, k)


# --------------------------------------------------------------------------- #
# embedded selectors
# --------------------------------------------------------------------------- #
def select_lasso(X, y, k, ctx):
    """LASSO / L1. clf -> L1-logistic (|coef|); reg -> LassoCV (|coef|).

    NOTE on the cache: the fitted coefficients depend only on the TRAINING rows
    (and the task), never on k -- k only decides how many of them we keep. So we
    fit once per (fold, seed) and reuse the |coef| vector for every k. This is a
    pure speed-up: the returned indices are identical to the un-cached version.
    """
    Z = _z(X)
    task = ctx.get("task")
    cache = ctx.setdefault("_cache", {})
    key = ("lasso_score", task)
    if key not in cache:
        if task == "clf":
            m = LogisticRegression(penalty="l1", solver=L1_SOLVER, C=1.0,
                                   random_state=L1_SEED, max_iter=L1_MAX_ITER,
                                   tol=L1_TOL)
            m.fit(Z, np.asarray(y).astype(int))
        else:
            m = LassoCV(cv=5, random_state=0, max_iter=5000)
            m.fit(Z, np.asarray(y, dtype=float))
        cache[key] = np.abs(m.coef_).ravel()
    return _topk(cache[key], k)


def select_enet(X, y, k, ctx):
    """Elastic Net (L1+L2) coefficient ranking (same k-independent cache)."""
    Z = _z(X)
    task = ctx.get("task")
    cache = ctx.setdefault("_cache", {})
    key = ("enet_score", task)
    if key not in cache:
        if task == "clf":
            m = LogisticRegression(penalty="elasticnet", solver="saga",
                                   l1_ratio=0.5, C=1.0, max_iter=L1_MAX_ITER,
                                   tol=L1_TOL, random_state=L1_SEED)
            m.fit(Z, np.asarray(y).astype(int))
        else:
            m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=0)
            m.fit(Z, np.asarray(y, dtype=float))
        cache[key] = np.abs(m.coef_).ravel()
    return _topk(cache[key], k)


def select_perm(X, y, k, ctx):
    """Permutation importance on an RF base model (more reliable than impurity).

    The importance vector is computed once per (fold, seed) and reused for every
    k (see the lasso note above) -- this is the single largest speed-up of the
    K sweep, because one call costs a 200-tree RF fit + 10 shuffles x 117
    features of prediction.
    """
    Z = _z(X)
    seed = int(ctx.get("seed", 42))
    task = ctx.get("task")
    cache = ctx.setdefault("_cache", {})
    key = ("perm_score", task, seed)
    if key not in cache:
        if task == "clf":
            base = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        else:
            base = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
        base.fit(Z, np.asarray(y))
        r = permutation_importance(base, Z, np.asarray(y), n_repeats=10,
                                   random_state=seed, n_jobs=-1)
        cache[key] = r.importances_mean
    return _topk(cache[key], k)


# --------------------------------------------------------------------------- #
# wrapper selector
# --------------------------------------------------------------------------- #
def select_rfe(X, y, k, ctx):
    """Recursive Feature Elimination with a linear SVM (clf) / SVR (reg).

    The elimination is run ONCE down to a single feature and the ranking is
    cached in ctx, so every k reuses the same elimination order instead of
    re-running RFE (which would be ~117 model fits per k).
    """
    Z = _z(X)
    task = ctx.get("task")
    cache = ctx.setdefault("_cache", {})
    key = ("rfe", task)
    if key not in cache:
        est = SVC(kernel="linear", C=1.0) if task == "clf" else SVR(kernel="linear", C=1.0)
        rfe = RFE(est, n_features_to_select=1, step=1)
        rfe.fit(Z, np.asarray(y))
        cache[key] = rfe.ranking_          # 1 = kept longest (most important)
    order = np.argsort(cache[key], kind="stable")
    return np.sort(order[:int(k)].astype(int))


# --------------------------------------------------------------------------- #
# meta: stability selection
# --------------------------------------------------------------------------- #
def _lasso_fast(X, y, k, task):
    """Fast fixed-alpha L1 ranking, used as a stability-selection base.

    Deliberately NOT LassoCV: stability selection invokes the base B=100 times,
    so a CV-in-the-loop base would be ~500x slower and would spam convergence
    warnings on collinear spectral features.
    """
    Z = _z(X)
    if task == "clf":
        m = LogisticRegression(penalty="l1", solver=L1_SOLVER, C=1.0,
                               random_state=L1_SEED, max_iter=L1_MAX_ITER,
                               tol=L1_TOL)
        m.fit(Z, np.asarray(y).astype(int))
        w = np.abs(m.coef_).ravel()
    else:
        m = Lasso(alpha=0.5, max_iter=2000)
        m.fit(Z, np.asarray(y, dtype=float))
        w = np.abs(m.coef_).ravel()
    return _topk(w, k)


def select_stability(X, y, k, ctx):
    """Stability selection (Meinshausen & Buhlmann 2010).

    B random subsamples (fraction `stab_frac`); a FAST base selector returns its
    top-q features on each; features are ranked by selection frequency and the
    top-k kept. Only fast bases are allowed (fs_cs for clf, lasso) because a
    slow base would multiply into hours.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(int(ctx.get("seed", 42)))
    B = int(ctx.get("stab_B", 100))
    frac = float(ctx.get("stab_frac", 0.5))
    q = int(ctx.get("stab_q", 30))
    base = ctx.get("stab_base", "fs_cs")
    n = X.shape[0]
    key = (base, B, round(frac, 3), q)
    cache = ctx.setdefault("_cache", {})
    if key in cache:
        counts = cache[key]
    else:
        counts = np.zeros(X.shape[1])
        for _b in range(B):
            idx = rng.choice(n, size=max(2, int(round(frac * n))), replace=False)
            if base == "fs_cs":
                if ctx.get("task") != "clf":
                    raise ValueError("stability base 'fs_cs' is classification-only")
                sel = select_fs_cs(X[idx], y[idx], q)
            elif base == "lasso":
                sel = _lasso_fast(X[idx], y[idx], q, ctx.get("task"))
            else:
                raise ValueError(f"unknown stability base: {base}")
            counts[sel] += 1.0
        cache[key] = counts
    return _topk(counts, k)


# --------------------------------------------------------------------------- #
# control
# --------------------------------------------------------------------------- #
def select_random(X, y, k, ctx):
    """Random feature subset — the noise floor every selector must beat."""
    rng = np.random.default_rng(int(ctx.get("seed", 42)))
    idx = rng.choice(X.shape[1], size=int(min(k, X.shape[1])), replace=False)
    return np.sort(idx.astype(int))


# --------------------------------------------------------------------------- #
# registries
# --------------------------------------------------------------------------- #
def _stab(base):
    return lambda X, y, k, ctx: select_stability(X, y, k, {**ctx, "stab_base": base})


CLF_SELECTORS = {
    "relieff": select_relieff,
    "mrmr": select_mrmr,
    "spa": select_spa,
    "lasso": select_lasso,
    "enet": select_enet,
    "perm": select_perm,
    "rfe_svm": select_rfe,
    "fs_cs": lambda X, y, k, ctx: np.sort(select_fs_cs(X, y, k)),
    "stab_fscs": _stab("fs_cs"),
    "stab_lasso": _stab("lasso"),
    "random": select_random,
}

REG_SELECTORS = {
    "relieff": select_relieff,
    "mrmr": select_mrmr,
    "spa": select_spa,
    "cars": select_cars,
    "lasso": select_lasso,
    "enet": select_enet,
    "perm": select_perm,
    "rfe_svr": select_rfe,
    "stab_lasso": _stab("lasso"),
    "random": select_random,
}

# which model each embedded/wrapper selector natively uses (for disclosure)
NATIVE_MODEL = {
    "lasso": "itself (L1)",
    "enet": "itself (L1+L2)",
    "perm": "RandomForest",
    "rfe_svm": "linear SVM",
    "rfe_svr": "linear SVR",
    "cars": "PLSR",
    "stab_fscs": "FS-CS (filter base)",
    "stab_lasso": "Lasso (base)",
}

# selectors that need the (expensive) MI redundancy matrix
NEEDS_MI = {"mrmr"}


if __name__ == "__main__":  # smoke self-test on synthetic data
    rng = np.random.default_rng(0)
    n, p = 60, 20
    X = rng.normal(size=(n, p))
    y_cls = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, .5, n) > 0).astype(int)
    y_reg = X[:, 0] * 2 + X[:, 1] + rng.normal(0, .5, n)
    ctx = {"task": "clf", "seed": 42, "mi_matrix": compute_mi_matrix(X)}
    print("== clf selectors (k=5) ==")
    for name, fn in CLF_SELECTORS.items():
        try:
            sel = fn(X, y_cls, 5, ctx)
            assert len(sel) == 5 and len(set(sel.tolist())) == 5, name
            print(f"  {name:12s} -> {sel.tolist()}")
        except Exception as e:
            print(f"  {name:12s} ERROR {type(e).__name__}: {e}")
    ctx2 = {"task": "reg", "seed": 42, "mi_matrix": ctx["mi_matrix"]}
    print("== reg selectors (k=5) ==")
    for name, fn in REG_SELECTORS.items():
        try:
            sel = fn(X, y_reg, 5, ctx2)
            assert len(sel) == 5 and len(set(sel.tolist())) == 5, name
            print(f"  {name:12s} -> {sel.tolist()}")
        except Exception as e:
            print(f"  {name:12s} ERROR {type(e).__name__}: {e}")
    print("expect feature 0/1 to appear often (true signal)")
