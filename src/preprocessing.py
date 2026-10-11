"""Leakage-safe preprocessing shared by all later experiments."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class IQRClipper(BaseEstimator, TransformerMixin):
    """Learn per-feature clipping bounds from training data only."""
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanquantile(X, 0.25, axis=0)
        q3 = np.nanquantile(X, 0.75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)


def build_preprocessor(clip_outliers=False):
    """Return an unfitted pipeline; it must be fitted inside each training fold."""
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if clip_outliers:
        steps.append(('iqr_clipper', IQRClipper(factor=1.5)))
    steps.append(('scaler', StandardScaler()))
    return Pipeline(steps)
