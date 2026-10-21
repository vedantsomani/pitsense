"""Pit hazard benchmark: "does the car stop within k laps?" with retirement censoring, k in ``labels.HORIZONS``.

Unlike ``pit_within_k`` (which drops every row of a car that retires within 3 laps, using the future),
``pit_hazard_k`` keeps every lap-end row whose outcome over k laps was observed (``y_pit_{k}_known``):
a car that retires before L+k without stopping is censored, finishers (lapped ones too) are known.

Also home of the generic contenders the new tasks share: :class:`FeatureGBM` / :class:`FeatureGBMRegressor`
(HistGradientBoosting on ``features.feature_columns()``, regularised like ``models.GBMHazard``).
"""

from __future__ import annotations

from dataclasses import fields
from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .features import feature_columns
from .labels import HORIZONS
from .models import BaseRate, TyreAgeLogit
from .tasks import Task


def task(*args, **kw) -> Task:
    """``Task(...)``, dropping the fields an older ``Task`` doesn't have (``classes``, ``primary``)."""
    have = {f.name for f in fields(Task)}
    return Task(*args, **{k: v for k, v in kw.items() if k in have})


def num(df: pd.DataFrame, cols) -> pd.DataFrame:
    """``cols`` of ``df`` as floats (NaN where a column is missing or not numeric)."""
    return pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") if c in df else np.nan for c in cols},
                        index=df.index).astype(float)


# ----------------------------------------------------------------------------- generic contenders
class FeatureGBM:
    """Boosted trees on every bench feature (``feature_columns()``) plus ``extra``; small, heavily regularised trees."""

    name = "feature_gbm"
    extra: tuple[str, ...] = ()
    params = dict(max_iter=150, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=200, l2_regularization=10.0,
                  random_state=0)

    def fit(self, df: pd.DataFrame, target: str) -> "FeatureGBM":
        self.cols = feature_columns() + [c for c in self.extra if c in df]
        y = df[target].astype(int).to_numpy()
        self.p = float(y.mean()) if len(y) else 0.0
        self.model = None
        if 0 < y.sum() < len(y):
            self.model = HistGradientBoostingClassifier(**self.params).fit(num(df, self.cols), y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(df), self.p)
        return self.model.predict_proba(num(df, self.cols))[:, 1]


class FeatureGBMRegressor:
    """Boosted trees on every bench feature plus ``extra``, learning the residual over ``_ref`` (default 0)."""

    name = "feature_gbm"
    extra: tuple[str, ...] = ()
    params = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0,
                  loss="absolute_error", random_state=0)

    def _ref(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df))

    def fit(self, df: pd.DataFrame, target: str) -> "FeatureGBMRegressor":
        self.cols = feature_columns() + [c for c in self.extra if c in df]
        y = df[target].to_numpy(float) - self._ref(df)
        ok = ~np.isnan(y)
        self.model = HistGradientBoostingRegressor(**self.params).fit(num(df, self.cols)[ok], y[ok])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._ref(df) + self.model.predict(num(df, self.cols))


# ----------------------------------------------------------------------------- pooled discrete-time hazard
def _k(target: str) -> int:
    return int(target.split("_")[2])


def _left(df: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(pd.to_numeric(df["laps_remaining"], errors="coerce").to_numpy(float), nan=99.0)


class PooledHazard:
    """Discrete-time hazard: one classifier on the stacked person-period table (row, step j = 1..K) predicts the
    per-lap hazard h_j; P(stop within k) = 1 - prod_{j<=k} (1 - h_j), so the answers over k are a monotone CDF.

    A row is at risk at step j while it has not stopped before j, was still seen (a retired car only up to its last
    lap) and the race has a lap L+j. Steps beyond the race distance have hazard 0. Needs the ``laps_to_stop`` labels
    (``y_stop_obs``, ``y_stop_event``, ``y_stop_retired``) on the training rows.
    """

    name = "pooled_hazard"
    K = max(HORIZONS)
    thin = 2  # train on every ``thin``-th lap (a car's consecutive rows are strongly correlated)
    params = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=12, min_samples_leaf=400, l2_regularization=10.0,
                  random_state=0)

    def _x(self, df: pd.DataFrame, rows: np.ndarray, h: np.ndarray) -> np.ndarray:
        x = num(df, self.cols).to_numpy(np.float32)[rows]
        left, age = _left(df)[rows], pd.to_numeric(df["tyre_age"], errors="coerce").to_numpy(float)[rows]
        return np.column_stack([x, h, left - h, age + h]).astype(np.float32)

    def fit(self, df: pd.DataFrame, target: str) -> "PooledHazard":
        self.k, self.cols = _k(target), feature_columns()
        d = df[df["lap"] % self.thin == 0] if self.thin > 1 else df
        obs, ev = d["y_stop_obs"].to_numpy(float), d["y_stop_event"].to_numpy(int)
        ret = d["y_stop_retired"].to_numpy(int)
        n_at = np.minimum(self.K, _left(d))
        n_at = np.where((ev == 1) | (ret == 1), np.minimum(n_at, obs), n_at).astype(int).clip(0)
        rows = np.repeat(np.arange(len(d)), n_at)
        h = np.concatenate([np.arange(1, m + 1) for m in n_at]) if n_at.sum() else np.array([], int)
        y = ((ev[rows] == 1) & (h == obs[rows])).astype(np.int8)
        X = self._x(d, rows, h)
        X[:, np.isnan(X).all(axis=0)] = 0.0  # a column missing everywhere breaks the binning
        self.model = HistGradientBoostingClassifier(**self.params).fit(X, y)
        return self

    def hazards(self, df: pd.DataFrame, k: int) -> np.ndarray:
        """Per-lap hazards [n, k], zero beyond the laps left in the race."""
        n = len(df)
        h = np.tile(np.arange(1, k + 1), n)
        hz = self.model.predict_proba(self._x(df, np.repeat(np.arange(n), k), h))[:, 1].reshape(n, k)
        return np.where(np.arange(1, k + 1)[None, :] <= _left(df)[:, None], hz, 0.0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return 1.0 - np.prod(1.0 - self.hazards(df, self.k), axis=1)


# ----------------------------------------------------------------------------- tasks
def hazard_rows(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Every lap-end decision whose outcome over k laps was observed (``y_pit_{k}_known``): all running cars,
    including those about to retire, as long as they stop or are still seen at L+k (finishers always are).
    Cars that retire before L+k without stopping are censored out, since "would it have stopped?" is unknown."""
    return df[(df["kind"] == "lap_end") & (df[f"y_pit_{k}_known"] == 1)].copy()


HAZARD_MODELS = (BaseRate, TyreAgeLogit, FeatureGBM, PooledHazard)


def hazard_tasks(**_) -> list[Task]:
    return [task(f"pit_hazard_{k}", "binary", f"y_pit_{k}", partial(hazard_rows, k=k), HAZARD_MODELS,
                 keep=("compound", "tyre_age", "laps_remaining"), primary="log_loss") for k in HORIZONS]
