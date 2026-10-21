"""Next-compound benchmark: which tyre does the car fit at its next stop?

Labels read the finished race (the future); they are only used to score.

* ``y_next_compound``  lap-end rows: the compound fitted at the car's next non-red stop after lap L; pit-entry
                       rows: the compound fitted at this stop. Read on the last lap of the stint after the stop
                       that has a known compound (the out-lap often says UNKNOWN until the feed confirms the
                       set). "" when there is no further stop or the compound can't be seen.

Tasks (kind "multiclass", ``CLASSES`` order): ``next_compound_entry`` (at pit entry) and ``next_compound_5``
(lap-end rows whose next stop is within 5 laps: "if it stops soon, on what?"). Both are conditional on a stop.
Class balance, 2025 test races (pit entries): see docs/models.md; hard and medium dominate, inters/wets are
a few percent and come in clusters (wet races).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..state import RaceState
from .features import feature_columns
from .hazard import num, task
from .tasks import Task

CLASSES = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
WET = ("INTERMEDIATE", "WET")


# ----------------------------------------------------------------------------- labels
class _Race:
    def __init__(self, final: RaceState) -> None:
        self.laps: dict[str, dict[int, object]] = {}
        for x in final.laps:
            self.laps.setdefault(x.driver, {})[x.lap] = x
        self.stops: dict[str, list[int]] = {}  # strategy stops (red-flag entries are not)
        self.ends: dict[str, list[int]] = {}  # every pit entry ends a set, red flag included
        for pe in final.pit_events:
            self.ends.setdefault(pe.driver, []).append(pe.in_lap)
            if not pe.under_red:
                self.stops.setdefault(pe.driver, []).append(pe.in_lap)

    def fitted(self, drv: str, s: int) -> str:
        """Compound fitted at the stop ending lap ``s``: last known compound of the stint that follows."""
        laps = self.laps.get(drv, {})
        nxt = [e for e in self.ends.get(drv, ()) if e > s]
        end = min(nxt) if nxt else max(laps, default=s)
        for k in range(end, s, -1):
            c = getattr(laps.get(k), "compound", None)
            if c in CLASSES:
                return c
        return ""

    def next_compound(self, drv: str, L: int) -> str:
        s = min((x for x in self.stops.get(drv, ()) if x > L), default=None)
        return "" if s is None else self.fitted(drv, s)


def compound_labels(rows: list[dict], final: RaceState) -> None:
    race = _Race(final)
    for row in rows:
        drv = row["driver"]
        if row.get("kind", "lap_end") == "pit_entry":
            row["y_next_compound"] = race.fitted(drv, final.pit_events[row["pit_event"]].in_lap)
        else:
            row["y_next_compound"] = race.next_compound(drv, row["lap"])


# ----------------------------------------------------------------------------- rows
def entry_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Pit entries (red-flag entries are never rows) whose fitted compound was seen on the following stint."""
    return df[(df["kind"] == "pit_entry") & df["y_next_compound"].isin(CLASSES)].copy()


def soon_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Lap-end rows whose next stop is within 5 laps (``y_pit_5``) and whose fitted compound was seen.
    Conditional on stopping: the question is "if it stops soon, on what?", not whether it stops."""
    return df[(df["kind"] == "lap_end") & (df["y_pit_5"] == 1) & df["y_next_compound"].isin(CLASSES)].copy()


# ----------------------------------------------------------------------------- models
def _onehot(y) -> np.ndarray:
    idx = {c: i for i, c in enumerate(CLASSES)}
    out = np.zeros((len(y), len(CLASSES)))
    out[np.arange(len(y)), [idx[c] for c in y]] = 1.0
    return out


def _wet(df: pd.DataFrame) -> np.ndarray:
    rain = pd.to_numeric(df["rainfall"], errors="coerce").fillna(0).to_numpy() > 0
    wet_run = pd.to_numeric(df.get("weather__wet_running", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    return rain | (wet_run.to_numpy() > 0) | df["compound"].isin(WET).to_numpy()


def _cells(df: pd.DataFrame) -> pd.DataFrame:
    frac = pd.to_numeric(df["race_frac"], errors="coerce").fillna(0.5).to_numpy()
    return pd.DataFrame({"cmp": df["compound"].to_numpy(), "third": np.minimum((frac * 3).astype(int), 2),
                         "wet": _wet(df)})


class CompoundFreq:
    """Empirical frequency of the fitted compound by (current compound, race third, wet now), fit on earlier races.

    Laplace smoothing with one pseudo-count per class, spread like the next coarser cell (current compound only,
    itself smoothed toward the overall frequency), so a rare cell falls back on its parent rather than on uniform.
    """

    name = "compound_freq"
    alpha = 1.0  # pseudo-counts per class

    def fit(self, df: pd.DataFrame, target: str) -> "CompoundFreq":
        Y, cells = _onehot(df[target]), _cells(df)
        a = self.alpha * len(CLASSES)
        self.p0 = (Y.sum(0) + self.alpha) / (len(Y) + a)
        self.p1, self.p2 = {}, {}
        for c, idx in cells.groupby("cmp").groups.items():
            n = Y[idx].sum(0)
            self.p1[c] = (n + a * self.p0) / (n.sum() + a)
        for key, idx in cells.groupby(["cmp", "third", "wet"]).groups.items():
            n = Y[idx].sum(0)
            self.p2[key] = (n + a * self.p1[key[0]]) / (n.sum() + a)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        cells = _cells(df)
        return np.array([self.p2.get((c, t, w), self.p1.get(c, self.p0))
                         for c, t, w in zip(cells.cmp, cells.third, cells.wet)], dtype=float)


class CompoundGBM:
    """Multiclass boosted trees on every bench feature, small and heavily regularised; trained on earlier races."""

    name = "compound_gbm"
    params = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=50, l2_regularization=10.0,
                  random_state=0)

    def fit(self, df: pd.DataFrame, target: str) -> "CompoundGBM":
        self.cols = feature_columns()
        y = df[target].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(int)
        self.seen = np.unique(y)
        self.freq = CompoundFreq().fit(df, target)
        self.model = None
        if len(self.seen) > 1:
            self.model = HistGradientBoostingClassifier(**self.params).fit(num(df, self.cols), y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return self.freq.predict(df)
        P = np.full((len(df), len(CLASSES)), 1e-4)
        P[:, self.model.classes_] = self.model.predict_proba(num(df, self.cols))
        return P / P.sum(1, keepdims=True)


MODELS = (CompoundFreq, CompoundGBM)


def compound_tasks(**_) -> list[Task]:
    keep = ("compound", "race_frac", "rainfall")
    return [task("next_compound_entry", "multiclass", "y_next_compound", entry_rows, MODELS, keep=keep,
                 classes=CLASSES, primary="log_loss"),
            task("next_compound_5", "multiclass", "y_next_compound", soon_rows, MODELS, keep=keep,
                 classes=CLASSES, primary="log_loss")]
