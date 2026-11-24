"""Expanding-window evaluation: for each test race, train on every earlier race only."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..asof import assert_trained_before
from .metrics import (binary_scores, calibration_table, ece, higher_is_better, multiclass_scores, paired_race_bootstrap,
                      position_scores, regression_scores)
from .tasks import Task, core_tasks

# a Grand Prix lasts at most ~4h (incl. red flags); used as a conservative end time
RACE_SPAN = timedelta(hours=4)


@dataclass
class EvalResult:
    task: str
    leaderboard: pd.DataFrame
    per_race: pd.DataFrame
    predictions: pd.DataFrame
    calibration: dict[str, pd.DataFrame]
    primary: str = ""  # the leaderboard's primary metric (per_race holds it per race)


def proba_col(model: str, cls: str) -> str:
    """Predictions-file column of a multiclass model's probability for one class."""
    return f"{model}:{cls}"


def _races(df: pd.DataFrame) -> pd.DataFrame:
    return df[["race_id", "start_utc", "year"]].drop_duplicates().sort_values("start_utc")


def _splits(df: pd.DataFrame, test_year: int, min_train_races: int):
    races = _races(df)
    for _, race in races[races.year == test_year].iterrows():
        train = df[df.start_utc + RACE_SPAN < race.start_utc]
        if train.race_id.nunique() < min_train_races:
            continue
        yield race.race_id, train, df[df.race_id == race.race_id]


def fit_predict(cls, train: pd.DataFrame, test: pd.DataFrame, target: str, timing: list | None = None) -> np.ndarray:
    """Fit a fresh model on ``train`` and predict ``test``.

    Every model in every task goes through here, so this is where the training
    cutoff is enforced: no training race may still be running when the test race starts.
    ``timing`` (optional) gets one (fit seconds, predict seconds, rows) tuple appended.
    """
    assert_trained_before([train.start_utc.max() + RACE_SPAN], test.start_utc.min())
    t0 = time.perf_counter()
    model = cls().fit(train, target)
    t1 = time.perf_counter()
    pred = model.predict(test)
    if timing is not None:
        timing.append((t1 - t0, time.perf_counter() - t1, len(test)))
    return pred


def fit_predict_cdf(cls, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """As :func:`fit_predict` for survival models: the [n, 15] CDF P(stop within k laps). Same cutoff guard."""
    assert_trained_before([train.start_utc.max() + RACE_SPAN], test.start_utc.min())
    return cls().fit(train).predict_cdf(test)


def evaluate_task(task: Task, df: pd.DataFrame, test_year: int = 2026, min_train_races: int = 10) -> EvalResult:
    if task.evaluate is not None:
        return task.evaluate(df, test_year=test_year, min_train_races=min_train_races)
    if task.kind not in _SCORERS:
        hint = "a survival task must supply its own evaluate (see bench/laps_to_stop.py)" if task.kind == "survival" \
            else f"generic kinds: {sorted(_SCORERS)}"
        raise ValueError(f"{task.name}: kind {task.kind!r} has no generic scorer; {hint}")
    if task.kind == "multiclass" and not task.classes:
        raise ValueError(f"{task.name}: a multiclass task needs classes")
    data = task.select(df)
    preds, timing = [], {cls.name: [] for cls in task.models}
    for _race_id, train, test in _splits(data, test_year, min_train_races):
        out = test[["race_id", "driver", "tla", "lap", "t", *task.keep, task.target]].copy()
        for cls in task.models:
            p = fit_predict(cls, train, test, task.target, timing[cls.name])
            if task.kind == "multiclass":
                p = np.asarray(p, dtype=float)
                if p.shape != (len(test), len(task.classes)):
                    raise ValueError(f"{task.name}/{cls.name}: predict gave shape {p.shape}, want ({len(test)}, {len(task.classes)})")
                out = out.assign(**{proba_col(cls.name, c): p[:, i] for i, c in enumerate(task.classes)})
            else:
                out[cls.name] = p
        preds.append(out)
    if not preds:
        raise ValueError(f"{task.name}: no {test_year} race has {min_train_races}+ earlier races to train on")
    return _score(task, pd.concat(preds, ignore_index=True), timing)


_SCORERS = {
    "binary": lambda task, y, p: binary_scores(y, p),
    "position": lambda task, y, p: position_scores(y, p),
    "regression": lambda task, y, p: regression_scores(y, p),
    "multiclass": lambda task, y, p: multiclass_scores(y, p, task.classes),
}


def _score(task: Task, P: pd.DataFrame, timing: dict[str, list]) -> EvalResult:
    """Micro (row-pooled) scores, the primary metric per race and its macro mean, the paired race-bootstrap
    delta vs the baseline (models[0]; negative = lower metric), and fit/predict cost."""
    names, metric, score, multi = [c.name for c in task.models], task.metric, _SCORERS[task.kind], task.kind == "multiclass"

    def ys(S):
        return S[task.target].to_numpy(object if multi else float)

    def pred(S, n):
        return S[[proba_col(n, c) for c in task.classes]].to_numpy(float) if multi else S[n].to_numpy(float)

    board = pd.DataFrame([{"model": n, **score(task, ys(P), pred(P, n))} for n in names])
    if task.kind == "binary":
        board["brier_skill"] = 1 - board["brier"] / board.loc[0, "brier"]  # vs the reference baseline
        board["ece"] = [ece(ys(P), pred(P, n)) for n in names]
    per_race = pd.DataFrame([{"race_id": r, "n": len(g), **{n: score(task, ys(g), pred(g, n))[metric] for n in names}}
                             for r, g in P.groupby("race_id", sort=False)])
    board[f"{metric}_macro"] = [float(per_race[n].mean()) for n in names]
    ci = [paired_race_bootstrap(per_race, n, names[0]) for n in names]
    board["d_vs_base"], board["d_lo"], board["d_hi"] = ([c[i] for c in ci] for i in range(3))
    board["fit_s"] = [float(np.mean([f for f, _, _ in timing[n]])) for n in names]
    board["pred_us"] = [1e6 * sum(p for _, p, _ in timing[n]) / max(sum(k for _, _, k in timing[n]), 1) for n in names]
    if task.kind == "position" and metric == "mae":
        board = board.sort_values(["mae", "exact"], ascending=[True, False])
    else:
        board = board.sort_values(metric, ascending=not higher_is_better(metric))
    cal = {n: calibration_table(ys(P), pred(P, n)) for n in names} if task.kind == "binary" else {}
    return EvalResult(task.name, board.reset_index(drop=True), per_race, P, cal, metric)


def evaluate_pit(df: pd.DataFrame, horizon: int = 3, test_year: int = 2026, min_train_races: int = 10) -> EvalResult:
    (task,) = [t for t in core_tasks(horizons=(horizon,)) if t.kind == "binary"]
    return evaluate_task(task, df, test_year, min_train_races)


def evaluate_rejoin(df: pd.DataFrame, test_year: int = 2026, min_train_races: int = 10) -> EvalResult:
    (task,) = [t for t in core_tasks(horizons=()) if t.name == "position_after_stop"]
    return evaluate_task(task, df, test_year, min_train_races)


def write_report(results: list[EvalResult], out_dir: Path, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# PitSense-Bench leaderboard", ""]
    lines.append(
        f"Test races: {meta['test_races']} ({meta['test_year']}), each scored with models trained only on "
        f"races that finished before it started. Decision points: {meta['decision_points']:,} "
        f"from {meta['races']} races. Feature version {meta['feature_version']}."
    )
    lines.append("")
    for r in results:
        lines.append(f"## {r.task}")
        lines.append("")
        lines.append(_md_table(r.leaderboard))
        lines.append("")
        if r.calibration:  # the best model's table, else the first one there is (custom evaluations calibrate a subset)
            best = r.leaderboard.iloc[0]["model"] if len(r.leaderboard) and "model" in r.leaderboard else None
            best = best if best in r.calibration else next(iter(r.calibration))
            lines.append(f"Calibration of `{best}` (predicted vs observed rate):")
            lines.append("")
            lines.append(_md_table(r.calibration[best].assign(bin=lambda x: x["bin"].astype(str))))
            lines.append("")
        r.predictions.to_parquet(out_dir / f"predictions_{r.task}.parquet", index=False)
        r.per_race.to_csv(out_dir / f"per_race_{r.task}.csv", index=False)
    path = out_dir / "leaderboard.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "leaderboard.json").write_text(
        json.dumps({"meta": {**meta, "primary": {r.task: r.primary for r in results if r.primary}},
                    **{r.task: r.leaderboard.to_dict(orient="records") for r in results}}, indent=1, default=str),
        encoding="utf-8",
    )
    return path


def _md_table(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.4f}"
        return str(v)

    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(out)
