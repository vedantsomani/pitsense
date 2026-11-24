"""Scores for probabilistic and positional predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6
CAL_BINS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def binary_scores(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = {"n": int(len(y)), "positives": int(y.sum()), "log_loss": log_loss(y, p), "brier": brier(y, p)}
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, p))
        out["avg_precision"] = float(average_precision_score(y, p))
    else:
        out["auc"] = out["avg_precision"] = float("nan")
    return out


def calibration_table(y: np.ndarray, p: np.ndarray, bins=CAL_BINS) -> pd.DataFrame:
    """Predicted vs observed rate per probability bin."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.cut(df["p"], bins=list(bins), include_lowest=True)
    g = df.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return g.reset_index()


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }


def position_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    return {
        "n": int(len(y_true)),
        "exact": float(np.mean(err == 0)),
        "within_1": float(np.mean(err <= 1)),
        "mae": float(np.mean(err)),
    }


# metrics where higher is better (everything else, e.g. log_loss, brier, mae, ibs, rps, ece: lower is better)
HIGHER_BETTER = frozenset({"exact", "within_1", "auc", "avg_precision", "accuracy", "top2", "c_index", "brier_skill",
                           "precision", "recall", "window_acc"})


def higher_is_better(metric: str) -> bool:
    return metric in HIGHER_BETTER or metric.startswith("recall_")


def multiclass_scores(y, P: np.ndarray, classes) -> dict:
    """``P``: [n, C] probabilities in ``classes`` order; ``y``: class names."""
    classes = list(classes)
    y = np.asarray(y, dtype=object)
    idx = pd.Series(y).map({c: i for i, c in enumerate(classes)})
    if idx.isna().any():
        raise ValueError(f"labels not in {classes}: {sorted(set(map(str, y[idx.isna().to_numpy()])))}")
    k = idx.to_numpy(int)
    P = np.clip(np.asarray(P, dtype=float).reshape(len(y), len(classes)), EPS, None)
    P = P / P.sum(1, keepdims=True)
    rows = np.arange(len(k))
    onehot = np.zeros_like(P)
    onehot[rows, k] = 1.0
    rank = (P > P[rows, k][:, None]).sum(1)  # classes strictly more likely than the true one
    out = {"n": int(len(k)), "log_loss": float(-np.mean(np.log(P[rows, k]))) if len(k) else float("nan"),
           "brier": float(np.mean(((P - onehot) ** 2).sum(1))) if len(k) else float("nan"),
           "accuracy": float(np.mean(P.argmax(1) == k)) if len(k) else float("nan"),
           "top2": float(np.mean(rank < 2)) if len(k) else float("nan")}
    for i, c in enumerate(classes):
        m = k == i
        out[f"recall_{c}"] = float(np.mean(P[m].argmax(1) == i)) if m.any() else float("nan")
    return out


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-mass bins: sum over bins of share * |observed - predicted|."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if not len(y):
        return float("nan")
    o = np.argsort(p, kind="stable")
    return float(sum(len(b) * abs(y[b].mean() - p[b].mean()) for b in np.array_split(o, min(n_bins, len(o))) if len(b)) / len(y))


def race_bootstrap(values_by_race, B: int = 2000, seed: int = 0, alpha: float = 0.10) -> tuple:
    """(mean, lo, hi) of one value per race, races resampled; NaN races dropped. Interval is (1 - alpha)."""
    v = pd.Series(values_by_race, dtype=float).dropna().to_numpy()
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    if len(v) < 2:
        return mean, float("nan"), float("nan")
    m = v[np.random.default_rng(seed).integers(0, len(v), (B, len(v)))].mean(1)
    return mean, float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_race_bootstrap(per_race: pd.DataFrame, model: str, base: str, B: int = 2000, seed: int = 0) -> tuple:
    """(delta_mean, lo, hi): per-race ``model - base`` metric, races resampled, 90 % interval."""
    return race_bootstrap(per_race[model].astype(float) - per_race[base].astype(float), B=B, seed=seed)
