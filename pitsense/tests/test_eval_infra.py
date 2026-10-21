"""Evaluation infrastructure: metrics, bootstrap CIs, the generic evaluate loop, bench-compare, bundle feature hash."""

import json
import logging
import pickle
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from pitsense.bench import compare
from pitsense.bench.evaluate import evaluate_task, proba_col, write_report
from pitsense.bench.metrics import ece, multiclass_scores, paired_race_bootstrap, race_bootstrap
from pitsense.bench.tasks import Task
from pitsense.modelstore import TrainedBundle, latest_bundle_for, load_bundle, save_bundle

T0 = datetime(2025, 3, 1, 13, 0, tzinfo=timezone.utc)
CLASSES = ("SOFT", "MEDIUM", "HARD")


# ----------------------------------------------------------------------------- metrics
def test_multiclass_uniform_and_perfect():
    y = np.array(["SOFT", "MEDIUM", "HARD", "SOFT"])
    s = multiclass_scores(y, np.full((4, 3), 1 / 3), CLASSES)
    assert s["log_loss"] == pytest.approx(np.log(3))
    assert s["brier"] == pytest.approx(2 / 3) and s["top2"] == 1.0  # ties: no class is strictly more likely
    onehot = (y[:, None] == np.array(CLASSES)).astype(float)
    p = multiclass_scores(y, onehot, CLASSES)
    assert p["accuracy"] == 1.0 and p["top2"] == 1.0 and p["log_loss"] < 1e-5
    assert p["recall_HARD"] == 1.0 and set(p) >= {"n", "log_loss", "brier", "accuracy", "top2", "recall_SOFT", "recall_MEDIUM"}
    with pytest.raises(ValueError):
        multiclass_scores(np.array(["SOFT", ""]), np.full((2, 3), 1 / 3), CLASSES)


def test_ece_perfect_calibration_near_zero():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=50_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    assert ece(y, p) < 0.01
    assert ece(y, np.full_like(p, 0.9)) > 0.3


def test_race_bootstrap_deterministic_and_paired():
    vals = {f"r{i}": v for i, v in enumerate([0.1, 0.3, 0.2, 0.5, 0.4, 0.25])}
    a, b = race_bootstrap(vals), race_bootstrap(pd.Series(vals))
    assert a == b and a[1] <= a[0] <= a[2] and a[0] == pytest.approx(np.mean(list(vals.values())))
    assert race_bootstrap(vals, seed=1) != a
    pr = pd.DataFrame({"race_id": list(vals), "base": list(vals.values())})
    pr["m"] = pr["base"] - 0.05
    d, lo, hi = paired_race_bootstrap(pr, "m", "base")
    assert d == pytest.approx(-0.05) and lo == pytest.approx(-0.05) and hi == pytest.approx(-0.05)
    assert np.isnan(race_bootstrap({"r0": 1.0})[1])


# ----------------------------------------------------------------------------- evaluate on a tiny fake bench
def _fake(n_races=8, n=60):
    rng = np.random.default_rng(0)
    parts = []
    for r in range(n_races):
        x = rng.normal(size=n)
        df = pd.DataFrame({"x": x, "driver": "1", "tla": "VER", "lap": np.arange(n), "t": np.arange(n, dtype=float)})
        df["y_bin"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-2 * x))).astype(float)
        df["y_cls"] = np.where(x < -0.4, "SOFT", np.where(x < 0.4, "MEDIUM", "HARD"))
        df["y_reg"] = 3 * x + rng.normal(size=n)
        df["race_id"] = f"r{r}"
        df["year"] = 2024 if r < 4 else 2025
        df["start_utc"] = pd.Timestamp(T0 + timedelta(days=7 * r))
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


class Mean:
    name = "mean"

    def fit(self, df, target):
        self.m = float(df[target].mean())
        return self

    def predict(self, df):
        return np.full(len(df), self.m)


class Logit:
    name = "logit"

    def fit(self, df, target):
        return self

    def predict(self, df):
        return 1 / (1 + np.exp(-2 * df["x"].to_numpy()))


class Linear:
    name = "linear"

    def fit(self, df, target):
        self.b = float(np.polyfit(df["x"], df[target], 1)[0])
        return self

    def predict(self, df):
        return self.b * df["x"].to_numpy()


class Uniform:
    name = "uniform"

    def fit(self, df, target):
        return self

    def predict(self, df):
        return np.full((len(df), 3), 1 / 3)


class Sharp:
    name = "sharp"

    def fit(self, df, target):
        return self

    def predict(self, df):
        x = df["x"].to_numpy()
        k = np.where(x < -0.4, 0, np.where(x < 0.4, 1, 2))
        return np.where(np.arange(3) == k[:, None], 0.8, 0.1)


def _sel(df):
    return df.copy()


def test_evaluate_binary_columns_and_ci():
    task = Task("toy_bin", "binary", "y_bin", _sel, (Mean, Logit))
    r = evaluate_task(task, _fake(), test_year=2025, min_train_races=2)
    b = r.leaderboard.set_index("model")
    assert r.primary == "log_loss" and r.leaderboard.iloc[0]["model"] == "logit"
    for c in ("log_loss_macro", "d_vs_base", "d_lo", "d_hi", "ece", "fit_s", "pred_us", "brier_skill"):
        assert c in b.columns
    assert b.loc["mean", "d_vs_base"] == 0
    assert b.loc["logit", "d_hi"] < 0 and b.loc["logit", "d_lo"] <= b.loc["logit", "d_vs_base"] <= b.loc["logit", "d_hi"]
    assert list(r.per_race.columns) == ["race_id", "n", "mean", "logit"] and r.per_race["n"].sum() == len(r.predictions)
    assert b.loc["logit", "log_loss_macro"] == pytest.approx(r.per_race["logit"].mean())
    again = evaluate_task(task, _fake(), test_year=2025, min_train_races=2).leaderboard
    cols = ["model", "log_loss", "log_loss_macro", "d_vs_base", "d_lo", "d_hi", "ece"]
    pd.testing.assert_frame_equal(r.leaderboard[cols], again[cols])


def test_evaluate_regression_and_multiclass(tmp_path):
    reg = evaluate_task(Task("toy_reg", "regression", "y_reg", _sel, (Mean, Linear)), _fake(), test_year=2025, min_train_races=2)
    assert reg.primary == "mae" and reg.leaderboard.iloc[0]["model"] == "linear" and "mae_macro" in reg.leaderboard
    task = Task("toy_cls", "multiclass", "y_cls", _sel, (Uniform, Sharp), classes=CLASSES)
    mc = evaluate_task(task, _fake(), test_year=2025, min_train_races=2)
    b = mc.leaderboard.set_index("model")
    assert b.loc["uniform", "log_loss"] == pytest.approx(np.log(3)) and mc.leaderboard.iloc[0]["model"] == "sharp"
    assert b.loc["sharp", "accuracy"] == 1.0 and b.loc["sharp", "d_hi"] < 0
    assert proba_col("sharp", "HARD") in mc.predictions
    with pytest.raises(ValueError, match="classes"):
        evaluate_task(Task("bad", "multiclass", "y_cls", _sel, (Uniform,)), _fake(), test_year=2025, min_train_races=2)
    # a calibration table without the best model must not crash the report
    bin_r = evaluate_task(Task("toy_bin", "binary", "y_bin", _sel, (Mean, Logit)), _fake(), test_year=2025, min_train_races=2)
    bin_r.calibration = {"mean": bin_r.calibration["mean"]}
    write_report([bin_r, mc, reg], tmp_path, {"test_races": 4, "test_year": 2025, "decision_points": 1, "races": 8,
                                               "feature_version": "t"})
    data = json.loads((tmp_path / "leaderboard.json").read_text(encoding="utf-8"))
    assert data["meta"]["primary"] == {"toy_bin": "log_loss", "toy_cls": "log_loss", "toy_reg": "mae"}
    assert {"log_loss_macro", "d_lo", "d_hi", "fit_s", "pred_us"} <= set(data["toy_cls"][0])
    assert "Calibration of `mean`" in (tmp_path / "leaderboard.md").read_text(encoding="utf-8")


def test_survival_without_evaluate_is_a_clear_error():
    with pytest.raises(ValueError, match="survival task must supply its own evaluate"):
        evaluate_task(Task("toy_surv", "survival", "y_reg", _sel, (Mean,)), _fake(), test_year=2025, min_train_races=2)


# ----------------------------------------------------------------------------- bench-compare
def test_compare_old_base_and_new_with_macro(tmp_path):
    old = {"meta": {"test_year": 2025},
           "pit_within_1": [{"model": "gbm", "log_loss": 0.20}, {"model": "base_rate", "log_loss": 0.25}],
           "position_after_stop": [{"model": "nc", "mae": 1.1, "exact": 0.4}],
           "pit_loss": [{"split": "all", "model": "a", "mae": 2.0}, {"split": "sc", "model": "b", "mae": 1.0}],
           "dropped": [{"model": "x", "log_loss": 0.5}]}
    new = {"meta": {"test_year": 2025, "primary": {"pit_within_1": "log_loss", "acc": "accuracy"}},
           "pit_within_1": [{"model": "gbm2", "log_loss": 0.18, "log_loss_macro": 0.19}],
           "position_after_stop": [{"model": "nc", "mae": 1.2, "exact": 0.5, "mae_macro": 1.0}],
           "pit_loss": [{"split": "all", "model": "a", "mae": 1.5}],
           "acc": [{"model": "m", "accuracy": 0.7}, {"model": "n", "accuracy": 0.6}]}
    (tmp_path / "a.json").write_text(json.dumps(old), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(new), encoding="utf-8")
    rows = {r["task"]: r for r in compare.compare(compare.load(tmp_path / "a.json"), compare.load(tmp_path / "b.json"))}
    p = rows["pit_within_1"]
    assert (p["base_model"], p["new_model"], p["better"]) == ("gbm", "gbm2", True)
    assert p["delta"] == pytest.approx(-0.02) and np.isnan(p["base_macro"]) and p["new_macro"] == 0.19
    assert rows["position_after_stop"]["metric"] == "mae" and rows["position_after_stop"]["better"] is False
    assert rows["pit_loss"]["base"] == 2.0 and rows["pit_loss"]["delta"] == pytest.approx(-0.5)
    assert rows["dropped"]["status"] == "missing in new" and rows["acc"]["status"] == "only in new"
    assert rows["acc"]["new_model"] == "m"
    md = compare.to_markdown(list(rows.values()))
    assert "missing in new" in md and "| pit_within_1 | log_loss | gbm |" in md


# ----------------------------------------------------------------------------- bundles
def _bundle(**kw):
    return TrainedBundle(T0 - timedelta(days=1), T0, ("r0",), "v0.1", None, **kw)


def test_bundle_features_hash_refused(tmp_path, monkeypatch, caplog):
    from pitsense import modelstore

    monkeypatch.setattr(modelstore, "FEATURE_VERSION", "v0.1")
    monkeypatch.setattr(modelstore, "current_features_hash", lambda: "aaa")
    good, bad = save_bundle(_bundle(features_hash="aaa"), tmp_path / "good.pkl"), save_bundle(_bundle(features_hash="bbb"), tmp_path / "bad.pkl")
    assert load_bundle(good).features_hash == "aaa"
    with pytest.raises(ValueError, match="feature columns bbb"):
        load_bundle(bad)
    with caplog.at_level(logging.WARNING, logger="pitsense.modelstore"):
        assert latest_bundle_for(T0 + timedelta(days=1), tmp_path).features_hash == "aaa"
    assert any("bad.pkl" in m for m in caplog.messages)
    assert load_bundle(save_bundle(_bundle(), tmp_path / "none.pkl")).features_hash is None  # not recorded: allowed


def test_old_pickle_gets_new_defaults(tmp_path):
    b = _bundle()
    del b.__dict__["features_hash"], b.__dict__["kinds"]
    path = tmp_path / "old.pkl"
    path.write_bytes(pickle.dumps(b))
    old = load_bundle(path)
    assert old.features_hash is None and old.kinds == {}
