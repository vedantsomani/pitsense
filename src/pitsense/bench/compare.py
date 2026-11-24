"""Compare two benchmark runs: per task, the best model of each run on the task's primary metric.

Reads ``leaderboard.json`` files written by ``bench.evaluate.write_report``, including ones written before
macro metrics existed (then only the micro number is compared). A task's primary metric comes from the
file's ``meta.primary``, else from the leaderboard's columns (``PREFERENCE`` order).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .metrics import higher_is_better

# primary metric when a file does not name one: the first of these the leaderboard has
PREFERENCE = ("log_loss", "ibs", "rps", "mae", "mae_laps", "exact", "auc", "accuracy", "brier")
_SUBSET = ("split", "subset")  # boards split into slices are compared on their "all" slice


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def primary_of(task: str, *boards: dict) -> str | None:
    for b in boards:
        p = ((b.get("meta") or {}).get("primary") or {}).get(task)
        if p:
            return p
    rows = [r for b in boards for r in (b.get(task) or []) if isinstance(r, dict)]
    return next((m for m in PREFERENCE if any(m in r for r in rows)), None)


def best_row(rows: list, metric: str) -> dict | None:
    """The leaderboard row with the best ``metric`` (on the "all" slice if the board is sliced)."""
    rows = [r for r in rows or [] if isinstance(r, dict) and not math.isnan(_num(r.get(metric)))]
    for col in _SUBSET:
        if any(col in r for r in rows):
            rows = [r for r in rows if r.get(col) == "all"]
    if not rows:
        return None
    sign = -1 if higher_is_better(metric) else 1
    return min(rows, key=lambda r: sign * _num(r[metric]))


def compare(base: dict, new: dict) -> list[dict]:
    """One row per task in either file: best model and primary metric (micro, macro) on each side, deltas."""
    tasks = [t for t in base if t != "meta" and isinstance(base[t], list)]
    tasks += [t for t in new if t != "meta" and isinstance(new[t], list) and t not in tasks]
    out = []
    for t in tasks:
        metric = primary_of(t, new, base)
        row = {"task": t, "metric": metric, "status": "ok"}
        if t not in base or t not in new:
            row["status"] = "only in new" if t not in base else "missing in new"
        if metric is None:
            row["status"] = "no metric"
            out.append(row)
            continue
        for side, board in (("base", base), ("new", new)):
            b = best_row(board.get(t), metric)
            row[f"{side}_model"] = b.get("model") if b else None
            row[side] = _num(b[metric]) if b else float("nan")
            row[f"{side}_macro"] = _num(b.get(f"{metric}_macro")) if b else float("nan")
        row["delta"] = row["new"] - row["base"]
        row["delta_macro"] = row["new_macro"] - row["base_macro"]
        row["better"] = None if math.isnan(row["delta"]) or row["delta"] == 0 else \
            (row["delta"] > 0) == higher_is_better(metric)
        out.append(row)
    return out


def to_markdown(rows: list[dict], base_name: str = "base", new_name: str = "new") -> str:
    def f(v):
        return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.4f}" if isinstance(v, float) else str(v)

    def g(v):
        return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"

    lines = [f"# Benchmark comparison: `{base_name}` -> `{new_name}`", "",
             "Best model of each run on the task's primary metric. Macro = mean over test races (blank when the run "
             "did not record it).", "",
             "| task | metric | base model | base | base macro | new model | new | new macro | delta | delta macro | "
             "better | status |", "|" + "---|" * 12]
    for r in rows:
        better = {True: "yes", False: "no", None: ""}[r.get("better")]
        lines.append(f"| {r['task']} | {r.get('metric') or ''} | {r.get('base_model') or ''} | {g(r.get('base'))} | "
                     f"{g(r.get('base_macro'))} | {r.get('new_model') or ''} | {g(r.get('new'))} | {g(r.get('new_macro'))} | "
                     f"{f(r.get('delta'))} | {f(r.get('delta_macro'))} | {better} | {r['status']} |")
    return "\n".join(lines) + "\n"


def cmd_compare(a) -> None:
    rows = compare(load(a.base), load(a.new))
    text = to_markdown(rows, a.base, a.new)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"saved {a.out}")


def add_commands(sub) -> None:
    s = sub.add_parser("bench-compare", help="compare two benchmark leaderboard.json files task by task")
    s.add_argument("--base", required=True, help="baseline leaderboard.json")
    s.add_argument("--new", required=True, help="new leaderboard.json")
    s.add_argument("--out", help="also write the table to this markdown file")
    s.set_defaults(fn=cmd_compare)
