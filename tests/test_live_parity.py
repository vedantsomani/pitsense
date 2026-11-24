"""Live/benchmark parity: the wall a live or followed source builds is the wall the benchmark builds."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from pitsense.archive import SessionRef
from pitsense.asof import HistoryStore, RaceSummary
from pitsense.bench.features import FeatureBuilder, feature_columns
from pitsense.bench.history import prior_for
from pitsense.bench.models import BaseRate, NoChange
from pitsense.modelstore import TrainedBundle
from pitsense.pitloss import PitLossPrior
from pitsense.pitwall import Context, PitWall
from pitsense.pitwall.runtime import PitWallRuntime, ReplaySource, session_meta
from pitsense.state import RaceState

from .conftest import make_log, synthetic_events

START = datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc)
INFO = {"Meeting": {"Key": 1300, "Name": "Test Grand Prix", "Location": "Testville", "Country": {"Key": 1, "Name": "Testland"},
                    "Circuit": {"Key": 7, "ShortName": "Testville"}},
        "Key": 9001, "Type": "Race", "Name": "Race", "StartDate": "2026-05-01T15:00:00", "GmtOffset": "02:00:00",
        "Path": "2026/2026-05-01_Test_Grand_Prix/2026-05-01_Race/"}


def _history() -> HistoryStore:
    end = START - timedelta(days=7)
    return HistoryStore([RaceSummary("2026-01-test", 2026, 7, end - timedelta(hours=2), end,
                                     [24.0 + 0.1 * i for i in range(20)], [12.0] * 12, [15.0] * 12, 57, 3, 2)])


def _same(a, b):
    return (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b)) or a == b


def _bundle():
    b = TrainedBundle(START - timedelta(days=1), START, ("r0",), "v0.1", None)
    b.models = {"pit_within_1": BaseRate().fit(pd.DataFrame({"y": [0, 0, 0, 1]}), "y"),
                "pit_within_3": BaseRate().fit(pd.DataFrame({"y": [0, 1]}), "y"), "position_after_stop": NoChange()}
    return b


def _ctx(meta, history):
    prior = prior_for(history, meta["circuit_key"], START)
    return prior, Context.for_race(prior, meta, history=history, race_start_utc=START)


def test_contextual_parity_with_history_and_realistic_meta(race_log):
    """With the same prior, meta, history and start, every shared column of the live lap_end row equals the benchmark's."""
    meta = {**race_log.meta, **session_meta(INFO), "slug": "2026-05-test-grand-prix-race"}
    history = _history()
    prior, bctx = _ctx(meta, history)
    assert prior != PitLossPrior()  # the history is not decorative
    _, lctx = _ctx(meta, history)
    fb = FeatureBuilder(prior, meta, ctx=bctx)
    wall = PitWall(lctx)
    me = wall.engineer("models")
    state, events, checked = RaceState(meta), race_log.events, 0
    queued = []
    for k, e in enumerate(events):
        state.apply(e)
        fb.observe(state)
        wall.observe(state)
        queued += state.new_laps
        if not (queued and (k + 1 == len(events) or events[k + 1].t > e.t)):
            continue
        view = wall.view(state)
        for rec in queued:
            bench, (live, _) = fb.row(state, rec), me.decision_rows(state, rec.driver, view)
            assert (bench is None) == (live is None)
            if bench is None:
                continue
            assert set(feature_columns()) <= live.keys()
            assert all(_same(bench[c], live[c]) for c in bench if c in live), [c for c in bench if c in live and not _same(bench[c], live[c])]
            checked += 1
        queued = []
    assert checked > 10


def test_follow_meta_equals_archive_meta_after_session_info():
    ref = SessionRef(year=2026, meeting_key=1300, meeting_name="Test Grand Prix", location="Testville", country="Testland",
                     circuit_key=7, circuit="Testville", round_index=5, session_key=9001, session_name="Race",
                     session_type="Race", start_local="2026-05-01T15:00:00", gmt_offset="02:00:00",
                     path="2026/2026-05-01_Test_Grand_Prix/2026-05-01_Race/")
    got = session_meta(INFO)
    assert got == {k: v for k, v in ref.to_dict().items() if k != "round_index"}
    assert session_meta({}) == {} and session_meta({"Meeting": {"Name": "x"}}) == {}


def test_runtime_enriches_a_refless_source_once_and_rebuilds_the_prior():
    ev = [(t, topic, INFO if topic == "SessionInfo" else d) for t, topic, d in synthetic_events()]
    history = _history()
    rt = PitWallRuntime(ReplaySource(make_log(ev), 0), models=False, history=history)
    rt.start()
    assert rt.wait(30)
    ctx = rt.wall.ctx
    assert ctx.meta["circuit_key"] == 7 and ctx.meta["year"] == 2026 and ctx.meta["meeting_name"] == "Test Grand Prix"
    assert ctx.race_start_utc == START
    assert ctx.prior == prior_for(history, 7, START) and ctx.prior != PitLossPrior()
    assert len(ctx.past_races) == 1


def test_memo_is_bounded_and_lap_end_predictions_are_per_lap(race_log):
    wall = PitWall(Context(prior=PitLossPrior(), models=_bundle()))
    me = wall.engineer("models")
    state, first, peak = RaceState(race_log.meta), {}, 0
    for e in race_log.events:
        state.apply(e)
        wall.observe(state)
        view = wall.view(state)
        for _ in range(3):  # several ticks per event
            me.prefetch(state, list(state.drivers), view)
        for key, vals in me._lap_memo.items():
            assert first.setdefault(key, vals) == vals  # as first computed, never recomputed later in the lap
        peak = max(peak, len(me._lap_memo))
        if state.current_lap != getattr(me, "_seen_lap", None):
            me._seen_lap = state.current_lap
            me._memo.clear()  # what replay.trim_memo does at each leader lap
    assert 0 < peak <= len(state.drivers) * 4
    assert len(me._memo) < 4 * len(state.drivers)
