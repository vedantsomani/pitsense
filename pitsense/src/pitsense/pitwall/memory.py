"""Indexes built incrementally from the race state, shared by every engineer.

They are filled from what the state already holds, one event at a time, so
they are as-of by construction. Each record of the state (laps, lap times, pit
events, track status, race control, weather) is read through a cursor, so the
cost per event is O(new items) and a decision row never rescans the race.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

from ..pitloss import LapIndex, measure_stops
from ..state import LapRecord, RaceState

NEUTRAL = ("4", "5", "6", "7")  # SC, red, VSC, VSC ending
_YELLOW = ("yellow", "double_yellow")
_INCIDENT = ("noted", "investigation", "investigation_after")


def is_clean(lap: LapRecord | None) -> bool:
    """A lap that says something about pace: timed, not lap 1, no pit stop, green throughout."""
    return (
        lap is not None
        and lap.lap_time is not None
        and lap.lap > 1
        and not lap.is_in_lap
        and not lap.is_out_lap
        and lap.track_status == "1"
    )


@dataclass(frozen=True)
class CarSnap:
    """A car's tyre and stint state at its last lap end: what a ``pit_entry`` row describes."""

    lap: int
    t: float
    stint: int
    stint_start_lap: int
    compound: str | None
    tyre_new: bool | None
    tyre_age: int | None
    compounds_used: tuple[str, ...]


def _snap(d, t: float) -> CarSnap:
    return CarSnap(d.laps, t, d.stint, d.stint_start_lap, d.compound, d.tyre_new, d.tyre_age, tuple(d.compounds_used))


class RaceMemory:
    """Laps by driver and lap number, the order at each pit entry, pit losses so far, and
    as-of race history (status onsets, race-control counts, a weather ring) for the features."""

    def __init__(self) -> None:
        from .engineers.rules.parser import parse  # pure function; a top-level import would be circular

        self._parse = parse
        self.index = LapIndex([])
        self.order_at_pit: dict[int, dict[str, int]] = {}  # pit event idx -> positions at entry
        self._n_laps_seen = 0
        self._n_pits_seen = 0
        self._loss_cache_key: tuple[int, int, int] | None = None
        self._loss_cache: dict[str, list[float]] = {"green": [], "sc": [], "vsc": []}
        # lap times (incl. late ones) and per-lap field medians
        self._n_timed = 0
        self.lap_rev: dict[int, int] = defaultdict(int)  # lap -> times set so far (cache key)
        self._fmed: dict[int, tuple] = {}
        # car state at each lap end; refreshed by later events at the same instant
        self.car_at_lap_end: dict[str, CarSnap] = {}
        self._snap_t: float | None = None
        self._snapped: list[str] = []
        # pit events
        self._open_pits: list[int] = []
        self._n_pit_out = 0
        self.pit_laps: dict[str, list[int]] = defaultdict(list)  # driver -> in-laps of every stop
        self.pits_by_lap: dict[int, list[str]] = defaultdict(list)  # in-lap -> drivers (not under red)
        # track status since the start
        self._n_status = 0
        self._status_prev = "1"
        self.n_sc = self.n_vsc = self.n_red = 0
        self.neutral_end_t: float | None = None
        self.neutral_end_lap: int | None = None
        # race control
        self._n_rc = 0
        self.yellow_t: list[float] = []
        self.incident_t: list[float] = []
        self.pit_exit_closed = 0
        # weather ring: one sample per WeatherData message
        self.w_t: list[float] = []
        self.w: dict[str, list[float | None]] = {k: [] for k in ("Rainfall", "TrackTemp", "AirTemp", "Humidity")}
        self.last_rain_t: float | None = None

    def observe(self, state: RaceState) -> None:
        """Call after every applied event: indexes new laps and freezes order at pit entry."""
        if self._snap_t == state.t:
            for n in self._snapped:
                self.car_at_lap_end[n] = _snap(state.drivers[n], state.t)
        if len(state.laps) > self._n_laps_seen:
            if self._snap_t != state.t:
                self._snap_t, self._snapped = state.t, []
            for lap in state.laps[self._n_laps_seen:]:
                self.index.add(lap)
                self._snapped.append(lap.driver)
                self.car_at_lap_end[lap.driver] = _snap(state.drivers[lap.driver], state.t)
            self._n_laps_seen = len(state.laps)
        if len(state.timed_laps) > self._n_timed:
            for lap in state.timed_laps[self._n_timed:]:
                self.lap_rev[lap.lap] += 1
            self._n_timed = len(state.timed_laps)
        if self._open_pits:
            still = [i for i in self._open_pits if state.pit_events[i].out_lap is None]
            self._n_pit_out += len(self._open_pits) - len(still)
            self._open_pits = still
        if len(state.pit_events) > self._n_pits_seen:
            positions = {d.number: d.position for d in state.drivers.values() if d.position is not None}
            for i in range(self._n_pits_seen, len(state.pit_events)):
                pe = state.pit_events[i]
                pre = dict(positions)
                # the stopping car may already have been demoted; use its last lap's position
                last = self.index.by_driver.get(pe.driver, {}).get(pe.in_lap - 1)
                if last is not None and last.position is not None:
                    pre[pe.driver] = last.position
                self.order_at_pit[i] = pre
                self.pit_laps[pe.driver].append(pe.in_lap)
                if not pe.under_red:
                    self.pits_by_lap[pe.in_lap].append(pe.driver)
                if pe.out_lap is None:
                    self._open_pits.append(i)
                else:
                    self._n_pit_out += 1
            self._n_pits_seen = len(state.pit_events)
        if len(state.status_log) > self._n_status:
            self._statuses(state)
        if len(state.rc) > self._n_rc:
            self._race_control(state)
        if state.weather_t is not None and state.weather_t == state.t:
            self._weather(state)

    # ------------------------------------------------------------------ cursors
    def _statuses(self, state: RaceState) -> None:
        for ts, code in state.status_log[self._n_status:]:
            prev, self._status_prev = self._status_prev, code
            if state.started_t is None or ts < state.started_t:
                continue  # before the start: not part of the race
            if code == "4" and prev != "4":
                self.n_sc += 1
            elif code == "6" and prev not in ("6", "7"):
                self.n_vsc += 1
            elif code == "5" and prev != "5":
                self.n_red += 1
            if prev in NEUTRAL and code not in NEUTRAL:
                self.neutral_end_t, self.neutral_end_lap = ts, state.current_lap
        self._n_status = len(state.status_log)

    def _race_control(self, state: RaceState) -> None:
        for m in state.rc[self._n_rc:]:
            kind = self._parse(m.message, m.category, m.flag).kind
            if kind in _YELLOW:
                self.yellow_t.append(m.t)
            elif kind in _INCIDENT:
                self.incident_t.append(m.t)
            elif kind == "pit_exit_closed":
                self.pit_exit_closed = 1
            elif kind == "pit_exit_open":
                self.pit_exit_closed = 0
        self._n_rc = len(state.rc)

    def _weather(self, state: RaceState) -> None:
        if self.w_t and self.w_t[-1] == state.weather_t:  # a second message at the same instant
            for k, xs in self.w.items():
                xs[-1] = state.weather.get(k)
        else:
            self.w_t.append(state.weather_t)
            for k, xs in self.w.items():
                xs.append(state.weather.get(k))
        if (state.weather.get("Rainfall") or 0) > 0:
            self.last_rain_t = state.weather_t

    # ------------------------------------------------------------------ queries
    @staticmethod
    def count_since(times: list[float], since: float) -> int:
        """Entries of a time-sorted list after ``since``."""
        return len(times) - bisect_right(times, since)

    def weather_at(self, key: str, t: float) -> float | None:
        """The weather value of the last sample at or before ``t`` (None if there is none)."""
        j = bisect_right(self.w_t, t)
        return self.w[key][j - 1] if j else None

    def weather_mean(self, key: str, since: float) -> float | None:
        """Mean of the samples after ``since`` (None if there are none)."""
        xs = [x for x in self.w[key][bisect_left(self.w_t, since):] if x is not None]
        return sum(xs) / len(xs) if xs else None

    def stops_from(self, driver: str, lap: int) -> int:
        """This car's stops with an in-lap at or after ``lap``."""
        return sum(1 for x in self.pit_laps.get(driver, ()) if x >= lap)

    def field_median(self, lap: int) -> tuple[float | None, dict[str, float]]:
        """(median clean lap time of the field on ``lap`` if >= 5 clean laps, {compound: median} if >= 3)."""
        recs = self.index.by_lap.get(lap, ())
        key = (len(recs), self.lap_rev.get(lap, 0))
        hit = self._fmed.get(lap)
        if hit is None or hit[0] != key:
            clean = [x for x in recs if is_clean(x)]
            by_c: dict[str, list[float]] = defaultdict(list)
            for x in clean:
                if x.compound:
                    by_c[x.compound].append(x.lap_time)
            hit = (key, median(x.lap_time for x in clean) if len(clean) >= 5 else None,
                   {c: median(v) for c, v in by_c.items() if len(v) >= 3})
            self._fmed[lap] = hit
        return hit[1], hit[2]

    def in_race_losses(self, state: RaceState) -> dict[str, list[float]]:
        """Pit losses measured in this race so far, by condition. Treat as read-only.

        Keyed on laps, lap times (late ones too) and completed stops seen, so the
        result depends only on the state at ``t``, not on when it was first asked."""
        key = (self._n_laps_seen, self._n_timed, self._n_pit_out)
        if key != self._loss_cache_key:
            samples = measure_stops(self.index, state.pit_events)
            out: dict[str, list[float]] = {"green": [], "sc": [], "vsc": []}
            for s in samples:
                out[s.condition].append(s.loss)
            self._loss_cache, self._loss_cache_key = out, key
        return self._loss_cache
