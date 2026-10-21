# PitSense research and engineering plan (v2)

Written 2026-10-21. This is the working plan for two goals that share one codebase:

1. **Race-day real:** a tool a strategy team would run in shadow mode on a Sunday and trust within a season.
2. **A paper:** physics-informed tyre-state estimation for real-time race strategy from public timing and
   telemetry, evaluated leak-free on held-out races and judged on decision value, not lap-time error.

It replaces the "Open" items of [ROADMAP.md](ROADMAP.md) with a sequenced programme. Each work package (WP) has
a goal, the files it touches, the acceptance test, and the agent it is sized for (`pit-architect` for hard
reasoning and measurement, `pit-engineer` for specified features, `pit-mechanic` for chores). Nothing in a WP is
done until its acceptance test passes on the benchmark with as-of training (`docs/evaluation.md`).

Rules that apply to every WP:

- No model sees the future. Every new feature goes through `base_row` in `bench/features.py` so
  `pitsense leakcheck` covers it. Every new simulator input goes through `strategy-leakcheck`.
- Every claim in the paper is a number produced by `pitsense results --readme` with its data version, cutoff and
  commit. No hand-typed numbers.
- A WP that does not move its metric is reported as such and kept off (as `rival_cover` was). Negative results go
  in the paper's ablation table.
- Work on a branch per WP, commit per WP, keep `reports/baseline/` as the frozen comparison.

---

## 0. Housekeeping first (pit-mechanic, half a day)

The working tree has 28 modified and 8 new files uncommitted from the October work, plus a stray
`tests/_rev_tmp_test.py`.

- [ ] Delete `tests/_rev_tmp_test.py`; run `pytest -q` and `ruff check`; commit as "October pit-wall work".
- [ ] Regenerate `reports/baseline/` with `pitsense bench run` and `pitsense results --readme` so the README
      results block is filled and tagged with the commit. This is the **paper baseline**; tag it `paper-baseline-v0`.
- [ ] Record the data version and the 15 test races in `reports/baseline/README.md`.

Acceptance: clean `git status`, README results block non-empty, tag exists.

---

## 1. Telemetry physics features (pit-architect, 3-4 days) — the paper's first experiment

**Gap.** Lap time in the simulator (`strategy/sim.py: lap_times_for`) is `pace + (deg - fuel) * age + cliff`.
The degradation slope comes from a GBM on timing-feed features only. The CarData stream (speed, throttle,
brake, RPM, gear, DRS at ~4 Hz) and Position stream are already downloaded by `pitsense fetch --telemetry`
and used only by the map, coach and mechanics. Tyre age is a poor proxy for tyre state: a lap behind a slow car
costs almost nothing, a lap in a 30-second fight costs a lot.

**Build.** A per-lap, per-car extractor `src/pitsense/bench/telemetry.py`:

| Feature | Definition | Why |
|---|---|---|
| `ft_frac` | share of samples at throttle >= 95% | fuel burn and rear-tyre thermal load |
| `brake_energy` | sum over braking samples of `v^2` drop, normalised by the circuit's clean-lap median | front-tyre load and wear |
| `lat_energy` | sum of `v^2 * kappa` over the lap, `kappa` = curvature from the track outline (`trackmap.py`) at the car's position | cornering load; the main tyre-wear driver |
| `slide_proxy` | `lat_energy` on laps where the car was within 1.0 s of a car ahead, split out | dirty-air laps wear more |
| `wear_cum` | running integral of `brake_energy + lat_energy` since the last stop, per compound | the latent "used energy" of the set |
| `v_max`, `v_min_corner` | top speed and minimum corner speed | grip loss shows in corner speed before lap time |
| `fuel_proxy` | `ft_frac * lap_distance` cumulated over the race | per-car fuel correction instead of one circuit constant |

Rows without telemetry (pre-2025 archive, no F1 TV auth) get NaN; the GBM handles missing values natively.

**Experiment A (tyre tasks).** Three arms on `y_deg_5`, `y_cliff_1/3/5`, `y_pace_5`, `fresh_tyre_pace`:

1. `gbm_timing` — today's model, unchanged (the baseline).
2. `physics_only` — a reduced-order model: `lap_time = p0 + a * wear_cum + b * fuel_proxy + cliff(wear_cum > w_c)`,
   parameters per compound and circuit fitted by least squares on earlier races only.
3. `gbm_physics` — the GBM with the telemetry features and the `physics_only` residual as extra inputs.

Acceptance: `gbm_physics` beats `gbm_timing` on `y_cliff_3` AUC and `y_deg_5` RMSE on the 15 held-out 2026 races,
with the gain reported per race. If it does not, the ablation is still reported.

Files: `bench/telemetry.py` (new), `bench/features.py` (`base_row` gets the columns), `bench/tyre.py` (two new
models), `bench/hazard.py` (if the physics residual needs a hook), `docs/engineers/tyre.md`, tests.

---

## 2. Cliff as a wear-driven hazard (pit-architect, 2-3 days)

**Gap.** The cliff is drawn empirically from an age distribution (`sim.py: cliff_ages`). This is where the
"calls 2-7 laps early" failure lives: an age-based cliff fires on schedule whether or not the tyre is actually
being hurt.

**Build.** A discrete-time survival model: hazard of the cliff on lap `L+1` given `wear_cum`, compound, circuit
and track temperature, fitted as-of with the existing `FeatureGBM` machinery or a logistic hazard. The simulator
samples cliff laps from this hazard per draw instead of from age.

**Experiment B (decision value).** Rerun `bench/decision.py` and `bench/callscore.py` on 2026 with the old and new
cliff. Report: mean call lead/lag versus the team's real stop (the "early" number), positions gained in the
counterfactual simulator, and Brier of calibrated confidence.

Acceptance: median call timing error on 2026 moves toward zero without decision value falling.

---

## 3. Rivals that react (pit-architect, 4-5 days)

**Gap.** Opponent stop laps are shared across all candidate plans (`sim.py: sample_schedules`), so an undercut is
never covered and an overcut is never answered. The earlier `rival_cover` attempt was a fixed cover rate and did
not help.

**Build.** Per-plan opponent trajectories: for each of our candidate plans, the car ahead and the car behind
re-sample their stop lap from a response model `P(stop next lap | our stop this lap, gap, their tyre age,
their team's historical cover rate)`, fitted as-of from `bench/rivals.py` rows. Keep the shared draws for the
rest of the field to control cost; measure simulator runtime (must stay under the live-cadence budget).

Acceptance: decision value on 2026 up versus WP2; p95 call latency under 1 s on a live recording.

---

## 4. Per-team pit-crew and fuel priors (pit-engineer, 1-2 days)

- Stationary-time prior per team from history (`pitloss.py`, `bench/pitstop.py`), as-of.
- Fuel correction per car from `fuel_proxy` (WP1) instead of one constant per circuit.

Acceptance: `position_after_stop` task log-loss improves; no leak-check regression.

---

## 5. Learning from the operator (pit-engineer, 2 days)

Rejections from `/api/ack` are graded but never used. Add a per-circuit, per-action bias on the call threshold
in the head, fitted from accept/reject history, shrunk toward zero with few samples. Report on shadow-score how
many later calls it would have changed.

---

## 6. Race-day operations (pit-engineer, 3 days)

- Second live source: FastF1's own live client as a hot standby behind `pitwall/sources.py`, switched by the
  health watchdog on 30 s of silence.
- Latency budget: log lap-end to call-emitted time; `/api/health` reports p50/p95; a test replays a live
  recording and asserts p95 < 1 s.
- Multi-screen: `?view=strategist|engineer|mechanic` on the dashboard, hiding panels per role.

---

## 7. Voice: fine-tuning a ~30B model (pit-architect to design, pit-engineer to run)

**Why.** The scratch 26M model and SmolLM2-360M write radio lines that pass the guard most of the time, but the
free-form answers ("why not the hard?", "what does the rival's radio mean for us?") are where a strategist
would actually use a language model, and the small models fall back to the template too often there.

**Model choice.** The right 30B-class target is a mixture-of-experts with ~3B active parameters (Qwen3-30B-A3B
class): it serves at 7B-dense speed, which keeps the 170 ms radio-call budget reachable, and it fine-tunes with
QLoRA. Record the model card's data cutoff in `docs/engineers/voice.md` before any result is claimed, as the
SmolLM2 note already requires; a model that may have read 2026 race reports cannot be scored on 2026 races.

**Hardware.** Two machines: the RTX 5070 (12 GB) for development and the pit wall itself, and a college node
with **8x H200 (141 GB each, 1.1 TB total)** for training and race-day serving of the big model.

- *Training.* With 8 H200s a 30B model does not need QLoRA. Run **full-parameter SFT** in bf16 with FSDP or
  DeepSpeed ZeRO-3 (30B weights + Adam states ~360 GB, fits with room for a 4k context and large batches).
  Budget: a corpus of ~500k short samples, 2 epochs, under 2 hours. Also run a LoRA arm on the same data
  so the paper can report full fine-tune vs LoRA vs base at equal data.
- *Scaling arm.* The same recipe on a 7-8B dense model (fits the 5070 at 4-bit for serving) and on a 70B-class
  dense model (one epoch, ~6 hours on 8 H200s). Three sizes on one corpus gives a scaling curve for the paper
  at essentially no extra engineering.
- *Serving.* vLLM on one H200 serves the 30B MoE at well under the 170 ms radio budget with batching for
  briefs and free-form questions. `voice/llm.py` gets an HTTP backend (OpenAI-compatible endpoint) so the pit
  wall on the 5070 calls the cluster, and falls back to the local 7-8B model, then the template, if the link
  drops. Measure round-trip latency over the college network and put it in `/api/health`.
- *Guard rails for the cluster.* Store nothing but weights and the corpus there; the corpus is public timing
  data. Pin the driver, CUDA and vLLM versions in `docs/engineers/voice.md` so a result can be reproduced.

**Data.** `voice/synth.py` builds the corpus from real benchmark rows with sampled calls. For the 30B:

1. Replace the sampled calls with the real head's calls now that the strategy engineer exists
   (`pitwall/engineers/strategy/engineer.py`), so "why this call" has a true reason chain.
2. Add free-form Q&A pairs generated from the simulator: question templates over `whatif.py` results, answers
   written by the template grammar then paraphrased, each checked by `guard.py` before it enters the corpus.
3. Add the transcribed real radio (`radio.py`, Whisper) as *context*, never as target: the model learns to read
   "box box, plan B" and state what it means for us.
4. Splits stay by race: train 2025, validate last two 2025, test all 2026.

**Evaluation.** Same harness as today (`voice/evaluate.py`): fact errors before guard, fallback rate, action and
length. Add for free-form: a strategist rubric scored by a frozen judge model on 200 held-out questions, with
human agreement measured on 50. Latency p50/p95 per backend.

Acceptance: fallback rate on free-form halves versus SmolLM2 with zero fact errors after guard.

Files: `voice/llm.py` (backend by model id, QLoRA config), `voice/synth.py` (real calls, Q&A), `voice/cli.py`
(`--model`), `docs/engineers/voice.md`.

---

## 8. The paper

**Working title.** Physics-informed tyre state estimation for real-time race strategy from public timing and
telemetry.

**Claims that hold up** (each tied to a WP and a results table):

1. An open, leak-free benchmark of race-strategy decisions on public data: as-of training, corrupted-future
   checks, 43k decision points, 15 held-out races (WP0).
2. Telemetry-derived tyre state improves degradation and cliff prediction over tyre age (WP1).
3. A wear-driven cliff hazard reduces early box calls and raises decision value (WP2).
4. The physics-to-decision pipeline runs at live cadence (WP3, WP6).
5. A guarded language model turns calls into radio with zero fact errors (WP7, secondary).

**Sections.** Introduction; related work (tyre models in motorsport engineering, strategy simulation,
decision-aware evaluation); data and benchmark; the hybrid tyre model; the simulator and head; experiments A, B
and the live test; ablations (timing-only, physics-only, hybrid; age-cliff vs wear-cliff; shared vs reacting
rivals); limitations (public feed only, no fuel or energy telemetry, 2026 rules); reproducibility (commit, data
version, one command per table).

**Order of work for the paper:** WP0, WP1, WP2 give a submittable result. WP3 and WP6 make it a systems paper.
WP7 is a separate short paper or an appendix.

---

## Timeline and agents

| Week | WP | Agent | Deliverable |
|---|---|---|---|
| 1 | 0, 1 | mechanic, architect | baseline tag; telemetry features; experiment A table |
| 2 | 2 | architect | wear hazard; experiment B table |
| 3 | 3 | architect | reacting rivals; latency numbers |
| 4 | 4, 5 | engineer | pit-crew and fuel priors; operator learning |
| 5 | 6 | engineer | failover, latency budget, role views |
| 6 | 7 | architect, engineer | corpus v2; 8B, 30B and 70B runs on the H200 node; vLLM endpoint |
| 7-8 | 8 | — | paper draft from generated tables |

Token discipline: one agent per WP, with the WP section above as its brief, and the acceptance test as its
definition of done. Do not spawn for anything below a WP.
