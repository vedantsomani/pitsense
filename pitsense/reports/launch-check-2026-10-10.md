# Launch verification - 10 October 2026

The reported model-loading error is fixed. The saved weights were valid; launching from `aiml` selected `aiml/data`, while the trained bundles and track caches live in `aiml/pitsense/data`.

## Focused fixes

- Editable installs consistently use the checkout's data folder. `--data` and `PITSENSE_DATA` still override it. Live recordings and the disk check use the same resolver.
- Archive dashboard replays request missing timing and telemetry streams, including positions, even when timing is already cached.
- When no earlier track outline exists, map-building retries follow completed race laps instead of elapsed wall time or a fixed retry limit. Replay checkpoints restore that retry state.
- Finished recordings can select models using the scheduled session start in their initial SessionInfo messages.
- Corrupt/incompatible model files produce diagnostics. A missing compatible bundle shows the directory, required cutoff and training command. Training-after-race protections remain enforced.
- The voice guard rejects points or gap values presented as penalty durations, and gap answers about a different neighbour from the one requested.

## Evidence

| Check | Result |
| --- | --- |
| Full test suite, Python 3.14 with optional voice dependencies | 522 passed; no skips or failures |
| Ruff and git diff whitespace checks | Passed |
| Installed dependency consistency (`python -m pip check`) | Passed |
| Hungary 2026 corruption-of-future check, three cuts, truncate + scramble | Six passes; zero mismatches |
| Complete Hungary 2026 replay at 20x, Ferrari | Completed; 37 call changes, including 6 BOX calls; 32 radio messages |
| Hungary 2026 browser and API | Model loaded, stored map, 22 position entries, no health alarms or event/engineer/snapshot errors |
| Hungary 2025 browser and API | Model trained before the race loaded; provisional map; 20 position entries; no health alarms or runtime errors |
| Replay controls | Pause, backward seek to lap 6, and resume worked; older-race map restored |
| Typed question in the final build | Gap-ahead answer returned successfully through the LLM and guard |
| Tyre analysis and speed trace APIs | HTTP 200, successful JSON |
| Spoken radio endpoint | HTTP 200, valid RIFF WAV audio |
| Browser console | No captured errors or warnings |

## Run locally

From `C:\Users\vedan\Downloads\aiml`, stop any existing server and run:

```powershell
pitsense pitwall --year 2026 --race hungary --speed 20 --team ferrari
```

On this machine `pitsense` currently resolves to the global Python 3.14 installation, which contains the voice, transcription and speech dependencies used in verification. The outer `.venv` is empty; activating it does not make those dependencies local. Use `Get-Command pitsense` to confirm the executable. No environment or dependency installations were changed during this repair.

## Remaining launch limits

- This validates archive replays and the automated recorded/live-feed tests. An actual authenticated live session, microphone capture and reconnects over a real network still need a session-day check.
- A map cannot be reconstructed when the source publishes no usable positions. With no earlier outline, a provisional map appears only after enough clean laps.
- Replays before the training data of every available bundle require an earlier-cutoff bundle; later-trained models are intentionally refused.
- The test run retains four non-failing warnings: one tensor-to-scalar warning in a training test and three pandas concatenation deprecation warnings in the benchmark loader.
- The voice guard catches the observed mistakes; it is not a guarantee that all generated wording is correct. The existing benchmark's strategy accuracy limitations still apply.
- No public deployment was performed.
