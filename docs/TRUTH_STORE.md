# The truth store — the operator's answers, collated into one trusted source

Updated 2026-09-15.

Every accuracy number in this project is scored against what the operator (David) says happened in
his own videos. This document says where those answers live, how a review becomes one, and what the
store holds today. The running record of what was measured with it is `docs/ACCURACY_LEDGER.md`.

## Why it exists

Reviews used to arrive as separate files — `labels.csv`, `truth.json`, `missed_review.json`, an
xlsx per review — one set per CLIP. The same video cut into two clips was reviewed twice, scorers
read whichever file they had been written against, and three harness metrics were scored against
stale snapshots while a fuller review sat unread. The operator's rule, in his words: *"the info I
provide on shots should be saved as truths and built upon so I don't have to keep reviewing the
same info"*, and *"use the last one I built as the truth if there is a conflict between any
reviews."*

So there is now exactly one home for it.

## Where it lives

`docs/truth/<SOURCE VIDEO>.json` — keyed by the SOURCE VIDEO, not the clip. A review of any clip
covers every clip cut from that video. `tools/truth_store.py` owns the file; nothing else writes it.

What a store holds:

| key | meaning |
|---|---|
| `shots` | every shot the operator accounts for: `t_sec`, `type`, `hitter`, `side`, `volley`, `detected` (false = one we missed), `source`, `authority`, `seq`, `key` |
| `false_positives` | detections he marked NOT a shot |
| `rally_ends` | the ending shot of each point, with `reason` (net / out / not-returned / serve-fault) |
| `rally_truth` | his own rally windows: start, end, server, shot count, end reason |
| `serve_strikes` | serve CONTACT times (`truth.json`'s `start_t_sec` is NOT the strike — early by ~1.06 s indoors, ~0.32 s outdoors) |
| `dead_intervals`, `third_shot_drops`, `free_notes`, `totals` | smaller facts from the same reviews |
| `superseded_shots`, `superseded_false_positives` | demoted by a later review — kept, never deleted, so a reading can be restored |
| `provenance` | every import: when, from which source, and what it changed |

Read it with `known(clip_dir)` / `known_shots(clip_dir)` from `tools.truth_store`. Never parse the
JSON directly in a scorer, and never read a per-clip `labels.csv` or `truth.json`: those are
absorbed sources, and reading one is how a scorer ends up grading itself against stale answers.

## How a review is produced

1. `python -m tools.annotate_full data/<clip>` — the WHOLE video, annotated in order: a running
   clock, every detected shot numbered and labelled with the type we assigned, and a live per-rally
   tally against his own counts. Cut-up snippet reels were rejected by the operator: a shot cannot
   be judged without seeing where the ball came from and where it went.
2. `python -m tools.shot_review_sheet data/<clip>` — a workbook with one row per detected shot,
   numbered to match the video, prefilled with what we say. He corrects only what is wrong:
   `NOT_A_SHOT`, `RALLY_END` + `END_REASON`, `CORRECT_TYPE`, `CORRECT_VOLLEY`, `CORRECT_HITTER`,
   `CORRECT_SIDE`. Shots we missed entirely go in the blank rows at the bottom (time + type, no
   number). Rows he has already answered in an earlier review come pre-marked green so he skips them.
3. `python -m tools.truth_store --import-review data/<clip>` — reads every `shot_review*.xlsx` for
   that clip, oldest first so the newest wins, and merges into the store.
4. `python -m tools.truth_store --import-all` — absorbs every source under `data/` and then runs
   the folding pass that removes an older label a newer review already covers.

Rules the importer enforces, each of which failed at least once first:
- A blank row means "I looked and agreed" — but only in a sheet with operator marks in it. An
  untouched, freshly built sheet is refused, or our own detections would be filed as his truth.
- Rows are matched to stored shots ONE-TO-ONE, shortest pair first, within 1.0 s. Loose many-to-one
  matching silently double-counted shots twice in 2026-09.
- A row marked NOT_A_SHOT retracts any earlier shot at that time; the retracted entry keeps
  `not_a_shot` and must be excluded from "real shot" counts.
- An older review's false-positive claim cannot stand where the latest sheet says a shot is real.
- Nothing is deleted. A superseded reading moves to `superseded_*` with what changed and who
  changed it.

**Import into a COPY first when a review is large.** Load the doc, run the import, and check the
per-rally counts against his own totals before saving. That is how court B's import was verified.

## What the store holds today (2026-09-15)

| video | real shots | typed | that we MISSED | junk detections | rally ends | his rally windows | serve strikes | conflicts |
|---|---|---|---|---|---|---|---|---|
| PB 5 minute outdoor | 103 | 103 | 19 | 54 | 17 | – | 4 | 3 |
| PB 3 min indoor 1 court B | 82 | 82 | 26 | 18 | 10 | 10 | 10 | 0 |
| PB 3 min indoor 1 court C | 58 | 58 | 10 | 23 | 10 | 10 | – | 0 |
| PB 5 min indoor 1 court A | 97 | 97 | 27 | 38 | 19 | – | – | 0 |

Totals: 340 real shots, 82 of them shots we never detected, 133 junk detections, 56 rally ends.

- **Court A is HELD OUT.** It was reviewed end to end on 2026-09-14 and nothing has been tuned on
  it since. It is the only honest read of how the app does on a court it has never seen.
- **Court B** was reviewed shot by shot on 2026-09-15; every rally matches his own shot counts.
- **Court C** carries a `corrections.csv` as well — corrections he gave in conversation rather than
  in a sheet, which outrank the sheets.
- **The outdoor video has 3 conflicts** recorded: two sources disagree and the original was kept.
  `python -m tools.truth_store --report` prints them.

## Who reads it

`tools/regression.py` (the standing suite), `tools/serve_score.py`, `tools/rally_end_score.py`,
`tools/shot_precision_score.py`, `tools/shot_type_score.py`, `tools/score_shots.py`,
`tools/gemini_video_test.py`, and the review sheet itself (to grey out answered rows).

If a scorer's numbers look good, check it is reading the store through `known()` before believing
them.
