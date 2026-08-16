# Session Handoff — Pickleball-Analyzer-v2 (updated 2026-08-15)

## 2026-08-15 — READ FIRST: NEXT ACTION IS SHOT-TO-PLAYER ATTRIBUTION

### The next job, and why it is worth doing

Per-player numbers on the indoor clip are wrong because individual shots land on the
wrong player. Identity is NOT the problem — the operator confirmed by render that the
`user` role is the right person. Roles are right; shots are attributed within them wrongly.

**It is provable without any new labelling.** The serving team cannot hit the return of
its own serve, and `data/pb_3_min_indoor_1_court_b` contains exactly that:

```
rally 3  1:00.7  server opp_a   2nd shot opp_a      <- impossible
```

Operator truth for that clip: they served 2 and returned 2. Detected: 2 rallies served by
`user` (correct) but 0 returns (partner credited with 2).

Build the rule as a check first — "same team serves and returns" is a contradiction that
can be counted on BOTH clips — then fix what it exposes. That gives a test before a fix,
which is what this codebase keeps needing.

Start from `tools/verify_identity.py` (render-based, already proves roles) and the rally
table produced by reading `classified.json` + `rallies.json` + `track_roles.json`.

### ⚠ Do NOT re-litigate these — measured and understood today

**Volleys over-count because bounces are MISSED, not because volleys are over-detected.**
`is_volley` means "no bounce since the previous shot", so every missed bounce becomes a
false volley. Indoor: 36 volleys detected vs the operator's 20, and 49 bounces detected
where ~68 are implied — **16 excess volleys against 19 missing bounces, the same error
from both ends**. This is the accepted bounce-detection limitation, worse indoors because
ball contrast is ~3.5x lower. Not an aggregation or identity problem.

**Serves: the count metric disagrees with the rally table.** `n_serves` counts shot TYPES;
the rally table counts servers. Indoor rally 0 is attributed to `user` but its shot was
typed `drive`, so the metric said 1 serve where the operator counted 2. Rally 0 is also a
one-shot "rally" at 0:00.0 — almost certainly a warm-up ball counted as a point (the known
between-point problem).

### Fixed today (do not re-investigate)

**The cumulative rating was LOWER than both its inputs** (3.23 vs 4.14 and 3.78). Two
faults in Stage 7.9, both now fixed:
1. `poses.parquet` was never unioned. Stage 8 silently falls back from the pose front foot
   to the bbox foot (which is the BACK foot for a net-facing player) and under-counts
   kitchen time; it also dropped ready-position and knee-bend entirely.
2. Pose pixels were projected through ONE homography. A union spans venues and carries one
   `court.json`, so every member after the first landed off-court. Stage 7.9 now
   pre-projects per member with that member's own court into
   `front_foot_court_x_ft/_y_ft`, and `compute_metrics` uses those when present.
   **Any future per-video pixel data added to the union needs the same treatment.**

Result: rating 3.82, between its inputs; every strategy driver between its inputs.

⚠ **The aggregate acceptance test cannot catch this class of bug** — it partitions ONE
clip, so every member shares a court and has poses. It proves losslessness, never
venue-independence. A multi-venue fixture is worth building.

**Court positioning was missing from EVERY app-produced report.** The per-video pipeline
had no render step at all (Stage 11 also renders a slow annotated video, so the whole
stage was skipped), and `build_report` silently omits images it cannot find. Now runs
`render --heatmaps-only` before the report.

**Three separate silent-degradation bugs in one day** (missing poses, missing heatmaps,
missing timeline duration). The report drops sections rather than saying "not available".
Worth a pass.

### Other things now working

- Cumulative videos end to end: choose the report when picking the video, analysis folds
  itself in server-side at `phase=done` (not browser-dependent), management view in the
  top bar, reports named per player.
- Drive auto-sync now carries the ball model too, verified against DriveFS silent
  truncation (a 45,508,290-byte model landed as 45,088,768 with a successful return).
- The vision hand-off tells the operator to WAIT for Drive to finish uploading before
  running Colab — starting early fails with "no *_vision_input.zip found on Drive root".

---

# Session Handoff — Pickleball-Analyzer-v2 (updated 2026-08-13)

## 2026-08-13 — READ FIRST: the unseen-venue failure is CONTRAST, and it is not fundamental

### Result

| venue | Run 3 | **Run 4** (contrast augmentation) |
|---|---|---|
| `home` | 0.916 / fp 0.053 | 0.887 / fp 0.053 |
| `indoor_seen` | 0.978 | 0.977 |
| `court2` | 0.713 | 0.640 |
| **`indoor_unseen`** | **0.087** | **0.589** |

`ball_model_v4_run4.pt` (epoch 5) is in MyDrive. Run 3's is `ball_model_v4_run2.pt`.
**Neither has been brought into the local pipeline yet** — see "next actions".

### How it was diagnosed (the method, not just the answer)

Ball-vs-background yellow contrast measured at every labelled ball, against Run 3 recall:

```
indoor_C1 45.0 -> 0.978   pb_2min        25.5 -> 0.916
indoor_B1 36.5 -> 0.978   court2         20.5 -> 0.713
pb_3min   30.0 -> 0.916   pb_3min_indoor  7.5 -> 0.087
```

Monotonic — and causal, not merely correlated: applying CLAHE at inference lifted
pb_3min_indoor 0.050 -> 0.360 on existing weights with no retraining. But the same CLAHE
COST pb_2min 0.870 -> 0.570, so fixed preprocessing trades one venue for another. That is
why the fix is trained invariance, not a preprocessing step.

⚠ **A wrong turn worth not repeating:** early-epoch fp spikes on that clip (0.833, 0.786)
looked like adjacent-court confusion. They were transient — at the saved checkpoint fp is
0.012 with recall 0.087. Contrast fits the evidence; adjacent courts do not.

### ⚠ Run 4 DIVERGED — read before trusting anything past epoch 5

Epoch 8 spiked (home fp 0.496), epoch 9 collapsed, epoch 11 saturated (fp 1.000 on every
venue), and it never recovered: loss plateaued ~1.29 and never beat epoch 5's 1.281.
Nineteen epochs wasted. Gradient clipping was already on at norm 2.0, so clipping alone
was not the answer.

The notebook is now **Run 5** with four fixes, each dry-run against Run 4's real numbers:

| fix | effect on Run 4's numbers |
|---|---|
| LR 1e-4 -> 5e-5 + 300-step warmup | the collapse had no warmup at all |
| clip 2.0 -> 1.0 | 2.0 did not hold |
| collapse rollback (reload best, halve LR) | would have fired at epochs 9 and 11 exactly |
| `INDOOR_FP_CAP` 0.20 + keep TOP_K=3 | picks epoch 4 over epoch 5, and surfaces epoch 7 |

**The fp caps change which checkpoint ships, deliberately.** Run 4 shipped epoch 5
(unseen 0.589) with `indoor_seen` fp **0.259** — recall bought with false positives, on
the venues that already worked. Only home fp was capped, so nothing stopped it. Under the
new rules the picks are:

```
1. epoch 4  mean_rec 0.833  home fp 0.044  indoor fp 0.172  unseen 0.515
2. epoch 7  mean_rec 0.828  home fp 0.044  indoor fp 0.103  unseen 0.483
3. epoch 3  mean_rec 0.808  home fp 0.035  indoor fp 0.190  unseen 0.387
```

i.e. it trades 0.07 of unseen recall for materially better precision. **fp becomes false
shots and bounces downstream** — the error the acceptance scorecard is already losing to
(shots 121 vs 98). Epoch 7 is arguably the best balance of the three and Run 4 discarded
it outright.

### Next actions

1. **Operator:** upload the Run 5 notebook, Run All. Bundle unchanged, no re-upload.
2. **Operator:** download the chosen checkpoint to `data/models/` **under its own name** —
   do NOT overwrite `ball_model_v4.pt`; every local stage loads it and swapping it
   silently changes Stage 4 output and every downstream count.
3. **Then:** re-run `tools/score_acceptance.py` and see what the new model does to the
   counts. Higher recall AND higher fp could make them worse. Measure, do not assume.

### In flight: cumulative videos + venue supportability

`stages/aggregate/contract.md` — cross-video aggregation, operator decisions D1–D5
settled, NOT implemented. Key design call: aggregation sits BELOW Stage 8 (union the
per-item streams, run Stage 8 once) because a median of medians is not the median and
averaging the confidence envelope gets small-sample handling exactly backwards.

**Build order (operator chose): venue supportability check FIRST**, because D4 (do not
merge a venue whose measurement quality is materially worse) gates on a number we cannot
currently compute for an unlabelled venue. It predicts from unlabelled signals, is
calibrated against the venues whose true recall we know, needs **no training and no
Colab**, and doubles as "is this clip worth labelling before I spend hours on it".
Honest limit: 8 clips but only **4 distinct venues**, so it can only be a coarse
supported / marginal / not-supported call.

---

## 2026-08-11 — Labels were corrupted; Run 3 fixed them (indoor 0.035 -> 0.978)

### Start here — operator step

`data/pb_v4_upload.zip` (3.28 GB, 8 clips, 23,171 jpegs) is rebuilt and ready.

1. Upload it to Drive, replacing the previous `pb_v4_upload.zip`.
2. Re-run `stages/finetune_ball_model/finetune_v4.ipynb` — **config unchanged from
   Run 3**, no notebook edit needed. `ball_model_v4_base.pt` is already in MyDrive.

Nothing else is pending on the operator side. The labelling does NOT need redoing.

### Why Run 3 failed, and it was not the model

Run 3 read `indoor_seen_rec 0.035` and looked like the model could not learn indoor.
It could not — it was being taught the wrong thing. **`tools/label_ball.py` displayed
one frame and recorded the click against a different one.** It seeked per frame with
`cap.set(CAP_PROP_POS_FRAMES, idx)`, which on long-GOP H.264 returns a frame NEAR idx.
Fixed in a98158b with an exact sequential reader; **do not reintroduce per-frame
seeking** (there is a warning in the class docstring).

Damage, and the fix (labels recovered by re-indexing, not re-clicking):

| clip | labels wrong | drift range | after fix |
|---|---|---|---|
| `indoor_B1_3min` | 92% | −1 … **+12** | 1.3px median, 90% ≤6px |
| `indoor_C1_3min` | 85% | −5 … 0 | 2.1px, 89% |
| `pb_3min_indoor` | **97%** | −2 … 0 | see caveat below |
| pb_2min / pb_3min / pb_4min / pb_5min / pb_3min_court2 | clean | — | 1.4–2.3px, 0% bad |

Drift is **not** a property of "raw 4K from Dropbox" — four of the clean clips were
labelled from exactly such files. It is specific to how individual files encode, so
**assume nothing; measure per file.**

### ⚠ The unseen-venue number is not trustworthy and must be re-measured

`pb_3min_indoor` is the **held-out val clip** — the one Run 3 scored **0.03** on, which
is the whole basis for "cannot generalise to a new indoor venue". 976 of its 1,002
labels were off by 1–2 frames ≈ 8–15 source px, about the hit tolerance. **That figure
was biased low and should not be quoted until Run 3 is repeated.** The venue is
genuinely hard (pale ball against a pale wall), so this does not explain 0.03 by
itself — but it was never a clean measurement.

### Method notes worth keeping

- **The model-based audit cannot judge a clip the model is blind on.** `pb_3min_indoor`
  read 241px median error but **0% confidently-wrong** — the detector is never confident
  anywhere on it. A blind model and a corrupt label look identical. Only the mechanical
  drift measurement, which never looks for the ball, could answer it.
- **Interpolating a sparse drift curve is not good enough.** The first recovery sampled
  every 400 frames and interpolated; the visual check caught it pushing an
  already-correct label off the ball. B1's curve is non-monotonic. Measuring every label
  moved 41%/47% of them AGAIN and found drift the sparse sampling never saw. Seeking is
  deterministic, so measure — never interpolate.
- **`verify_remap` samples the fastest-ball frames**, because a 1–2 frame shift on a slow
  ball is invisible and a random sheet is unreadable.
- **Read the crops at full resolution.** On `pb_3min_indoor` the downscaled sheet looked
  like it CONTRADICTED the remap; at full resolution the same tiles confirm it.
- **Negative result, do not rebuild:** scoring "is there a yellow ball at the label"
  across candidate offsets fails its own control (pb_2min, known correct, peaks at
  offset 0 with 11% against a 7.7% chance floor). No discriminating power.

### Tools this left behind (all under `tools/`)

| tool | use |
|---|---|
| `measure_seek_drift.py` | per-label drift; run on ANY clip labelled before a98158b |
| `remap_labels.py` | idempotent — always rebuilds from `ball_labels.PRE_REMAP.json` |
| `audit_labels.py` | the label acceptance test: median px / ≤6px / confidently-wrong |
| `verify_remap.py` | before/after contact sheet, stratified across drift values |

`audit_labels.py` is the standing check: **any new labelled clip should read near
pb_2min's 1.5px / 97% before it is trained on.**

### Not yet checked

`indoor_b`, `indoor_c`, `outdoor`, `test_clip` were labelled with the buggy tool but
have no `frames_720` cache, so they were not measurable. `indoor_b`/`indoor_c` are
superseded 1080p/30 and not in the bundle; `test_clip` backs the Stage 2.5 smoke test.

---

## 2026-08-04 — NEXT ACTION IS BALL LABELLING (indoor, 4K/60) [DONE — see above]

### Start here tomorrow

Five NEW indoor clips at the required capture standard (**4K / 60 fps**) are
materialised and ready. **Label two of them, then the retrain can run.**

```
python tools/label_ball.py --video data/indoor_B1_3min/video.mp4 --out data/indoor_B1_3min/ball_labels.json
python tools/label_ball.py --video data/indoor_C1_3min/video.mp4 --out data/indoor_C1_3min/ball_labels.json
```

Left-click = ball here (advances) · Spacebar/right-click = not visible · Backspace =
undo · Esc = save+quit. Auto-saves every 25 labels and RESUMES if relaunched with the
same `--out`, so it can be done in sittings.

- **Keep the default `--sample-every 3`.** `densify_labels` interpolates the ball
  between consecutive visible labels up to 4 frames apart, so labelling every 3rd
  frame yields **~2.7x that many visible training samples**. Labelling sparser than
  every 4th frame loses the multiplier entirely.
- **Target ~1000 labels per clip** (≈1,860 samples, ≈1,350 visible each). Two clips
  ≈ **2,700 visible indoor samples vs the 1,354 that exist today — roughly 3x.**
- **Spread across the clip**, don't label one contiguous block: varied ball positions,
  speeds and occlusions matter more than density.
- **Do NOT label `indoor_A_5min`** — it is the held-out unseen test clip.

Court calibration is NOT needed for ball labelling (only the video). It is needed
later to run the full pipeline on these clips.

### Why this, and why it is measurable

Indoor ball recall is **0.61 against a 0.80 bar** and the July analysis concluded it is
a DATA problem, not a technique problem. This is the one workstream where measurement is
objective — per-venue recall against labelled frames, a defined threshold, a held-out
clip — unlike shot type, where per-type samples are n=1-3 and two principled rule fixes
produced no measurable movement (34% -> 38%, all of it from adding `return`).

### Clip inventory (materialised into data/ as hardlinks, no disk cost)

| folder | source | length | use |
|---|---|---|---|
| `indoor_A_5min` | PB 5 min indoor 1 court A | 5.1 min | **HELD OUT — do not label** |
| `indoor_B1_3min` | PB 3 min indoor 1 court B | 3.1 min | **label this** |
| `indoor_C1_3min` | PB 3 min indoor 1 court C | 3.0 min | **label this** |
| `indoor_B2_3min` | PB 3 min indoor 2 Court B | 3.3 min | reserve |
| `indoor_C2_3min` | PB 3 min indoor 2 court C | 3.1 min | reserve |

Operator note: these are the **same indoor court**, different angles and some different
players. So the held-out clip tests ANGLE/PLAYER generalisation, **not** new-venue
generalisation. Fixing indoor recall here is still worth it (it is where the operator
plays); cross-venue remains a later question needing a different facility.

⚠ **`indoor_b` and `indoor_c` (lowercase) are 30 fps / 1080p — SUPERSEDED, do not
train on them.** They were captured before the 4K/60 standard. Their 3,279 visible
labels are not usable: the v4 breakthrough came from the capture change to 4K/60, and
at 1080p the ball is ~half the pixels (cf. the "512x288 trap").

### After labelling (my side, then one Colab step)

1. `python -m stages.finetune_ball_model.prepare_v4 data/<clip> --clip <clip>` — 720p
   frame cache + manifest.
2. Add the clips to `CLIPS` in `tools/build_v4_train_bundle.py`, rebuild the bundle.
3. Colab: warm-start from the 0.90 baseline, all venues, **`indoor_A_5min` held out
   entirely**. **GATE: home (pb_2min) must NOT regress** — Run-2 showed multi-venue
   training costs same-court precision (home fp 0.018 -> 0.10-0.24 for >=0.90 recall);
   the fp-capped selection exists for this.
4. `reality_check_v4.ipynb` — per-venue recall, target >= 0.80 each.

### State of the pipeline (unchanged by the above)

`tools/score_acceptance.py` is the one-command scorecard. Current: **shots 121 vs 98,
mean error 38%** — high because between-point balls are counted, which is an ACCEPTED,
documented limitation (KNOWN_ISSUES "Stage 7 - RALLY END is undetectable"). Serves
**12/14**. Bounces 77 vs 81, the best recorded. Do not chase the counts without reading
that entry first — six rally-end routes are recorded as measured-and-failed.

---

# Session Handoff — Pickleball-Analyzer-v2 (updated 2026-08-01)

## 2026-08-01 — READ FIRST: identity RENDER-VALIDATED; the foundation gap is the FAR SIDE

### The 07-27 "next = user/partner identity" theory was tested and is WRONG

Built `tools/verify_identity.py` (cheap test before build, per the operator method) —
automatic consistency report + per-role track timelines + contact sheets of sampled
frames with every player boxed and labelled `role / track_id / confidence / basis`.
Sampled at the **detected serve frames** (where operator truth already exists) plus a
uniform sweep. Full detail in `docs/ACCURACY_LEDGER.md` ("IDENTITY VALIDATED BY RENDER").

1. **user/partner is NOT flipped.** Despite Stage 2.5's `seed is ambiguous (dx=0.2ft)`
   warning, the render shows the woman serving at **1:16 labelled `partner`, matching
   operator truth**; the man is consistently `user`. Duplicate-role frames (provably
   impossible) are rare: user 8, partner 53 of 18,862.
2. **ROOT CAUSE (operator-confirmed): Stage 2.5 discarded far-side SERVERS by
   construction.** THE RULE — **players are NOT confined to the 20×44 court.** They serve
   from BEHIND the baseline and chase wide. Operator's play envelope: **5 ft beyond each
   sideline, 15 ft beyond each baseline** → `court_x ∈ [-5, 25]`, `court_y ∈ [-15, 59]`.
   The old noise filter cut at `med_y ≤ 44` AND floored `in_court_frac` against the strict
   rectangle — so a server behind the baseline failed both. At EVERY operator-identified
   opponent serve (0:47, 3:39, 4:05, 4:43) **both opponents were detected at court_y 45–54
   and both were classified `noise`.** Adjacent-court people separate at 59–115 ft.
   **v0.2.0:** noise judged against the play envelope; `in_court_frac` floor → `in_env_frac`.

### Two wrong turns on the way — DO NOT REPEAT (detail in ACCURACY_LEDGER)
1. **A too-clever rule** (a `(44,52]` "drift zone" gated on reaching the far kitchen +
   a depth-span cap) passed smoke 6/6 and raised `opp_a` presence 59%→87%, but recovered
   only ONE of the two opponents at 0:47/3:39/4:05 and NEITHER at 4:43.
2. **Misreading which court is ours, then reverting a correct fix.** Our court's far half
   is a thin foreshortened sliver near the top of frame (image x 1801→2739, y 1217→1370)
   while a NEIGHBOURING court dominates the view; boxes judged "past the fence" were on
   our court. **`tools/verify_identity.py` now projects the court outline onto every
   frame** — never judge "is that player on our court" by eye again.

### Operator truth CLARIFIED (supersedes the older serve list)
**0:47 opponent serves; 0:48 is the operator RETURNING** (so the man IS the user, near-side
labels confirmed). **3:39, 4:05, 4:43 are all opponent serves.** The earlier "0:48(partner)"
note meant the return, not a serve.

### OPEN
- Residual **over-count** (shots 99→118, serves 13→15 on the first far-side run): an
  adjacent-court figure reads `dist_from_net` 21.8–33.8 ft, satisfying the serve rule's
  "behind the baseline (≥21 ft)", so contamination can **steal serve status**. The generous
  envelope admits more far tracks (83→131 kept), so opponent-role contamination is the
  live risk to measure — NOT a reason to revert.
- **Near-side identity confidence** (latent, render-checked as not flipped): 72% of `user`
  frames rest on two conf-0.53–0.56 assignments (tid 1452 @1:28–4:19, tid 4127 @4:23–5:14).

### Method lesson (do not lose)
Both wrong turns came from judging geometry by eye, or from a coordinate that was itself
the thing in question. **Project the court, then look. Render before building.**

---

## 2026-07-27 — unified point-boundary detector SHIPPED; NEXT = user/partner IDENTITY foundation

### What shipped this session (commit 8424963)
A **unified point-boundary detector** (`detect_shots.structure_points`) replacing isolated
serve heuristics, built the operator's way: **combine weak cues + anchor on the known COUNTS
+ solve the shared dependency (point boundaries) ONCE.** A rally is SERVE → … → POINT-END:
- **POINT-END** = opponent does NOT return within 2.5s + dead time follows (≥3s).
- **SERVE** = struck from BEHIND the baseline (`hitter dist_from_net ≥ 21 ft`; baseline is
  22 ft from the net — an IN-FRONT shot is never a serve, operator rule) + opens a point (≥3s gap).
- **MUTUAL CONSTRAINT** = one serve per point (accept a serve only if a point ended since the
  last accepted serve, OR ≥10s elapsed) → drops between-point balls/returns that look serve-like.
- `segment_rallies`: rally starts at a serve; the FIRST burst is kept even if its serve was
  missed (a missed serve must not delete real play); between-point balls dropped.

**Result vs operator truth on `pb_5_minute_outdoor-2`: 14 rallies (13), 99 shots (98), 13
serves — 11/14 real, 2 false.** 20/20 tests pass. Report auto-regenerates from the pipeline
(build_report needed no changes; report.html/metrics.json are NOT git-tracked — regenerated).

### The diagnosis that sets the NEXT target (cheap-test-first, do NOT re-derive)
The remaining serve/attribution errors are NOT ball-association errors — **the ball is linked
to the correct hitter track.** Two distinct causes:
1. **Serve DETECTION misses (0:03, false 2:24):** the server *steps forward through contact*
   so the instantaneous depth reads shallow (0:03: partner at dist 26 pre-contact but 20 at the
   detected frame); and at 2:24 the front thrower **isn't tracked**, so the behind player is the
   only near track. A windowed "deepest-recent-position" tweak recovers 0:03 but trades precision
   (12/14 but 3 false, count 15) — marginal, NOT adopted.
2. **WHO-served errors (0:48, 3:39) = ROLE IDENTITY, not association.** At 0:48 we picked the
   RIGHT hitter (tid=669, deep server, closest to ball) but it's LABELED "user" when the operator
   says the server was the PARTNER. Root: **track fragmentation** (over 5 min: partner=29 track-IDs,
   user=8, opp_a=19, opp_b=27) → fragment→role assignment churns and swaps user↔partner (same side).

### NEXT (operator-chosen): FIX THE user/partner IDENTITY FOUNDATION (Stage 2.5) FIRST
Operator: "make sure the foundation is correct otherwise we risk building additional function off
a shaky foundation." The true lever for who-served AND every per-player stat is **user-vs-partner
identity in Stage 2.5 (classify_tracks / track_roles)**: de-fragment tracks and reliably pin which
near-side player is the user vs the partner. Also improve **far-side shot recall** (opponent serves
44 ft away, e.g. 4:04 not detected as a shot) and the clip-start shots (operator: 0:03 serve by
partner + 0:06 return by opponent are BOTH missed/mis-attributed today).

### Method to follow (operator's, saved as memory feedback_avoid_rabbit_hole_cycle)
Combine weak cues; anchor on known counts; **cheap-test → gate → then build** (don't deep-build on
faith); solve the shared dependency once; when a "camera limit" appears, first check it's not a
mis-set rule. Multi-venue generalization is a priority AFTER critical fixes — prefer game-structure
signals that travel across courts over per-court thresholds; derive court values from calibration.

### Operator ground truth for `pb_5_minute_outdoor-2` (the acceptance test)
98 shots (24 user) · 13 rallies = 13 serves · 18 dinks · 17 volleys (5 user) · 81 bounces
(shots = volleys + bounces). **Net hits: 8 total, 3 user** (BH dink @0:20, FH drive @2:36,
BH drive @3:49); all 8 @ 0:20/0:38/1:09/1:51/2:19/2:36/3:25/3:49. **Serves (~14):**
0:03(partner)/0:33/0:48(partner)/1:04(partner)/1:16(partner)/1:29/2:08/2:32(me)/2:45/3:06/
3:39(opp)/4:04(opp)/4:43(opp)/4:58. Camera-BLOCKED (need side/2nd camera): net-hit counting,
per-shot outcome (net/out/winner), ball height, true speed, spin.

---

# Session Handoff — Pickleball-Analyzer-v2 (updated 2026-07-22b)

## 2026-07-22b — REPORT VALIDATED PER-USER; next = TECHNIQUE (body mechanics) — READ FIRST

### State: the report is now internally consistent and validated against operator counts

USER-level counts (rating is per-user) vs operator truth on `pb_5_minute_outdoor-2`:
drive 14 (12) · serve 4 (4) · dink 6 (6) · drop 1 (2) · volley 6 (6) · returns 3 (3) ·
FH 13 (15) · BH 10 (9). **Dink/serve/volley/returns are EXACT.** Rating 3.77, band 4.0,
measurement coverage 36%. Match-level counts also shown ("22 in the match, 6 by you").

Report fixes done this session (operator's 12-question review): per-user + cleaned
third shot (deep drop-or-drive only; gated at <4 decisions), returns counted, header
shots in-rally (96), volley relabelled "your shots", confidence relabelled
"measurement coverage", bounce map notes net/volley shots aren't shown + separates
between-point from missed, plan drills refiled (resets->Strategy, soft-game->Dink),
7/10 target removed, coaching gate 10->6.

### OPERATOR DIRECTION (2026-07-22b): technique BEFORE multi-clip

Operator wants to **enrich the report + close any remaining CURRENTLY-DOABLE accuracy
items first**, then multi-clip. Named priority: **TECHNIQUE / BODY MECHANICS** — which
is derivable from POSE (94% detected, 33 joints/frame) and does NOT need ball height,
so it is achievable on this one 6ft camera. This adds shot-QUALITY the camera can
actually see (unlike pace/spin/dink-height, which are height-limited).

### ROADMAP (priority order set by operator)

1. **TECHNIQUE / BODY MECHANICS from pose (NEXT).** Camera-feasible quality signals:
   ready position / paddle-up between shots; split-step timing (hop as opponent
   contacts); athletic stance / knee bend; contact point in front vs late; shoulder
   turn / rotation on groundstrokes; balance at contact; reach-vs-move; follow-through.
   Maps to USAPA technique language. Scope which are robust from a 6ft corner view.
2. **Report enrichment** — surface more of what we already compute (per-rally detail,
   position quality, movement) with honest confidence.
3. **Remaining doable accuracy** — bounce recall (close the ~2-10 shot identity gap);
   reduce "unknown" strokes (recover the 2 user FH); tighten opponent-side dink
   over-count (match totals only; does not affect the per-user rating).
4. **Net-hit detection** — a ball into the net doesn't bounce; detect the ball
   stopping/dropping at the net line so net errors show on the bounce map + counts
   (operator hit several into the net: dink, FH, BH, drives).
5. **Multi-clip aggregation over time** — accumulate per-player stats across many
   clips/sessions so thin per-category samples (dink 6, serve 4 in 5 min) become
   coachable. Operator wants this AFTER technique/enrichment.
6. **HEIGHT-LIMITED quality (needs camera change; deferred)** — true shot speed, dink
   pop-up height/depth control, return depth/location, volley block/put-away/speed-up
   classification, spin, direct bounce-vs-volley. All need ball height; three
   height-free methods tested and defeated (see ACCURACY_LEDGER). Revisit only if a
   higher/second camera is added.

### Fast test rig + acceptance test (unchanged)

`data/pb_outdoor2_excerpt` (44s, source f16200-18861, excerpt f == source f+16200).
Acceptance test = operator COUNTS (see ACCURACY_LEDGER top). Validate every change
against them, never the previous run. All work committed + pushed (HEAD 47c25ef);
20/20 stage tests pass.

---


## 2026-07-22 — ACCURACY DRIVEN BY OPERATOR COUNTS → next = dinks, volleys, rallies — READ FIRST

### THE ONE THING TO KNOW

**The operator's counts are the acceptance test. Validate every change against THEM,
never against the previous run.** Full detail in `docs/ACCURACY_LEDGER.md` (top).

`pb_5_minute_outdoor-2` operator truth: **98 shots · 13 serves = 13 rallies · 18 dinks
· 17 volleys · 81 bounces**, and the identity **`shots = volleys + bounces`** (98 = 17
+ 81) must hold.

> **Why this matters.** Per-shot shot-TYPE accuracy was tuned on a single rally for
> several days while nobody checked the whole-clip COUNTS — which is all the operator
> sees in the report. He supplied one count and a 25 % adjacent-court contamination bug
> fell out in minutes. Do not optimise a stage in isolation of the report's objectives.

### Scorecard now (start of session → now, vs truth)

| item | truth | start | now |
|---|---|---|---|
| Shots | 98 | 155 | **108** |
| Bounces | 81 | 146 | **75** |
| Serves | 13 | 4 | **11** |
| Volleys | 17 | 51 | **27** |
| Dinks | 18 | 8 | **35** |
| Rallies | 13 | 19 | 18 |
| Forehand+backhand | — | 26 shots | **76 shots** |

### Fixed this session (commits `ad83df5`, `e1d9518`, and earlier)

1. **Adjacent-court contamination** — `detect_shots`/`detect_bounces` never excluded
   `role='noise'` tracks, so people on NEIGHBOURING courts generated shots/bounces
   (38 of 155 shots = 25 %). Both now restrict association to the four participants.
2. **Serves detected then thrown away** — the serve test `continue`d when the frame was
   already an impulse shot, leaving `is_serve=False`; 11 of 18 rallies had no serve.
   Now promotes that shot to a serve.
3. **Handedness discarded** — `roster.json` has all four players' handedness and
   `stroke_side()` derives facing from pose, but only the user's was used (74 of 108
   shots "unknown"). Now per-player.
4. **Bounce physics** — capped at ONE landing per shot (intervals had up to 12).
5. Rally model: **a serve STARTS a rally; a stall/gap/dead ball ENDS it** (does not
   start a new one); a non-serve shot may open a rally only if struck from DEEP.
6. Earlier in session: Stage 4 candidate+continuity ball tracking (teleports 20→0),
   confidence calibration (fallback + volley) to measured accuracy.

### NEXT — in this order, each validated against the counts above

1. **Dinks 35 vs 18** — over-calling. Recalibrate thresholds against the operator's 18.
2. **Volleys 27 vs 17** — should follow once bounces are exact (use the identity).
3. **Rallies 18 vs 13** — 11 serve-starts + 7 deep-shot restarts. **Sweeping the
   stall / gap / dead-ball thresholds does NOT move the count** (verified), so the
   current theory is wrong — trace the 7 actual restart frames instead of tuning.
4. **Then** rebuild the report.

### Operator directives (do not lose)

- **Report chart:** keep EVERY USA-Pickleball-rated item listed under "what's behind
  each category" — do not delete rows. The filled/unfilled circles must honestly show
  measured / partial / not-yet.
- **Camera stays as-is** (one camera, ~6 ft, corner). A side mount was tried before and
  caused other problems; a higher mount may be possible later, not now.
- Everything must be driven off what USA Pickleball rates; no stage work in isolation.

### Capability verdict (full matrix in the ledger)

- **ACHIEVABLE:** shot/serve/rally/return counts, dink VOLUME, third-shot
  drop-vs-drive CHOICE, forehand/backhand per player, positioning/kitchen time, and
  volley COUNT via the identity (`volleys = shots − bounces`).
- **BLOCKED by the single low camera** (needs ball HEIGHT — three independent
  height-free methods tested and defeated): dink QUALITY, true shot speed, direct
  bounce-vs-volley at the feet, spin.
- So the report can honestly do VOLUME / CHOICE / POSITIONING, **not stroke quality**.

### Fast test rig (use this, not the 4.7 GB clip)

`data/pb_outdoor2_excerpt` — 44 s, source frames 16200-18861 (**excerpt frame f ==
source f + 16200**; rally 10 = excerpt f1684-2544). Bundle already on Drive as
`pb_outdoor2_excerpt_vision_input.zip`. Full vision pass ≈ 2 min on A100.
Colab notebook has `CLIP` / `RERUN` knobs — `RERUN='ball'` redoes ONE stage.
⚠ The notebook run cell needs all three lines including `run_all(REPO, clip=CLIP,
rerun=RERUN)`; pasting only the two assignments does nothing (runs in 0.006 s).

### Repo state

All work committed and pushed to `main` (HEAD `e1d9518`). 20/20 stage tests pass.
Pipeline order: shots → bounces → **trajectory (5.7)** → classify → rallies → metrics
→ rate → plan → report.

---


## 2026-07-12 — SETUP UI PHASE 1 DONE (setup wizard) → next = UI Phase 2 (run & progress) — READ FIRST

Built the **input/setup UI Phase 1** per `docs/UI_PLAN.md` — a local FastAPI web
app + vanilla-JS single-page wizard that replaces the Tkinter `mark_court.py` /
`mark_user.py` tools. New top-level **`app/`** package. Two commits: backend
skeleton (`6cf935f`) + front end (`69b42b2`). On branch **`feat/setup-ui-phase1`**
(not yet merged to main).

- **Backend (`app/`):** `video.py` serves EXACT source frames as JPEGs via OpenCV
  (same indexing the pipeline uses → marked coords map to original-video pixels);
  `sessions.py` owns per-video `data/<name>/` folders and writes the input JSONs,
  calling `stages.calibrate.calibrate()` **in-process** (no reimplementation);
  `browse.py` = server-side local file browser (the "server" is the user's laptop);
  `server.py` = FastAPI routes; `__main__.py` = `python -m app` launches uvicorn +
  opens the browser. Session name auto-derives from the parent folder when the
  filename is generic (`data/<clip>/video.mp4` → "<clip>", not "video").
- **Front end (`app/static/`):** 5-step wizard — **Video** (server-side file
  browser AND browser upload w/ progress; lists prior setups) → **Court** (served-
  frame canvas marking of the 8 pts in order, zoom loupe, live lines, undo/clear,
  scrubber, the 3 hand/baseline/corner dropdowns; "Check calibration" runs Stage 1
  and shows the top-down warp + RMSE/kitchen-error metrics + warnings for
  confirm-or-redo) → **Players** (per-role handedness → `roster.json`, keys
  `user/partner/opp_a/opp_b`) → **You** (optional self-ID clicks → `user_clicks.json`;
  skippable) → **Review** (reads back from disk) → Finish. Theme-aware, court-teal.
- **Data contracts UNCHANGED** — the wizard produces `markers.json` →
  `court.json`/`court_zones.json` (via Stage 1), `roster.json`, `user_clicks.json`,
  exactly as before. Nothing downstream changes.
- **Validated end-to-end by driving the REAL app in a browser** against the 4K
  pb_2min clip + **self-contained pytest smoke tests** (`app/test_app.py`, 9 tests,
  synthesizes a tiny video — no dependency on gitignored `data/`).
- **Operator-feedback UX revision (`e471242`, merged to main):** (1) Video — dropped
  the full filesystem browser + upload for a **single designated drop folder**
  (`PB_VIDEOS_DIR`, default `./videos`; user copies the clip in and picks it) + an
  on-screen **recording-requirements** card (corner-nearest-player, whole-court view,
  tripod ≥6 ft, shutter 1/1000s+, 60 fps, 4K UHD, ISO 800, 2–5 min). (2) Court — the
  calibration confirm is now an **inline panel below the still-visible marked frame**
  (was a modal that hid it). (3) "Point yourself out" (per-frame clicking) → a
  **visual left/right "which side do you start on"** picker that patches
  `user_starting_corner` into markers/court (new `POST …/starting-corner`); the
  no-click geometric seed is the default (user_clicks dropped from the wizard,
  backend endpoint retained but dormant). (4) Review — simplified to "Setup complete."
  + a **"Start Analysis →"** button (Phase 2 wires the run).

**NEXT — UI Phase 2 (run & progress + the Stage-4 hand-off).** Backend job
orchestration: run Stages 1–3 locally as a background job with per-stage progress
(SSE/websocket), the guided Stage-4 GPU/Colab hand-off (the one operator-assisted
step), then resume Stages 5–11 + `build_report.py` + `compress_video.py`, and serve
the finished report/video. `session.json` already records the resolved video path
for the orchestrator. Then Phase 3 (library/polish). See `docs/UI_PLAN.md`.

**Also still open / parallel (unchanged):** the ADD metrics are data-gated on the
cross-venue detector / more footage (bounce recall C4, serve C3, stroke-side F16,
shot speed F7/F8); operator labeling indoor clips → cross-venue retrain is the
standing gate (`DATA_COLLECTION_PLAN.md`).

---

## 2026-07-13 — SETUP UI PHASE 2 (run & progress) — backbone built + REAL post-ball loop proven — READ FIRST

Built Phase 2 (run & progress) on top of the Phase-1 wizard. New `app/pipeline.py`
+ a **Run** view (6th wizard step). Commits on `main`: backbone (`8244fb9`) +
real-loop fixes (this session's latest).

- **Runner:** each stage runs as an isolated subprocess (`python -m stages.<x>.<x>
  <folder>`), one at a time, stdout streamed into a live per-job log. State
  machine: **pre** (materialize video.mp4 via hardlink → Stage 2 track → 2.5
  classify_tracks → 3 pose) → **pause at ball** → **post** (5→5.5→6→7→8→9→10→11 →
  compress_video → build_report) → done. Progress via **SSE** (`…/run/stream`).
- **Stage 4 = GPU/Colab hand-off, decoupled + auto-resume:** the run PAUSES and
  resumes the instant `ball.parquet` (+ `ball.meta.json`) is uploaded back
  (`POST …/ball`). No local GPU here (`torch.cuda`=False). **Operator Q answered:**
  the *app* can't autonomously drive Colab (no Colab API; auto-login is brittle +
  against the safety rules) — the prior "autonomous" runs were *Claude* driving via
  the Chrome extension in dev sessions. The backbone makes it hands-off regardless:
  whoever fills ball.parquet (guided Colab / Claude-via-Chrome / future cloud GPU)
  triggers the rest automatically.
- **REAL post-ball loop PROVEN (2026-07-13):** pb_2min already had a real
  `ball.parquet` + full Stage 1-3 outputs from June, so per operator we skipped a
  redundant 40-min Colab run. Seeded a session with those real upstream files,
  uploaded the real ball, and the app **auto-resumed real Stages 5→11 + compress +
  build_report**, producing the real consumer report (**rating 3.95 / band 4.0**,
  matching the documented pb_2min result) + a 6 MB web video, served in-app and
  confirmed rendering in the browser. Analytical stages ran in seconds; render
  (capped to 15s via `PB_RENDER_MAX_SECONDS`) 137s; compress 41s.
- **Fixes surfaced by the real run:** Stage 5 needs `ball.meta.json` → `/ball`
  accepts it + `ensure_ball_meta()` synthesizes one from the video probe if absent;
  runner passes `--force` to stages (render fails on an existing annotated.mp4
  without it) for re-runnability; `PB_RENDER_MAX_SECONDS` caps the slow full render.
  `PB_FAKE_STAGES` simulates stages fast to preview the Run UI without a GPU.
- **Tests:** 14 app tests (incl. 4 pipeline: state machine, hardlink materialize,
  failure-stops-run, real subprocess plumbing). Run: `python -m app`.

**REAL PRE-BALL STAGES NOW VALIDATED + throughput wall confirmed (2026-07-14).**
Ran a real end-to-end attempt through the app on a **5-min 4K/60fps outdoor clip**.
Finding: **Stage 2 (YOLO) runs at ~1 fps on CPU (no GPU) → ~5 h for tracking alone,
~8–12 h with pose** — the app's "run locally" path is **not viable for real clips on
CPU**. A **20-s trim** ran Stages 2→2.5→3 correctly in **~17 min** (players.parquet
8297 rows/56 tracks, all 4 roles, poses 4637 rows), paused at the ball hand-off — so
the logic is right; the wall is CPU throughput. Logged in KNOWN_ISSUES ("local CPU
can't process real clips") + UI_PLAN.
- **The autonomous Colab drive hit a hard limit:** the Chrome-extension file-upload
  tool caps at **10 MB**, and a 20-s 4K clip is **877 MB** — so Claude **cannot push
  the video to Drive** to run Colab autonomously. Claude can drive the *notebook* +
  download the small result, but the operator must get the video onto Drive first
  (prior "Claude drove Colab" runs had the video pre-placed). Reconfirms: the GPU step
  isn't fully automatable by the agent for real-size clips → **cloud GPU is the real
  self-serve fix.**
- Trimmed test clip: `videos/PB_5min_test_20s.mp4` (2:00–2:20 of the 5-min clip,
  full 4K/60fps, re-encoded high-bitrate via OpenCV since no ffmpeg). Session
  `data/pb_5min_test_20s/` (setup reused from David's real `pb_5_minute_outdoor`
  markers — same static court). Server was left running on :8180 paused at ball.

**DONE — MOVE STAGES 2/3 (+2.5, +4) TO GPU/COLAB + APP INTEGRATION (2026-07-14).**
The combined vision pass is built, **validated on Colab** (Stages 2→2.5→3→4, ~7 min
on an A100, outputs match local: players 8309/local 8297, poses 4637 identical, roles
identical, ball detect_frac 0.9275), AND **wired into the app**:
- **Runner reworked** (`app/pipeline.py`): `prepare` (materialize video) → `vision`
  (local if a CUDA GPU is present, else a **GPU hand-off** that pauses) → `post`
  (5→11 + compress + report). The vision hand-off auto-resumes when the outputs
  arrive — new **multi-file `POST …/vision`** (guarded to known output filenames)
  replaces the single-file `/ball`. Run view: **vision hand-off card** w/ multi-file
  upload; phases prepare/vision/post.
- **Render length-capped by default** (`DEFAULT_RENDER_MAX_SECONDS=90`, override
  `PB_RENDER_MAX_SECONDS`; "0"/"full"=uncapped) so a 5-min clip's 4K render doesn't
  dominate. This is the local-side "reasonable time" lever.
- **`track_ball_v4` now BATCHES** (`detect_batch` + auto batch-size by GPU memory,
  `--batch` flag; CPU stays per-frame). Identical results, fast enough for full-length
  clips (per-frame ~100 min on an 18k-frame clip). The vision notebook runs this module
  → batched automatically. Bundle regenerated with it.
- **Verified end-to-end via the NEW runner** on real data: start → vision hand-off →
  `/vision` upload of the real outputs → auto-resume → real Stages 5-11 + capped render
  → report (http 200). 15 app tests + 5 track_ball tests pass.

**Operator hand-off note (unchanged limit):** the agent **cannot push files to the
user's Drive** (browser-upload security boundary — rejected scratchpad + @-referenced
files; binary not shareable; private+stale GitHub). So the operator uploads the bundle
+ setup files to Drive and runs `infer_vision.ipynb` (the ~7-min GPU pass); everything
around it is automatic. A cloud-GPU runner is the eventual full-self-serve fix.

**REMAINING (next):** the vision notebook needs the setup files on Drive alongside
video.mp4 — the app could package a per-clip bundle (video + court/roster) to make the
hand-off one download; and Phase 3 (library / reopen a paused run / polish). For 5-min+
clips: Colab vision (~few min w/ batched ball) + local post w/ capped render = reasonable.

---

## (superseded) 2026-07-14 — vision pass BUILD notes (pre-integration)
**IN PROGRESS — MOVE STAGES 2/3 (+2.5, +4) TO GPU/COLAB (combined vision pass).**
Operator chose **one combined Colab pass** for all heavy vision (2 track → 2.5
roles → 3 pose → 4 ball) so the app offloads it in one GPU trip and keeps only the
light analytical stages (5→11 + report) local. **Built this session (commit below):**
- `tools/build_vision_bundle.py` → `data/pb_vision_upload.zip` (34 KB; the real
  stage code — track/classify_tracks/pose/track_ball_v4 + _tracknet_model + inits).
  Confirmed self-contained (only track_ball_v4 imports _tracknet_model). Both models
  auto-download on Colab (yolo11s via ultralytics, pose model via MediaPipe); ball
  weights already in Drive root.
- `tools/build_vision_nb.py` → `stages/infer_vision.ipynb` (17 cells, GPU). Mounts
  Drive, pip-installs ultralytics+mediapipe, unzips the bundle, and runs the REAL
  modules `python -m stages.<x>.<x> <CLIP_DIR>` for 2 → 2.5 → 3 → 4 (YOLO auto-GPU
  confirmed; track_ball_v4 auto-CUDA confirmed; pose on Colab CPU). Reads `pb_infer/
  <CLIP>/` (needs video.mp4 + court.json/court_zones.json/roster.json[/user_clicks])
  and writes players/roles/poses/ball (+metas) back to Drive.
- **Ball note:** the combined pass runs the per-frame `track_ball_v4` module (auto-
  CUDA, simple, no code dup) — a bit slower than the standalone batched `infer_v4.ipynb`;
  folding the batched loop in is a noted future optimization.

**NEXT (not done):** (1) **TEST `infer_vision.ipynb` on Colab** end-to-end (upload
the bundle to Drive root + the setup files to `pb_infer/<CLIP>/`, Run All, verify
outputs) — drive it via Chrome like the ball run. (2) **App integration:** rewire the
runner's local "pre" phase into a **vision GPU hand-off** — get the setup files +
video to the GPU step, run the combined pass, upload all vision outputs back, then
auto-resume Stages 5→11 (same decouple/auto-resume mechanism as the ball). Then
Phase 3 (library/reopen a paused run/polish). See `docs/UI_PLAN.md` + KNOWN_ISSUES.

---

## 2026-07-11 — USAPA REALIGN + CONSUMER REPORT DONE → next = build the UI (Phase 1) — READ FIRST

Big session. The build program is now through **FIX → REALIGN → (partial) ADD →
report**, with the input UI scoped and next. In order:

- **USAPA REALIGN done (Stage 9 v0.4.0 `52c9a56` + Stage 10 v0.5.0 `bc85e80`).**
  Rating rewritten from 6 homegrown dims to the **7 official USAPA categories**
  (strategy/third_shot/dink/volley/serve_return/forehand/backhand), each with a
  `coverage_status` (measured/partial/not_assessable); count-only strokes + no-serve
  serve_return capped to not_assessable; a zero-event guard in Stage 10 (0 dinks →
  not assessable); single heavily-caveated estimate that leans on Strategy. Design
  in `docs/USAPA_REALIGN_DESIGN.md`. Contracts fully rewritten (`21eac19`). Also
  fixed `score_volley` reading the wrong path (always NEUTRAL) `c2a703f`. pb_2min:
  3.95 / band 4.0, only Strategy `measured`. Smoke 9/9 both stages.
- **ADD step — court-plane ball speed (F7) TESTED + REJECTED.** Validated before
  building: naive image→ground projection of the airborne ball explodes (drop read
  157 ft/s, one 5626 ft/s; court_y to 902 ft). Real pace needs ball HEIGHT (F8
  z-recovery, gated on recall). Logged in KNOWN_ISSUES. Stroke-side (A2) also
  examined — improves counts but does NOT flip forehand/backhand off not_assessable
  (quality-gated), so deferred. **The remaining ADD metrics are data-gated** (bounce
  recall C4, serve detection C3, stroke-side F16, shot speed F7/F8) → they wait on
  the cross-venue detector / more footage (operator labeling indoor clips).
- **CONSUMER REPORT done — `tools/build_report.py`.** Self-contained `report.html`
  from the pipeline JSONs: rating hero, session-at-a-glance stats, the 7-category
  table (what USAPA rates × your level × coverage badge), category detail (plain-
  English metrics + %, ●◐○ legend), improvement plan, USAPA ladder (user band
  highlighted), positioning heatmaps + a ball-landing map (dots only), annotated-
  video embed, technique/trends placeholders, footnotes. Polished visual (court-teal
  + serif/sans, both themes). Two operator review rounds folded in. `tools/
  compress_video.py` (new) makes the 462 MB 4K render → 50 MB 720p web clip.
- **NEW bounce-recall KNOWN_ISSUES entry (2026-07-11):** ~50% recall — 30
  groundstrokes (39 shots − 9 volleys) imply ~30 bounces but only 15 detected;
  thins the landing map + depth metrics. Same root cause as Stage 4 recall.

**NEXT — build the input/setup UI, Phase 1 (spec: `docs/UI_PLAN.md`).** Operator
decisions locked: **local guided web UI (FastAPI + browser) replacing the Tkinter
`mark_*` tools + orchestrating the local pipeline; GPU ball step = guided Colab
hand-off; audience = early outside users** (setup + report polished; GPU step
operator-assisted for v1). Phase 1 = the **setup wizard** (frame-serving + in-browser
8-point court marking with validation, player setup, optional self-ID, writes the
same input JSONs + runs Stage 1). Best started fresh — it's a multi-hour,
outside-user-polish build; UI_PLAN.md is the spec. Data contracts unchanged.

**Also open / parallel:** operator labeling 2–3 indoor clips today → cross-venue
retrain (the standing data-limited gate, `DATA_COLLECTION_PLAN.md`); the ADD metrics
above unlock as recall/venues improve. Report follow-ups: annotated video is still
462 MB at full res (compress step added); landing map thin until bounce recall.

---

## 2026-07-09 — CONSUMER-OUTPUT FIX STEP COMPLETE (5 fixes) → next = USAPA REALIGN — READ FIRST

The build program's **FIX step is done**: the live pb_2min consumer output is now
TRUE and readable end-to-end, each fix operator-validated on rendered output and
committed. This session, in order:

1. **Net-play zone bug → front foot (Stage 8 v0.3.0, `88ff309`).** The prior
   hypothesis ("position→zone mapping is off") was WRONG — `zone_from_court_y` is
   correct. Root cause: court position came from the bbox bottom = the **back foot**;
   a net-facing player with a staggered stance reads several feet behind where they
   play, so a kitchen-line player mis-classified as transition. Fix: position = the
   **net-most ankle** (front foot) from `poses.parquet`, bbox fallback. Operator's
   rule ("front foot within 2 ft of the line = kitchen") is already the 2 ft buffer
   in `KITCHEN_MAX_DIST_FT`. user kitchen 5.4%→26.2%; opponents unchanged (far side's
   bbox bottom already = front foot). Validated on a frame-532 overlay.
2. **Rally over-segmentation (Stage 7 v0.3.0, `13b629c`).** Minimum-rally filter:
   drop a segment only when it's BOTH < `MIN_RALLY_SEC` (2.0s) AND < `MIN_RALLY_SHOTS`
   (3). Note rally 7 was a **falsely detected serve**, so a serve-flag guard alone
   can't catch it — size is the separator. Lone serve-faults (n_shots==1) guarded.
   Dropped shots → `unassigned_shots` (reconciles). Real-ball only. **8→6 rallies**
   (matches operator). mean rally length 5.19→5.67.
3. **Rally-scope position metrics (Stage 8 v0.4.0, `bc4df48`).** Operator-confirmed:
   between-point frames (~42% of clip = baseline standing) must not count. All
   position views now scope to in-rally frames (`position.scope`); movement never
   bridges a rally boundary. Needed step 2's clean boundaries first. user kitchen
   26.2%→**33.6%**, both-at-kitchen 22.6%→**33.3%**.
4. **Movement jitter-floor bug (Stage 8 v0.5.0, `4be7ccf`) — found while doing #3.**
   `MOVE_MIN_STEP_FT=0.25` was per-frame, never fps-scaled → at 60fps a 15 ft/s floor
   that rejected 84% of real movement and summed noise spikes. A speed floor can't
   fix jitter (jitter has high instantaneous speed). Fix: integrate from a **0.2s
   downsample** (window-mean positions), gated by a jitter floor + a 24 ft/s cap.
   user `distance_ft_per_min` **492→192** (plausible ~3 ft/s). Same "confidently
   wrong at conf 1.0" class as net-play.
5. **Finding language (Stage 10 v0.4.0, `fa09f59`).** Findings stated raw numbers
   with no verdict + jargon ("court coverage of your half", "transition zone",
   "N shot types used"). Rewrote to plain second-person English pairing each number
   with a good/bad verdict (`_verdict` bands). Where a metric isn't inherently
   good/bad (court coverage/distance), it SAYS so and points at the lever instead of
   faking a verdict. Numbers still straight from rating.json (can't drift).

**pb_2min after the 5 fixes:** rating **3.8, band 4.0**; the two real-position dims
are now TRUSTWORTHY at conf 1.0 (net_play 3.89, movement 3.48) — they were the two
"confidently wrong" ones. Smoke: Stage 7 9/9, Stage 8 16/16, Stage 10 9/9.

**Two follow-ups flagged, NOT done (in KNOWN_ISSUES):**
- **Stage 2.5 near-side role gap** — at some frames both near tracks resolve to one
  role (pb_2min f6420: both `partner`, user unidentified), slightly UNDER-counting the
  user's kitchen time. Deferred to a Stage 2.5 continuity pass.
- (movement bug above was found-and-fixed, not deferred.)

**NEXT — USAPA REALIGN (build program step 2): rewrite Stage 9's 6 homegrown dims to
USAPA's 7 categories.** Scoping started this session — see the mapping below / in
`docs/PRODUCT_VISION.md`. Stage 9 today = `net_play, movement, error_control,
shot_skill, serve, rally_consistency` (see `stages/rate/rate.py` `WEIGHTS` + the
`score_*` fns). Target 7 = `Forehand, Backhand, Serve/Return, Dink, Third-Shot,
Volley, Strategy`. The realign is design-heavy (most USAPA criteria map to ◐/○
not-yet-measured metrics — the legitimacy gap); scope with the operator before coding.

---

## 2026-07-07 — Cross-venue = data-limited; stats layer 8→11 DONE on pb_2min (real+confidence) — READ FIRST

**Item #3 (cross-venue detector) pushed to its current-data ceiling; item #5 (stats layer
8→9→10→11 + confidence propagation) COMPLETED on pb_2min (provisional, one venue) — the
06-21 "confidence validated on synthetic only" gap is now closed on real data.**
David bought Colab Pro+ and ran two warm-start training runs this session. This session:

- **Reconciled the notebook.** The live Colab `finetune_v4.ipynb` had DIVERGED from the
  repo copy (stronger photometric aug + a from-scratch training loop + no resume block;
  the repo copy was stale and even had a `resume_best` NameError bug). Downloaded the
  live copy, made it the repo source of truth, and rebuilt it for Run 2.
- **Two findings from the live notebook's cached Run-1 output:** (a) the documented
  **"0.90→0.858 same-court regression" is largely a model-SELECTION artifact** —
  recall actually hit **0.96**, but the `score = recall − fp` selector kept a
  low-recall/low-fp epoch. There is *also* a real precision cost (fp 0.10–0.24 for ≥0.90
  recall vs baseline 0.018). (b) **held-out indoor recall maxed at 0.126** → augmentation
  alone doesn't generalize indoor; indoor must be trained on.
- **Run 2 design (commit `ed3d02d`):** warm-start from the clean 0.90 baseline
  (`MyDrive/ball_model_v4_base.pt`) at LR 1e-4; **all 3 venues in training with per-venue
  held-out slices** (pb_2min home guardrail; court2 + indoor 88/12 leakage-free split);
  **fp-capped selection** (max mean per-venue recall s.t. home fp ≤ 0.05) → saves
  `ball_model_v4_run2.pt` + `validation_report_run2.json` (0.90 baseline never overwritten).
  Data-split logic validated locally; warm-start confirmed loading the 0.90 base on the
  live run (recall 0.9024).

- **RESULTS (Run 2a 15-epoch, Run 2b 30-epoch + fp cap 0.06):** warm-start held precision
  (fp stayed low). Best saved model `ball_model_v4_run2.pt` (Run 2b ep9): **home 0.892 /
  court2 0.625 / indoor 0.448** raw recall. More epochs did NOT help (best ~ep9; the 30-ep
  back half went unstable). **Reality check** (`reality_check_v4.ipynb`) measured EFFECTIVE
  coverage after Stage-4 trajectory post-proc: **home 0.935 · court2 0.691 · indoor 0.608**.
- **VERDICT: court2 + indoor still below the ~0.80 bar → detector is DATA-LIMITED** (home
  has 4 clips and works; the others have 1 each). court2's misses cluster into long
  (>8-frame) gaps = **hard-hit motion blur, a capture-side limit**; indoor's misses are
  short/isolated and respond to more data. **Lever = more footage per venue, NOT more
  training** — see `docs/DATA_COLLECTION_PLAN.md` (faster shutter + 2-3 varied clips/venue).
  Operator is capturing that footage; retrain loop is in the plan doc.

**NEXT (resume here) — work-order #5 stats layer on pb_2min (provisional, one venue):**
- Real-ball chain **5→5.5→6→7 confirmed reproducible** this session (39 shots/4 serves,
  15 bounces, 0 unknown types, 8 rallies — the "stale shots.json" note was WRONG, corrected).
- **Stage 8 DONE:** compute_metrics + **confidence propagation (C9) validated on REAL ball
  for the first time** (was synthetic-only since 06-21). 54 metrics carry
  `{value,confidence,n,limited_by}`; durable ones (rally length/duration, position) honest-
  moderate, noisy ball families correctly distrusted (serve 0.15, end_reason 0.15, shot_mix
  0.50, stroke_side 0.09). *Flag for downstream-sufficiency review:* `match.serve.n_serves=8`
  vs Stage-5 `is_serve`=4 (metric counts one serve/rally) — reconcile when serve detection (C3) lands.
- **Stages 9→10→11 DONE (v0.3.0 each) — the same C9 gap fixed at every layer:** Stage 8
  computed honest per-event/dimension confidence, but 9/10/11 each ignored it identically.
  - **Stage 9 (`9c85079`):** the estimate was confidence-BLIND — `error_control` scored 4.5
    at confidence 0 (errors undetectable → "no data" read as flawless), inflating pb_2min to
    3.61. Now **confidence-weighted** (`weight × confidence`, renormalized) → recenters to
    **2.79**, leaning on the measured dims (position/movement).
  - **Stage 10 (`e78475a`):** gated "provisional" on coarse `ball_source`, so on real ball it
    coached off data gaps (serve = weakness, error_control = strength). Now **gates on per-dim
    confidence**: near-zero-confidence dims route to `developing_capability.not_assessable_now`;
    focus areas = net_play + movement only.
  - **Stage 11 (`f103568`):** timeline events dropped per-shot confidence (each rendered as
    certain). Now shot/bounce events carry `shot_type_confidence`/`is_volley_confidence`/etc.
    (62/70 events). Watermark correctly drops (ball_source real).
- **NET: the C9 confidence machinery is now BUILT + VALIDATED ON REAL DATA end-to-end on
  pb_2min** (the 06-21 "synthetic-only" gap is closed) — flagged **provisional (pb_2min only)**
  per §0 rule 6. Each stage smoke-passed + operator-reviewed.

**NEXT (resume here):** the pb_2min real-ball pipeline (5→11) is complete + confidence-honest.
The gating item is again **cross-venue data** (work-order #3, data-limited). When new
court2/indoor footage is labeled → run the retrain loop in `DATA_COLLECTION_PLAN.md`
(prepare_v4 → build bundle → warm-start finetune → reality_check_v4) → re-run 5→11 across
venues to lift the provisional flag. Other open parallel tracks: serve detection (C3, would
reconcile the n_serves=8-vs-4 flag + unlock the serve dimension), bounce recall (C4, unlocks
end_reason + error_control), z-recovery spike, input-UI + reporting skeleton (the timeline.json
per-event confidence is now ready for it).

**CONSUMER OUTPUT + USAPA VISION (2026-07-07, second half — READ):** Operator rendered the
real Stage 8–11 output for the first time and it exposed what confidence numbers CANNOT
catch — **confidence ≠ correctness.** Confirmed bugs: (a) **net-play is wrong** — kitchen
time reads ~5% / both-at-line 0.3% while players clearly live at the line; the position→zone
logic is systematically off, AND it's a "99% confidence" dim the rating LEANS on (undercuts
the Stage 9 fix). (b) **rally over-segmentation** — Stage 7 makes 8 rallies (two are 0.8s/1.1s
micro-splits); real count is 6. (c) finding language unclear. Lesson: Stage 8–11 were validated
on schema/smoke, NOT on operator-viewing-numbers — that's the validation gap; the consumer
view is the missing instrument. ALSO: the rating's 6 homegrown dims **do not match the official
USAPA standard** (7 categories: forehand/backhand/serve-return/dink/third-shot/volley/strategy).
Captured the full USAPA-aligned target spec in **`docs/PRODUCT_VISION.md`** (skill ladder +
criteria→metric alignment [most planned = the legitimacy gap] + body-mechanics-as-supporting-
pose-layer + build program). **Operator-chosen order: vision (DONE, captured) → FIX bugs
(net-play zones, rally filter, finding language) → REALIGN Stage 9 to USAPA's 7 categories →
ADD the ◐/○ metrics (map to C4 bounce recall, F7 court-plane speed, F16 FH/BH, F12 opponents,
F17 pose) → COMPLETE UI (`tools/build_report.py` skeleton → full report).** Resume the build
program at the FIX step (net-play zone bug first — it drags the rating).

**Deployment note:** Colab needs, in `MyDrive/` root: `pb_v4_upload.zip` (bundle, current)
+ `ball_model_v4_base.pt` (= local `data/models/ball_model_v4.pt`, the 0.90 model). The
G4/RTX-PRO-6000 GPU auto-sizes BATCH to 12. Pro+ background execution runs it 24h even if
the tab closes.

---

## 2026-06-21 — COURSE CORRECTION + plan reset (read this first)

**What happened:** this session built **Foundation #3 (confidence propagation)** as
inline `{value, confidence, n, limited_by}` wrappers across Stages **8→9→10→11**
(commits `0d116b2` S8, `8350724` S9, `39b2c41` S10; **S11 uncommitted**), plus the
**operator-vs-player separation** in Stage 10 (`operator_considerations`, surfaced
only when a real limiter bites; David's call). The mechanism is sound and smoke-tested.

**BUT it was validated only on the SYNTHETIC `test_clip`, and the session started
declaring stages "done" + validating Stage 11 while Stage 7 was still unvalidated on
real data** — rebuilding the v1–v3 compounding-error failure. **Operator stopped it.**

**Corrections committed this session:**
- **SYSTEM_DESIGN §0 rules 5–9** added (synthetic ≠ validation; real = all venues;
  strict dependency order; one stage at a time; downstream-sufficiency review is part
  of "done"). **Read §0 before any work.**
- **§6/§7 roadmap REORDERED:** the real foundation gap is the **ball detector across
  venues**, not confidence. v4 trained on **outdoor same-court only** — different-court
  (0.54) + indoor (0.13) **never trained** (contract_v4.md). Confidence work is
  reframed as **built-but-UNVALIDATED**, deferred to after the real-ball upstream is locked.

**NEXT (agreed work order):**
1. **Stage 4 cross-venue retrain — NEEDS COMPUTE (operator funding Colab).** All 6
   clips already LABELED; **Run 2 (3-venue) is CONFIGURED in `finetune_v4.ipynb`**,
   blocked on compute. So: re-run Run 2, achieve per-venue recall on all 6 clips
   **WITHOUT regressing same-court** (Run 1 regressed pb_2min 0.90→0.858 — the open
   challenge; may need venue-balancing / 1080p / per-venue heads). Production detector
   today = original same-court v4. *Gates real validation of Stages 5–11 on 2 of 6 clips.*
2. Lock real-ball upstream **5→5.5→6→7**, one stage at a time, operator-validated,
   each with a downstream-sufficiency review.
3. Stats layer **8→9→10→11 + confidence**, re-validated on real (reuse this session's code).

**Decide before resuming:** whether to commit the Stage 11 confidence code as
work-in-progress (smoke-passed, unvalidated) or hold it.

---

# Session Handoff — Pickleball-Analyzer-v2 (prior: updated 2026-06-19)

> **READ `SYSTEM_DESIGN.md` (repo root) FIRST.** It is the authoritative source of
> truth: dependency map, per-stage accuracy ledger, the honest trust-map (what's
> real vs noise *today*), fundamental-limits decisions, the foundations-first
> roadmap, and the F1–F32 future register. This handoff is the session *log*;
> SYSTEM_DESIGN.md is the live *state*. It's also pinned in auto-memory.

## The design philosophy (NEW — how we work now)

v1–v3 (and v4 was repeating it) failed from **deferred decisions becoming
downstream blockers, lost cross-session rationale, and stats reported with no
honest accuracy accounting.** The countermeasure is SYSTEM_DESIGN.md §0:
1. A stage isn't "done" until it meets the accuracy its downstream needs,
   **validated on REAL data**. No "good enough for now, fix later."
2. **No deferral without recording its blast radius** in the ledger.
3. **Every session reads SYSTEM_DESIGN.md first** and updates it when decisions change.
4. **Fundamental limits are decided, not deferred** (accept-with-confidence /
   fix-at-capture / scope-out).

## Status (end of 2026-06-19 session)

This session pivoted from symptom-patching to a **full whole-system parallel audit**
→ SYSTEM_DESIGN.md, then began the **foundations-first roadmap**. Commits:
- `fcddcb6` — **SYSTEM_DESIGN.md** (the audit / source of truth).
- `94d5b1f` — **Stage 6 v0.4.0**: landing-aware shot type (drive/drop/dink from the
  bounce landing — sound where the airborne ball's depth-corrupted speed is not;
  ~21% landing coverage, honest confidence).
- `4b9c25d` — **Foundation #1**: Stage 2 far-side drift + **role-based pose scope**.
  Opponents were deleted from pose by a `court_y.max()≤44` gate (far-side jitter
  spikes past the baseline); now Stage 3 scopes by Stage-2.5 role. Opponents
  restored (validated pb_2min). Far-side absolute position is **zone-precision
  (~±5 ft)** — a camera-geometry limit, flagged via `court_pos_reliable`.
- `736b567` — **Foundation #2 (core)**: opponents grouped into two stable
  IDENTITIES **`opp_a`/`opp_b`** by appearance + continuity re-id (NOT position
  L/R — they switch sides), honest moderate confidence. System-wide rename
  opp_left/opp_right → opp_a/opp_b.

**Roadmap (SYSTEM_DESIGN §6): #1 done · #2 core done · #3 = NEXT.**

## NEXT SESSION: Foundation #3 — confidence propagation (design LOCKED)

Thread honest per-event confidence through Stages 6→11 so **every reported number
carries its reliability** (the audit's #1 architectural finding: no stage
propagates per-event confidence; every stat renders as certain even when it rests
on noise). **Decided with David 2026-06-19:**
- **Option 2 — inline `{value, confidence, n}` wrappers** on every metric
  (confidence inseparable from the value; no orphan numbers; chosen over a parallel
  block because that drifts).
- **All three stages in one pass: 8 → 9 → 11.**
- **Stage 8:** `conf_n()` aggregator (mean per-event confidence × small-sample
  penalty) + `mv()` wrapper; a per-metric **confidence-source map** — clean sources
  exist (`shot_mix`←`shot_type_confidence`, `bounce_in_out`←bounce confidence,
  `serve_fault`←`end_reason_confidence`); metrics with **no** per-event source
  (`rally_length_shots`, `match_span`) need a deliberate decision on what their
  confidence *means*. That per-metric mapping is the real work.
- **Stage 9:** read `.value`; set per-dimension confidence from each metric's real
  `.confidence` (retire the coarse synthetic/real heuristic).
- **Stage 11/timeline:** surface per-metric confidence so the report gates each number.
- **Honest caveat to document:** captures classification-noise + sample-size, **NOT
  recall-bias** (Stage 8 can't see missed events — that stays a documented limit).

## Parallel tracks (David chose; queued, not started)
- **z-recovery feasibility spike** — parabola/gravity ball-height from a single
  camera; informs the SYSTEM_DESIGN §5 ball-height/3D decision (currently "investigate first").
- **Input-UI + reporting skeleton** — #2 surfaced a concrete need: non-user
  handedness needs a UI to show the operator "who is `opp_a`" before they can label it.

---

_Below: the prior session log (history; superseded by SYSTEM_DESIGN.md for current state)._

## DONE 2026-06-16: foundation hardening (Stage 5 v0.3.0 + Stage 7 v0.2.0)

Operator review of the FULL pb_2min clip (not just the per-shot overlay) exposed
data-quality issues that corrupt stats. Fixed + validated + committed:
- **Stage 5 v0.3.0** (`066d63e`): **adjacent-court contamination gates**
  (serve-must-launch-a-sustained-run + impulse-impact-must-not-teleport-in —
  rejects neighbouring-court phantom shots/serves the single-ball detector grabs
  when ours is occluded); **reliable `hitter_court_xy_ft`/`hitter_side`** from the
  hitting player's GROUND position (the airborne `impact_court_xy_ft` projection
  is garbage — court_y up to ~1900 ft on a 44-ft court); **serve de-duplication**.
- **Stage 7 v0.2.0** (`14def29`): **rally boundaries from the ball going OUT OF
  PLAY** (sustained not-in-play run), NOT `is_serve`/time-gap. KEY INSIGHT: during
  a point the ball is in flight (visible ~every frame, <0.25s absences); between
  points it's dead 3-4s. A missed shot leaves the ball flying → no false split.
  This is a **general physical signal**, the thing that finally made David's rally
  boundaries correct ("top labels are correct"). Side from `hitter_side`;
  zero-bounce end_reason → `unknown` on real ball.

**Residuals (all tied to deferred work, NOT new hacks):** serve→drive labels +
courtesy-feed-as-rally-start (need serve detection); drive↔drop/dink type errors
(2D depth-speed limit, need court-plane/3D ball speed); missed shots + mostly-
`unknown` end_reason (ball-detection recall). See `KNOWN_ISSUES.md`.

**Real-vs-synthetic adaptation pattern (applies to every remaining stage 6–11):**
1. **`is_user` from `track_roles.json`** (role 'user'), NOT players.parquet's
   click-only flag (empty in the no-clicks flow). Every stage reading is_user.
2. **Resolution scaling**: px thresholds × `frame_width/1920` (4K = 2×).
3. **fps scaling**: frame-count windows × `fps/30` (60fps = 2×).
4. **Real-world-phenomenon filters gated to real ball** (`ball_source=="real"`):
   the synthetic placeholder lacks the noise/handling, so gating keeps the
   synthetic smoke bars valid (e.g. Stage 5 net-side ball-handling rejection;
   Stage 5.5 y-flip-for-all + apex filter).
5. **Validation = operator spot-check overlays** (render markers on the video,
   David confirms) — there is no real-data ground truth to auto-grade against.
6. **Gotcha:** `(x > thresh)` on numpy floats yields a **numpy bool**; `if b is
   not True` then rejects everything (numpy True ≠ Python True). Wrap in `bool()`.

## Where the project is

- **Stages 1–11:** implemented, end-to-end runnable. Last run on the **synthetic**
  placeholder ball — every ball-derived output is a validated scaffold until
  re-run on the real ball.
- **Stage 4/4.5:** **v4 WORKING.**
  - Trained detector `data/models/ball_model_v4.pt` (720p TrackNet, 3-frame/9-ch):
    **val recall 0.90 same-court / 0.54 cross-court**, fp 0.02.
  - Inference: `stages/track_ball/track_ball_v4.py` (720p + trajectory
    post-processing) + smoke `test_track_ball_v4.py`. Validated vs ground truth on
    pb_2min frames 300–420: 39/40 balls, **median 4.9px at 4K**, 100% within 25px.
  - Production (full-clip) inference: `stages/track_ball/infer_v4.ipynb` (GPU/Colab,
    built by `tools/build_infer_v4_nb.py`). **Real full-clip `ball.parquet`
    produced for `data/pb_2min/`**: 7164 frames, 4418 visible + 426 interp + 2320
    not-visible, detect_frac 0.676, coords in-bounds, conf mean 0.78. Visually
    spot-checked on the longest rally — looks good.

## What was done this session (2026-06-11/12)

- **Drove `infer_v4.ipynb` on Colab end-to-end via the Claude-in-Chrome browser
  MCP** (Claude *can* drive Colab once the Chrome extension is connected) and
  produced the first real full-clip `ball.parquet` + `ball.meta.json` for pb_2min;
  downloaded + validated them locally; rendered a local overlay spot-check.
- **Fixed a T4 OOM:** the notebook hardcoded `BATCH=16`, which OOMs a 15GB T4 at
  720×1280. The builder now scales BATCH to GPU memory (T4→4, >20GB→8, >32GB→16).
  Committed **`1621541`** ("Stage 4 v4: T4-safe GPU batch size").
- **Logged two now-first-class requirements** (product reality: many ≥5-min videos,
  varied courts) → see `KNOWN_ISSUES.md`:
  - **Throughput:** full-clip inference is **CPU-decode-bound at ~2.9 fps**
    (~40 min for 2 min of 4K/60; ~100 min for 5 min). A background task to switch
    to GPU/hardware decode (NVDEC) was spawned.
  - **Cross-court generalization:** 0.90 same-court vs **0.54 cross-court** — must
    close before relying on the detector across indoor/outdoor venues.
- **Updated docs** (this commit): ARCHITECTURE.md (Stage 4/4.5 + pipeline status no
  longer "paused"), KNOWN_ISSUES.md (v4-landed update, synthetic-caveat-still-applies
  note, two new issues), this handoff.

## DONE 2026-06-13: Stages 1–3 on pb_2min (no clicking) + user-tracking fixes

pb_2min now has the full Stage 1–3 set: `court.json`/`court_zones.json` (operator
court clicks via `tools/mark_court.py`), `players.parquet`, `track_roles.json`,
`poses.parquet`. Three improvements landed, driven by "track the user extremely
well" + "no user-clicking":
- **No-click user ID** (`4b8e4b8`): operator only sets handedness/baseline/
  starting-corner; Stage 2.5 seeds the user geometrically from
  `user_starting_corner`. `user_clicks.json` is now an optional override (Stage 2
  + 2.5). `tools/mark_user.py` is the override clicker.
- **Appearance re-id** (`b348d98`): Stage 2.5 follows the user across ByteTrack
  ID swaps / gaps / side-switches by clothing-color + height. pb_2min user
  coverage 68% -> **85.5%** (visually verified on a role-overlay clip).
- **Role-aware pose** (`f349141`): Stage 3 takes `is_user` from the role `user`
  and poses every user track (incl. behind-baseline 1663). pb_2min: user pose
  6125 rows @ 99.1% detection, all 3 user tracks `[1, 1554, 1663]`.

**Product note (David):** the final app's input/setup flow must let the user
select their own handedness; the court/user inputs are product UI, not dev
fixtures (see memory `project_product_requirements`).

## DONE 2026-06-14: Stages 5 + 5.5 on the real ball (pb_2min)

Both operator-validated via spot-check overlays and committed (real-ball
adaptations per the pattern above):
- **Stage 5 (detect shots)** `8aa9164`, v0.2.0: 304 → 45 real shots (all real
  over-net strikes by David's eye). Adaptations: teleport-drop (don't crash on
  outliers), 4K resolution + 60fps scaling (the fps scaling collapsed 2–3
  duplicate detections/strike), is_user-from-roles, and a **net-side
  ball-handling filter** (real players catch/bounce/hold the ball between points
  = a sharp dir-change at a hand; every rally shot crosses the net, so consecutive
  same-side impacts = handling — keep the LAST of each run).
- **Stage 5.5 (detect bounces)** `740fac9`, v0.2.0: 135 → 16 bounces (4/4
  validated). Adaptations: scaling, **apex/off-court filter** (reject bounces
  projecting far off-court = ball in the air, not on the ground),
  **ground-contact refinement** (snap to lowest pixel_y for accurate far-court
  zones), and **y-flip-for-all on real ball** (a real bounce reverses vertical
  down→up; impulse-with-no-reversal = mid-air wobble). Deferred: bounce occluded
  behind the net is missed (ball-quality cap); is_at_feet edge case.

## DONE 2026-06-15: Stage 6 (classify shots) on the real ball (pb_2min)

Operator-validated via spot-check overlay and committed. `classified.json`: 45
shots, types {drive:14, drop:12, serve:8, dink:6, lob:4, reset:1}, **0 unknown**,
volleys 9. Real-ball adaptations (v0.2.0 → **v0.3.0**):
- **Volley decoupled from the bounce LIST → recall-focused local trajectory scan.**
  The precision-tuned Stage 5.5 bounce list under-detects → false volleys. The
  volley flag now scans the inter-shot ball directly for a **ground bounce =
  interior local peak in pixel_y** (ball momentarily lowest on screen, descends
  in + rebounds out). **Gotcha that cost a retry:** do NOT use the *global*
  pixel_y max — the segment starts at high pixel_y (previous contact is low on
  screen) and the arc apex is a pixel_y *minimum*; both must be ignored. Bounce
  list kept only as an occlusion fallback. Pipeline volleys 27 → 9; operator
  confirmed volley/bounced on a 7-shot window.
- **Lob requires below-drive speed** (a lob is lofted AND slow; fixed fast drives
  reading as lobs on the noisy ball).
- **Tweener arc-shape tiebreak** (16–25 ft/s dead-zone): flat=drive, lofted=drop;
  drained all 7 "unknown" types into the right bucket.
- **fps + resolution scaling** of the px/frame thresholds (4K/60fps).

**Three residuals logged in KNOWN_ISSUES (NOT fixable in Stage 6):**
1. **Depth/height corrupts pixel-speed** → a drive hit down-court reads as slow
   and mistypes as a drop (f3541: a real drive measured 4.2 px/f). Proper fix =
   **homography-projected court-plane ball speed** (also feeds Stage 8 metrics)
   or 3D. The arc-tiebreak only covers the 16–25 ft/s band.
2. **Serve labeling** depends on Stage 5 `is_serve` (f3470 missed → "drive"). Fix
   in **Stage 5**.
3. **Courtesy/between-point feeds** read as volleys (f3148) — correct but not a
   rally shot. Exclude in **Stage 7 (rally segmentation)**.

## NEXT STEPS (me, next session)

1. **Improve Stage 4 ball detection — the agreed high-impact "do it now" work**
   (before building Stages 8–11, to avoid compounding errors into every stat).
   **Refined diagnosis (2026-06-16):** the "62% ball-visible" was misleading —
   **in-rally recall is ~92%**; the dead-time drags the average down. The real
   miss is **FAST-BALL under-detection**: at a hard hit the ball moves ~250 px/f
   and is **lost to motion blur** (tracked max only ~67 px/f in the missed region
   vs 252 in clean rallies), so the shot has no ball at impact. An **impact-recovery
   experiment (gap-based) was built and REVERTED** — it only recovers *gap-hidden*
   shots, not these fast-ball misses (the ball is "visible" but slow/jittery, not
   absent). So the fix is genuinely **detector quality**, not a Stage 5 heuristic.
   **Plan = ONE combined retrain:**
   (a) **fast-ball / motion-blur recall** — label more hard-hit / blurred-ball
       frames (the same-court outdoor clips are a rich source);
   (b) **cross-court generalization** — add the different-court + indoor clips
       (closes the 0.90 same-court vs **0.54 cross-court** gap, a hard product
       requirement). Do both in one training pass, not several.
   **Operator data on hand (David, 2026-06-16):** 4 outdoor videos of the SAME
   (pb_2min) court + 1 outdoor video at a DIFFERENT court + 1 indoor video. The
   different-court + indoor are the generalization set; the same-court clips add
   fast-ball examples. (Training is GPU/Colab, operator-driven, like `infer_v4`.)
   Also fold in the **adjacent-court contamination** root cause (Stage 4
   single-ball — KNOWN_ISSUES); a court-aware/continuity detector helps recall +
   contamination together.
   **Resume here:** quantify the fast-ball failure modes on pb_2min (speed/blur
   correlation, where misses cluster) to target labeling, set up the
   label→retrain→validate loop, then drive Colab.
2. **Then Stages 8 → 9 → 10 → 11** on the real ball. In **Stage 8**, build
   **court-plane / height-aware ball speed** (KNOWN_ISSUES Stage 6 depth-speed) —
   the right speed signal for metrics; retro-improves Stage 6 drive/drop typing.
3. **Calibrate Stages 9/10** against real rallies (uncalibrated until now).
4. Stage 11 synthetic-ball watermark drops automatically once `ball_source != synthetic`.

**Real-ball boundary lesson (carry forward):** the GENERAL rally-boundary signal
is **ball-out-of-play** (sustained not-in-play run), not serves or time-gaps —
robust to missed shots. The same "use a physical signal, gate real-only, validate
by operator spot-check" pattern applies to every remaining real-ball adaptation.

Notes carried forward: Stage 2.5 user coverage 85.5% (rest is genuine off-frame
time); partner/opponent role-awareness + opp L/R continuity still geometric
heuristics (KNOWN_ISSUES) — revisit if downstream opponent stats look off.

Parallel / larger efforts (tracked in KNOWN_ISSUES + a spawned task):
- **Inference speedup** (GPU decode) — required for the real ≥5-min workload.
- **Cross-court training diversity** — add more indoor/outdoor courts.

## Key facts / gotchas

- **Production inference is GPU-only.** Local CPU torch is ~11 s/frame (~23h for a
  2-min clip). Use `infer_v4.ipynb` on Colab. Claude can drive it via Chrome MCP
  once the extension is connected.
- **`data/` is gitignored.** `ball.parquet`, `ball.meta.json`, `video.mp4`,
  `frames_720/`, the `ball.val_300-419.*` backups — all local, regenerable, NOT
  committed. The 120-frame validation slice is preserved as
  `data/pb_2min/ball.val_300-419.parquet`.
- **Colab upload gotcha:** uploading the whole `data\pb_2min` folder into
  `MyDrive/pb_infer/pb_2min/` doubles the path → `.../pb_2min/pb_2min/video.mp4`.
  Either upload just `video.mp4`, or set `CLIP='pb_2min/pb_2min'` in the notebook.
- **Chrome downloads** land in `C:\Users\hochh\Dropbox\My PC (DESKTOP-94DNBCT)\Downloads`
  (symlinked from `~/Downloads`); Dropbox briefly renames in-flight files to
  `<guid>.tmp` then restores the real name.
- **The 512×288 trap** (still true): never let inference silently downscale 4K to
  512×288 — it reshrinks the ball to ~2px. v4 runs at 1280×720.

## Things to NOT touch

- Don't re-attempt ball-detection v1/v2/v3; failures well understood (KNOWN_ISSUES).
- v1/v2 weights on Drive retained for reference.

## Bring this to the next session

    Continuing Pickleball-Analyzer-v2. Read docs/SESSION_HANDOFF.md,
    ARCHITECTURE.md, KNOWN_ISSUES.md, stages/finetune_ball_model/contract_v4.md
    before proposing anything.

    pb_2min has real ball (synthetic:false) run through Stages 1,2,2.5,3,5,5.5,6
    (court.json, players.parquet, track_roles.json, poses.parquet, shots.json,
    bounces.json, classified.json), each operator-validated. Next: Stage 7 (segment
    rallies) on the real ball, then 8-11. Follow the real-vs-synthetic adaptation
    pattern at the top of this handoff (is_user-from-roles, 4K/fps scaling,
    real-only filter gating, spot-check validation, numpy-bool gotcha). In Stage 7
    own the courtesy-feed exclusion; in Stage 8 build homography-projected
    court-plane ball speed (see KNOWN_ISSUES Stage 6 depth-speed). Then calibrate
    Stages 9/10. Also open: inference throughput (GPU decode), cross-court
    generalization, partner/opponent role-awareness, Stage 5 serve-flagging.

---

Generated at session end on June 14, 2026.
