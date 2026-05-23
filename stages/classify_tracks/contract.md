# Stage 2.5 — Classify Tracks (player roles)

**Status:** Contract APPROVED (2026-05-22), v1 IMPLEMENTED. NEW stage (not in
the original 11); runs after Stage 2, consumed by Stages 6+. Re-ID is multi-cue
(click-anchored motion continuity + perspective-normalized height [+ multi-region
clothing color]), so matching team kit doesn't break user/partner separation.
Opponents get provisional `opp_left`/`opp_right`. Folder `stages/classify_tracks/`,
output `track_roles.json`. Smoke test 5/5; user coverage 60% -> 98.6% on
test_clip.

**v1 scope note:** v1 ships the VIDEO-FREE cues (continuity + height + the
simultaneity "two people at once" constraint), which is what addresses the
matching-kit case. **Multi-region clothing-color matching is a documented
fast-follow** (not yet implemented) — it mainly helps the easier
different-colour case. The stage runs on `players.parquet` + `court.json` only.

## Purpose

Map the many ByteTrack `track_id`s in `players.parquet` to the four **logical
player roles** of a doubles match — `user`, `partner`, `opp_left`, `opp_right`
— plus `noise`, consistently across the whole match. One person is split across
many track_ids (ByteTrack swaps IDs on every crossing), so a "role" is a set of
track_ids over time, not a single id.

This is the stage `KNOWN_ISSUES.md` repeatedly points to (the Stage 2
court-switch ID-swap, the Stage 3 hard-coded scope filter, the Stage 6 non-user
handedness gap). It unlocks:
- **Complete user labeling.** Today `is_user` covers only ~60% of the clip
  (the clicked segments); linking the user's other track segments raises that,
  improving every user-centric metric in Stages 8–9.
- **Per-player analysis.** Stable roles let Stage 6 apply `roster.json`
  handedness to opponents/partner ("hit to the opponent's backhand") and let
  Stage 8 compute per-player stats.
- **Noise rejection** in one place (adjacent-court contamination), instead of
  each stage re-deriving a geometric filter.

**It runs on real player tracking only — no dependency on the (synthetic)
ball.** So unlike Stages 5–7, its output is durable real value now.

> **DECISION (folder/output name).** Proposed code folder `stages/classify_tracks/`
> (importable), output `track_roles.json`. `KNOWN_ISSUES.md` called the output
> `track_classification.json` — flag if you prefer that name.

## Place in the architecture

Inserts between Stage 2 and the rest (a true "Stage 2.5"), taking the 11-stage
pipeline to 12. Stage 3 currently re-derives its own scope filter; it could
later consume `track_roles.json` instead (a follow-up — Stage 3 already works,
won't be re-wired now).

```
players.parquet (S2) + court.json + user_clicks.json + roster.json [+ video.mp4]
        │
        ▼
   [2.5] classify_tracks ──► track_roles.json
```

## Inputs (per-video folder)

| File | Used for |
|---|---|
| `players.parquet` (S2) | per-track court positions, lifetime, `is_user`, `transient`, bbox |
| `court.json` (S1) | net line (`length_ft`/2), `user_baseline` (near/far), court bounds |
| `court_zones.json` (S1) | tracking-zone buffers (noise bounds) |
| `user_clicks.json` | the operator's user identifications → seed the `user` role |
| `roster.json` | (carried through for downstream; handedness isn't needed to assign roles, but the file's roles define the output vocabulary) |
| `video.mp4` | OPTIONAL — multi-region clothing-color features. Height + motion-continuity cues are video-free, so the stage still runs (lower confidence) without it. |

CLI: `python -m stages.classify_tracks.classify_tracks <video_folder> [--force]`
(+ threshold flags; + `--no-appearance` if appearance matching is the default).

## Output — `track_roles.json`

```json
{
  "schema_version": 1,
  "roles": {
    "user":      {"track_ids": [2, 1393, 2857, 4074], "n_frames": 7200, "court_frac": 0.0},
    "partner":   {"track_ids": [1, 3597], "n_frames": 6800},
    "opp_left":  {"track_ids": [1260, 3236], "n_frames": 3100},
    "opp_right": {"track_ids": [4, 3973], "n_frames": 2900}
  },
  "track_roles": {
    "2":   {"role": "user",    "confidence": 0.95, "basis": "click"},
    "1":   {"role": "partner", "confidence": 0.7,  "basis": "near-side-largest"},
    "4":   {"role": "opp_right","confidence": 0.6, "basis": "far-side-x"},
    "2863":{"role": "noise",   "confidence": 0.9,  "basis": "out-of-court"}
  },
  "noise_track_ids": [2863, "...hundreds of short/adjacent tracks..."],
  "stats": {
    "n_tracks": 835,
    "n_assigned": 18,
    "n_noise": 817,
    "user_frame_coverage": 0.82,
    "user_frame_coverage_was_is_user": 0.60
  },
  "params": {"...": 0},
  "warnings": [],
  "stage_version": "0.1.0",
  "completed_at_utc": "..."
}
```

Downstream reads `track_roles[track_id].role`. `is_user` in `players.parquet`
stays as the raw click-seed; the authoritative role is here.

## Method (v1 heuristic — honest confidence, not perfect re-id)

1. **Per-track summary + features.** For each track: frame count / lifetime,
   median `court_x_ft`/`court_y_ft`, `in_court` fraction, `is_user` fraction,
   time span, position-vs-frame path; plus the re-id features used in step 5 — a
   perspective-normalized height proxy (robust-percentile bbox height ÷
   pixels-per-foot at the track's court position) and, if video is available,
   upper- and lower-body HSV color histograms.
2. **Noise filter.** Mark `noise` if any: lifetime below a floor (e.g. < ~1 s),
   OR median `court_y_ft` outside `[-8, 44]` (adjacent-court / behind-gym),
   OR `in_court` fraction below a floor. (The data has 835 tracks; ~18 survive.)
3. **Side split.** Net at `length_ft/2` (= 22). Non-noise tracks with median
   `court_y_ft` < 22 → **near side** (user/partner pool); ≥ 22 → **far side**
   (opponent pool). (`user_baseline` confirms near = user's side.)
4. **Seed the user.** Tracks with any `is_user=True` rows → `user` (basis
   "click", high confidence).
5. **Extend the user + split partner — multi-cue, click-anchored.** The user is
   anchored at the click frames (hard ground truth). The hard part is linking
   the user's other near-side segments and separating them from the partner —
   **especially when teammates wear matching colors**, so color alone is not
   enough. Combine cues, in priority:
   - **Click anchors + motion continuity (always available, no video):**
     segments must chain into a physically-continuous path through the click
     anchors — a person moves smoothly and can't teleport, so each near-side
     person traces a continuous court path. This is the backbone; the two
     near-side people form two continuous paths, and the clicks pin which is the
     user.
   - **Perspective-normalized height / build (no video):** median/upper-
     percentile bbox height ÷ pixels-per-foot at the track's court position ≈ a
     real-height proxy (in feet), independent of clothing. Separates a taller
     user from a shorter partner even in identical kit. Noisy (players
     crouch/lunge) so used as a soft cue via a robust percentile.
   - **Multi-region clothing color (needs video):** SEPARATE HSV histograms for
     the upper body (top) and lower body (shorts/skirt), not one blended
     histogram — teammates often share a top color but differ in bottoms.
     Strong when colors differ; uninformative when identical (height +
     continuity then carry it).
   A near-side track joins `user` if its combined, click-anchored score beats
   the `partner` alternative; else `partner`. Confidence reflects cue agreement;
   when all cues are ambiguous (identical kit + height), fall back to continuity
   from the nearest click anchor at low confidence (see operator-in-the-loop
   follow-up).
   > **DECISION (resolved per review):** multi-cue — motion continuity + height
   > + multi-region color — NOT color-only, so matching team kit doesn't break
   > user/partner separation. Color needs the video; continuity + height are
   > video-free, so the stage still runs (lower confidence) without video.
6. **Opponents L/R.** Far-side tracks split by median `court_x_ft`: lower x →
   `opp_left`, higher → `opp_right`.
   > **DECISION (resolved per review):** provisional `opp_left` / `opp_right` by
   > median x (matches `roster.json`'s labels), at low confidence — opponents
   > also switch sides, so the L/R label is approximate until appearance
   > matching is extended to them.
7. **Confidence + emit.** Each track gets role + confidence + basis. Roles
   aggregate their track_ids and frame counts.

## Smoke test

`stages/classify_tracks/test_classify_tracks.py`, against `data/test_clip/`.
There is **no full role ground truth**, so the test combines partial truth +
consistency + the core value metric:

1. `track_roles.json` parses; every track has a role in
   {user,partner,opp_left,opp_right,noise} + confidence in [0,1]; roles
   aggregate consistently with `track_roles`.
2. **Click agreement:** every track with `is_user=True` rows is role `user`.
   (Hard requirement — contradicting the operator's clicks is a failure.)
3. **The 4 playing roles are populated**, and: user+partner tracks are
   predominantly near-side (median `court_y_ft` < 22), opponents predominantly
   far-side (≥ 22).
4. **Noise rejection:** tracks with median `court_y_ft` > 44 (adjacent court,
   e.g. tid 2863 at ~55) are `noise`.
5. **Core value — user coverage rises:** `user_frame_coverage` >
   `user_frame_coverage_was_is_user` (linking extended the ~60% click coverage).
   Bar: a meaningful increase (e.g. ≥ +10 percentage points), proving the
   re-identification actually did something.

> Hardcoded track_ids are avoided (they're ByteTrack-run-dependent); checks key
> off `is_user`, court position, and coverage deltas, which are stable.

## Edge cases / failure modes (loud where it matters)

- **Required input missing/malformed** → fail loudly. Output exists w/o
  `--force` → `FileExistsError`.
- **No `is_user` rows at all** (no user seed) → fail with a clear message
  (Stage 2 must have resolved at least one click first).
- **Fewer/more than 4 non-noise playing tracks pooled** (e.g. a singles clip, or
  heavy contamination) → still emit best-effort roles + a warning; don't crash.
- **Appearance matching unavailable** (no video / decode error) → fall back to
  position-only with a warning, don't fail.
- **A near-side track ambiguous between user/partner** → assign the
  higher-likelihood role with reduced confidence; never silently force.

## Out of scope (deferred)

- **Perfect multi-object re-identification.** v1 is a documented heuristic with
  honest confidence, not a solved re-id system.
- **Re-wiring Stage 3** to consume roles (it re-derives its own scope filter and
  works; a follow-up).
- **Back-propagating roles into a corrected `is_user`** in players.parquet
  (Stage 2 output is immutable; downstream reads `track_roles.json`).
- **Opponent appearance separation** if Option B (single `opponent`) is chosen.

## Known follow-ups

- When real ball detection (v4) + better footage arrive, re-validate roles on
  real data; appearance matching may need lighting-robust features.
- Consider letting the operator confirm/correct a few role assignments
  (operator-in-the-loop) for matches where appearance is ambiguous (similar
  clothing) — far cheaper than per-gap click-fixing.
- Once Stage 2.5 exists, Stage 3's hard-coded scope filter and Stage 6's
  is_user-only handedness mapping should both switch to consuming roles.
- **Multi-region clothing-color matching is not yet implemented** (v1 is
  video-free). Add it to strengthen user/partner and opponent separation in the
  different-colour case; height + continuity already cover the matching-kit case.
- **Opponent roles are contaminated.** Far-side adjacent-court players (court_y
  in 22–44, on a court behind) survive the noise filter, so `opp_left`/
  `opp_right` collect many extra tracks (~19 each on test_clip). Tighten with a
  per-track in-court-fraction threshold, far-side simultaneity/continuity (like
  the near side), or appearance once color matching exists. Opponent confidence
  is low (0.5) to reflect this.

## Architecture note

Adds Stage 2.5; update `ARCHITECTURE.md` from 11 to 12 stages on approval.
