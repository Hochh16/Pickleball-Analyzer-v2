# Stage 5 — Detect Shots

**Status:** Contract APPROVED (2026-05-22), IMPLEMENTED. Rally hits are detected
by an impulse signal (single-frame turn-rate / speed-jump), not raw windowed
angle, so free-flight arcs (e.g. a lob's apex over a player's head) are not
mistaken for shots. Serves — which have no incoming ball — are detected by a
separate appearance-after-dead-time signal and flagged `is_serve`. Association
radius scales with player on-screen size.

## Purpose

Find every moment a player strikes the ball ("a shot") and emit a per-shot
record: which frame, which player, where the impact happened, and the ball's
motion immediately before/after. This is the event backbone the rest of the
pipeline hangs off — Stage 6 classifies each shot, Stage 7 groups shots into
rallies, Stage 8 computes per-shot metrics (speed, shot-type mix, error rate).

A shot is detected as a **sharp change in the ball's trajectory direction that
coincides, in space and time, with a tracked player**. This is the heuristic
named in the original stub ("ball direction change within ~50px of a tracked
player, ~0.2s window"); the rest of this contract makes it precise and
defensible against noisy/gappy real ball data.

## Place in the architecture

Consumes the outputs of Stages 1–4 for one video; produces `shots.json`.
Per-video, file-path I/O, standalone CLI — same rules as every other stage.

```
court.json (S1) + players.parquet (S2) + poses.parquet (S3) + ball.parquet (S4)
        │
        ▼
   [5] detect_shots ──► shots.json
```

> **DECISION (folder name).** Implemented Stages 1–4 live in *importable*
> folders (`stages/track_ball/`, not `stages/04_track_ball/`) because Python
> modules can't start with a digit. This contract sits at
> `stages/05_detect_shots/` to match the existing stub, but the **code** must
> live at `stages/detect_shots/`. Proposal: when this contract is approved,
> move it to `stages/detect_shots/contract.md` and implement there. Flag if
> you'd rather keep the numbered folder and use a different entry mechanism.

## Inputs

Per-video folder positional argument, matching Stages 2 & 3
(`python -m stages.detect_shots.detect_shots <video_folder>`).

The folder must contain:

| File | From | Stage 5 reads |
|---|---|---|
| `court.json` | Stage 1 | `homography.image_to_court` (impact → court ft), `video.fps`, frame size |
| `players.parquet` | Stage 2 | `frame`, `track_id`, `is_user`, `bbox_*`, `foot_x/y`, `court_x_ft/y_ft`, `transient` |
| `poses.parquet` | Stage 3 | `frame`, `track_id`, `left_wrist_*`, `right_wrist_*` (+visibility) for precise impact-to-hand association |
| `ball.parquet` | Stage 4 (or `synth_ball.py`) | `frame_idx`, `pixel_x`, `pixel_y`, `visible`, `interpolated` |
| `ball.meta.json` | Stage 4 (or `synth_ball.py`) | `video_fps`, `video_width/height`, and the **`synthetic`** flag (see Defenses) |

Optional tuning flags (all have defaults; see Configuration):
`--min-direction-change-deg`, `--impact-window-frames`, `--assoc-max-px`,
`--velocity-window-frames`, `--max-ball-speed-px-per-frame`, `--force`,
`--log-level`.

Stage 5 **does not** read `video.mp4` — everything it needs is in the parquet
files and `ball.meta.json`. (Cheaper, and keeps it runnable without the video.)

## Outputs

### `shots.json`

```json
{
  "schema_version": 1,
  "video_path": "data/test_clip/video.mp4",
  "fps": 30.0,
  "frame_width": 1920,
  "frame_height": 1080,
  "ball_source": "synthetic",
  "params": {
    "min_turn_rate_deg": 45.0,
    "min_speed_change_ratio": 0.35,
    "min_direction_change_deg": 45.0,
    "impact_window_frames": 6,
    "velocity_window_frames": 3,
    "assoc_bbox_height_frac": 0.5,
    "assoc_max_px": 120.0,
    "assoc_max_px_min": 30.0,
    "max_ball_speed_px_per_frame": 400.0,
    "serve_gap_frames": 21
  },
  "shots": [
    {
      "shot_id": 0,
      "frame": 1052,
      "t_sec": 35.07,
      "track_id": 2,
      "is_user": false,
      "is_serve": false,
      "detection_method": "impulse",
      "impact_pixel_xy": [613.4, 171.2],
      "impact_court_xy_ft": [15.9, 3.2],
      "player_distance_px": 24.6,
      "assoc_basis": "wrist",
      "pre_velocity_px_per_frame": [12.1, -3.4],
      "post_velocity_px_per_frame": [-9.8, -6.1],
      "speed_pre_px_per_frame": 12.6,
      "speed_post_px_per_frame": 11.5,
      "direction_change_deg": 138.7,
      "turn_rate_deg": 121.4,
      "speed_change_ratio": 0.09,
      "confidence": 0.82
    }
  ],
  "stats": {
    "n_shots": 14,
    "n_serves": 2,
    "n_candidate_inflections": 41,
    "n_rejected_no_player": 19,
    "n_rejected_ball_gap": 6,
    "n_rejected_low_speed": 2,
    "n_merged_duplicates": 5,
    "ball_visible_frac": 0.97,
    "analyzed_frame_range": [1000, 8124]
  },
  "warnings": [
    "ball_source is 'synthetic': shots are derived from PLACEHOLDER ball data and are not real detections."
  ],
  "stage_version": "0.1.0",
  "completed_at_utc": "2026-05-22T20:00:00Z"
}
```

Field notes:
- `shots` is **ordered by `frame` ascending**; `shot_id` is the index in that
  order, stable for Stage 7 to reference.
- `track_id` / `is_user` identify the striking player (from players.parquet).
- `is_serve` / `detection_method`: `detection_method` is `"impulse"` for a
  direction-change hit or `"serve_appearance"` for a serve. `is_serve=true`
  marks rally-opening serves (detected by a separate appearance signal — see
  Detection method §9). For serves, the impulse fields are null
  (`turn_rate_deg`, `speed_change_ratio`, `direction_change_deg`,
  `pre_velocity_px_per_frame`, `speed_pre_px_per_frame`), since a serve has no
  incoming ball; `post_velocity_px_per_frame` carries the launch velocity.
- `impact_pixel_xy` is the ball position at the impact frame.
- `impact_court_xy_ft` is that pixel projected through `image_to_court`.
  **Caveat:** the ball is in the air at impact, so this ground-plane
  projection is approximate (it answers "roughly where on the court", not a
  true 3D position). `NaN` if the homography projection is non-finite.
- `pre/post_velocity_px_per_frame` are the smoothed ball velocity vectors just
  before/after impact — handed to Stage 6 (drive vs dink vs lob) and Stage 8
  (ball speed) so they don't re-derive them.
- `direction_change_deg` is the *windowed* angle between pre and post velocity
  (0–180). Reported for downstream use; it is **not** the detection trigger.
- `turn_rate_deg` is the *single-frame* turn at impact (the impulse signature —
  see Detection method); `speed_change_ratio` is the fractional speed jump. One
  of these crossing threshold is what makes a frame a candidate shot.
- `confidence` ∈ [0,1]: blends impulse sharpness (`turn_rate_deg` /
  `speed_change_ratio`), player proximity, and ball-data quality around the
  impact (lower if the impact sits on/near interpolated or missing ball
  frames). Exact weighting is an implementation detail tuned against the smoke
  test.

There is **no separate `.meta.json`** — Stage 5's run metadata (params, stats,
warnings) lives inside `shots.json` because shots are a small JSON list, not a
big parquet. (Consistent in spirit with the other stages' "data + sidecar"
split; here the sidecar fields are folded in since the output is already JSON.)

## Detection method

1. **Load & validate the ball track** (see Defenses). Build an array indexed by
   `frame_idx` of `(x, y)` for frames where `visible OR interpolated`; mark the
   rest as gaps.
2. **Velocity estimation.** For each frame with a known position, compute
   `v_in` from positions over the preceding `velocity_window_frames` and
   `v_out` over the following window (finite differences, gap-aware). Windows
   that straddle a gap are shortened or, if too short, the frame is skipped as
   an impact candidate.
3. **Impact signature (impulse, not just total angle).** A paddle strike is an
   *impulse*: the ball's velocity turns sharply in ~1 frame and/or its speed
   jumps. Free-flight gravity also bends the path, but *gradually* — greatest at
   a lob's apex — and that must NOT be mistaken for a hit. So per frame compute:
   - `turn_rate_deg` = angle between the single-frame velocities `v[i-1→i]` and
     `v[i→i+1]` (how much the path turns in ONE frame), and
   - `speed_change_ratio` = `|speed_out − speed_in| / max(speed_in, speed_out)`.

   The windowed `direction_change_deg` (`v_in` vs `v_out`) is still computed and
   reported, but is **not** the trigger. A frame is a **candidate impact** if:
   - `turn_rate_deg >= min_turn_rate_deg` **OR** `speed_change_ratio >=
     min_speed_change_ratio` (the strike impulse), AND
   - ball speed on at least one side `>=` a small floor (rejects jitter when the
     ball is nearly stationary; counts toward `n_rejected_low_speed`), AND
   - it is a **local maximum** of `turn_rate_deg` within `impact_window_frames`
     (the ~0.2s window; at 30 fps, 6 frames).

   Because gravity's curvature is spread across many frames, its *per-frame*
   `turn_rate_deg` stays low even when the *windowed* `direction_change_deg` is
   large. So a lob sailing over a player's head (a smooth apex) is rejected,
   while the impulsive strike that launched it — near the striking player — is
   detected. A ground bounce IS impulsive, but is rejected by the player-
   proximity filter (step 6) because it happens away from any player; bounces
   landing at a player's feet are a known limitation (see Known follow-ups).
4. **Player association.** For each candidate, among players present on that
   frame in players.parquet (excluding `transient`), compute the distance from
   `impact_pixel_xy` to each player's **nearest wrist** (`left_wrist`/
   `right_wrist` from poses.parquet, using only landmarks with
   `visibility >= 0.5`). If no usable wrist, fall back to **bbox** (distance to
   bbox rectangle) then **foot point**. Associate to the closest player whose
   distance `<= assoc_max(player)`.
5. **Perspective-aware threshold.**
   `assoc_max(player) = clamp(assoc_bbox_height_frac * bbox_height,
   assoc_max_px_min, assoc_max_px)`. A flat 50px (the stub's number) is too
   tight for near players (bbox ~600px tall) and too loose for far ones;
   scaling by bbox height adapts to perspective. Defaults below; tuned against
   the smoke test.
   > **DECISION (association).** Wrist-first, bbox/foot fallback, threshold
   > scaled by bbox height. Alternative considered: project the ball impact to
   > court-space and threshold in feet — rejected because the in-air ball's
   > ground projection is unreliable, whereas pixel proximity to the hand is
   > what physically happens at contact.
6. **Reject non-player inflections.** Candidates with no player in range are
   **dropped** (counted as `n_rejected_no_player`). These are mostly **ground
   bounces** (direction change away from any player). Bounce detection proper
   is out of scope (see below).
7. **Deduplicate.** Collapse candidate impacts within `impact_window_frames` of
   each other to the single highest-scoring frame (`n_merged_duplicates`).
8. **Emit** impulse shots ordered by frame, projecting each impact to court
   coords, computing confidence (`detection_method="impulse"`, `is_serve=false`).
9. **Serve detection (separate signal).** A serve has no incoming ball
   trajectory (the ball starts at the server's paddle), so steps 1–8 are
   structurally blind to it. Instead, find the **start of a ball-visible run**
   that follows a not-visible gap longer than `serve_gap_frames` (dead time
   between rallies — distinct from a short mid-rally detection gap), that has an
   **outgoing launch** trajectory (speed ≥ the jitter floor) and a **player in
   range** (same association as step 4). Emit it as a shot with
   `is_serve=true`, `detection_method="serve_appearance"`, impulse fields null
   and `post_velocity` = launch. A serve frame that already coincides
   (±`impact_window_frames`) with an impulse shot is not double-emitted.

No multi-frame physics fit, no Kalman smoothing. Light gap-aware finite
differencing only — over-smoothing would erase the very inflection we detect.

## Defenses against placeholder / bad ball data

The handoff is explicit: downstream stages **must not silently accept whatever
the placeholder ball produces**. Stage 5 therefore:

1. **Requires `ball.meta.json`.** Missing/unparseable → fail loudly
   (`FileNotFoundError`/`ValueError`). It carries fps, frame size, and the
   `synthetic` flag.
2. **Surfaces the source.** `ball_source = "synthetic"` if
   `ball.meta.json["synthetic"]` is truthy, else `"real"`. When synthetic, a
   loud `warnings[]` entry and a `WARNING` log line state that all shots are
   placeholder-derived. Downstream stages read `ball_source` and propagate it.
3. **Validates physical plausibility** before detection:
   - **Teleport outliers → dropped, not fatal.** Any consecutive known pair
     displaced `> max_ball_speed_px_per_frame` is impossible motion. Real ball
     detection (unlike a clean synthetic placeholder) legitimately leaves a few
     residual outlier frames that survive Stage 4's postprocess; crashing the
     whole stage on one bad detection is wrong. So Stage 5 **drops** the later
     frame of each impossible pair (marks it not-visible — a gap), counts it
     (`n_teleport_dropped`), and warns; it does not raise. Processed
     left-to-right, this removes isolated spikes (both of a spike's pairs
     resolve from the one drop). The cap is resolution-scaled (see Configuration)
     and deliberately generous — a real fast smash is ~tens of px/frame at 1080p,
     well under it; only physically-impossible jumps are dropped. (Letting them
     through instead would fabricate shots from the velocity spike.)
   - **Schema invariants.** Re-assert Stage 4's invariants (exactly one of
     `visible`/`interpolated`, NaN rules). Violation → raise.
   - **Coverage warning.** If `ball_visible_frac` over the analyzed range is
     below a floor (default 0.30), still run but warn that shot **recall** will
     be poor (can't detect impacts the ball tracker never saw).
4. **Honest about gaps.** An impact that occurs *inside* a ball-detection gap
   is **undetectable** — and worse, Stage 4's short-gap linear interpolation
   replaces the sharp inflection with a straight line. Stage 5 lowers
   confidence for impacts adjacent to interpolated/gap frames and never invents
   a shot where it has no ball motion to measure. This is a known recall
   ceiling tied to Stage 4 quality (see Known follow-ups).

## Edge cases (loud where it matters, honest otherwise)

- **Ball all-NaN / no usable track.** Complete successfully, `shots: []`,
  `warnings` notes zero ball data. (Not a crash — an empty but valid result.)
- **No players on a candidate frame.** Candidate rejected (`n_rejected_no_player`).
- **Multiple players equidistant.** Pick the minimum distance; tie-break
  `is_user` first, then lowest `track_id`. (Rare; recorded deterministically.)
- **Impact during a ball gap / on interpolated frames.** Detect if possible
  with reduced confidence; otherwise miss it (counted toward gap rejects).
- **Degenerate homography** (court projection non-finite). Emit the shot with
  `impact_court_xy_ft = [NaN, NaN]`, add a warning. Do **not** fail the stage —
  pixel-space impact is still useful.
- **fps disagreement** between `ball.meta.json` and `court.json` → fail loudly
  (frame↔time conversions would be wrong downstream).
- **Required input missing/malformed** (any of the parquet/json) → fail loudly
  naming the file. No partial `shots.json`.
- **Output exists without `--force`** → `FileExistsError`.

## Configuration (defaults; tuned against smoke test)

```python
MIN_TURN_RATE_DEG          = 45.0   # single-frame turn = strike impulse signature
MIN_SPEED_CHANGE_RATIO     = 0.35   # sudden speed jump = strike impulse signature
MIN_DIRECTION_CHANGE_DEG   = 45.0   # windowed angle; reported + weak sanity, not trigger
IMPACT_WINDOW_FRAMES       = 6      # ~0.2s at 30fps; local-max + dedup window
VELOCITY_WINDOW_FRAMES     = 3      # frames each side for the reported windowed velocity
ASSOC_BBOX_HEIGHT_FRAC     = 0.5    # perspective-scaled association radius
ASSOC_MAX_PX               = 120.0  # upper clamp on association radius
ASSOC_MAX_PX_MIN           = 30.0   # lower clamp
MIN_BALL_SPEED_PX_PER_FRAME = 1.5   # jitter floor
MAX_BALL_SPEED_PX_PER_FRAME = 400.0 # teleport / corruption cap (~0.2*width)
BALL_COVERAGE_WARN_FRAC    = 0.30   # warn below this visible+interp fraction
MIN_SERVE_GAP_S            = 0.7    # not-visible gap before a serve appearance
REFERENCE_WIDTH_PX         = 1920   # resolution at which the px defaults were tuned
```

**Resolution scaling.** The px-space defaults above were tuned on 1080p footage
(`data/test_clip`). Because the app ingests footage at varied resolutions (e.g.
pb_2min is 4K/3840-wide), the pixel thresholds — `assoc_max_px`,
`assoc_max_px_min`, `min_ball_speed_px_per_frame`, `max_ball_speed_px_per_frame`
— are scaled by `res_scale = frame_width / REFERENCE_WIDTH_PX` (so 2× at 4K). The
**angle/ratio** triggers (`min_turn_rate_deg`, `min_speed_change_ratio`,
`min_direction_change_deg`) are scale-invariant and NOT scaled. An explicit CLI
override (`--assoc-max-px`, `--max-ball-speed-px-per-frame`) is taken as an
absolute px value (not re-scaled). `res_scale` and `reference_width_px` are
recorded in `shots.json.params`.

## Real-ball adaptations (v0.2.0)

Stage 5 was developed and tuned on the clean 1080p/30fps synthetic ball. v0.2.0
adds the adaptations needed for the real v4 ball (4K/60fps, noisy, gappy), all
validated on `data/pb_2min` (operator-confirmed via shot-overlay review). The
synthetic smoke bars are unchanged (these adaptations are no-ops or gated off on
synthetic data, so the logic is still validated there).

1. **Resolution scaling** (above): px thresholds scale by `frame_width/1920`.
2. **fps scaling**: the frame-count windows (`impact_window_frames`,
   `velocity_window_frames`) were tuned at 30fps; they scale by `fps/30` (→ 12
   and 6 at 60fps). At 60fps the unscaled 6-frame merge window was too short and
   emitted **2–3 duplicate detections per strike**; scaling collapses them.
   `fps_scale`/`reference_fps` recorded in params.
3. **Teleport outliers dropped, not fatal** (above): the residual impossible-
   motion frames real detection leaves are dropped (counted `n_teleport_dropped`).
4. **`is_user` from roles**: `is_user` is taken from `track_roles.json`'s role
   `user` (Stage 2.5), not players.parquet's click-only flag (empty in the
   no-clicks flow). Without it, every shot was mislabeled `is_user=false`.
5. **Ball-handling rejection (net-side alternation, real ball only).** The real
   ball includes the **between-points ritual** the synthetic never had — players
   **catching / holding / bouncing** the ball (each a sharp direction-change at a
   hand → a false "shot"). Physical rule: every rally shot **crosses the net**, so
   the striker's net side must alternate; a run of consecutive **same-side**
   impacts means the ball stayed on one side = handling. The filter keeps the
   **last** impact of each same-side run (handling precedes the real shot — you
   catch/bounce, then serve/hit) and drops the rest. Net side comes from each
   track's **median `court_y_ft`** vs the net (`length_ft/2`) — robust for every
   track, role-independent. Runs split on a side change or a gap >
   `handling_reset_frames` (= `HANDLING_RESET_S`·fps; a new rally). **Gated to
   real ball** (`ball_source=="real"`) because the synthetic generator doesn't
   model strict net-crossing alternation. Count: `n_rejected_handling`.
   - *Known limitation:* with very gappy ball, if an opposite-side shot falls
     entirely in a detection gap, the two real same-side shots around it can look
     consecutive and one may be dropped — and a player catching the ball *without*
     then hitting it leaves one residual false positive (kept as the run's last).
     Rally-context cleanup (Stage 7) is the natural place to refine this.

## Real-ball adaptations (v0.3.0): contamination, hitter position, serve dedup

Operator review of the full pb_2min clip (via Stages 6–7 overlays) surfaced
issues the per-shot overlay missed. v0.3.0 adds three fixes, all **gated to the
real ball** so synthetic smoke is unchanged.

1. **Adjacent-court contamination gates.** On a multi-court venue the single-ball
   detector grabs a **neighbouring court's ball** when ours is occluded, yielding
   phantom shots/serves (e.g. a "serve before the point", a "lob" from the court
   behind). Two trajectory-coherence gates reject them — calibrated on pb_2min
   with clean separation from real shots:
   - **Serve must launch a sustained run** (`min_serve_run_frames` =
     `MIN_SERVE_RUN_S`·fps): a real serve launches a ball run that persists; an
     other-court blip appears for 2–5 frames then vanishes. (Real serves ran
     12–267 frames; phantom blips 2–5.) Count `n_rejected_serve_blip`.
   - **Impulse impact must not TELEPORT in** (`teleport_in_px_per_frame`,
     res-scaled): the rally ball is spatially continuous; a neighbouring-court
     ball picked up mid-gap jumps in from far away. (Real impulse impacts: ≤49
     px/f run-entry jump; a phantom: 168 px/f.) Count `n_rejected_teleport_in`.
2. **Reliable hitter court-position (`hitter_court_xy_ft`, `hitter_side`).** The
   ball-contact projection `impact_court_xy_ft` is **garbage for shots**: the
   contact is **airborne**, and projecting an elevated point through the ground
   homography explodes toward the horizon (observed court_y up to ~1900 ft on a
   44-ft court). Every shot now also carries the **hitting player's GROUND
   position** (from `players.parquet`) and a `near`/`far` `hitter_side` derived
   from it. Downstream side logic (Stage 7) uses `hitter_side`, not the ball
   projection. (`impact_court_xy_ft` is retained for debugging only.)
3. **Serve de-duplication.** A point has one serve; two serve detections within
   `serve_dedup_frames` (`SERVE_DEDUP_S`·fps) with **no rally shot between** =
   a pre-serve artifact (e.g. the server bouncing the ball) plus the real serve.
   Keep the one whose ball run is longer (the launch that starts the rally).
   Count `n_serve_deduped`.

*Recall is ball-detection-limited.* On pb_2min the ball is detected in only ~62%
of frames (occlusion / motion blur), so some real shots have no ball at the
impact and are missed (`n_rejected_ball_gap` are the identifiable ones). This is
a **Stage 4** limit, not a Stage 5 algorithm gap; forcing shots out of gaps
reintroduces contamination. Closing it needs better ball-detector recall.

## Test fixture: synthetic ball generator (`tools/synth_ball.py`)

Because real ball detection is paused (Stage 4.5), Stage 5 is developed and
smoke-tested against a **synthetic** ball whose impacts are placed at **real**
player positions (so the heuristic has something true to find). `synth_ball.py`
is specified here only as far as Stage 5's test depends on it; full details go
in its own header.

Inputs: a video folder with `players.parquet` + `court.json` (+ frame
count/dims from either). Outputs into the folder:
- `ball.parquet` — **exactly** Stage 4's schema (`schema_version, frame_idx,
  pixel_x, pixel_y, visible, confidence, interpolated`), one row per frame.
- `ball.meta.json` — Stage-4-shaped **plus `"synthetic": true`** and a
  `"generator"` block (seed, params). No `weights_*` fields.
- `ball_synth_truth.json` — **ground truth** Stage 5's smoke test grades
  against. Each hit carries `is_serve` (true for a rally's first hit):
  ```json
  {
    "schema_version": 1, "synthetic": true, "fps": 30.0, "seed": 1234,
    "n_hits": 273, "n_serves": 46, "n_detectable": 227,
    "hits": [
      {"hit_id": 0, "frame": 1052, "track_id": 2, "is_user": false,
       "pixel_xy": [613.4, 171.2], "is_serve": false}
    ]
  }
  ```
  **Serves are undetectable by design.** A serve has no *incoming* ball
  trajectory (the ball starts at the server's paddle), so a direction-change
  detector cannot see it — physically true, not a synth artifact. Recall is
  graded on the **non-serve** population. The generator gives every non-serve
  hit both an incoming and an outgoing segment (including a follow-through
  after a rally's last hit) so they are all detectable.

Behavior: builds rallies (~one per 3–5 s of video) by choosing a sequence of
in-scope, non-transient, on-court players from players.parquet (using the
poses.parquet in-scope track set plus a court-position filter to exclude
adjacent-court contamination) and routing the ball between them, preferring
opposite sides of the net so the ball crosses the court. Each "hit" is placed
at paddle height within the striking player's bbox at a frame where that player
is actually detected, with smooth (gravity-flavored, sinusoidal-bump) arcs
between hits whose apexes are deliberately *gradual* (so they exercise Stage
5's impulse-vs-arc discrimination). Optionally injects realistic **gaps**
(`--gap-frac` drops a fraction of in-flight frames to `visible=False`) so Stage
5 is exercised against missing data, not just a perfect track. Deterministic
via `--seed`.

## Smoke test

`stages/detect_shots/test_detect_shots.py`, against `data/test_clip/` (which now
has real `players.parquet` + `poses.parquet`):

1. Run `synth_ball.py` (fixed seed) → `ball.parquet` + `ball_synth_truth.json`.
2. Run Stage 5 → `shots.json`.
3. Assert:
   1. `shots.json` parses, schema_version/fields/dtypes correct, shots sorted
      by frame, `shot_id` contiguous from 0.
   2. `ball_source == "synthetic"` and the placeholder warning is present.
   3. **Recall ≥ 0.80 on non-serve hits**: ≥80% of truth hits with
      `is_serve=false` have a detected impulse shot within
      ±`IMPACT_WINDOW_FRAMES` frames.
   4. **Player match** ≥ 0.80: of recovered non-serve hits, ≥80% have the
      correct `track_id`.
   5. **Precision** ≥ 0.70: ≤30% of detected shots are spurious (not matching
      any truth hit — serve or non-serve — within the window). In particular,
      arc apexes must not be emitted as shots.
   6. **Serve recall ≥ 0.70**: ≥70% of truth hits with `is_serve=true` have a
      detected shot flagged `is_serve=true` within ±`IMPACT_WINDOW_FRAMES`
      frames. (Lower bar than rally hits — the appearance signal is less robust
      than the impulse signal, and serve recall on real gappy ball data will be
      harder still.)
   7. With an **injected-gap** variant (`--gap-frac > 0`), the stage still
      completes, recall degrades gracefully (no crash), and impacts inside gaps
      are honestly missed rather than fabricated.

> Thresholds (non-serve recall 0.80 / player-match 0.80 / precision 0.70 /
> serve recall 0.70, ±6 frames) are the proposed acceptance bar on *synthetic*
> data — generous, because the point is to validate Stage 5's logic, not ball
> quality. They'll be revisited when real ball detection (v4) exists.

## Stage version

`0.3.0`. (0.1.0 initial → 0.2.0 real-ball adaptations [resolution/fps scaling,
teleport-drop, is_user-from-roles, ball-handling rejection] → 0.3.0
adjacent-court contamination gates + reliable `hitter_court_xy_ft`/`hitter_side`
+ serve de-duplication.) Increment minor for behavior changes preserving the
`shots.json` schema; bump `schema_version` for breaking schema changes. (v0.3.0
**adds** fields `hitter_court_xy_ft`/`hitter_side` — additive, schema kept.)

## Out of scope (deferred)

- **Bounce detection / in-out determination** — Stage 7 territory; needs the
  ground-contact + court-polygon reasoning Stage 5 deliberately avoids.
  Direction changes away from players are dropped, not classified.
- **Shot classification** (forehand/backhand, drive/dink/lob/serve/volley) —
  Stage 6. Stage 5 only localizes impacts, which player, and whether it's a
  serve.
- **3D / z-height reconstruction** of the ball — not from a single corner cam.
- **Multi-ball** — single ball by rule.

## Known follow-ups

- **Recall is capped by Stage 4 ball quality.** On real footage with gappy
  detection, impacts inside gaps are missed. Re-validate Stage 5 against real
  (noisy) ball trajectories when ball detection v4 lands; the synthetic-only
  acceptance bar will need a real-data counterpart.
- **Association threshold** may need per-region tuning if perspective is
  extreme; revisit if smoke-test player-match is low for far-court players.
- **Bounce/hit disambiguation** currently rests entirely on player proximity.
  If bounces near a player's feet get mis-attributed as shots, add a
  vertical-motion-signature check.
- **Serve detection on real data.** The appearance-after-gap signal works well
  on synthetic data, but on real gappy ball detection a long mid-rally
  occlusion could masquerade as a serve, and a serve could be missed if the
  ball isn't detected as it leaves the paddle. Consider adding a "server is
  behind the baseline" court-position check to reduce false serves, and
  re-tune `serve_gap_frames` against real footage.
```
