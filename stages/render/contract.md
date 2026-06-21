# Stage 11 — Render Annotated Video

**Status:** DRAFT for review. The capstone presentation stage: draws everything
the upstream stages decided onto the **actual source video** and emits a
synchronized `timeline.json` event stream + standalone heatmap PNGs. **Pure
consumer** — it recomputes nothing; it only renders what other stages already
produced. Same pipeline philosophy (file-path I/O, loud failures, honest
synthetic-ball flagging — here as a burned-in watermark).

## Scope decisions (settled with the operator before drafting)

> **DECISION (annotate the real video + minimap inset).** The primary
> `annotated.mp4` is the ACTUAL footage with AR-style overlays (pixel space),
> PLUS a small top-down **minimap inset** in a corner (court-feet space) showing
> player-role dots + bounces. The schematic/top-down view also ships as
> standalone heatmap PNGs. (Alternatives considered: real-video-only, or a fully
> schematic top-down video — rejected; the hybrid keeps real-game context and
> the data-viz both.)

> **DECISION (heatmaps = standalone PNGs).** Stage 8 emits numeric heatmap grids
> and deferred *rendering* here. Stage 11 renders them as PNG files over a court
> diagram (player-position per role + ball-landing) — reusable by the deferred
> dashboard, no video bloat.

> **DECISION (rating/plan = HUD card + timeline).** A compact corner HUD card
> (rating band + estimate + top focus area, tagged provisional/synthetic) is
> burned into the video; the full rating + plan also go into `timeline.json` for
> the deferred dashboard.

> **DECISION (default layers).** Core layers always on: court lines, player
> boxes + role labels, ball marker, **ball trajectory trail**, shot markers,
> bounce markers, rally/end_reason banner, HUD card, synthetic-ball watermark.
> Optional, OFF by default (flag to enable): pose skeleton (`--pose`),
> shot-type text labels (`--labels`).

> **DECISION (older test video is fine for now).** Stage 11 is built + smoke-
> tested against the existing `data/test_clip/video.mp4` and its
> synthetic-ball-derived JSON. The annotated video carries a persistent
> SYNTHETIC-BALL watermark so a rendered demo is never mistaken for validated
> output. When real footage + ball v4 land, Stage 11 re-runs unchanged.

## Purpose

Make the analysis visible: a coach/player watches the real rally with the
court, players, ball, shots, bounces, rally outcomes, and their rating/plan
drawn on top — and a UI can later scrub a timeline of events. Stage 11 turns the
pile of JSON into something a human actually looks at.

## Place in the architecture

```
video.mp4 + ALL upstream JSON (court, players, track_roles, poses, ball,
  classified, bounces, rallies, metrics, rating, improvement_plan)
        │
        ▼
   [11] render ──► annotated.mp4 + timeline.json + heatmap_*.png
```

Per-video, file-path I/O, standalone CLI:
`python -m stages.render.render <video_folder>`.

> **DECISION (folder name).** Code + contract live at `stages/render/`
> (importable). This contract sits at the stub `stages/11_render/` for review;
> on approval it moves to `stages/render/contract.md` and the stub is deleted.

## Inputs

Per-video folder positional argument. **`video.mp4` + `court.json` are
required**; every other input is **optional** — if missing/malformed, its layer
is skipped with a warning (Stage 11 renders a partial overlay rather than
failing, since it may run on an incomplete pipeline).

| File | From | Used for |
|---|---|---|
| `video.mp4` | source | frames to draw on (required) |
| `court.json` | S1 | `homography.court_to_image` → court-line overlay + minimap; `video` dims/fps (required) |
| `players.parquet` | S2 | per-frame bboxes (pixel) + `court_x/y_ft` (minimap dots) |
| `track_roles.json` | S2.5 | `track_id → role` for box colors/labels + minimap |
| `poses.parquet` | S3 | pose skeleton (only with `--pose`) |
| `ball.parquet` + `ball.meta.json` | S4/synth | ball marker + trail (pixel); `synthetic` flag → watermark |
| `classified.json` | S6 | shot markers + type/stroke/volley (labels, timeline) |
| `bounces.json` | S5.5 | bounce markers (pixel + court) in/out/zone |
| `rallies.json` | S7 | rally banner (current rally + end_reason), timeline rally events |
| `metrics.json` | S8 | heatmap grids (v2-wrapped → unwrapped) → PNGs; headline metric confidence → timeline `metrics_confidence` |
| `rating.json` | S9 | HUD card + timeline summary (incl. per-dimension confidence/`limited_by`) |
| `improvement_plan.json` | S10 | HUD card top focus area + timeline summary (incl. `operator_considerations`, kept separate) |

CLI flags: `--force`, `--log-level`, `--start-frame`, `--end-frame`,
`--max-seconds` (cap render length; default full), `--fps-out` (default source
fps), `--pose`, `--labels`, `--no-trail`, `--no-minimap`, `--no-hud`,
`--heatmaps-only` (write PNGs + timeline, skip the video).

## Outputs

### `annotated.mp4`
The source frames (within the selected range) with overlays composited. Same
resolution as the source; encoded with OpenCV `VideoWriter` (`mp4v`). Layers:

- **Court lines** — project the court rectangle, sidelines, baselines, kitchen
  lines, and net through `court.json.homography.court_to_image`. Pure drawing
  (geometry from S1; not recomputed).
- **Player boxes + role labels** — `players.parquet` bbox at the frame, colored
  by `track_roles.json` role (user/partner/opp_left/opp_right; noise faint or
  off). Label text = role.
- **Ball marker + trail** — `ball.parquet` `pixel_x/y` where `visible`; a short
  fading trail of the last `TRAIL_FRAMES` positions. Drawn in a distinct
  "synthetic" style when `ball_source == synthetic`.
- **Shot markers** — at each shot's `impact_pixel_xy` (classified.json), shown
  for a few frames around the shot frame; with `--labels`, the `shot_type`
  (+ volley tag) is drawn.
- **Bounce markers** — at each bounce's `pixel_xy`, green=in / red=out, shown
  briefly around the bounce frame.
- **Rally banner** (top strip) — current `rally_id`, shot count, server role;
  at/after a rally's end frame, its `end_reason` (+ confidence).
- **HUD card** (corner) — rating `band` + `estimate` + range, and the #1 focus
  area; tagged `provisional`/`synthetic` when applicable. `--no-hud` disables.
- **Minimap inset** (corner) — a small top-down court (court-feet → mini
  canvas): role-colored player dots from `court_x/y_ft`, bounce markers from
  `court_xy_ft` (valid at z=0), net + kitchen lines. `--no-minimap` disables.
  (The mid-air ball is NOT projected to court here — it would be geometrically
  wrong; the ball appears on the minimap only at bounce frames.)
- **Synthetic-ball watermark** (persistent) — when `ball_source == synthetic`,
  a fixed banner "SYNTHETIC BALL — placeholder analysis" so no rendered demo is
  mistaken for validated output.
- **Pose skeleton** — only with `--pose`, from `poses.parquet` landmarks.

### `timeline.json`
The synchronized event stream for a scrubbable UI bar — pure data, the dashboard
contract.

```json
{
  "schema_version": 1,
  "source_video": "data/test_clip/video.mp4",
  "fps": 30.0,
  "frame_count": 8125,
  "rendered_range": [0, 8125],
  "ball_source": "synthetic",
  "duration_sec": 270.8,
  "summary": {
    "rated_role": "user",
    "rating": {"estimate": 3.69, "band": "3.5", "range": [3.0, 4.5], "confidence": 0.545},
    "rating_dimensions": [{"name": "net_play", "subscore_level": 2.78, "confidence": 0.99, "data_source": "real", "limited_by": "measurement"}],
    "target_band": "4.0",
    "focus_areas": [{"priority": 1, "dimension": "net_play", "confidence": "high"}],
    "operator_considerations": [],
    "metrics_confidence": {"rally_length_shots": {"confidence": 0.84, "n": 42, "limited_by": "sample_size"}, "shot_mix.by_shot_type": {"confidence": 0.62, "n": 218, "limited_by": "measurement"}},
    "synthetic_ball": true
  },
  "events": [
    {"frame": 1052, "t_sec": 35.07, "type": "rally_start", "rally_id": 0, "server_role": "user"},
    {"frame": 1060, "t_sec": 35.33, "type": "shot", "shot_id": 0, "role": "user", "shot_type": "serve", "is_serve": true, "is_volley": false},
    {"frame": 1080, "t_sec": 36.0, "type": "bounce", "bounce_id": 0, "is_in_court": true, "court_zone": "kitchen"},
    {"frame": 1183, "t_sec": 39.43, "type": "rally_end", "rally_id": 0, "end_reason": "ball-out", "end_reason_confidence": 0.85}
  ],
  "layers_rendered": ["court", "players", "ball", "trail", "shots", "bounces", "rally_banner", "hud", "minimap", "watermark"],
  "warnings": ["..."],
  "stage_version": "0.1.0",
  "completed_at_utc": "..."
}
```

- `events` sorted by `frame` then a stable type order. Event types:
  `rally_start`, `rally_end`, `shot`, `bounce`. Fields copied verbatim from the
  source JSON (pure consumer — no recomputation).
- `summary` carries the rating + plan top focus areas (the dashboard renders
  these; the video also burns the HUD card). **Foundation #3:** it also surfaces
  `rating_dimensions` (per-dimension `confidence`/`limited_by`/`data_source`) and
  `metrics_confidence` (headline match-metric `{confidence, n, limited_by}`) so a
  report can gate each number, and `operator_considerations` — the plan's
  operator items, carried verbatim and kept SEPARATE from player coaching (empty
  list ⇒ the report hides that section; empty on the synthetic ball).
- Events are emitted for the full clip by default (not just the rendered video
  range), so the timeline is complete even when the video is range-limited;
  `rendered_range` records what the video covers.

### `heatmap_*.png`
Standalone PNGs over a court diagram: `heatmap_position_<role>.png` per role
(from `metrics.json.heatmaps.player_position`) and `heatmap_ball_landing.png`
(from `heatmaps.ball_landing`). **Stage 8 `schema_version 2` wraps each grid as
`{value, confidence, n, limited_by}`; Stage 11 unwraps to `.value` before
rendering.** Normalized intensity colormap; court outline +
kitchen/net drawn for reference. These ARE the top-down/schematic view.

## Method

1. **Load + validate.** Require `video.mp4` + `court.json`. Load every other
   input defensively (missing → skip layer + warn; schema_version checked where
   present). Pull `ball_source` (classified/bounces/ball.meta) → watermark.
2. **Build per-frame indices** (so drawing is O(1)/frame): `frame → player
   rows`, `frame → shot`, `frame → bounce`, ball position array, rally lookup
   by frame range, role color map.
3. **Render video** over the selected frame range: read each frame
   (`VideoCapture`), draw enabled layers, composite minimap + HUD + watermark,
   write (`VideoWriter`).
4. **Build timeline.json** from rallies/classified/bounces + rating/plan
   summary (full clip, independent of video range).
5. **Render heatmap PNGs** from `metrics.json` grids over a court diagram.
6. **Write** outputs (refuse to overwrite without `--force`).

> **Pure consumer.** Stage 11 never recomputes a decision: rally `end_reason`,
> `shot_type`, roles, in/out, rating, plan — all copied from upstream JSON. The
> only geometry Stage 11 does is *drawing* (projecting known court coordinates
> through the known homography, colormapping known grids).

## Defenses against placeholder / bad data

- **Persistent synthetic-ball watermark** on the video + `synthetic_ball: true`
  in timeline + a `warnings[]` entry whenever `ball_source == synthetic`. The
  rendered video is the most "real-looking" output, so the caveat is the most
  important here.
- **Pure consumer** — cannot introduce new claims; if upstream is wrong, the
  overlay shows exactly what upstream decided (auditing value).
- **Missing optional input** → layer skipped + warning + omitted from
  `layers_rendered`; never a crash.
- **Missing `video.mp4`/`court.json`** → fail loudly.
- **Output exists without `--force`** → `FileExistsError`.
- **Codec/writer failure** → fail loudly naming the issue.

## Edge cases

- **Frame with no detections** → just the court lines + watermark.
- **Range flags** beyond clip bounds → clamp + warn.
- **Ball not visible / interpolated** at a frame → trail thins; no marker drawn.
- **Empty rallies/shots/bounces** → video still renders (court + players +
  watermark); timeline `events: []`.
- **No rating/plan** → HUD card shows "rating unavailable"; timeline `summary`
  minimal.
- **Very long clip** → use `--max-seconds`/range; full-clip render is allowed
  but slow (CPU draw + encode).

## Configuration (defaults)

```python
TRAIL_FRAMES        = 10       # ball trajectory trail length
SHOT_MARKER_FRAMES  = 6        # how long a shot marker stays visible (+/-)
BOUNCE_MARKER_FRAMES = 6
ROLE_COLORS = {"user": green, "partner": blue, "opp_left": red,
               "opp_right": orange, "noise": gray}
MINIMAP_W_PX        = 220      # minimap inset width
HUD_*, BANNER_* sizes = documented constants in code
HEATMAP_COLORMAP    = "inferno-like" (documented)
VIDEO_FOURCC        = "mp4v"
```

## Smoke test

`stages/render/test_render.py`, against `data/test_clip/`. Rendering can't be
graded pixel-exactly, so the test gates on **runs-without-crash + output
well-formedness + timeline reconciliation + pure-consumer invariants**, on a
SHORT frame range for speed (e.g. 150 frames).

Pipeline prefix: regenerate the chain (synth → … → S10) then run Stage 11 on a
short range.

1. **Runs + outputs exist.** `annotated.mp4`, `timeline.json`, and at least one
   `heatmap_*.png` are produced; exit 0.
2. **Video well-formed.** `annotated.mp4` opens with `VideoCapture`; frame
   count == the rendered range length; width/height == source dims.
3. **Timeline schema.** parses, `schema_version == 1`, events sorted by frame,
   every event `type` in the allowed set, required per-type fields present.
4. **Timeline reconciliation (pure consumer).** `shot` event count == shots in
   `classified.json`; `bounce` events == bounces in `bounces.json`;
   `rally_start`/`rally_end` counts == rallies in `rallies.json`; each event's
   copied fields equal the source (e.g. a shot event's `shot_type` ==
   classified.json's). No fabricated events.
5. **Synthetic propagation.** `ball_source == "synthetic"`, `summary.synthetic_
   ball == true`, watermark in `layers_rendered`, synthetic warning present.
6. **Heatmap PNGs.** Each written PNG opens, is non-empty, and has plausible
   dims (> court-diagram minimum).
7. **HUD/summary.** `summary.rating` + `summary.focus_areas` match
   `rating.json`/`improvement_plan.json` (copied, not recomputed).
7b. **Confidence surfacing (Foundation #3).** `summary.rating_dimensions` carry a
   valid `limited_by` + `confidence ∈ [0,1]`; `summary.metrics_confidence` has the
   headline match metrics as `{confidence, n, limited_by}`; `operator_considerations`
   is present and empty (suppressed) on the synthetic ball.
8. **Pure-consumer invariant.** Stage 11 does not modify any input file (assert
   the input JSON/parquet mtimes/sizes are unchanged across the run).
9. **Degradation.** Hiding an optional input (e.g. `rating.json`) → still
   renders, that layer omitted from `layers_rendered`, warning present, no
   crash.

## Stage version

`0.2.0` (Foundation #3): unwraps Stage 8 `schema_version 2` heatmap grids before
rendering, and surfaces per-dimension confidence/`limited_by`, headline
`metrics_confidence`, and the separate `operator_considerations` into
`timeline.json`'s `summary`. `timeline.json` output `schema_version` stays `1`
(additive summary fields). `0.1.0` was the initial version.

## Out of scope (deferred)

- **The dashboard / interactive scrubber UI** — `timeline.json` is its data
  contract; the UI itself is deferred to post-4.5 (ARCHITECTURE future section).
- **Audio, slow-mo, highlight reels, per-shot clip export.**
- **Schematic-only top-down video** (rejected in favor of minimap + heatmap
  PNGs).
- **Re-deciding anything** — Stage 11 only draws.
- **Multi-role HUD** — HUD shows the user (matches Stage 9/10).

## Known follow-ups

- **Render performance** — CPU draw + encode on a full clip is slow; a future
  pass could parallelize or GPU-encode. v1 supports range/`--max-seconds`.
- **Real footage** — when ball v4 + new footage land, re-run unchanged; the
  watermark drops automatically once `ball_source != synthetic`.
- **Overlay tuning** — colors/sizes/marker durations are constants; expose as
  flags if needed after visual review.

## Architecture note

Stage 11 was the final stage in the original pipeline diagram; this contract
takes it from "not started" to "implemented + smoke-tested" and **completes the
13-stage pipeline** (modulo the paused Stage 4/4.5 ball detection). On approval,
`ARCHITECTURE.md`'s "Stage 11: not started" becomes "Stage 11 implemented" and
the pipeline is end-to-end runnable on synthetic-ball data.
