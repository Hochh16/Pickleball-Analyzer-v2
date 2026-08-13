# Stage 7.9 — Aggregate (cross-video union)

**Status:** CONTRACT DRAFT, not implemented. Open questions at the end need operator
answers before code.

Turns N analysed videos into ONE virtual session that Stages 8–11 then run on
**unchanged**. Produces no statistics of its own.

## The operator's principle

> "I'd like to see it work the same as if the multiple videos were all in one video."

That is the whole specification, and it is testable:

```
compute_metrics(union(A, B))  ==  compute_metrics(concat_video(A, B))
```

Everything below follows from taking it literally.

## Why this sits BELOW Stage 8, not above it

`ARCHITECTURE.md` § "Proposed — Cross-video trend tracking" proposes aggregating
`metrics.json` across sessions. **That cannot satisfy the principle**, and the reason is
arithmetic, not engineering:

- `rally_length_shots.median` — a median of medians is not the median. Same for `max`
  (recoverable) and any percentile (not recoverable).
- `mean_post_speed_ftps` — a mean of means is wrong unless re-weighted by each video's
  `n`, and `n` differs per metric.
- `confidence` — the `{value, confidence, n, limited_by}` envelope exists so small
  samples are marked untrustworthy. Two 6-rally videos aggregated should read as
  12 rallies with HIGHER confidence. Averaging confidences would keep it low, which is
  exactly backwards, and `limited_by: "sample_size"` would persist when the sample is no
  longer small.

The per-item streams do not have this problem. Union the streams, run Stage 8 once, and
every one of these is correct **by construction** rather than by a re-derivation we would
have to keep in sync with Stage 8 forever. Stage 8's formulas stay the single source of
truth; this stage adds no arithmetic that could disagree with them.

Consequence: Stages 8/9/10/11 need **no changes**. A cumulative report is the existing
report over a bigger input.

## Inputs

Per member video, the streams Stage 8 already consumes:

| file | key list | per-video IDs |
|---|---|---|
| `classified.json` | `shots[]` | `shot_id`, `track_id` |
| `rallies.json` | `rallies[]` | `rally_id`, `shot_ids[]`, `server_track_id` |
| `bounces.json` | `bounces[]` | `bounce_id`, `between_shots[]` |
| `players.parquet` | rows | `track_id`, `frame` |
| `track_roles.json` | `track_roles{}`, `roles{}` | `track_id` |
| `court.json`, `roster.json` | — | — |

Plus `session.json` for provenance and `metrics.json` only for version checks.

## Output

```
data/_collections/<collection_id>/
    collection.json      membership, order, provenance, warnings
    classified.json      unioned  ─┐
    rallies.json         unioned   ├─ byte-identical schemas to a real session,
    bounces.json         unioned   │  so Stage 8 cannot tell the difference
    players.parquet      unioned   │
    track_roles.json     unioned  ─┘
    court.json           copied from the reference member (see Venues)
    metrics.json         written by Stage 8, unchanged code
    rating.json          Stage 9
    improvement_plan.json Stage 10
    report.html          the cumulative report
```

Per-video folders are **never modified**. A collection is derived data and can be
deleted and rebuilt from its members at any time.

## Union rules

**IDs are renumbered, never reused.** Members are processed in chronological order and
each gets an offset block. `shot_id`, `rally_id`, `bounce_id`, `track_id` are rewritten
and every cross-reference (`rallies.shot_ids`, `bounces.between_shots`,
`rallies.serve_shot_id`, `rallies.server_track_id`, `track_roles` keys,
`players.track_id`) is rewritten with them. A dangling reference after renumbering is a
hard failure, not a warning.

**Time is offset, frames are not comparable.** Each member's `t_sec` and `frame` are
shifted by the running total so ordering is global and monotonic. `frame` is retained
only for provenance — after union it indexes nothing, because members have different
`fps` and different videos. Every stream row gains `video_id` and `video_frame` so any
number in the cumulative report can be traced back to a frame in a real file.

**`match_span_sec` is the SUM of member spans**, not wall-clock from first to last
video. Cumulative stats describe time on court, not calendar time. (A literal
concatenation would agree; the videos have no gap between them.)

**Roles are the join key, not tracks.** `track_id` is meaningless across videos; `user`
/ `partner` / `opp_a` / `opp_b` are what carry through. Stage 8 already attributes by
role, so this works unchanged — but see the open questions, because only `user` is
reliably the same human in every video.

**Reliability propagates worst-case, never averaged.** If any member has
`synthetic_ball: true`, the union is synthetic. `synthetic_gated` is the UNION of member
lists — a metric untrustworthy in one video is untrustworthy in the total. This is the
easiest thing here to get quietly wrong and the most damaging, because it would present
a contaminated number as clean.

## Collections and the operator's control

Requirements as stated: a video may join the running analysis or stand alone, and the
operator may start a fresh cumulative analysis from any point forward.

```
collection.json
{
  "schema_version": 1,
  "id": "...", "name": "...",
  "created_at": "...", "closed_at": null,
  "members": [{"session_id": "...", "captured_at": "...",
               "video_sha256": "...", "added_at": "..."}],
  "pipeline_fingerprint": {...},
  "warnings": [...]
}
```

- A session gains `collection_id` (absent/null = **standalone**, which stays the
  default — a video is never silently absorbed into a collection).
- Exactly one collection is **active** at a time; new videos offer to join it.
- "Start a new cumulative analysis" closes the active collection (`closed_at`) and opens
  a new one. Closed collections stay readable and rebuildable forever; nothing is
  deleted and no history is rewritten.
- A member can be removed. Membership is an ordered list, so add/remove is cheap.

**Adding a video triggers a FULL rebuild of the union, not an incremental update.**
Aggregation is arithmetic over a few thousand rows — a second or two, no video decoding.
Incremental accumulation would drift from the "as if one video" invariant the first time
a member is removed or re-run, and the drift would be silent. Full recompute makes the
invariant hold by construction.

## Ordering

Members are ordered by capture time, needed both for deterministic ID assignment and for
any future trend view. Source, in priority order: operator-supplied date → video
container creation metadata → file mtime. mtime is unreliable (copying a file resets it),
so when we fall back to it the collection records a warning and the operator can correct
the order. **Order affects only IDs and time offsets, never any statistic** — but it must
be stable, or two rebuilds of the same collection would produce different `shot_id`s.

## Guards

**Duplicate member.** `video_sha256` per member; adding the same video twice is refused.
Without this, double-counting is invisible and inflates every count.

**Pipeline version skew.** Every member records the ball model and stage versions it was
processed with. Mixing a video processed with the 0.90 baseline and one processed with
the run-4 model means mixing two different accuracy regimes into one number. The
collection records a `pipeline_fingerprint`; a mismatch raises a warning naming the
stale members and the re-run needed. **Open question below: warn or block.**

**Stale member.** If a member's own pipeline is re-run after it joined, the union is
stale. Detected by comparing each member's `completed_at_utc` against the union's build
time; stale members trigger a rebuild.

## Acceptance test (no new labelling needed)

The invariant is directly testable on data we already have:

1. Split `pb_2min` **at a rally boundary** into two clips.
2. Run the pipeline on each half independently.
3. Union them; run Stage 8.
4. Compare against Stage 8 on the whole clip.

Counts (`n_shots`, `n_rallies`, `n_bounces`), sums, distributions and heatmap grids must
match **exactly**. Means/medians must match to floating-point tolerance.

Splitting at a rally boundary is deliberate: split mid-rally and one rally becomes two,
so the halves genuinely do not sum to the whole. That is a property of cutting video, not
an aggregation bug, and the test must not confuse the two.

Second test, cheaper and stronger on the ID logic: union two real sessions, then verify
every `shot_ids` reference in `rallies` resolves, every `between_shots` resolves, every
`track_id` in `players.parquet` has a role, and member-wise sums equal the union's counts.

### Fixtures already on disk

These cover the awkward cases without preparing anything:

| session | shots | rallies | bounces | tracks | fps | ball | exercises |
|---|---|---|---|---|---|---|---|
| `pb_2min` | 39 | 6 | 15 | 335 | 60 | real | the split-and-recombine invariant |
| `pb_5_minute_outdoor-2` | 121 | 16 | 77 | 789 | 60 | real | scale; the acceptance-count clip |
| `pb_outdoor2_excerpt` | 22 | 5 | 18 | 146 | 60 | real | a second venue in one collection (Q4) |
| `test_clip` | 218 | 42 | 54 | 835 | **30** | **synthetic** | worst-case reliability + mixed fps |

`test_clip` is the important one for the two rules most likely to be got wrong quietly:
unioning it with any real-ball session MUST mark the whole collection synthetic, and its
30 fps against the others' 60 proves `frame` is not comparable after union while `t_sec`
is.

## Known limitations to record up front

- **Rallies cannot span videos.** Correct behaviour, but if the operator ever splits one
  session across two files mid-point, that point is lost. Worth a warning if a member
  starts or ends mid-rally.
- **A cumulative rating is only as good as Stage 9's category alignment**, which is
  already flagged as needing rework (`docs/PRODUCT_VISION.md`). Aggregation will make
  ratings *more* stable by growing `n`; it will not fix a mis-specified rating.
- **Between-point balls are still counted** (the accepted Stage 7 rally-end limitation).
  Aggregating N videos aggregates that error N times. Cumulative numbers will not be
  more accurate than per-video ones — only better sampled.

---

## DECISIONS (operator, 2026-08-13)

**D1. People are pooled by ROLE; individuals are not identified across videos.**

| cumulative bucket | contents | why |
|---|---|---|
| `user` | one human | the only person guaranteed constant; asserted per video in the UI |
| `partners` | every partner, pooled | team-side; who they were is not tracked |
| `opponents` | `opp_a` + `opp_b`, pooled | `opp_a` in video 1 is not `opp_a` in video 2 |

The operator asked not to distinguish partners. Taken as "do not identify individuals",
**not** as "merge partner into opponents" — the partner is on the user's team, and
`team.near` / `team.far` and error attribution ("my side's errors" vs "theirs") depend on
that split. Collapsing it would silently corrupt those.

Merging `opp_a`/`opp_b` cumulatively is a change FROM the per-video schema and the one
place this stage may not simply pass Stage 8's output through. Per-video reports keep
`opp_a`/`opp_b` unchanged.

**D2. A collection is all-time. Recency is the operator's control, not a weighting.**
No time decay, no "last N sessions". If a collection has gone stale the operator closes
it and starts a new one, which is exactly what the start-new-collection requirement is
for. Capture dates are still recorded, so a trend view remains possible later without
reprocessing.

**D3. Mixed pipeline versions produce a report FOOTNOTE, never a warning.** Too technical
to put in a player's face. The `pipeline_fingerprint` is still recorded in full so the
footnote can be generated and so the operator can see which members are stale; that
detail stays operator-facing (`collection.json`), not player-facing.

**D4. A venue whose measurement quality is materially worse is NOT merged.** It is
excluded from the collection and the report, and the UI says the venue is not supported
yet. Rationale: merging a venue at 0.98 recall with one at 0.59 yields a number that
describes neither, and a wrong number is worse than a missing one.

This needs a mechanism that does not exist yet — see **Venue supportability** below.

**D5. Identity is asserted by the operator, per video.** The UI already asks who to
analyse (`user_clicks.json`); it asks again for each added video. No appearance-based
re-identification across sessions — clothing, venue and lighting all change, and it is a
research problem we do not need to take on.

Consequences for the UI, since multiple people can be analysed:
- The collection list is **selectable** — there will be one collection per person.
- The list carries a persistent reminder that **each collection must be the same
  person**. Nothing in the data can verify this; it is an operator promise, so it must be
  visible at the moment of choosing.
- Adding a video offers: join the active collection / start a new collection / analyse
  standalone. Standalone remains the default.

## Venue supportability (required by D4, NOT yet built)

D4 gates on measurement quality, but a brand-new venue has no labels, so its recall is
unknown — the number we would gate on cannot be computed. It has to be **predicted from
unlabelled signals**:

- ball-vs-background contrast measured at detections (the metric that already predicts
  recall monotonically across six venues: 45→0.978, 36→0.978, 30→0.916, 25→0.916,
  20→0.713, 7.5→0.087)
- detection density versus expected shot rate
- track continuity / fragmentation
- mean and spread of detector confidence

Calibrate against the six venues whose TRUE recall is known, so the threshold is fitted,
not invented. Two uses for one mechanism: gating a venue out of a collection, and
answering "is this clip worth labelling?" before the operator spends hours on it.

Being honest about its limits: this predicts from six calibration points, one of which
is the failing venue. It will be a coarse three-bucket judgement (supported / marginal /
not supported), not a recall estimate, and it should be presented that way.

---

## RESOLVED — the questions these decisions answered

**Q1. Partners and opponents change between videos.** → **D1.** Pool by role, do not
identify individuals. Operator asked not to distinguish partners; read as "no person
identity", not "merge partner into opponents", since that split carries the team stats.

**Q2. Career-to-date or recent form?** → **D2.** All-time. Recency is handled by the
operator closing a collection and starting a new one, which is what that control is for.

**Q3. Mixed pipeline versions — warn or block?** → **D3.** Neither: a report footnote.
Too technical for a player-facing warning.

**Q4. Different venues in one collection.** → **D4.** Do not merge a materially worse
venue; exclude it and say so in the UI. Stronger than my default of merge-with-warning,
and better: a wrong number is worse than a missing one. Requires the venue
supportability check, which does not exist yet.

**Q5. What identifies "the same player" across videos?** → **D5.** The operator says so,
per video, through the existing identify-the-user step. Surfaced a requirement I had
missed: collections are per-person, so the UI needs collection selection plus a standing
reminder that a collection must not mix people.
