# Player report redesign — decision record

**Status: IMPLEMENTED in `tools/build_report.py` on 2026-09-16.**
Designed and reviewed 2026-09-10; operator review: David, across both sessions.
The canvas remains the design record; the code is now the product.

The design lives as a Claude Design canvas:
<https://claude.ai/code/artifact/13043cb7-3b3a-44b3-81fd-8562bb992649>

Working files that produce it are in `docs/design/` — six artboards plus `canvas.json`.
To change anything: edit the `.dc.html`, re-seed, republish. The canvas is **not** the
implementation; `tools/build_report.py` still emits the old layout.

```bash
# re-seed after editing docs/design/*.dc.html  (needs the bundled design skill's helper)
node "<design-skill>/seed-canvas.mjs" --template "<design-skill>/payload.template.html" \
  --out player-report-redesign.html --title "Player Report Redesign" \
  --artboard Main.dc.html --artboard Categories.dc.html --artboard Heatmaps.dc.html \
  --artboard Points.dc.html --artboard Ladder.dc.html --artboard Phone.dc.html \
  --canvas canvas.json
```

Every number on the canvas is real `David2` output, refreshed 2026-09-11 after the rally
end-time cap (estimate **4.01**, band 4.0; it was 3.92 before rallies stopped counting the
seconds after each point as play). Nothing on it is invented or illustrative — an earlier
draft carried a placeholder "rating over time" chart and the operator cut it.

---

## 1. Who the report is for

Operator, asked directly: *"combination. Needs to be focused on a player using it once,
or they won't come back to it. And also, have them see progress over time. but in both of
those, I see the how we measured it in a collapsed section — good info, but only for those
interested."*

So: **a single-session reader is the primary audience**, progress is second, and every
measurement caveat goes behind a disclosure. That is why the `●◐○` element chips and the
"measured on N of M" lines — which used to sit in the main table — are now inside a
`<details>` block per category titled *How we measured this*.

**Progress is the cumulative estimate, not per-video.** Operator: *"agree, progress should
be the cumulative estimate, not by individual videos."* The per-video estimates across
David2 span 3.47–4.40, and the user gets 7–28 shots per video (median 11) — a per-video
trend line would show measurement noise and read as skill change.

## 2. The five sections

| artboard | section | what changed |
|---|---|---|
| `Main` | Hero + how settled it is | estimate, likely range, evidence accumulating |
| `Categories` | The seven categories | every measurement, ordered by leverage, coaching folded in |
| `Heatmaps` | Court + ball views | three-zone positioning, five toggleable ball views |
| `Points` | Watch the points | unchanged in substance |
| `Ladder` | USAPA ladder + how it's made | footnotes become four collapsibles |
| `Phone` | phone width (390) | proves the responsive behaviour |

The design system is lifted verbatim from the shipped `tools/build_report.py` CSS —
`--court #0f766e`, Iowan Old Style / Palatino headings, 14px card radii, 840px column.
This is a re-layout of the existing product, not a new visual language.

### Hero — "how settled this is"

The estimate plus an **evidence strip**: your shots accumulating 28 → 38 → 45 → 55 → 67 →
80 across the six sessions, per-session bars under a cumulative line. All real.

**A running estimate per session is NOT stored anywhere.** There is no history to plot,
which is why the placeholder was cut rather than fixed. If the rating-over-time view is
ever wanted, the pipeline needs to persist the estimate on each collection rebuild. Small
addition; nobody has asked for it yet.

### Categories — ordered by measured leverage

Operator: *"I assume you will include all the measurements for each category with those
within each category ordered by what moves rating fastest, as well as across categories."*

Every driver metric appears in a per-category table with three columns: **measurement /
where you are / worth, per 10 points**. Rows sort by worth; so do the categories.

The "worth" column is **measured, not judged** — see `tools/rating_leverage.py`, which
nudges one driver and re-runs the real scorers from `stages/rate/rate.py`. Reproduce with:

```bash
python -m tools.rating_leverage data/_collections/david2
```

This ordering disagrees with `improvement_plan.json`, which ranks by gap-to-target ×
weight and puts third shot first. Both are honest answers to different questions. The
canvas uses leverage because that is what was asked for, and keeps the plan's coaching
copy inside the cards.

### The four buckets — why a measurement can move nothing

This went through two wrong versions before landing. The final taxonomy separates the
**one** case that is a deliberate product decision from the **three** that are our
limitations:

| bucket | meaning | members (David2) |
|---|---|---|
| 🔴 USAPA rates it, we cannot see it yet | **a real hole in the rating** | unforced errors, put-aways, speed-ups, pace, spin, dink height |
| 🟠 What we have is a stand-in | related, but not the thing | court covered, ready position |
| 🟢 Nothing to gain right now | properly measured, temporarily inert | serves-in (ceiling), returns-in (floor), drop rate (too few) |
| ⚪ Context, not a skill grade | genuinely not graded by USAPA | shot counts |

**Only the last is a judgement about what USAPA cares about.** An earlier draft labelled
unforced errors "not rated on purpose — and never will be", which tells a player a real
weakness is not graded. It is graded: the ladder in this very report reads "still frequent
unforced errors" at 3.0, "fewer" at 3.5, "very few" at 5.0. See §4.

Inside 🟢, two opposite cases wear the same dash and must never be conflated:

- **Ceiling** — serves landing in, 9 of 9. The in-play scale runs 70%→100% and the player
  is *on* 100%. Improvement is arithmetically impossible; only slipping is.
- **Floor** — returns landing in, 4 of 7 = **57%**, which is *under* where the scale
  begins. Ten points reads as zero, but crossing 70% is worth up to **+0.06** on the
  overall rating — the largest single gain available anywhere in the report.

`tools/rating_leverage.py` distinguishes these automatically by probing further when the
10-point step reads zero.

### Heatmaps

Operator's direction, verbatim: *"a) court/player positioning should just show 3 colored
sections for use, partner, opponents and % time in each section... b) Ball landing heatmap:
need ability to toggle to different views... Color code: bounce = blue, Out - Red,
volleys - green"* — with the caveat *"need to verify that the info is there for these
views - if not, will need to skip those for now."*

**Positioning.** Three zones with percentages, per player. You and your partner are real.
**Opponents render an honest empty state** — `opp_b` has zero tracked frames and `opp_a`
reads 3% kitchen time, which is not a thing a pickleball player does. Role attribution on
the far side is not good enough to publish.

**Ball views are plotted marks, not heat.** Verified counts, in-rally only:

| view | points | basis |
|---|---|---|
| Serves & returns | 13 landings of 21 shots | landing bounce |
| Your volleys | 18 contacts, 10 with landings | hitter position (100% available) |
| Opponent mistakes | 17 | hitter position |
| Your errors | 14 | rally end + landing |
| Your winners | 5 | rally end + landing |

5–20 points per view is far too few for density shading, and plotting each mark is better
anyway: out-of-court and net balls plot naturally outside the lines, which is what was
asked for. Colours as specified — blue bounce, red out/net, green volley.

**Winners and errors rest on rally-end attribution and are labelled as weak** (see §4).

## 3. Category weights: intended vs actual

`rate.py` assigns fixed weights — strategy .20, third_shot .18, dink .15, volley .13,
serve_return .12, forehand .12, backhand .10 — then **confidence-weights the estimate**
(`weight × confidence`, renormalised). So the static weight is not what a category
contributes:

| category | meant | confidence | **actually carrying** |
|---|---|---|---|
| strategy | 20% | 0.98 | **39%** |
| third_shot | 18% | 0.48 | 17% |
| volley | 13% | 0.47 | 12% |
| dink | 15% | 0.38 | 11% |
| forehand | 12% | 0.30 | 7% |
| serve_return | 12% | 0.28 | 7% |
| backhand | 10% | 0.30 | 6% |

**David2's 3.92 is 39% positioning.** Strategy is the only category measured near
completely (32,487 frames of the user's feet), so it absorbs double its intended share
while the six ball-dependent categories collapse toward the 3.0 neutral prior.

The canvas shows **both numbers** on every card ("meant to carry 20% — carrying 39%
today") plus a bar strip at the top of the section. An earlier draft printed only the
static weight, which overstated how balanced the rating is.

Two standing risks recorded here because nothing else records them:

- **The weights are uncalibrated.** `rate.py` says so: *"UNCALIBRATED heuristics for rough
  skill importance"*, and `USAPA_REALIGN_DESIGN.md` agrees — *"encode rough skill
  importance at rec levels"*. They have never been checked against a known-rated player.
  This is the softest number in the rating.
- **Improving ball tracking will MOVE every rating, not just sharpen it.** Six categories
  are currently pulled toward the 3.0 prior by low confidence. As confidence rises they
  stop being pulled, so historical estimates will not be comparable across a tracking
  improvement. Worth understanding before real users accumulate history.

## 4. Unforced errors — the hole this design exposed

USA Pickleball rates unforced errors plainly; the ladder rows in this report turn on them.
`score_strategy` computes `unforced_error_rate` and deliberately excludes it:

> *"they're undetectable until ball recall improves; folding them in would either zero the
> confidence or read 'no errors = perfect'"*

That exclusion is **correct today** and should not be reversed by tuning:

- Rally-end reason is **8 of 23 (35%)** on outdoor-12 after the 2026-09-10 Stage 9 rebuild
  (`python -m tools.rally_end_score data/pb_5_minute_outdoor-12`).
- The dominant confusions are `6 net → not-returned` and `4 not-returned → net` — **ten of
  twenty-three pairs flip the winner/error attribution**, which is precisely the quantity
  the metric needs.
- The error is **biased, not random**: a missed error looks like clean play, so including
  it would systematically flatter every player, and flatter most the ones tracked worst.

The canvas shows the count (14 of 64 rallies) in the red bucket, stating that it belongs
in the rating and is missing from it.

**Rally-end accuracy unlocks four things at once**: unforced errors as a rated element,
the winners view, the errors view, and error attribution generally. That is the argument
for treating it as a foundation fix rather than an accuracy chore.

> **Update, later on 2026-09-10: end reason is now 14/23 (61%)**, after wiring the trusted
> net detector into the reason (see the ledger entry "RALLY END REASON 8/23 -> 14/23"). The
> 8/23 figures in this section and on the canvas are the pre-fix state. The exclusion of
> unforced errors from the rating has **not** been revisited: 61% is a large gain but still
> wrong on 9 of 23, and whether that is enough to score on is the operator's call. The
> canvas numbers that depend on end reasons (the unforced-errors row, the winners and
> errors views) need re-cutting against the rebuilt collection before implementation.

## 5. What was fixed to get here

`pb_5_minute_outdoor-12` had a **stale `rallies.json`** referencing shot ids 111–116 that
`classified.json` no longer contained — classify had been re-run during the pass-by filter
work, `segment_rallies` had not. Consequences: `stages.aggregate` crashed with `KeyError:
111`, so **the David2 collection could not be rebuilt at all** and "refresh the cumulative
report" was broken.

Fixed 2026-09-10 by re-running `ends → rallies → metrics → rate → plan → heatmaps →
report` on that member (the ends step feeds rallies, so both were needed), then rebuilding
the collection. An audit of all 20 session folders found **only this one** affected.

| | before | after |
|---|---|---|
| estimate | 3.90 | 3.92 |
| your volleys | 16 | 18 |
| volley subscore | 3.5 | 3.6 |
| your returns | 13 | 14 |
| third shots typeable | 3 | 2 |
| in-rally shots / bounces | 422 / 278 | 418 / 277 |
| rally-end reason | 7/23 | 8/23 |

Band unchanged at 4.0. **Lesson worth keeping: re-running `classify` without re-running
`ends` and `rallies` silently corrupts a session folder.** Nothing checks this. A cheap
guard would be for `segment_rallies` (or the aggregate stage) to assert that every
`shot_ids` entry exists in `classified.json`.

## 6. Implementation checklist — `tools/build_report.py`

Done 2026-09-16, operator decisions: all five ball views with the weak ones labelled, and one
review at the end rather than after each step. `tools/test_build_report.py` covers each piece
(10 tests) and the full suite is green.

- [x] Hero: evidence strip (sessions / minutes / rallies / your shots + cumulative SVG), bars
      labelled by the day filmed, drawn to their own scale so one session stays visible
- [x] Categories: per-category cards, ordered by measured leverage within and across categories
- [x] Leverage column via `tools.rating_leverage.leverage()` — never hardcoded
- [x] Intended-vs-actual weight strip via `category_shares()`, and on every card
- [x] `●◐○` chips and "measured on N of M" moved into `<details>` "How we measured this"
- [x] `improvement_plan.json` focus areas folded into their category cards
- [x] The four buckets replace `live`/`partial`/`planned` for what a measurement can move:
      `BUCKET` + `INERT_NOTE` in build_report.py say which of the three limitations, or the
      one deliberate exclusion, applies — including unforced errors as a stated hole
- [x] Positioning: three zones with percentages; a role is published only when its position
      confidence is >= 0.80 and its role is not contaminated (`ZONE_MIN_CONF`), so David2's
      opponents render the honest empty state rather than 3% kitchen time
- [x] Ball views: five plotted views with toggles (`ball_views`, `ball_views_svg`), replacing
      `landing_diagram_uri`; blue bounce, red out/net, green volley; the three that rest on
      rally-end attribution are labelled weak
- [x] Ladder + notes: four collapsibles, with the footnote ids kept so the in-report links work

**Bug found in review and fixed:** the layers were switched with the `hidden` attribute, which
browsers ignore on SVG elements — the buttons changed and all five views stayed drawn at once.
They toggle by CSS class now, with a test.

## 7. Still open

- The UI (7-panel wizard) design has not started. `docs/UI_PLAN.md` is the existing plan.
- More videos are being prepared for labelling.
- Whether to persist a running estimate per rebuild (§ Hero).
- Whether to calibrate the category weights against a known-rated player (§3).
