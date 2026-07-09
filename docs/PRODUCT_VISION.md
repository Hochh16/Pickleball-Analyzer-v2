# Product Vision — the analysis output a player should get

_Created 2026-07-07. The target spec for the consumer-facing output, grounded in
the OFFICIAL USA Pickleball skill framework. Drives the fix → realign → add → UI
program. Sources: [USA Pickleball skill levels](https://usapickleball.org/skill-level/)
(level pages 2–5)._

## Why this exists

The pipeline produces trustworthy-ish JSON, but a player can't judge or use JSON.
Rendering the real pb_2min output (Stage 8–11) immediately exposed correctness bugs
the confidence numbers missed (net-play kitchen % wrong; 8 vs 6 rallies) AND that the
rating's 6 homegrown dimensions **do not match the official USAPA standard.** This doc
fixes the target: what the output should be, aligned to USAPA, with every element
mapped to a metric we can or will measure.

## The official USA Pickleball framework (the standard we align to)

USAPA rates players across **7 categories** at levels **2.0 → 5.5+**:
**Forehand · Backhand · Serve/Return · Dink · Third Shot · Volley · Strategy.**
(Verbatim per-level criteria for 3.0/3.5/4.0/4.5 captured from the official level pages
— see the alignment table below for the descriptive elements.)

Skill ladder (measurable markers the app can use to place a player):

| Level | Defining criteria (official, condensed) | App-measurable marker |
|---|---|---|
| 2.0 | true beginner; can't reliably direct the ball | rally length <3, high miss rate |
| 2.5 | sustains a short rally; serves in; moves to NVZ late; pops up dinks | kitchen arrival late, dink pop-up rate |
| 3.0 | keeps ball in play; knows 3rd-shot-drop idea (inconsistent); chooses drop vs drive | unforced-error rate, drop-vs-drive mix |
| 3.5 | dinks moderately consistent; 3rd-shot drop with kitchen plan; basic stacking | dink-rally length, kitchen time % |
| 4.0 | deeper returns; drops with clean transition; resets from transition; reads attackable balls | reset success %, return depth |
| 4.5 | absorbs pace (blocks/resets); speed-ups at right targets; disciplined dinks | speed-up target accuracy, block % |
| 5.0 | drop/drive/hybrid selected correctly; resets under stress; precise speed-ups | shot-selection accuracy |
| 5.5+ | outcomes tier — dominance + tournament results | results-based |

## Rating aligned to USAPA — criteria → metric (● live · ◐ partial · ○ planned)

| Category | What USAPA rates (official descriptive elements) | Metric we map to it |
|---|---|---|
| **Forehand** | pace, directional control, consistency, depth | FH count ● · error/consistency ◐ · pace mph ○ · placement/depth ○ |
| **Backhand** | consistency, direction, tendency to avoid, depth/pace | BH count ● · avoids-BH ratio (run-arounds) ◐ · error rate ◐ · pace/depth ○ |
| **Serve/Return** | in-play consistency, depth, direction, pace, speed/spin variation | serve/return count ● · in-play % / faults ◐ · depth ○ · pace/spin ○ |
| **Dink** | rally sustain, height/depth control, consistency, pace variation, attackable recognition, offensive intent | dink count ◐ · dink-rally length ◐ · pop-up rate ○ · height/depth ○ · attackable attempts ○ |
| **Third Shot** | drop-to-net, soft/power mix, direction, placement "not easily returned" | 3rd-shot count ● · drop vs drive mix ◐ · drop landing depth ○ · transition success ○ |
| **Volley** | pace handling, control, block/re-set volley, swinging volley, overhead put-aways | volley count ◐ · block/reset ○ · put-away ○ · speed-up/counter ○ |
| **Strategy** | court positioning/NVZ approach, hard-vs-soft game, move as a team + coverage, stacking, target weakness, poaching, resets, unforced errors | zone times (kitchen/transition/baseline) ◐ · covers-middle / moves-as-unit ○ · stacking ○ · targeting ○ · resets ○ · unforced errors ○ |

**Body mechanics:** NOT a USAPA rating category. Footwork/weight-transfer live inside
Strategy + shot consistency. Kept as a planned **supporting** pose layer (split-step
timing, knees-bent-on-dinks, contact-point consistency, ready-position recovery,
paddle-up) that FEEDS the categories — not a standalone rated dimension.

## Full output sections (target report)

1. **USAPA rating** — band + estimate + range + confidence, by the 7 categories (● overview / ◐ most categories partial).
2. **Skill ladder** — where you sit + what the next level needs (●).
3. **Per-category detail** — each category's constituent stats (the alignment table above).
4. **Shot taxonomy** — every shot type × forehand/backhand × count × pace (mph) × depth/landing (serve, return, 3rd-shot drop, drive-baseline, drive-transition, dink, speed-up, reset, volley, lob) (◐/○ — needs court-plane speed + landing).
5. **Court-zone time** — kitchen / transition / baseline (◐ — needs the net-play/zone fix).
6. **Partner strategy** — covers-middle, moves-as-unit, stacking, shot selection (○).
7. **Technique (pose)** — the supporting layer (○).
8. **Opponent breakdown + targeting** — per-opponent weaknesses (○ — role-awareness + opponent pose).
9. **Coaching plan** — focus areas + drills, tied to the weakest categories (●).
10. **Annotated video + scrubbable timeline** — per-event, confidence-carrying (●).
11. **Trends across sessions** — rating + key stats over time (○ — cross-video identity).

## Build program (fix → realign → add → UI)

The vision reorders the roadmap around USAPA legitimacy. Dependency-correct order:

1. **FIX the live output so it's TRUE** — the ● items are currently buggy:
   - net-play zone %/positioning wrong (players at the kitchen read as ~0%) — court-zone + position→zone.
   - rally over-segmentation (8 vs 6 — 0.8s micro-rallies) — Stage 7 min-duration/shots filter.
   - finding language ("court coverage", "recover to middle") → plain English + good/bad context.
2. **REALIGN the rating to USAPA's 7 categories** — rewrite Stage 9 dims from the 6 homegrown ones to Forehand/Backhand/Serve-Return/Dink/Third-Shot/Volley/Strategy, scored from the ● metrics available, ◐/○ flagged not-yet-measured (confidence-gated as already built).
3. **ADD the ◐/○ metrics** toward alignment — each maps to a known roadmap item:
   depth/landing → bounce recall (C4); pace → court-plane/3D ball speed (F7/F8);
   FH/BH split → contact side from pose (F16); dink/speed-up/reset sub-types → Stage 6
   extension; opponents/targeting → role-awareness (F12) + opponent pose (C6);
   technique → pose stage (F17); trends → cross-video identity (F28).
4. **COMPLETE the UI** — the reporting skeleton (`tools/build_report.py`, JSONs → HTML)
   grows into the full report above; plus the input/setup UI.

Gated throughout by the cross-venue detector (#3, data-limited) for validation across
venues; pb_2min stays the provisional single-venue proving ground.
