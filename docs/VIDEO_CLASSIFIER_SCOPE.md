# Scope: a learned "is this a paddle strike?" video classifier

Status: SCOPED 2026-09-30, not started. Decision rules below are fixed BEFORE any result is seen.

## Why this, and why now

Ten ideas were measured against unseen video in September and none beat the rules
(`docs/ACCURACY_LEDGER.md`). They all worked from hand-built numbers — kinks, speeds, arcs,
sound, pose. The last pre-check reframed the problem: **our junk is not a tracking failure.**
59% of it sits on ball tracking as clean as the real shots', with detector confidence that does
not separate the two (AUC 0.49). It is a real ball, really moving, that was not struck by a
paddle — a bounce, a pick-up, a hand, a contact on the next court.

That is a semantic question, and it is the one a pretrained video model is built for: learn from
the PIXELS around the moment, not from features we chose. It is also the only candidate
technology that is available today and aimed at this failure (WHAM was assessed and rejected the
same day; see the ledger).

## The data we already have

Stage 5 records every impulse candidate before any filter (`shot_discards.json`, "candidates"),
and the truth store labels each one real or junk. Nothing new has to be reviewed to start.

| video | role | candidates | real among them |
|---|---|---|---|
| outdoor-7 | dev | 716 | 110 |
| court C | dev | 372 | 59 |
| court B | dev | 351 | 80 |
| court A | HELD OUT | 637 | 95 |

Dev total: about **1,440 clips, about 250 real** (1 : 5). Court A is scored once, at the end.

## Is that enough data? — measured, not guessed

It cannot be known in advance, but it can be measured cheaply, and Phase 1 exists to measure it.
Two regimes matter:

- **Frozen pretrained features + a small classifier on top.** The video model is used as-is to
  turn each clip into a vector; only a logistic regression is trained. This commonly works with
  **hundreds** of labelled examples, IF the pretrained model already encodes the distinction (a
  hand or paddle meeting a ball, and what the ball does next). 250 positives is in range.
- **Fine-tuning the video model itself.** Typically wants **thousands** of examples per class to
  beat a frozen probe. We do not have that, and would not start there.

The instrument is a **learning curve**: train the probe on 25%, 50%, 75% and 100% of the dev
data, score each on the video left out, and read the slope.

| what the curve shows | what it means | what we do |
|---|---|---|
| beats the rules at 100%, still rising | signal is there; more data would add | Phase 2, and consider more reviewed video |
| beats the rules, flat | signal is there; data is not the limit | Phase 3 (court A) directly |
| below the rules, rising steeply | signal may be there; data-starved | extrapolate how many more videos it would take, then decide whether the reviewing is worth it |
| below the rules, flat | the pixels at this camera do not carry it either | stop; record it |

Each additional fully reviewed video adds roughly 60-110 real contacts and 350-700 candidates;
50 session folders exist locally, 4 reviewed. So "more data" has a known unit cost: one review
of the kind done for courts A and B.

## Phases

**Phase 0 — build the clip set (local, no GPU, ~half a day).** For every candidate, cut 16 frames
spanning about 0.5 s centred on it (every 2nd frame at 60 fps), cropped around the impact pixel
to include the ball and the nearest player, resized to 224×224. Label from the truth store
one-to-one within 0.35 s — the same matcher every other experiment used. Save as small clips plus
a manifest; zip for Drive. Seek to the frames needed rather than decoding whole 4K videos.

**Phase 1 — the sufficiency test (Colab GPU, ~2 hours).** Frozen pretrained backbone → one vector
per clip → logistic regression. Leave-one-video-out over the three dev videos, with the learning
curve above. Emit shots through the SAME emit rule the earlier classifier test used, and compare
found and junk against today's rule stack and against the earlier per-candidate classifier
(64 found / 26 junk on court A at its dev-chosen threshold).

**Phase 2 — only if Phase 1 says so.** Fine-tune the last blocks with heavy augmentation, and/or
add reviewed videos.

**Phase 3 — court A, once.** Threshold chosen on dev only. The bar, carried over from the
September classifier test: clearly beat the rules on both missed and junk shots (today 72 found /
36 junk of 97), and do not lose serves.

**Only after a court A win:** decide how it enters the pipeline — most likely as a scorer consulted
by the Stage 5 filters, run on Colab beside the existing vision pass (`tools/colab_vision.py`
already orchestrates GPU stages there).

## Choices and risks, decided up front

- **Backbone licence.** A shipped app cannot depend on a non-commercial model. The original
  VideoMAE release is, as far as I know, non-commercial — verify before using it for anything
  beyond the experiment. Default to torchvision's Kinetics-pretrained video models (MViT-v2-S,
  S3D, R(2+1)D; BSD-licensed code) so a win does not have to be redone.
- **Throughput.** A video model per candidate is cost on top of the vision pass: ~500-700
  candidates per 5-minute video. Measure time per clip in Phase 1.
- **What a win cannot do.** It chooses among existing candidates; it cannot recover the 2-4% of
  real shots that never become candidates, and it does not by itself fix shot TYPE.
- **Held-out discipline.** Court A is not read in Phases 0-2 except to build its clips. The drop
  override on 2026-09-22 gained 2 points on dev and lost 7 on court A; that is the failure this
  rule exists to catch.
