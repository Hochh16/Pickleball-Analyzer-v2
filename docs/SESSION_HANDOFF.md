# Session Handoff: mid-Stage-4.5

This document captures everything needed to resume Pickleball-Analyzer-v2
development at the natural break point in Stage 4.5: after the labeling
tool is delivered and the four-video corpus is set up and calibrated,
before the offline labeling work begins.

## Context for the next session

If you're a future Claude reading this, here's the project's working
agreement and what's been done so far.

### Project conventions
- Repo: github.com/Hochh16/Pickleball-Analyzer-v2 (local: `C:\Users\hochh\pickleball-analyzer-v2`)
- Windows + PowerShell + Python 3.14 main env
- Files sent ONE AT A TIME as single PowerShell copy-paste blocks
- File writes use `[System.IO.File]::WriteAllText(...)` with UTF-8 no BOM
- Working agreement: contract -> code -> smoke test -> commit
- Each stage is a standalone CLI script with file-path I/O. No DB, no Celery, no shared state.
- Read ARCHITECTURE.md and KNOWN_ISSUES.md before proposing anything.

### Stages complete
- Stages 1, 2, 3: committed and smoke-tested in earlier sessions.
- Stage 4 (track ball): code-complete and committed. Smoke test fails on detection rate due to Dettor's pre-trained weights not generalizing to user footage. See KNOWN_ISSUES.md.
- Stage 4.5 (fine-tune ball model): contract committed at `stages/finetune_ball_model/contract.md`. Labeling tool committed at `tools/label_ball.py`. Markers-clicker tool committed at `tools/mark_court.py` (replaces the React frontend originally specified in Stage 1's contract — that frontend was never built). Other Stage 4.5 sub-pieces (data prep, training notebook, validation) not yet written.

### What's queued for the next session
1. User has labeled 1000+ frames across 4 videos using `tools/label_ball.py`.
2. Build remaining Stage 4.5 sub-pieces: `prepare_training_data.py`, `finetune.ipynb`, `validate.py`, `smoke_test.py`.
3. User runs the data prep, uploads to Drive, runs Colab notebook, downloads new weights.
4. Re-run Stage 4 smoke test with new weights.

### Current corpus state (all calibrated)
| Folder | Source filename | Length | Role |
|---|---|---|---|
| `data/test_clip/` | indoor 2 vs ira.mp4 | ~4.5 min, 8125 frames | Stage 4 smoke-test target; primary training video |
| `data/indoor_b/` | indoor 1 diff court.mp4 | ~4 min, 7095 frames | Training video (different court) |
| `data/indoor_c/` | Indoor 3 vs vicky.mp4 | ~4.3 min, 7823 frames | Training video (different opponents) |
| `data/outdoor/` | Outdoor 1.mp4 | ~5 min, 9065 frames | Held-out validation video |

All four have video.mp4, markers.json, court.json, court_zones.json. None have ball_labels.json yet — that's the offline work below.

### Important discovery from this session
The earlier `data/test_clip/` was a 2-minute camera-adjustment clip with non-stable framing. Calibration on it succeeded but the camera moved during recording, making the homography invalid for most frames. That clip has been deleted and replaced with indoor 2 (stable camera throughout). The previous test_clip's diagnostic outputs (ball.parquet, smoke results, the 4.5% detection rate finding) are documented in KNOWN_ISSUES.md and were the basis for committing to Stage 4.5.

---

## What you (David) need to do offline before the next session

You'll do all of the following on your own time. Estimated: 4-8 hours of labeling spread across days. Calibration is already done.

### Step 1: Label each video with `tools/label_ball.py`

The Stage 4.5 contract specifies:
- Sample every 3rd frame (`--sample-every 3`, the default)
- Per-video target: 250-400 labels
- Per-video minimum: 200 labels
- Total target: 1000-1500 labels across all 4 videos

Commands (one per video):

    python tools\label_ball.py --video data\test_clip\video.mp4 --out data\test_clip\ball_labels.json
    python tools\label_ball.py --video data\indoor_b\video.mp4   --out data\indoor_b\ball_labels.json
    python tools\label_ball.py --video data\indoor_c\video.mp4   --out data\indoor_c\ball_labels.json
    python tools\label_ball.py --video data\outdoor\video.mp4    --out data\outdoor\ball_labels.json

Tool UX:
- Left-click = mark ball at click position; advance to next sampled frame.
- Spacebar (or right-click) = mark "ball not visible"; advance.
- Backspace (or left arrow) = go back one frame to fix a misclick.
- Esc (or close window) = save and quit.
- Auto-saves every 25 labels. Re-running with the same `--out` resumes from where you stopped.

Labeling pace: expect 10-15 frames per minute once you're in flow. 250 frames per video = ~20-30 minutes per video. Do them in chunks; don't try to do all four in one sitting.

### Step 2: Verify your labels before declaring done

Check that you've hit the per-video minimum and overall target:

    Get-ChildItem data -Recurse -Filter ball_labels.json | ForEach-Object {
        $j = Get-Content $_.FullName | ConvertFrom-Json
        Write-Host "$($_.Directory.Name): $($j.labels.Count) labels"
    }

Expected output: four lines, each showing >= 200 labels. Total should be 1000+.

If any video is below 200 labels, label more before declaring step 1 done.

### Step 3 (optional but recommended): Visual sanity check

Open one of your `ball_labels.json` files and spot-check a few entries by hand. The schema is:

    {
      "labels": [
        {"frame_idx": 30, "ball_visible": true, "pixel_x": 1234.5, "pixel_y": 678.9},
        {"frame_idx": 33, "ball_visible": false, "pixel_x": null, "pixel_y": null},
        ...
      ]
    }

If you accidentally clicked something far off the court, re-run the labeling tool with the same `--out` — it will resume mid-list and you can navigate backward with Backspace to fix specific frames.

---

## What to bring to the next session

Once steps 1-2 are done (step 3 optional), open a new Claude session and paste this as your first message:

    Continuing Pickleball-Analyzer-v2. Read docs\SESSION_HANDOFF.md, ARCHITECTURE.md, KNOWN_ISSUES.md, and stages\finetune_ball_model\contract.md before proposing anything.

    Status: I have completed Steps 1-2 of the handoff document. All four videos at data\<folder>\video.mp4 already have court.json from Stage 1. ball_labels.json files now exist for each.

    Per-video label counts (paste the output of the verification command from Step 2):
    test_clip: NNN labels
    indoor_b:  NNN labels
    indoor_c:  NNN labels
    outdoor:   NNN labels

    Ready for Stage 4.5 sub-piece 2 onwards: prepare_training_data.py, finetune.ipynb, validate.py, smoke_test.py.

The next session's Claude will pick up from there. Stage 4.5 sub-piece 2 (training data prep) is the next file to deliver.

---

## Things to NOT touch between sessions

- Don't modify `stages/track_ball/` (Stage 4 is code-complete; weights swap is the only change planned).
- Don't modify `stages/finetune_ball_model/contract.md` (already approved).
- Don't delete `data/models/tracknet_v2_dettor.pt` — even though it doesn't work well, we use it as a starting point for fine-tuning.
- The `.venv-convert/` venv can stay or be deleted — it's not needed until/unless we re-run the converter for new Dettor weights. Either way is fine.
- Don't re-run mark_court on any of the four videos unless you intentionally want to recalibrate. All four court.json files are good.

---

## How calibration was done (for reference if you ever need to add a fifth video)

This project has no React frontend. Stage 1 is a two-piece system: a clicker tool that produces markers.json, and a CLI that consumes markers.json and produces court.json. Both are wired together in `tools/mark_court.py`:

    python tools\mark_court.py --video data\<folder>\video.mp4 --out data\<folder>\markers.json

The tool opens a Tkinter window. Workflow:

1. Three dropdowns at top: dominant_hand (right/left), user_baseline (near/far — which baseline the user is on), user_starting_corner (left/right — which side of the court the user is on at the start of the video).
2. Frame slider at bottom — scrub to a frame where all 4 court corners and both kitchen lines are clearly visible (not blocked by players).
3. Click 8 points in order, watching the status bar for the next-point hint:
   1. Court corner: bottom-LEFT (in image)
   2. Court corner: bottom-RIGHT
   3. Court corner: top-RIGHT
   4. Court corner: top-LEFT
   5. User-kitchen line: LEFT endpoint
   6. User-kitchen line: RIGHT endpoint
   7. Opponent-kitchen line: LEFT endpoint
   8. Opponent-kitchen line: RIGHT endpoint
4. As points are added, dots and connecting lines render. Backspace undoes the last point.
5. Click "Save" — the tool writes markers.json AND runs Stage 1 calibrate to produce court.json + court_zones.json. A popup shows success/failure. The terminal prints the full calibrate output (`=== calibrate result ===` block).
6. Close the mark_court window (Esc or X) before running other PowerShell commands. The window blocks the launching shell.

Time per video: ~5-8 minutes including the click-corners work.

---

## Quick troubleshooting

- **Labeling tool won't open / Tkinter error:** verify with `python -c "import tkinter; print('ok')"`. Should print `ok`.
- **Video frame looks tiny on screen:** add `--max-display-fraction 0.95` to the launch command.
- **Video frame too big / off-screen:** add `--max-display-fraction 0.6` (or smaller).
- **Misclicked, want to fix:** Backspace to go back, then re-click.
- **Need to take a break:** Esc closes and saves. Re-launch later with the same `--out` to resume.
- **Calibrate failed (mark_court popup says FAILED):** the window stays open with your points placed; check the terminal for the full stderr from Stage 1, then re-mark points (try a clearer frame or more accurate corner clicks) and click Save again.
- **PowerShell blocked while mark_court window is open:** that's expected. Open a second PowerShell window if you need to inspect files concurrently. The terminal that launched mark_court is reserved for the calibrate output.

---

## What's working as of session end

- Repository structure verified.
- Stage 4 code complete; smoke test fails on detection rate due to weights, documented in KNOWN_ISSUES.md.
- Stage 4.5 contract approved and committed.
- Labeling tool delivered, smoke-tested, working on user's monitor.
- Markers-clicker tool delivered (replacing the React frontend that was never built); successfully calibrated all four videos with no warnings.
- Four-video corpus assembled at `data/{test_clip,indoor_b,indoor_c,outdoor}/`; all have video.mp4 + markers.json + court.json + court_zones.json.
- Dettor's weights converted to PyTorch (`data/models/tracknet_v2_dettor.pt`); will be the starting point for fine-tuning.

---

Generated at session end on May 10, 2026.