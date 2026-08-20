"""Build `pb_v4_upload.zip` for the Colab training notebook
(`stages/finetune_ball_model/finetune_v4.ipynb`).

The notebook unzips `MyDrive/pb_v4_upload.zip` to `/content/pb_v4/` and expects:
  repo/stages/track_ball/_tracknet_model.py   (the only code import: TrackNet)
  data/<clip>/v4_manifest.json + frames_<H>/*.jpg   (the training caches)

The frame directory is read from each clip's manifest rather than hardcoded, so a 720p and a
1080p cache can coexist and this builds whichever the clip was last prepared at.

JPEGs are already compressed, so the zip is STORED (no recompression) — faster
to build and the same size. Re-run after re-caching any clip.

    python tools/build_v4_train_bundle.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pb_v4_upload.zip"
# The bundle carries EVERY clip; the notebook decides which are train vs held-out.
# indoor_B1/C1 added 2026-08-04 (operator labelled 1,555 frames each -> 3,407 + 3,629
# visible samples after densification). They are a DIFFERENT INDOOR FACILITY from
# pb_3min_indoor -- dark walls and blue courts ("PICKLR") vs a bright white-ceilinged
# gym with green courts -- so holding pb_3min_indoor out is a TRUE UNSEEN-VENUE test,
# not merely a different camera angle. Indoor visible samples 1,354 -> 8,390 (6.2x),
# taking indoor from ~13% of the training set to ~44%.
# 2026-08-20, the 1080p escalation: five clips (pb_3min, pb_4min, pb_5min, pb_3min_court2,
# pb_3min_indoor -- 4,136 labels, 22% of the set) have a frames_720 cache but NO video.mp4
# and no record of its source path, on disk or in Drive. They CANNOT be re-extracted, so the
# 1080p set drops them. Losing pb_3min_indoor also costs the true unseen-venue held-out clip
# described below; indoor_C1 has to serve that role instead.
# test_clip is EXCLUDED deliberately: it carries ball_synth_truth.json, so its labels come
# from the SYNTHETIC ball fixture, and it is the smoke test's own clip. Training on it would
# mean training on generated data and contaminating the fixture that validates the stage.
#
# indoor_b / indoor_c / outdoor are added. They are real footage at 1920x1080@30 rather than
# the 4K60 the product asks for, and were not in the 720p set. Two reasons to include them
# now. First, arithmetic: without them the recoverable set is 3,583 labels against the
# original 7,719, and with them it is 11,576 — larger than what the current model saw.
# Second, they mix cleanly at this resolution: a 4K source downscaled to 1080p puts the ball
# at 5.7-12 px, and these sources are ~6-12 px natively, so the model sees one size
# distribution rather than two.
CLIPS = ["pb_2min", "indoor_B1_3min", "indoor_C1_3min",
         "indoor_b", "indoor_c", "outdoor"]
REPO_FILES = ["stages/__init__.py",
              "stages/track_ball/__init__.py",
              "stages/track_ball/_tracknet_model.py"]


def main() -> int:
    n_jpg = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
        for rf in REPO_FILES:
            p = ROOT / rf
            assert p.exists(), f"missing repo file: {rf}"
            z.write(p, f"repo/{rf}")
        for clip in CLIPS:
            d = ROOT / "data" / clip
            man = d / "v4_manifest.json"
            assert man.exists(), f"missing cache (run prepare_v4): {clip}"
            z.write(man, f"data/{clip}/v4_manifest.json")
            import json as _json
            fdir = _json.loads(man.read_text(encoding="utf-8")).get("frames_dir", "frames_720")
            jpgs = sorted((d / fdir).glob("*.jpg"))
            assert jpgs, f"no {fdir} jpegs for {clip} (run prepare_v4)"
            for jpg in jpgs:
                z.write(jpg, f"data/{clip}/{fdir}/{jpg.name}")
                n_jpg += 1
            print(f"  + {clip}: {len(jpgs)} jpegs from {fdir}")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e9:.2f} GB): "
          f"{len(CLIPS)} clips, {n_jpg} jpegs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
