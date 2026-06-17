"""Build `pb_v4_upload.zip` for the Colab training notebook
(`stages/finetune_ball_model/finetune_v4.ipynb`).

The notebook unzips `MyDrive/pb_v4_upload.zip` to `/content/pb_v4/` and expects:
  repo/stages/track_ball/_tracknet_model.py   (the only code import: TrackNet)
  data/<clip>/v4_manifest.json + frames_720/*.jpg   (the training caches)

JPEGs are already compressed, so the zip is STORED (no recompression) — faster
to build and the same size. Re-run after re-caching any clip.

    python tools/build_v4_train_bundle.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pb_v4_upload.zip"
CLIPS = ["pb_2min", "pb_3min", "pb_4min", "pb_5min",
         "pb_3min_court2", "pb_3min_indoor"]
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
            jpgs = sorted((d / "frames_720").glob("*.jpg"))
            assert jpgs, f"no frames_720 jpegs for {clip}"
            for jpg in jpgs:
                z.write(jpg, f"data/{clip}/frames_720/{jpg.name}")
                n_jpg += 1
            print(f"  + {clip}: {len(jpgs)} jpegs")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e9:.2f} GB): "
          f"{len(CLIPS)} clips, {n_jpg} jpegs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
