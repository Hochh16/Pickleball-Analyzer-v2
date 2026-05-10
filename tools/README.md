# tools/

One-time scripts and helpers that are not pipeline stages. Stages live in
`stages/` and produce versioned outputs; tools live here and produce
artifacts that stages consume but that we don't regenerate every run.

## Inventory

### `verify_rally_frames.py`

Reads `data/<clip>/active_rally_frames.json` and writes annotated
thumbnails to `data/<clip>/rally_verification/` so the rally-range labels
can be eyeballed before they're used in a Stage 4 smoke test.

Usage:

    python tools\verify_rally_frames.py --clip data\test_clip

### `convert_dettor_weights.py`

Converts Andrew Dettor's pickleball-trained TrackNetV2 weights from
TensorFlow SavedModel format into a PyTorch .pt state-dict compatible
with the vendored model class at stages/track_ball/_tracknet_model.py.

This is a ONE-TIME conversion. After it succeeds, Stage 4 only ever
loads the resulting .pt file. Re-run only if Dettor publishes new
weights.

Format note:

Dettor's published weights are a TensorFlow SavedModel directory,
not a single .h5 file. The directory contains:

    <savedmodel_folder>/
        saved_model.pb
        fingerprint.pb
        keras_metadata.pb
        variables/
            variables.index
            variables.data-00000-of-00001

When downloading from his Google Drive folder, ensure all five files
land intact. The variables.index file is small (~15KB) and easy to
miss if Drive's bulk-download skips it; re-download individually if so.

Prerequisites:

1. Stage 4 code must be in place — specifically
   stages/track_ball/_tracknet_model.py must exist. The converter
   imports from this module to know the exact PyTorch architecture.

2. Dedicated venv with TensorFlow. Stage 4's main runtime venv
   (Python 3.14) does NOT include tensorflow and the project does not
   need it for normal operation. The converter is a one-time tool, so
   we use a separate Python 3.12 venv at .venv-convert/:

       py -3.12 -m venv .venv-convert
       .\.venv-convert\Scripts\python.exe -m pip install tensorflow torch numpy h5py

   The .venv-convert/ folder is gitignored. After conversion succeeds,
   it can be deleted to reclaim disk space; the produced .pt file is
   what Stage 4 actually uses.

3. Dettor's SavedModel folder. Download from
   https://github.com/AndrewDettor/TrackNet-Pickleball ("New Weights"
   Google Drive link). Extract to data/models/dettor_savedmodel/.
   This folder is gitignored.

Usage:

From the project root, using the conversion venv's Python:

    .\.venv-convert\Scripts\python.exe tools\convert_dettor_weights.py `
        --savedmodel data\models\dettor_savedmodel `
        --out        data\models\tracknet_v2_dettor.pt

On success, the script writes:
- data/models/tracknet_v2_dettor.pt — the converted PyTorch state-dict
- data/models/tracknet_v2_dettor.pt.meta.json — sidecar metadata
  (source SavedModel SHA-256, conversion timestamp, sanity-check log)

Stage 4 reads the .pt file via its --weights argument:

    python -m stages.track_ball.track_ball `
        --video data\test_clip\video.mp4 `
        --court data\test_clip\court.json `
        --weights data\models\tracknet_v2_dettor.pt `
        --out data\test_clip\ball.parquet

Failure modes:

The converter is built to fail loudly rather than produce silently-wrong
weights. Any of these will halt the script:

- SavedModel folder missing or unloadable — path wrong, or one of the
  five SavedModel files is missing.
- Layer count out of plausible range — the folder is not a TrackNetV2
  SavedModel.
- Conv/BN counts mismatch — architecture variant differs from
  _tracknet_model.py.
- Per-layer shape mismatch — specific layer has different filter count.
- Forward-pass produces all-NaN — conversion logic is broken.
- Forward-pass output abs-max greater than 100 — conv weights likely
  scrambled.

If any of these triggers, STOP AND DEBUG rather than working around
the error. A successfully-running Stage 4 backed by quietly-wrong
weights will produce plausible-looking parquet files of meaningless
ball positions, which is much harder to detect downstream than an
outright crash here.

Why this is needed:

Dettor published his model in TensorFlow SavedModel format (TF 2.x
default). Stage 4 uses PyTorch (mareksubocz/TrackNet model port,
MIT-licensed, exactly matches the V2 3-in-3-out architecture). The
two ecosystems use different tensor layouts for convolutional weights
(Keras: HWIO, PyTorch: OIHW) and different naming for batch-norm
parameters (Keras: gamma/beta/moving_mean/moving_variance, PyTorch:
weight/bias/running_mean/running_var). The converter handles both.

We chose conversion-once over runtime-tensorflow because it eliminates
a TensorFlow dependency from the runtime pipeline. Stage 4 only needs
PyTorch.