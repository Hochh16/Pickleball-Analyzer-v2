"""
Convert Andrew Dettor's pickleball-trained TrackNetV2 weights from
TensorFlow SavedModel format into a PyTorch state-dict (.pt) compatible
with the mareksubocz/TrackNet model class (vendored at
stages/track_ball/_tracknet_model.py).

NOTE on format: Dettor's Google Drive distributes weights as a SavedModel
DIRECTORY (containing saved_model.pb, keras_metadata.pb, fingerprint.pb,
and a variables/ subfolder), NOT as a single .h5 file. This converter
takes the SavedModel directory path, not a single file.

Usage (from a venv with tensorflow + torch installed; the project's
main Python 3.14 venv does NOT have tensorflow — use .venv-convert):

    .\.venv-convert\Scripts\python.exe tools\convert_dettor_weights.py ^
        --savedmodel data\models\dettor_savedmodel ^
        --out        data\models\tracknet_v2_dettor.pt

Output: a single .pt file containing the converted state-dict, plus a
sidecar .meta.json with the source folder path, conversion timestamp,
and the loud sanity-check results.

Failure modes (all loud, all halt the script):
  1. SavedModel folder missing or unloadable.
  2. Layer count plausibility check (Conv2D + BN total in expected range).
  3. Conv/BN counts mismatch between Keras and PyTorch reference.
  4. Per-layer shape mismatch (after Keras->PyTorch transpose).
  5. Forward-pass sanity: load converted state-dict into PyTorch model
     and run dummy input. If forward pass raises or output is all-NaN
     or absurdly out of range, halt.

This is a one-time conversion. After it succeeds, Stage 4 only loads
the resulting .pt file (via the project's main Python 3.14 venv, which
does NOT need tensorflow). The .venv-convert venv can be deleted after
this script succeeds.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

# Reduce TF chatter
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def info(msg: str) -> None:
    print(f"[convert] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_savedmodel(folder: Path) -> str:
    """Hash the saved_model.pb + variables.index + variables.data-* files
    in the folder. Stable identifier for the SavedModel as a whole."""
    h = hashlib.sha256()
    files_to_hash = ["saved_model.pb",
                     "variables/variables.index"]
    # Add all variables.data-* shards
    var_dir = folder / "variables"
    if var_dir.is_dir():
        for f in sorted(var_dir.iterdir()):
            if f.name.startswith("variables.data-"):
                files_to_hash.append(f"variables/{f.name}")
    for rel in files_to_hash:
        p = folder / rel
        if not p.exists():
            continue
        h.update(rel.encode("utf-8"))
        h.update(b":")
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


# ---------- SavedModel loading ----------

def load_keras_model_from_savedmodel(folder: Path):
    """Load a Keras model from a SavedModel directory and return it.
    Tries three load paths in order:
      1. tf_keras.models.load_model — the legacy-Keras compat package.
         Works for most TF 2.x SavedModels regardless of save-time TF
         version.
      2. tf.keras.models.load_model — modern Keras 3 path. Rejects
         legacy SavedModel format in TF 2.16+, useful only if Dettor
         re-saved in modern format.
      3. tf.saved_model.load + TFSMLayer wrapper — last resort, gives
         us the model as a raw object-graph that we walk via .variables.
    """
    try:
        import tensorflow as tf
    except ImportError:
        fail("tensorflow is required. This script must be run from a venv "
             "that has tensorflow installed (.venv-convert), not the main "
             "project venv. See module docstring for the exact command.")

    # Path 1: tf_keras (the legacy-API compat package)
    try:
        import tf_keras
        info("attempting load via tf_keras.models.load_model")
        model = tf_keras.models.load_model(str(folder), compile=False)
        info(f"loaded as Keras model via tf_keras.models.load_model")
        return model, "keras"
    except ImportError:
        info("tf_keras not installed; skipping that load path")
        e_tfk = None
    except Exception as e:
        info(f"tf_keras.models.load_model failed ({type(e).__name__}: {e})")
        e_tfk = e

    # Path 2: modern tf.keras (Keras 3 in TF 2.16+)
    try:
        info("attempting load via tf.keras.models.load_model")
        model = tf.keras.models.load_model(str(folder), compile=False)
        info(f"loaded as Keras model via tf.keras.models.load_model")
        return model, "keras"
    except Exception as e_k3:
        info(f"tf.keras.models.load_model failed ({type(e_k3).__name__})")

    # Path 3: tf.saved_model.load — gives us a raw object, not a Keras model.
    # We unconditionally try this because the TFSMLayer wrapper around a
    # raw SavedModel exposes `.variables` even when the older .layers /
    # .get_weights() API is unavailable.
    try:
        info("attempting load via tf.saved_model.load (raw SavedModel)")
        sm = tf.saved_model.load(str(folder))
        # Validate it has variables we can walk
        if not hasattr(sm, "variables") or len(sm.variables) == 0:
            fail(
                f"raw SavedModel at {folder} has no variables; "
                f"cannot extract weights."
            )
        info(f"loaded raw SavedModel with {len(sm.variables)} variables")
        return sm, "savedmodel"
    except Exception as e_sm:
        fail(
            f"could not load SavedModel at {folder} via any path.\n"
            f"  tf_keras error: {e_tfk!r}\n"
            f"  tf.keras (Keras 3) error: {e_k3!r}\n"
            f"  tf.saved_model.load error: {e_sm!r}\n"
            f"\n"
            f"This usually means the SavedModel was saved with a TF version "
            f"too different from the one in this venv. Options:\n"
            f"  - Try pinning the venv to TF 2.13 (matches the era Dettor "
            f"likely trained on).\n"
            f"  - Inspect the SavedModel with `saved_model_cli show` to see "
            f"what's actually in it."
        )


def extract_weighted_layers_from_keras(model) -> list:
    """Walk a Keras model in forward execution order, return a flat list
    of (kind, layer_name, weights_dict) where:
      - kind is 'conv' or 'bn'
      - weights_dict for conv: {'kernel': ndarray, 'bias': ndarray or None}
      - weights_dict for bn:   {'gamma', 'beta', 'moving_mean', 'moving_var'}
    Layers without trainable weights (ReLU, MaxPool, UpSampling, Concat,
    Input) are skipped."""
    flat = []
    for layer in model.layers:
        cls = layer.__class__.__name__
        weights = layer.get_weights()  # list of np.ndarrays
        if cls == "Conv2D" or cls == "Convolution2D":
            if len(weights) == 0:
                continue
            kernel = weights[0]
            bias = weights[1] if len(weights) >= 2 else None
            flat.append(("conv", layer.name, {"kernel": kernel, "bias": bias}))
        elif cls in ("BatchNormalization", "BatchNormalizationV1"):
            if len(weights) < 4:
                fail(
                    f"BN layer {layer.name!r} has only {len(weights)} weight "
                    f"tensors; expected 4 (gamma, beta, moving_mean, moving_var)"
                )
            flat.append(("bn", layer.name, {
                "gamma": weights[0],
                "beta": weights[1],
                "moving_mean": weights[2],
                "moving_var": weights[3],
            }))
        elif len(weights) > 0:
            info(f"  skipping unrecognized weighted layer "
                 f"{layer.name!r} ({cls}) with shapes "
                 f"{[w.shape for w in weights]}")
    return flat


def extract_weighted_layers_from_raw_savedmodel(model) -> list:
    """Fallback: walk variables of a raw SavedModel (no Keras structure).
    Returns the same flat-list format as extract_weighted_layers_from_keras.

    Variable names in a SavedModel typically look like:
      "conv2d_3/kernel:0"
      "batch_normalization_5/gamma:0"
    But TF 2.x sometimes uses names like:
      "tracknet/conv2d_3/kernel:0"  (with a model-name prefix)
      "Variable:0", "Variable_1:0"  (anonymous variables)

    We strip any leading prefix-up-to-the-last-second-slash, then group
    by the layer-name component. Anonymous variables are reported but
    cannot be matched to layer types, so the script halts.

    Variables are returned in the order they appear in `model.variables`,
    which TF guarantees to be deterministic across runs of the same
    SavedModel (matching the order they were created in the original
    Keras model definition — i.e. forward execution order).
    """
    flat = []
    if not hasattr(model, "variables") or len(model.variables) == 0:
        fail("SavedModel has no variables; cannot extract weights")

    # Group vars by layer name, preserving first-seen order across layers.
    by_layer = {}  # OrderedDict-like; in Py3.7+ dict preserves insertion order
    anonymous = []
    for v in model.variables:
        full = v.name
        # Drop ":N" suffix
        no_colon = full.split(":", 1)[0]
        # Need at least one slash to split into layer/param
        if "/" not in no_colon:
            anonymous.append(full)
            continue
        # Split into layer-path + param-name. Use rightmost slash so that
        # nested name scopes (e.g. "tracknet/conv2d_3/kernel") still group
        # correctly under "tracknet/conv2d_3".
        layer_path, suffix = no_colon.rsplit("/", 1)
        if layer_path not in by_layer:
            by_layer[layer_path] = {}
        by_layer[layer_path][suffix] = v.numpy()

    if anonymous:
        fail(
            f"SavedModel contains {len(anonymous)} variable(s) without a "
            f"layer-name prefix (first 5: {anonymous[:5]}). Cannot determine "
            f"which layer they belong to. Likely the model was saved with "
            f"unconventional naming."
        )

    info(f"grouped {len(model.variables)} variables into "
         f"{len(by_layer)} layer-paths")

    # Classify each layer-path by which param suffixes are present.
    # Conv2D: 'kernel' (+ optional 'bias')
    # BatchNorm: 'gamma', 'beta', 'moving_mean', 'moving_variance'
    n_skipped = 0
    for layer_path, params in by_layer.items():
        keys = set(params.keys())
        if "kernel" in keys:
            # Conv2D-like
            if params["kernel"].ndim != 4:
                # Could be a Dense layer or something else with a 'kernel'
                info(f"  skipping {layer_path!r}: 'kernel' is "
                     f"{params['kernel'].ndim}D, not 4D Conv2D")
                n_skipped += 1
                continue
            flat.append(("conv", layer_path, {
                "kernel": params["kernel"],
                "bias": params.get("bias"),
            }))
        elif {"gamma", "beta", "moving_mean", "moving_variance"} <= keys:
            flat.append(("bn", layer_path, {
                "gamma": params["gamma"],
                "beta": params["beta"],
                "moving_mean": params["moving_mean"],
                "moving_var": params["moving_variance"],
            }))
        else:
            info(f"  skipping {layer_path!r} with param set {sorted(keys)}")
            n_skipped += 1

    if n_skipped > 0:
        info(f"skipped {n_skipped} non-Conv/non-BN layer-paths")
    return flat


def build_pytorch_reference():
    """Build a fresh PyTorch TrackNetV2 model (3-in-3-out, 9-in, 3-out)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        fail("torch is required. Install with: pip install torch")

    # Add project root to sys.path so we can import the vendored model.
    # The script is in tools/, project root is one level up.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from stages.track_ball._tracknet_model import TrackNet
    except ImportError as e:
        fail(
            f"Could not import stages.track_ball._tracknet_model: {e}\n"
            f"Project root added to sys.path: {project_root}\n"
            f"Verify that {project_root / 'stages' / 'track_ball' / '_tracknet_model.py'} exists."
        )

    return TrackNet(in_channels=9, out_channels=3)


def keras_to_pytorch_conv_weight(arr: np.ndarray) -> np.ndarray:
    """Keras Conv2D kernel layout: (H, W, in_channels, out_channels).
    PyTorch Conv2d weight layout: (out_channels, in_channels, H, W)."""
    if arr.ndim != 4:
        fail(f"Expected 4D conv kernel, got shape {arr.shape}")
    return np.transpose(arr, (3, 2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--savedmodel", type=Path, required=True,
                    help="Path to Dettor's SavedModel folder "
                         "(e.g. data/models/dettor_savedmodel).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output .pt path. A sidecar .meta.json is also written.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output if present.")
    args = ap.parse_args()

    if not args.savedmodel.is_dir():
        fail(f"SavedModel folder not found or not a directory: {args.savedmodel}")
    if not (args.savedmodel / "saved_model.pb").exists():
        fail(f"{args.savedmodel} does not contain saved_model.pb; "
             f"is this really a SavedModel directory?")
    if args.out.exists() and not args.force:
        fail(f"Output file exists: {args.out}. Use --force to overwrite.")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    info(f"loading SavedModel from {args.savedmodel}")
    model, load_kind = load_keras_model_from_savedmodel(args.savedmodel)

    if load_kind == "keras":
        flat = extract_weighted_layers_from_keras(model)
    else:
        flat = extract_weighted_layers_from_raw_savedmodel(model)

    n_conv_keras = sum(1 for k, _, _ in flat if k == "conv")
    n_bn_keras = sum(1 for k, _, _ in flat if k == "bn")
    info(f"extracted {n_conv_keras} Conv2D + {n_bn_keras} BatchNorm "
         f"= {len(flat)} weighted layers from SavedModel")

    # SANITY CHECK 1: layer count plausibility
    total = n_conv_keras + n_bn_keras
    if not (15 <= total <= 50):
        fail(
            f"SavedModel has {total} weighted layers (Conv2D + BN); expected "
            f"~18 + ~17 = 35 for TrackNetV2 3-in-3-out. This file may not be "
            f"a TrackNetV2 model."
        )
    info(f"sanity check 1 passed: {total} weighted layers in plausible range")

    info("building PyTorch reference model")
    import torch
    pt_model = build_pytorch_reference()
    pt_state = dict(pt_model.state_dict())

    # Walk PyTorch model in forward execution order. Note: BN is custom
    # (BatchNormOverWidth) to match Dettor's training convention; see
    # stages/track_ball/_tracknet_model.py for why.
    pt_layers = []
    for module_name, module in pt_model.named_modules():
        cls = module.__class__.__name__
        if cls == "Conv2d":
            pt_layers.append(("conv", module_name, module))
        elif cls == "BatchNormOverWidth":
            pt_layers.append(("bn", module_name, module))

    n_conv_pt = sum(1 for k, _, _ in pt_layers if k == "conv")
    n_bn_pt = sum(1 for k, _, _ in pt_layers if k == "bn")
    info(f"PyTorch model has {n_conv_pt} Conv2d + {n_bn_pt} "
         f"BatchNormOverWidth = {len(pt_layers)} weighted layers")

    # SANITY CHECK 2: matching counts
    if n_conv_keras != n_conv_pt or n_bn_keras != n_bn_pt:
        fail(
            f"Layer count mismatch.\n"
            f"  Keras:   {n_conv_keras} Conv2D + {n_bn_keras} BN\n"
            f"  PyTorch: {n_conv_pt} Conv2d + {n_bn_pt} BatchNorm2d\n"
            f"Cannot proceed. Either Dettor's model is a different TrackNet "
            f"variant, or stages/track_ball/_tracknet_model.py needs to be "
            f"adjusted to match. See _tracknet_model.py block sizes."
        )
    info("sanity check 2 passed: Conv2D and BatchNorm counts match")

    # SANITY CHECK 3: per-layer shape parity, also performs the conversion.
    # Walk both lists in lockstep. They should align kind-for-kind.
    keras_iter = iter(flat)
    for kind_pt, name_pt, mod_pt in pt_layers:
        try:
            kind_k, name_k, arrs = next(keras_iter)
        except StopIteration:
            fail(f"Ran out of Keras layers before PyTorch layer {name_pt!r}")

        if kind_k != kind_pt:
            fail(
                f"Layer kind mismatch at PyTorch layer {name_pt!r}: "
                f"PyTorch is {kind_pt}, next Keras is {kind_k} ({name_k!r}). "
                f"This usually means the encoder/decoder block sizes in "
                f"_tracknet_model.py don't match Dettor's architecture."
            )

        if kind_pt == "conv":
            kernel_pt_shape = tuple(mod_pt.weight.shape)
            kernel_k = keras_to_pytorch_conv_weight(arrs["kernel"])
            if tuple(kernel_k.shape) != kernel_pt_shape:
                fail(
                    f"Conv weight shape mismatch at PyTorch {name_pt!r} "
                    f"(Keras {name_k!r}): PT {kernel_pt_shape}, "
                    f"Keras-transposed {tuple(kernel_k.shape)}"
                )
            pt_state[f"{name_pt}.weight"] = torch.from_numpy(kernel_k.copy()).float()

            has_bias = mod_pt.bias is not None
            keras_has_bias = arrs["bias"] is not None
            if has_bias and keras_has_bias:
                if mod_pt.bias.shape[0] != arrs["bias"].shape[0]:
                    fail(
                        f"Conv bias shape mismatch at {name_pt!r}: "
                        f"PT {tuple(mod_pt.bias.shape)}, "
                        f"Keras {arrs['bias'].shape}"
                    )
                pt_state[f"{name_pt}.bias"] = torch.from_numpy(arrs["bias"].copy()).float()
            elif has_bias and not keras_has_bias:
                info(f"  {name_pt}: PT has bias, Keras does not — leaving PT bias at init")
            elif not has_bias and keras_has_bias:
                info(f"  {name_pt}: Keras has bias, PT does not — discarding Keras bias")

        elif kind_pt == "bn":
            # BatchNormOverWidth has the same 4-parameter API as nn.BatchNorm2d:
            # weight (gamma), bias (beta), running_mean, running_var.
            # The parameter size is the WIDTH of the tensor at this layer's
            # position, NOT the channel count, because Dettor used
            # axis=-1 BN on NCHW data.
            for sub_key, expected_param, k_sub in [
                ("weight", mod_pt.weight, "gamma"),
                ("bias", mod_pt.bias, "beta"),
                ("running_mean", mod_pt.running_mean, "moving_mean"),
                ("running_var", mod_pt.running_var, "moving_var"),
            ]:
                k_arr = arrs[k_sub]
                if expected_param.shape[0] != k_arr.shape[0]:
                    fail(
                        f"BN {sub_key} shape mismatch at {name_pt!r} "
                        f"(Keras {name_k!r}): PT {tuple(expected_param.shape)}, "
                        f"Keras {k_arr.shape}\n"
                        f"  This usually means the input_shape (288, 512) in "
                        f"_tracknet_model.py does not match what Dettor "
                        f"trained at, OR the encoder/decoder block sizes "
                        f"are still off."
                    )
                pt_state[f"{name_pt}.{sub_key}"] = torch.from_numpy(k_arr.copy()).float()
            # num_batches_tracked stays at 0 (default from BatchNormOverWidth.__init__)

    leftover = list(keras_iter)
    if leftover:
        fail(
            f"Conversion consumed all PyTorch layers but {len(leftover)} "
            f"Keras layers remain unmatched. First leftover: {leftover[0][1]!r}"
        )
    info("sanity check 3 passed: all layers shape-matched and converted")

    # Load and forward-pass sanity check
    info("running forward-pass sanity check")
    try:
        pt_model.load_state_dict(pt_state, strict=True)
    except RuntimeError as e:
        fail(f"state_dict load failed: {e}")

    pt_model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 9, 288, 512)
        try:
            out = pt_model(dummy)
        except Exception as e:
            fail(f"forward pass failed: {e}")

    # SANITY CHECK 4: output not all NaN
    if torch.isnan(out).all():
        fail("forward-pass output is entirely NaN. Conversion is broken.")
    if out.shape[1] != 3:
        fail(f"forward-pass output has {out.shape[1]} channels, expected 3")
    info(f"sanity check 4 passed: forward-pass output shape {tuple(out.shape)}, "
         f"not-all-NaN, range [{out.min().item():.4f}, {out.max().item():.4f}]")

    # SANITY CHECK 5: output range plausible
    abs_max = out.abs().max().item()
    if abs_max > 100:
        fail(
            f"forward-pass output abs-max is {abs_max:.2f} which is "
            f"implausibly large for TrackNetV2. Conv weights may be wrong."
        )
    info(f"sanity check 5 passed: output abs-max {abs_max:.4f} is plausible")

    # Save
    info(f"writing converted weights to {args.out}")
    torch.save(pt_state, args.out)

    src_sha = sha256_savedmodel(args.savedmodel)
    dst_sha = sha256_file(args.out)
    meta = {
        "schema_version": 1,
        "source_savedmodel_path": str(args.savedmodel),
        "source_savedmodel_sha256": src_sha,
        "output_pt_path": str(args.out),
        "output_pt_sha256": dst_sha,
        "load_kind": load_kind,
        "model_class": "stages.track_ball._tracknet_model.TrackNet",
        "in_channels": 9,
        "out_channels": 3,
        "n_conv_layers": n_conv_keras,
        "n_bn_layers": n_bn_keras,
        "converted_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "sanity_checks_passed": [
            "weighted_layer_count_in_range",
            "conv_bn_counts_match",
            "shape_parity_per_layer",
            "forward_pass_no_nan",
            "forward_pass_output_range_plausible",
        ],
    }
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    info(f"wrote sidecar metadata: {meta_path}")
    info("conversion complete")


if __name__ == "__main__":
    main()