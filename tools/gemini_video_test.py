r"""Score a Gemini video model against the operator's truth, beside today's pipeline.

Why this exists. The hand-built detector plateaued on unseen video (docs/ACCURACY_LEDGER.md,
2026-09-14), so a general-purpose video model was tested as a completely different approach:
no training, one fixed prompt, structured JSON back. On 2026-09-15 it scored BELOW the
pipeline on court C at 720p and at 4K. This tool keeps that test repeatable, so each new model
release can be re-scored in one command instead of rebuilt.

What was learned, and is built in:
  * Ask about SHORT WINDOWS (20 s, 4 s overlap), not the whole video: asked about three
    minutes at once, the models stopped listing after a few points.
  * CROP TO THE COURT at native resolution. Gemini rescales every frame to a fixed token budget
    (127,556 prompt tokens per 20 s window whether the upload was 4K or not), so a 4K upload
    adds nothing by itself; a crop puts more of that budget on the players and ball. Flash on
    the crop found 21/29 shots against 14/29 on the full 4K frame.
  * 24 fps is the API maximum (30 is rejected). High media resolution.
  * Upload through the Files API (2 GB free / 20 GB paid). The 100 MB limit is inline data only.

The crop comes from court.json alone -- the operator's four court corners plus pixels-per-foot --
so it is the same for any video: 5 ft beyond each sideline, 8 ft above the far baseline (a
standing far player with a raised paddle), 3 ft behind the near baseline.

Held-out rule: court A (pb_5_min_indoor_1_court_a) is refused unless --allow-heldout is passed.
Iterate prompts on dev clips only; score court A once, deliberately.

The API key is read from GEMINI_API_KEY in the environment, else from the Windows user
environment (set with `setx`, which a running process does not see). It is never printed.

Usage:
    python -m tools.gemini_video_test data/pb_3_min_indoor_1_court_c --start 56 --end 119
    python -m tools.gemini_video_test data/pb_3_min_indoor_1_court_c --model gemini-pro-latest
    python -m tools.gemini_video_test data/pb_3_min_indoor_1_court_c --start 56 --end 119 \
        --windows-json data/pb_3_min_indoor_1_court_c/_gemini/<saved>.json     # re-score, no API

Needs `google-genai` (tools only; not an app requirement) and imageio-ffmpeg (already one).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.rally_end_score import note_reason  # noqa: E402
from tools.truth_store import known  # noqa: E402

HELD_OUT = {"pb_5_min_indoor_1_court_a"}
MAX_FPS = 24.0
WINDOW_S, STEP_S = 20.0, 16.0
SHOT_MERGE_S, POINT_MERGE_S = 0.3, 2.0
SHOT_TOL_S, SERVE_TOL_S, END_TOL_S = 0.35, 1.0, 2.0
SIDELINE_FT, FAR_ABOVE_FT, NEAR_BELOW_FT = 5.0, 8.0, 3.0

PROMPT = """This is a clip from a recreational doubles pickleball video from a fixed camera behind one baseline.
"Near" means the half of the court closest to the camera; "far" means the half beyond the net.
"Left"/"right" are as seen in the image. Ignore people and balls on neighbouring courts.

Report, with times in SECONDS from the start of THIS CLIP (decimals, e.g. 12.4):

1. Every POINT (rally) that has its serve inside this clip: the serve contact time, which player served
   (side near/far and left/right), the time of the last paddle contact of the point, the time the point was
   over, and how it ended: "net", "out", "not-returned" (landed in, opponent failed to return it),
   "serve-fault", or "unsure". If the point continues past the end of the clip, give your best estimate.
2. Every SHOT (every paddle contact with the ball during play, including serves and returns) inside this
   clip: contact time, hitter side (near/far) and position (left/right), shot type (serve, return, drive,
   drop, dink, lob, smash), and whether it was a volley (hit before the ball bounced).
   Do not report ball handling between points (bouncing, catching, picking up, feeding to the server).

List EVERY shot; do not summarise or skip. Be as precise with contact times as the video allows; the
sound of the paddle can help.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "points": {"type": "array", "items": {"type": "object", "properties": {
            "serve_time_s": {"type": "number"},
            "server_side": {"type": "string", "enum": ["near", "far"]},
            "server_position": {"type": "string", "enum": ["left", "right"]},
            "last_contact_time_s": {"type": "number"},
            "point_over_time_s": {"type": "number"},
            "end_reason": {"type": "string",
                           "enum": ["net", "out", "not-returned", "serve-fault", "unsure"]},
        }, "required": ["serve_time_s", "server_side", "last_contact_time_s",
                        "point_over_time_s", "end_reason"]}},
        "shots": {"type": "array", "items": {"type": "object", "properties": {
            "time_s": {"type": "number"},
            "side": {"type": "string", "enum": ["near", "far"]},
            "position": {"type": "string", "enum": ["left", "right"]},
            "type": {"type": "string",
                     "enum": ["serve", "return", "drive", "drop", "dink", "lob", "smash"]},
            "volley": {"type": "boolean"},
        }, "required": ["time_s", "side", "type", "volley"]}},
    },
    "required": ["points", "shots"],
}


# --- pure helpers (tested) ---------------------------------------------------------------------
def one_to_one(a: Sequence[float], b: Sequence[float], tol: float) -> List[Tuple[int, int]]:
    """Pairs (i, j) with |a[i] - b[j]| <= tol, shortest first, each index used once.

    Loose many-to-one matching inflated two findings in the ledger; every count here is one-to-one.
    """
    pairs = sorted((abs(x - y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)
                   if abs(x - y) <= tol)
    ui, uj, out = set(), set(), []
    for _, i, j in pairs:
        if i in ui or j in uj:
            continue
        ui.add(i)
        uj.add(j)
        out.append((i, j))
    return out


def court_crop(court: dict, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """(x, y, w, h) around the court from court.json, even-sized for the H.264 encoder."""
    ui = court["user_inputs"]["court_corners_image"]
    der = court["derived"]
    ppf_near = float(der["pixels_per_foot_at_near_baseline"])
    ppf_far = float(der["pixels_per_foot_at_far_baseline"])
    xs = [float(p[0]) for p in ui]
    ys = [float(p[1]) for p in ui]
    x0 = max(0.0, min(xs) - SIDELINE_FT * ppf_near)
    x1 = min(float(frame_w), max(xs) + SIDELINE_FT * ppf_near)
    y0 = max(0.0, min(ys) - FAR_ABOVE_FT * ppf_far)
    y1 = min(float(frame_h), max(ys) + NEAR_BELOW_FT * ppf_near)
    w = int(x1 - x0) // 2 * 2
    h = int(y1 - y0) // 2 * 2
    return int(x0), int(y0), w, h


def window_starts(duration: float, window: float = WINDOW_S, step: float = STEP_S) -> List[float]:
    out, s = [], 0.0
    while s < max(duration - (window - step), 0.001):
        out.append(round(s, 3))
        s += step
    return out


def merge_windows(windows: List[dict], clip_start: float) -> Tuple[List[dict], List[dict]]:
    """Shots and points in VIDEO time. Where windows overlap, an item is kept from the window in
    which it sits most centrally; shots closer than SHOT_MERGE_S and points whose serves are
    closer than POINT_MERGE_S are the same item."""
    shots, points = [], []
    for w in windows:
        if "error" in w:
            continue
        st, en = float(w["start"]), float(w["end"])
        for s in w["result"].get("shots", []):
            t = st + float(s["time_s"])
            shots.append(dict(s, t=clip_start + t, centrality=min(t - st, en - t)))
        for p in w["result"].get("points", []):
            t = st + float(p["serve_time_s"])
            points.append(dict(p, t=clip_start + t,
                               last=clip_start + st + float(p["last_contact_time_s"]),
                               centrality=min(t - st, en - t)))

    def dedupe(items, tol):
        kept: List[dict] = []
        for it in sorted(items, key=lambda x: -x["centrality"]):
            if all(abs(it["t"] - k["t"]) > tol for k in kept):
                kept.append(it)
        return sorted(kept, key=lambda x: x["t"])

    return dedupe(shots, SHOT_MERGE_S), dedupe(points, POINT_MERGE_S)


def score(shots: List[dict], serve_times: List[float], truth: dict,
          points: Optional[List[dict]] = None) -> Dict[str, object]:
    """Counts against the operator's truth inside [start, end]. `shots` need t, type, side."""
    tt = [float(s["t_sec"]) for s in truth["shots"]]
    st = [float(s["t"]) for s in shots]
    m = one_to_one(st, tt, SHOT_TOL_S)
    pairs = [(shots[i], truth["shots"][j]) for i, j in m]
    typed = [(g, t) for g, t in pairs if t.get("type")]
    sided = [(g, t) for g, t in pairs if t.get("side")]
    serves = [float(s["t_sec"]) for s in truth["shots"] if s.get("type") == "serve"]
    ms = one_to_one(serve_times, serves, SERVE_TOL_S)
    out: Dict[str, object] = {
        "truth_shots": len(tt), "found": len(m), "junk": len(st) - len(m),
        "found_within_1s": len(one_to_one(st, tt, 1.0)),
        "type_right": sum(1 for g, t in typed if g.get("type") == t["type"]), "typed": len(typed),
        "side_right": sum(1 for g, t in sided if g.get("side") == t["side"]), "sided": len(sided),
        "truth_serves": len(serves), "serves_right": len(ms), "serves_false": len(serve_times) - len(ms),
    }
    if points is not None:
        end_t = [float(e["t_sec"]) for e in truth["ends"]]
        reasons = [e.get("reason") or note_reason(e.get("notes", "")) for e in truth["ends"]]
        me = one_to_one([p["last"] for p in points], end_t, END_TOL_S)
        out.update(truth_ends=len(end_t), ends_right=len(me),
                   end_reason_right=sum(1 for i, j in me if points[i]["end_reason"] == reasons[j]))
    return out


# --- I/O ---------------------------------------------------------------------------------------
def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = str(winreg.QueryValueEx(k, "GEMINI_API_KEY")[0]).strip()
        except OSError:
            key = ""
    key = key.strip('"')
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set (setx GEMINI_API_KEY \"...\")")
    return key


def make_clip(video: Path, out: Path, start: float, end: float,
              crop: Optional[Tuple[int, int, int, int]]) -> Path:
    import imageio_ffmpeg
    vf = ["-vf", "crop={2}:{3}:{0}:{1}".format(*crop)] if crop else []
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(video),
           "-t", f"{end - start:.3f}", *vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-c:a", "aac", "-b:a", "128k", str(out)]
    subprocess.run(cmd, check=True)
    return out


def ask_gemini(clip: Path, model: str, fps: float, duration: float) -> List[dict]:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key())
    f = client.files.upload(file=str(clip))
    while not f.state or f.state.name != "ACTIVE":
        if f.state and f.state.name == "FAILED":
            raise SystemExit("Gemini could not process the uploaded clip")
        time.sleep(5)
        f = client.files.get(name=f.name)

    def one(start: float) -> dict:
        end = min(start + WINDOW_S, duration)
        err = ""
        for _ in range(3):
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=types.Content(parts=[
                        types.Part(file_data=types.FileData(file_uri=f.uri, mime_type="video/mp4"),
                                   video_metadata=types.VideoMetadata(
                                       fps=fps, start_offset=f"{start:.1f}s", end_offset=f"{end:.1f}s")),
                        types.Part(text=PROMPT)]),
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", response_json_schema=SCHEMA,
                        temperature=0.0, media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH))
                return {"start": start, "end": end, "result": json.loads(r.text),
                        "usage": str(r.usage_metadata)[:300]}
            except Exception as e:  # noqa: BLE001 -- report per window, never lose the others
                err = repr(e)
                time.sleep(10)
        return {"start": start, "end": end, "error": err}

    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            return list(ex.map(one, window_starts(duration)))
    finally:
        try:
            client.files.delete(name=f.name)
        except Exception:  # noqa: BLE001
            pass


def truth_between(clip_dir: Path, start: float, end: float) -> dict:
    store = known(clip_dir)
    inside = lambda t: start <= float(t) <= end  # noqa: E731
    return {"shots": sorted((s for s in store["shots"] if not s.get("not_a_shot") and inside(s["t_sec"])),
                            key=lambda s: float(s["t_sec"])),
            "ends": sorted((e for e in store.get("rally_ends", []) if inside(e["t_sec"])),
                           key=lambda e: float(e["t_sec"]))}


def fmt(label: str, r: Dict[str, object]) -> str:
    s = (f"  {label:30s} shots {r['found']}/{r['truth_shots']} junk {r['junk']} (within 1s {r['found_within_1s']})"
         f" | type {r['type_right']}/{r['typed']} | side {r['side_right']}/{r['sided']}"
         f" | serves {r['serves_right']}/{r['truth_serves']} false {r['serves_false']}")
    if "ends_right" in r:
        s += f" | rally ends {r['ends_right']}/{r['truth_ends']} reason {r['end_reason_right']}/{r['ends_right']}"
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path, help="data/<clip> folder (needs session.json, court.json, classified.json)")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None, help="default: the whole video")
    ap.add_argument("--fps", type=float, default=MAX_FPS)
    ap.add_argument("--no-crop", action="store_true", help="send the full frame instead of the court crop")
    ap.add_argument("--windows-json", type=Path, default=None, help="re-score a saved response; no API call")
    ap.add_argument("--allow-heldout", action="store_true")
    a = ap.parse_args(argv)

    clip = a.clip
    if clip.name in HELD_OUT and not a.allow_heldout:
        raise SystemExit(f"{clip.name} is the held-out clip; pass --allow-heldout to score it once, deliberately")
    if a.fps > MAX_FPS:
        raise SystemExit(f"--fps above {MAX_FPS:g} is rejected by the API")
    session = json.loads((clip / "session.json").read_text(encoding="utf-8"))
    video = Path(session["video_path"])
    end = a.end if a.end is not None else float(session["duration_sec"])
    duration = end - a.start

    if a.windows_json:
        windows = json.loads(a.windows_json.read_text(encoding="utf-8"))
        label = f"{a.windows_json.stem}"
    else:
        court = json.loads((clip / "court.json").read_text(encoding="utf-8"))
        crop = None if a.no_crop else court_crop(court, int(session["frame_width"]), int(session["frame_height"]))
        out_dir = clip / "_gemini"
        out_dir.mkdir(exist_ok=True)
        tag = f"{a.model}_{'full' if crop is None else 'crop'}_{a.start:g}-{end:g}_fps{a.fps:g}"
        clip_file = make_clip(video, out_dir / f"{tag}.mp4", a.start, end, crop)
        t0 = time.time()
        windows = ask_gemini(clip_file, a.model, a.fps, duration)
        saved = out_dir / f"{tag}.json"
        saved.write_text(json.dumps(windows, indent=1), encoding="utf-8")
        errors = [w for w in windows if "error" in w]
        print(f"{a.model}: {len(windows)} windows in {time.time() - t0:.0f}s, {len(errors)} failed; saved {saved}")
        for w in errors:
            print(f"  window {w['start']:g}s failed: {w['error'][:200]}")
        label = f"{a.model} {'full' if crop is None else 'crop'}"

    shots, points = merge_windows(windows, a.start)
    truth = truth_between(clip, a.start, end)
    print(f"\n{clip.name} {a.start:g}-{end:g}s: truth {len(truth['shots'])} shots, "
          f"{sum(1 for s in truth['shots'] if s.get('type') == 'serve')} serves, {len(truth['ends'])} rally ends")
    print(fmt(label, score(shots, [p["t"] for p in points], truth, points)))
    ours = [dict(t=float(s["t_sec"]), type=s.get("shot_type"), side=s.get("hitter_side"))
            for s in json.loads((clip / "classified.json").read_text(encoding="utf-8"))["shots"]
            if a.start <= float(s["t_sec"]) <= end]
    print(fmt("pipeline today", score(ours, [s["t"] for s in ours if s["type"] == "serve"], truth)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
