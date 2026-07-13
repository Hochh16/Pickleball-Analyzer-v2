"""Server-side (local) filesystem browsing for the video picker.

The 'server' is the user's own laptop, so this navigates their local folders to
find a video they've already copied over. Not a security boundary — a personal
local app — but it hides dotfiles and only surfaces video files + directories.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .video import VIDEO_EXTS


def quick_roots(data_root: Path) -> List[Dict]:
    """Convenient starting points for the browser."""
    roots = []
    home = Path.home()
    roots.append({"label": "Home", "path": str(home)})
    for sub in ("Videos", "Downloads", "Desktop"):
        p = home / sub
        if p.is_dir():
            roots.append({"label": sub, "path": str(p)})
    roots.append({"label": "App data", "path": str(data_root.resolve())})
    # De-dup by path, preserve order.
    seen = set()
    out = []
    for r in roots:
        if r["path"] not in seen:
            seen.add(r["path"])
            out.append(r)
    return out


def listing(path: Path) -> Dict:
    """List sub-directories and video files under `path`."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    dirs: List[Dict] = []
    videos: List[Dict] = []
    try:
        children = sorted(path.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        children = []

    for child in children:
        name = child.name
        if name.startswith("."):
            continue
        try:
            if child.is_dir():
                dirs.append({"name": name, "path": str(child), "is_dir": True})
            elif child.suffix.lower() in VIDEO_EXTS:
                size = child.stat().st_size
                videos.append({
                    "name": name,
                    "path": str(child),
                    "is_dir": False,
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 1),
                })
        except (OSError, PermissionError):
            continue

    parent = str(path.parent) if path.parent != path else None
    return {
        "path": str(path),
        "parent": parent,
        "dirs": dirs,
        "videos": videos,
    }
