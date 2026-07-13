"""Pickleball-Analyzer-v2 — local setup/analysis web app.

Phase 1 (setup wizard): a local FastAPI app + vanilla browser UI that replaces
the Tkinter mark_* tools. It lets a user pick a video, mark the 8 court points
in-browser (with live validation + a top-down sanity check), set up players,
optionally self-identify, and writes the exact input JSONs the pipeline already
consumes (markers.json -> court.json/court_zones.json via Stage 1, roster.json,
user_clicks.json). Data contracts are unchanged.

Run locally:  python -m app   (opens http://127.0.0.1:8000)

See docs/UI_PLAN.md for the scoped plan and milestones.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
