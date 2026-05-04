# Stage 2 — Track Players

Status: NOT STARTED. Contract will be written when Stage 1 is complete and approved.

Outline (for context only — not the final contract):
- Input: video.mp4 + court.json
- Output: players.parquet
- Tool: ultralytics YOLO + ByteTrack
- Identity: user is identified by selecting from tracked IDs in the first valid frame, using `user_starting_corner` from court.json as a hint