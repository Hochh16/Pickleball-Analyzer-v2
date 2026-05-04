# Stage 5 — Detect Shots

Status: NOT STARTED.

Outline:
- Input: players.parquet + ball.parquet + poses.parquet
- Output: shots.json (frame, player_id, impact_xy)
- Heuristic: ball direction change within 50px of a tracked player, 0.2s window