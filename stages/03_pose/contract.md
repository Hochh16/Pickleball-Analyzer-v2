# Stage 3 — Pose Estimation

Status: NOT STARTED.

Outline:
- Input: video.mp4 + players.parquet
- Output: poses.parquet (33 keypoints per player per frame)
- Tool: MediaPipe Pose
- Only runs on the user's bbox crops (not all detected players)