# Stage 4 — Ball Tracking

Status: NOT STARTED. Likely the hardest stage.

Outline:
- Input: video.mp4 + court.json
- Output: ball.parquet (frame, x, y, conf)
- Approach: TBD. Options include Roboflow pickleball model (already on disk), custom-trained YOLO, motion-based detection, or fusion. To be decided when we get there based on what works on actual test footage.