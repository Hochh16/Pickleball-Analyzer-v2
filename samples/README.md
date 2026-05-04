# Samples

Place test videos here for development and smoke tests.

## Naming convention

- `sample_short.mp4` — a 30-60 second clip used for fast iteration
- `sample_full.mp4` — a full match used for end-to-end validation

## Not committed

Video files are excluded by `.gitignore`. Each developer keeps their own copy locally.

## What makes a good test video

- Camera mounted in one position, not moving
- All four court corners visible in frame at start
- At least one rally where the user crosses to the opposite side of their court half (the failure case from v1)
- At least one moment where the user stands outside the court near the baseline (between rallies)