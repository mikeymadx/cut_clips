# Workflow: Cut Clips

**Objective**: Take raw 4K iPhone footage and a CSV of timestamps, cut them into YouTube-ready clips (horizontal 16:9 and vertical 9:16), then upload approved clips to the Holdfast YouTube channel.

---

## Inputs Required

- Raw video file(s) placed in `footage/` — filename must match the `initial_video_title` column in the CSV
- `clips.csv` populated with clip definitions

---

## CSV Format

```
initial_video_title,start,end,cut_title,approved,title,description,tags
holdfast_show_night1,00:01:10,00:01:50,opening_riff,false,,,
```

| Column | Required | Notes |
|--------|----------|-------|
| `initial_video_title` | Yes | Matches `footage/{title}.mp4` (or .mov, .MOV, .m4v) |
| `start` | Yes | `HH:MM:SS` or seconds |
| `end` | Yes | `HH:MM:SS` or seconds |
| `cut_title` | Yes | Slug for this clip (used in output filename) |
| `approved` | Yes | `false` → review → change to `true` before uploading |
| `title` | No | Auto-generated as `Cut Title \| Holdfast` if blank |
| `description` | No | Auto-generated with Holdfast links/hashtags if blank |
| `tags` | No | Comma-separated; auto-generated from clip name + Holdfast base tags if blank |

### Output filenames
```
.tmp/clips/{initial_video_title}-{cut_title}-horizontal.mp4   ← YouTube (1920×1080)
.tmp/clips/{initial_video_title}-{cut_title}-vertical.mp4     ← YouTube Shorts (1080×1920)
```

---

## Step 1: Install Dependencies (one-time)

```bash
pip install -r requirements.txt
```

Also requires **ffmpeg** as a system dependency:
- macOS: `brew install ffmpeg`
- Ubuntu/WSL: `sudo apt install ffmpeg`

> **Subject tracking dependencies** (`ultralytics`, `opencv-python`, `numpy`) are included in `requirements.txt`. On first run, YOLOv8 will download `yolov8n.pt` (~6MB) automatically.

---

## Step 2: Fill Out clips.csv

Add one row per clip you want to cut. You can have multiple clips from the same source file.

---

## Step 3: Cut Clips

```bash
python tools/cut_clips.py clips.csv
```

- Outputs horizontal + vertical MP4s to `.tmp/clips/`
- Skips clips that already exist (safe to re-run)
- Warns if a clip exceeds 60s (YouTube Shorts limit)
- **Vertical output uses subject tracking**: YOLOv8-nano detects the largest person in frame and pans the 9:16 crop window to follow them. Falls back to center crop if no person is detected.

---

## Step 4: Review

Open the clips in `.tmp/clips/` and review both versions. For clips you want to upload, set `approved=true` in `clips.csv`. Optionally fill in `title`, `description`, or `tags` to override the auto-generated values.

---

## Step 5: YouTube Auth (one-time setup)

You need a `credentials.json` from Google Cloud Console:

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project, enable **YouTube Data API v3**
3. Create OAuth 2.0 credentials → Desktop app
4. Download and save as `credentials.json` in the project root

Then run:
```bash
python tools/setup_youtube_auth.py
```

A browser window will open for Google login. This saves `token.json` locally — only needs to be done once (token refreshes automatically).

---

## Step 6: Upload Approved Clips

```bash
python tools/upload_youtube.py clips.csv
```

- Uploads only rows where `approved=true`
- Uploads both horizontal and vertical versions
- All videos are uploaded as **private** — publish from YouTube Studio when ready
- Prints the YouTube URL for each upload

---

## Edge Cases

- **Source file not found**: Ensure the filename in `footage/` exactly matches `initial_video_title` (case-sensitive on Linux)
- **Clip > 60s vertical**: Tool warns but still cuts — YouTube may not classify it as a Short
- **Token expired**: Delete `token.json` and re-run `setup_youtube_auth.py`
- **Quota exceeded**: YouTube Data API has a daily upload quota. If hit, wait 24h or request a quota increase in Google Cloud Console
- **No person detected in clip**: Tracking falls back to static center crop automatically (logged as a warning)
- **Source is already vertical**: Tracking is skipped, static crop is used
- **Tracking looks jittery**: The EMA smoothing (alpha=0.15) should prevent most jitter; if a clip is still jumpy, the subject may be moving very fast or going in/out of frame frequently
