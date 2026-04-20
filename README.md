# Holdfast Cut Clips

An AI-assisted video pipeline for cutting raw 4K iPhone band footage into YouTube-ready horizontal and vertical clips. Vertical clips use computer vision to track the subject and auto-pan the 9:16 crop window — no manual keyframing.

Built on the **WAT framework** (Workflows, Agents, Tools): markdown SOPs define what to do, Python scripts do the deterministic execution, and Claude coordinates between them.

---

## What this does

1. Takes raw footage and a CSV of timestamps
2. Cuts each entry into a horizontal (1920×1080) and vertical (1080×1920) MP4
3. Vertical clips use subject tracking — a face/body detector finds the main performer and pans the crop to follow them
4. Uploads approved clips to the Holdfast YouTube channel as private drafts

---

## Current workflow

### Setup (one-time)

```bash
pip install -r requirements.txt
sudo apt install ffmpeg          # or: brew install ffmpeg on macOS
```

### 1. Plan your video (optional but recommended)

Before cutting, run the planning tool on the raw file to see what the tracker sees and get an auto-generated scene breakdown:

```bash
python3 tools/plan_video.py footage/show_night1.mp4 --audio
```

Outputs to `.tmp/planning/{video_stem}/`:
- `{stem}_debug_planning.mp4` — full video with detection boxes, crop window lines, segment labels, and boundary flashes drawn on top
- `{stem}_scenes_planning.csv` — one row per detected segment with timing, dominant detection mode, crop stability score, and blank columns for your notes
- `{stem}_scenes_planning.json` — same data in JSON

Watch the debug video, fill in the blank columns (`label`, `subject_focus`, `description`, `pan_plan`), then use the completed CSV as your shot list.

See [PLAN_VIDEO_README.md](PLAN_VIDEO_README.md) for the full field reference.

### 2. Define your clips

Populate `clips.csv`:

```
initial_video_title,start,end,cut_title,approved,title,description,tags
show_night1,00:01:10,00:01:50,opening_riff,false,,,
```

- `initial_video_title` matches `footage/{title}.mp4`
- `start` / `end` in `HH:MM:SS` or seconds
- `approved` — set to `true` before uploading
- `title`, `description`, `tags` — auto-generated if left blank

### 3. Cut

```bash
python tools/cut_clips.py clips.csv
```

Outputs horizontal + vertical MP4s to `.tmp/clips/`. Safe to re-run — skips existing files.

### 4. Review

Open the clips in `.tmp/clips/`. For keepers, set `approved=true` in `clips.csv`.

### 5. Upload

```bash
python tools/upload_youtube.py clips.csv
```

Uploads only approved clips, both orientations, as private. Prints the YouTube URL for each.

> First-time YouTube auth: run `python tools/setup_youtube_auth.py` and follow the browser prompt.

---

## Subject tracking — how it works

The vertical crop pipeline has gone through several iterations:

| Version | What changed |
|---|---|
| v3 | Hybrid detector: OpenCV Haar cascade for faces, YOLOv8-nano body fallback. Debug flag added. |
| v4 | Replaced Haar with MediaPipe BlazeFace (confidence ≥ 0.7) — eliminates mic stand false positives. Detector logic split into `detect_mediapipe.py` and `detect_yolo.py`. |
| v5 | Three-stage smoothing: linear interpolation between sampled frames → median filter (kernel=7) to kill outlier spikes → Gaussian smooth (sigma=3.0) replacing EMA. No more lag artifact. |
| v6 | `plan_video.py` planning tool built. Adaptive segmentation on mode changes and crop jumps. Scene CSV + debug video output. |

Current stack: **MediaPipe BlazeFace** (primary) → **YOLOv8-nano body** (fallback) → **center crop** (no detection). Smoothing is Gaussian with a median pre-pass.

---

## Future workflow

The planning tool (v6) is the foundation for a more automated edit pipeline:

1. **Run `plan_video.py`** on the raw file — get a scene breakdown automatically
2. **Review the debug video + CSV** — fill in shot labels and pan intentions
3. **Feed the annotated CSV to Claude** as a shot list — it coordinates which segments to cut, in what order, with what crop behavior
4. **`cut_clips.py` executes** with per-clip crop instructions derived from the plan (subject hints, manual pan overrides, skip flags)
5. **Upload approved clips** as before

Planned improvements not yet built:
- `--sigma` and `--kernel` CLI flags for per-clip smoothing tuning
- Per-clip subject hints in `clips.csv` (tell the tracker who to follow if multiple people are in frame)
- Claude-assisted shot selection: use scene CSV + audio/energy data to auto-rank which segments are worth cutting

---

## File layout

```
footage/          # Raw source files (not committed)
tools/
  cut_clips.py        # Main cutting tool
  plan_video.py       # Pre-cut analysis and scene breakdown
  track_subject.py    # Smoothing pipeline
  detect_mediapipe.py # BlazeFace detector
  detect_yolo.py      # YOLOv8 body detector (fallback)
  upload_youtube.py   # YouTube upload
  setup_youtube_auth.py
workflows/
  cut_clips.md        # Full SOP for the cut → upload workflow
.tmp/
  clips/              # Output MP4s
  planning/           # Plan tool outputs
versioning/           # Per-version debug videos and session notes
clips.csv             # Clip definitions
```

---

## Dependencies

- Python 3.10+
- `ffmpeg` (system)
- `mediapipe`, `ultralytics` (YOLOv8), `opencv-python`, `numpy` — via `requirements.txt`
- `google-api-python-client`, `google-auth-oauthlib` — for YouTube upload
- `blaze_face_short_range.tflite` — bundled MediaPipe model file
