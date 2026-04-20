# Video Planning Tool — README

## What this tool does

`tools/plan_video.py` analyzes a raw video file using the same subject-tracking model used for cutting vertical clips. Instead of producing a finished clip, it produces three planning outputs:

- **`{stem}_debug_planning.mp4`** — the original video with detection boxes, crop window lines, segment labels, and boundary flashes drawn on top. Watch this to see exactly what the tracker sees.
- **`{stem}_scenes_planning.csv`** — a spreadsheet-ready breakdown of every detected scene segment. Fill in the blank columns after reviewing the debug video.
- **`{stem}_scenes_planning.json`** — same data in JSON format for programmatic use.

All outputs go to `.tmp/planning/{video_stem}/`.

---

## Usage

```bash
python3 tools/plan_video.py <video_path> [--jump-threshold 150] [--min-segment 2.0] [--audio]
```

**Options:**

| Flag | Default | What it does |
|---|---|---|
| `--jump-threshold` | 150 | Pixel distance the crop window has to jump in one frame to trigger a new segment |
| `--min-segment` | 2.0 | Minimum segment length in seconds — prevents noise from creating tiny segments |
| `--audio` | off | Mux the original audio into the debug video (useful for syncing to music) |

**Examples:**

```bash
# Basic analysis
python3 tools/plan_video.py footage/show_night1.mp4

# With audio so you can sync to the track
python3 tools/plan_video.py footage/show_night1.mp4 --audio

# More sensitive segmentation (smaller jumps trigger breaks)
python3 tools/plan_video.py footage/show_night1.mp4 --jump-threshold 80 --min-segment 1.5
```

---

## Scene Field Reference

Each row in the CSV / each object in the JSON represents one segment.

---

### `segment`
Sequence number, 1-based. Segment 1 is the top of the video.

---

### `start_s` / `end_s` / `duration_s`
When this segment starts and ends, in seconds from the beginning of the video. `duration_s` is `end_s - start_s`.

---

### `trigger`
What caused the tool to start a new segment at this point.

| Value | Meaning |
|---|---|
| `start` | First segment — no trigger, just the beginning of the video |
| `mode_change` | The tracker switched between face-detection and body-detection. Usually means a cut happened, the subject turned away, or someone moved in/out of frame |
| `crop_jump` | The crop window snapped hard to a different horizontal position without a mode change. Usually fast subject movement or the tracker briefly losing and re-acquiring the subject |

---

### `dominant_mode`
Whether face-detection or body-detection was running for most of this segment.

| Value | Meaning |
|---|---|
| `face` | The tracker found a face and centered on it. Tighter, more accurate framing — usually a close-up or subject facing camera |
| `body` | No reliable face found; tracker used the full person silhouette. Wider framing centered on torso — usually a wide shot, profile angle, or heavy movement |

---

### `detection_rate`
Fraction of sampled frames where the tracker found *something* (face or body). Range is 0.0–1.0.

| Value | Meaning |
|---|---|
| `1.0` | Locked on a subject for every sampled frame. Very confident. |
| `0.9–0.99` | Nearly perfect — one or two frames missed. |
| Below `0.7` | Struggling. The crop path for this segment is partly interpolated guesswork. Watch the debug video carefully before using this segment. |

---

### `avg_crop_x`
Average horizontal position of the **left edge** of the 9:16 crop window across this segment, in pixels measured from the left edge of the original wide frame.

The crop window slides left and right to follow the subject. This value tells you where the crop was sitting on average.

| Value range (1920px wide source) | Meaning |
|---|---|
| 0–400 | Subject was mostly on the **left** side of the frame |
| 600–900 | Subject was roughly **centered** |
| 1000+ | Subject was on the **right** side of the frame |

Use this as a reference if you're planning a manual pan or crop override in your edit.

---

### `crop_x_std`
Standard deviation of the crop window position — how much it moved around during this segment. Think of it as a **horizontal shakiness score**.

| Value | Meaning |
|---|---|
| Low (< 100) | Crop window was stable. Subject held position. Clean, easy edit. |
| Medium (100–200) | Moderate movement. Normal for an active performer. |
| High (> 200) | Lots of horizontal motion. Either the subject moved hard across frame, or the tracker was bouncing. Watch the debug video. |

---

### Blank fields — fill these in after watching the debug video

| Field | What to put here |
|---|---|
| `label` | Short name for this segment. Examples: `intro`, `chorus build`, `guitar solo`, `crowd cutaway` |
| `subject_focus` | Who or what is in the shot. Examples: `lead singer`, `drummer wide`, `guitarist close`, `crowd` |
| `description` | Free-text notes about what's happening. Examples: `Singer steps to the front`, `Guitar riff facing left`, `Energy builds before drop` |
| `pan_plan` | What you want the crop to do in the final cut. Examples: `hold center`, `follow subject`, `hard cut here`, `slow pan left`, `skip this segment` |

---

## What to look for in the debug video

The debug video overlays four things on the original frame:

- **Colored rectangle** around the detected subject — green for face, cyan for body
- **Orange vertical lines** marking the left and right edges of the 9:16 crop window
- **White text** in the top-left corner: `Seg {N} | {mode} | {time}s`
- **White flash bar** across the top of the frame at each segment boundary

Use this to verify the tracker is following the right subject, catch segments where it got confused, and decide which moments are worth cutting into a vertical clip.

---

## Workflow: from raw footage to edit plan

1. Run the tool on your raw video file with `--audio`
2. Watch the debug video while the CSV is open alongside it
3. Fill in `label`, `subject_focus`, `description`, and `pan_plan` for each segment as you watch
4. Use the completed CSV as your shot list when cutting clips with `cut_clips.py`
