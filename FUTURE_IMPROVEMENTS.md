# Future Improvements

Ideas distilled from a tooling survey (cut_clips_Improvement_Ideas.pdf). Broken into phases — none of this replaces existing work, all additive.

---

## Phase 1 — Audio Energy in `plan_video.py`

**The gap:** `plan_video.py` is currently blind to audio. A massive chorus hit, crowd eruption, or drum fill looks the same as a quiet moment in the scenes CSV.

**What to add:**
- Add `librosa` to `requirements.txt`
- After frame analysis, extract per-second RMS energy from the video's audio track using ffmpeg + librosa
- Add two columns to `{stem}_scenes_planning.csv`:
  - `audio_rms_mean` — average energy for the segment (normalized 0–1)
  - `audio_rms_peak` — peak energy within the segment
- No structural changes to the existing workflow; CSV just gets richer

---

## Phase 2 — Auto-Scored Clip Suggestions

**The gap:** After `plan_video.py` you still have to watch the full debug video and manually pick timestamps. For a long show this is a lot of work.

**What to add:**
- At the end of `plan_video.py`, score each segment by combining:
  - `audio_rms_mean` (Phase 1, weighted highest)
  - `detection_rate` (already in CSV) — rewards clearly visible subjects
  - `crop_stability` (already in CSV, inverted) — rewards locked-on shots
- Output `{stem}_suggested_clips.csv` — top N segments sorted by score, pre-filled in `clips.csv` format:
  ```
  initial_video_title, start, end, cut_title, approved, title, description, tags
  ```
- `approved` defaults to `false` — human still reviews and approves
- `cut_title` auto-generated from rank (e.g. `top_moment_01`)
- New flag: `--suggestions N` (default 10)

User workflow: copy `suggested_clips.csv` into `clips.csv` as a starting point, review/trim, then run `cut_clips.py` as normal.

---

## Phase 3 — Whisper Transcript → YouTube Metadata

**The gap:** `clips.csv` has `title`/`description`/`tags` columns but they're always blank and manually filled. For banter/speech clips, Whisper can auto-suggest titles.

**What to add:**
- New tool: `tools/transcribe_clips.py`
- For each row in `clips.csv` where `approved=true` and `title` is blank:
  - Extract audio from the horizontal clip via ffmpeg
  - Run Whisper `tiny` model (CPU-capable, fast)
  - Truncate transcript to first meaningful sentence
  - Write suggested title back to CSV with `[whisper]` prefix so it's clearly a suggestion
- Flags:
  - `--model tiny|base|small` — trade speed for accuracy (default `tiny`)
  - `--dry-run` — print suggestions without writing

`upload_youtube.py` is unchanged — it just picks up whatever is in the `title` column.

---

## What We Evaluated and Skipped

| Idea | Why skipped |
|------|-------------|
| PySceneDetect | Our custom segmentation in `plan_video.py` is already tuned for continuous live footage; PySceneDetect is built for cut-based editing |
| YAMNet audio classification | Heavy 521-class neural net; librosa RMS gives 80% of the value with zero model weight |
| PyTorchVideo action recognition | Requires GPU, complex fine-tuning — not practical for this workflow |
| BART / HuggingFace sentiment | Overkill for a band's use case |
| Generative B-roll, trend scraping | Out of scope |
| Continuous learning / engagement feedback | Too complex, too early |
