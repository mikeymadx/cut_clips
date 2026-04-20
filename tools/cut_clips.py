import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.track_subject import write_tracked_vertical, write_debug_horizontal

FOOTAGE_DIR = Path("footage")
OUTPUT_DIR = Path(".tmp/clips")
VIDEO_EXTENSIONS = [".mp4", ".mov", ".MOV", ".MP4", ".m4v", ".M4V"]
TRIM_THRESHOLD = 2.0  # seconds; gap must exceed this to trigger a trim rule


def find_source(initial_video_title):
    for ext in VIDEO_EXTENSIONS:
        for candidate in FOOTAGE_DIR.rglob(f"{initial_video_title}{ext}"):
            return candidate
    return None


def to_seconds(time_str):
    parts = time_str.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(time_str)


def get_video_duration(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    return 0.0


def load_planning_hints_json(json_path):
    """Load a planning JSON and return list of (seg_num, start_s, end_s, subject_focus)."""
    with open(json_path, encoding="utf-8") as f:
        scenes = json.load(f)
    return [
        (int(s["segment"]), float(s["start_s"]), float(s["end_s"]),
         s.get("subject_focus", "").strip().lower())
        for s in scenes
    ]


def load_planning_hints(planning_csv_path):
    """Load a planning CSV and return list of (seg_num, start_s, end_s, subject_focus)."""
    hints = []
    with open(planning_csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sf = row.get("subject_focus", "").strip().lower()
            hints.append((int(row["segment"]), float(row["start_s"]), float(row["end_s"]), sf))
    return hints


def get_hint_for_clip(hints, clip_start_s, clip_end_s):
    """Return the subject_focus from the planning segment with the largest overlap with this clip."""
    best_hint = ""
    best_overlap = 0.0
    for (_, seg_start, seg_end, hint) in hints:
        overlap = max(0.0, min(clip_end_s, seg_end) - max(clip_start_s, seg_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_hint = hint
    return best_hint or None


def cut_clip(row, horizontal=True, vertical=True, source_override=None, debug=False,
             planning_hints=None, max_subjects=3, all_subjects=False, trim_range=None):
    initial = Path(row["initial_video_title"].strip()).stem
    start = row["start"].strip()
    end = row["end"].strip()
    cut = row["cut_title"].strip()

    source = Path(source_override) if source_override else find_source(initial)
    if not source:
        print(f"[SKIP] No source file found for '{initial}' in footage/")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start_s = to_seconds(start)
    end_s = to_seconds(end)
    source_duration = get_video_duration(source)

    if trim_range is not None:
        trim_start, trim_end = trim_range
        effective_start = trim_start if trim_start > TRIM_THRESHOLD else 0.0
        effective_end = trim_end if (source_duration - trim_end) > TRIM_THRESHOLD else source_duration
        if effective_start > 0.0 and start_s < effective_start:
            print(f"[TRIM] Scene 1 starts at {effective_start:.2f}s — trimming clip start from {start_s:.2f}s")
            start_s = effective_start
        if effective_end < source_duration and end_s > effective_end:
            print(f"[TRIM] Last scene ends at {effective_end:.2f}s — trimming clip end from {end_s:.2f}s")
            end_s = effective_end

    duration = end_s - start_s

    print(f"[INFO] Source: {source.name}  ({source_duration:.1f}s total)")
    print(f"[INFO] Cut window: {start_s:.2f}s → {end_s:.2f}s  ({duration:.1f}s)")

    if vertical and duration > 60:
        print(f"[WARN] '{cut}' is {duration:.1f}s — exceeds YouTube Shorts 60s limit for vertical")

    h_out = OUTPUT_DIR / f"{initial}-{cut}-horizontal{'_DEBUG' if debug else ''}.mp4"
    v_out = OUTPUT_DIR / f"{initial}-{cut}-vertical{'_DEBUG' if debug else ''}.mp4"

    if horizontal:
        if h_out.exists():
            print(f"[SKIP] {h_out.name} already exists")
        elif debug:
            subject_hint = None
            if planning_hints:
                subject_hint = get_hint_for_clip(planning_hints, start_s, end_s)
                if subject_hint:
                    print(f"[INFO] Subject hint from planning CSV: '{subject_hint}'")
            print(f"[CUT]  {h_out.name}")
            write_debug_horizontal(str(source), start_s, end_s, str(h_out),
                                   subject_hint=subject_hint, max_subjects=max_subjects,
                                   all_subjects=all_subjects, planning_segments=planning_hints)
            print(f"[DONE] {h_out}")
        else:
            print(f"[CUT]  {h_out.name}")
            subprocess.run([
                "ffmpeg", "-y", "-i", str(source),
                "-ss", str(start_s), "-to", str(end_s),
                "-vf", "scale=1920:1080",
                "-c:v", "libx264", "-c:a", "aac", "-crf", "18",
                str(h_out)
            ], check=True, stderr=subprocess.DEVNULL)
            print(f"[DONE] {h_out}")

    if vertical:
        if v_out.exists():
            print(f"[SKIP] {v_out.name} already exists")
        else:
            subject_hint = None
            if planning_hints:
                subject_hint = get_hint_for_clip(planning_hints, start_s, end_s)
                if subject_hint:
                    print(f"[INFO] Subject hint from planning CSV: '{subject_hint}'")
            print(f"[CUT]  {v_out.name}")
            write_tracked_vertical(str(source), start_s, end_s, str(v_out), debug=debug,
                                   subject_hint=subject_hint, max_subjects=max_subjects,
                                   all_subjects=all_subjects, planning_segments=planning_hints)
            print(f"[DONE] {v_out}")


def main():
    args = sys.argv[1:]
    horizontal = "-v" not in args
    vertical = "-h" not in args
    debug = "-d" in args
    all_subjects = "--all-subjects" in args

    source_override = None
    planning_csv = None
    from_json = None
    max_subjects = 3
    filtered = []
    i = 0
    while i < len(args):
        if args[i] == "-s" and i + 1 < len(args):
            source_override = args[i + 1]
            i += 2
        elif args[i] == "--planning-csv" and i + 1 < len(args):
            planning_csv = args[i + 1]
            i += 2
        elif args[i] == "--from-json" and i + 1 < len(args):
            from_json = args[i + 1]
            i += 2
        elif args[i] == "--max-subjects" and i + 1 < len(args):
            max_subjects = min(int(args[i + 1]), 5)
            i += 2
        elif args[i] not in ("-h", "-v", "-d", "--all-subjects"):
            filtered.append(args[i])
            i += 1
        else:
            i += 1

    if not filtered:
        print("Usage: python tools/cut_clips.py [-h] [-v] [-d] [-s /path/to/video]")
        print("                                 [--planning-csv path/to/scenes.csv]")
        print("                                 [--from-json path/to/scenes.json]")
        print("                                 [--max-subjects N] [--all-subjects]")
        print("                                 clips.csv")
        print("  -h                   horizontal only")
        print("  -v                   vertical only")
        print("  -d                   debug mode (annotated boxes, _DEBUG in filename)")
        print("  -s /path/to/video    override source video (skips footage/ lookup)")
        print("  --planning-csv PATH  planning CSV from plan_video.py; subject_focus")
        print("                       column used as per-scene color hint")
        print("  --from-json PATH     planning JSON from plan_video.py; applies trim rules")
        print("                       and per-scene subject_focus hints (takes precedence")
        print("                       over --planning-csv)")
        print("  --max-subjects N     max people to detect per frame (default: 3, max: 5)")
        print("  --all-subjects       detect all people in frame, no cap")
        sys.exit(1)

    planning_hints = None
    trim_range = None
    if from_json:
        planning_hints = load_planning_hints_json(from_json)
        with open(from_json, encoding="utf-8") as f:
            _scenes = json.load(f)
        trim_range = (float(_scenes[0]["start_s"]), float(_scenes[-1]["end_s"]))
        print(f"[INFO]  Planning JSON: {from_json} ({len(planning_hints)} segments)")
        print(f"[INFO]  Trim range   : {trim_range[0]:.2f}s → {trim_range[1]:.2f}s")
    elif planning_csv:
        planning_hints = load_planning_hints(planning_csv)
        print(f"[INFO]  Planning CSV : {planning_csv} ({len(planning_hints)} segments)")

    csv_path = filtered[0]
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"[START] cut_clips running{' [DEBUG]' if debug else ''}")
    print(f"[INFO]  Date/time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO]  CSV       : {csv_path}")
    print(f"[INFO]  Clips     : {len(rows)}")
    print(f"[INFO]  Subjects  : {'all' if all_subjects else max_subjects}")
    if source_override:
        print(f"[INFO]  Source override: {source_override}")

    for row in rows:
        cut = row["cut_title"].strip()
        t0 = time.time()
        cut_clip(row, horizontal=horizontal, vertical=vertical, source_override=source_override,
                 debug=debug, planning_hints=planning_hints, max_subjects=max_subjects,
                 all_subjects=all_subjects, trim_range=trim_range)
        elapsed = time.time() - t0
        print(f"[TIME]  '{cut}' completed in {elapsed:.1f}s")

    print("[DONE]  cut_clips finished.")


if __name__ == "__main__":
    main()
