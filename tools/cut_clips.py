import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.track_subject import write_tracked_vertical

FOOTAGE_DIR = Path("footage")
OUTPUT_DIR = Path(".tmp/clips")
VIDEO_EXTENSIONS = [".mp4", ".mov", ".MOV", ".MP4", ".m4v", ".M4V"]


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


def cut_clip(row, horizontal=True, vertical=True, source_override=None, debug=False):
    initial = Path(row["initial_video_title"].strip()).stem
    start = row["start"].strip()
    end = row["end"].strip()
    cut = row["cut_title"].strip()

    source = Path(source_override) if source_override else find_source(initial)
    if not source:
        print(f"[SKIP] No source file found for '{initial}' in footage/")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    duration = to_seconds(end) - to_seconds(start)
    if vertical and duration > 60:
        print(f"[WARN] '{cut}' is {duration:.1f}s — exceeds YouTube Shorts 60s limit for vertical")

    h_out = OUTPUT_DIR / f"{initial}-{cut}-horizontal.mp4"
    v_out = OUTPUT_DIR / f"{initial}-{cut}-vertical.mp4"

    if horizontal:
        if h_out.exists():
            print(f"[SKIP] {h_out.name} already exists")
        else:
            print(f"[CUT]  {h_out.name}")
            subprocess.run([
                "ffmpeg", "-y", "-i", str(source),
                "-ss", start, "-to", end,
                "-vf", "scale=1920:1080",
                "-c:v", "libx264", "-c:a", "aac", "-crf", "18",
                str(h_out)
            ], check=True, stderr=subprocess.DEVNULL)
            print(f"[DONE] {h_out}")

    if vertical:
        if v_out.exists():
            print(f"[SKIP] {v_out.name} already exists")
        else:
            print(f"[CUT]  {v_out.name}")
            write_tracked_vertical(str(source), to_seconds(start), to_seconds(end), str(v_out), debug=debug)
            print(f"[DONE] {v_out}")


def main():
    args = sys.argv[1:]
    horizontal = "-v" not in args
    vertical = "-h" not in args
    debug = "-d" in args

    source_override = None
    filtered = []
    i = 0
    while i < len(args):
        if args[i] == "-s" and i + 1 < len(args):
            source_override = args[i + 1]
            i += 2
        elif args[i] not in ("-h", "-v", "-d"):
            filtered.append(args[i])
            i += 1
        else:
            i += 1

    if not filtered:
        print("Usage: python tools/cut_clips.py [-h] [-v] [-s /path/to/video] clips.csv")
        print("  -h              horizontal only")
        print("  -v              vertical only")
        print("  -s /path/to/video  override source video (skips footage/ lookup)")
        sys.exit(1)

    print("[START] cut_clips running..." + (" [DEBUG]" if debug else ""))
    with open(filtered[0], newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cut_clip(row, horizontal=horizontal, vertical=vertical, source_override=source_override, debug=debug)
    print("[DONE]  cut_clips finished.")


if __name__ == "__main__":
    main()
