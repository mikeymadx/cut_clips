#!/usr/bin/env python3
"""Video planning tool: analyze subject tracking across a full video and output
a debug video + scene breakdown (CSV + JSON) to inform an edit/panning plan.

Usage:
    python tools/plan_video.py <video_path> [--jump-threshold 150] [--min-segment 2.0] [--audio]
"""

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.detect_mediapipe import detect_subject
from tools.track_subject import _median_filter, _smooth

OUTPUT_DIR = Path(".tmp/planning")

SCENE_FIELDS = [
    "segment", "start_s", "end_s", "duration_s", "trigger",
    "dominant_mode", "detection_rate", "avg_crop_x", "crop_x_std",
    "label", "subject_focus", "description", "pan_plan",
]


def _detect_all_frames(cap, total_frames, fw, fh, crop_w, sample_every):
    default_x = (fw - crop_w) // 2
    raw_xs = [None] * total_frames
    raw_modes = [None] * total_frames
    annotations = [None] * total_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    actual_total = total_frames
    for i in range(total_frames):
        if i % 300 == 0 and i > 0:
            print(f"[PLAN]  Frame {i}/{total_frames}...")
        ret, frame = cap.read()
        if not ret:
            actual_total = i
            break
        if i % sample_every != 0:
            continue
        result = detect_subject(frame, fw, fh, crop_w)
        if result is not None:
            raw_xs[i] = result[0]
            raw_modes[i] = result[1][2]
            annotations[i] = result[1]

    raw_xs = raw_xs[:actual_total]
    raw_modes = raw_modes[:actual_total]
    annotations = annotations[:actual_total]
    detection_mask = [v is not None for v in raw_xs]

    # Linear interpolation for crop_x gaps
    known = [i for i, v in enumerate(raw_xs) if v is not None]
    if not known:
        raw_xs = [float(default_x)] * actual_total
    else:
        for i in range(known[0]):
            raw_xs[i] = float(default_x)
        for i in range(known[-1] + 1, actual_total):
            raw_xs[i] = float(raw_xs[known[-1]])
        for a, b in zip(known, known[1:]):
            if b - a <= 1:
                continue
            steps = b - a
            for j in range(1, steps):
                raw_xs[a + j] = raw_xs[a] + (j / steps) * (raw_xs[b] - raw_xs[a])

    # Forward-fill modes
    last = raw_modes[known[0]] if known else "body"
    filled_modes = list(raw_modes)
    for i in range(actual_total):
        if filled_modes[i] is not None:
            last = filled_modes[i]
        else:
            filled_modes[i] = last

    return raw_xs, filled_modes, detection_mask, annotations, actual_total


def _rolling_dominant_mode(modes, window=11):
    half = window // 2
    n = len(modes)
    result = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_modes = [m for m in modes[lo:hi] if m is not None]
        result.append(Counter(window_modes).most_common(1)[0][0] if window_modes else "body")
    return result


def _build_segments(smooth_xs, dom_modes, fps, jump_threshold, min_segment_s):
    n = len(smooth_xs)
    min_frames = max(1, int(min_segment_s * fps))
    breaks = [(0, "start")]

    for i in range(1, n):
        last_break = breaks[-1][0]
        if i - last_break < min_frames:
            continue
        if dom_modes[i] != dom_modes[i - 1]:
            breaks.append((i, "mode_change"))
        elif abs(smooth_xs[i] - smooth_xs[i - 1]) > jump_threshold:
            breaks.append((i, "crop_jump"))

    segments = []
    for idx, (start_f, trigger) in enumerate(breaks):
        end_f = breaks[idx + 1][0] if idx + 1 < len(breaks) else n
        segments.append((start_f, end_f, trigger))

    return segments


def _compute_metrics(segments, smooth_xs, filled_modes, detection_mask, fps, sample_every):
    rows = []
    for seg_idx, (start_f, end_f, trigger) in enumerate(segments):
        xs = smooth_xs[start_f:end_f]
        seg_modes = filled_modes[start_f:end_f]
        sampled = [i for i in range(start_f, end_f) if i % sample_every == 0]
        detected = sum(1 for i in sampled if i < len(detection_mask) and detection_mask[i])
        detection_rate = detected / len(sampled) if sampled else 0.0
        mode_counts = Counter(seg_modes)
        dominant = mode_counts.most_common(1)[0][0] if mode_counts else "none"
        rows.append({
            "segment": seg_idx + 1,
            "start_s": round(start_f / fps, 2),
            "end_s": round(end_f / fps, 2),
            "duration_s": round((end_f - start_f) / fps, 2),
            "trigger": trigger,
            "dominant_mode": dominant,
            "detection_rate": round(detection_rate, 3),
            "avg_crop_x": round(float(np.mean(xs)), 1),
            "crop_x_std": round(float(np.std(xs)), 1),
            "label": "",
            "subject_focus": "",
            "description": "",
            "pan_plan": "",
        })
    return rows


def _write_debug_video(video_path, total_frames, smooth_xs, filled_modes, annotations,
                       segments, fps, fw, fh, crop_w, output_path, audio):
    flash_frames = max(1, int(0.5 * fps))

    # Precompute per-frame segment index
    frame_seg = [0] * total_frames
    for seg_idx, (start_f, end_f, _) in enumerate(segments):
        for f in range(start_f, min(end_f, total_frames)):
            frame_seg[f] = seg_idx

    # Precompute flash frame set (frames right after a segment boundary)
    boundary_starts = {start_f for start_f, _, _ in segments if start_f > 0}
    flash_set = set()
    for bf in boundary_starts:
        for f in range(bf, min(bf + flash_frames, total_frames)):
            flash_set.add(f)

    fps_str = str(fps)
    size_str = f"{fw}x{fh}"

    if audio:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", size_str, "-pix_fmt", "bgr24", "-r", fps_str,
            "-i", "pipe:0",
            "-i", video_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-crf", "18",
            "-shortest",
            str(output_path),
        ]
    else:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", size_str, "-pix_fmt", "bgr24", "-r", fps_str,
            "-i", "pipe:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(output_path),
        ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    last_annotation = None

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        crop_x = int(smooth_xs[i])
        mode = filled_modes[i]
        seg_idx = frame_seg[i]
        t = i / fps

        if annotations[i] is not None:
            last_annotation = annotations[i]

        # Detection boxes
        if last_annotation is not None:
            all_boxes, selected_box, ann_mode = last_annotation
            box_color = (0, 255, 0) if ann_mode == "face" else (0, 255, 255)
            for box in all_boxes:
                cv2.rectangle(frame,
                    (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                    (180, 180, 180), 1)
            cv2.rectangle(frame,
                (int(selected_box[0]), int(selected_box[1])),
                (int(selected_box[2]), int(selected_box[3])),
                box_color, 3)

        # Crop window left/right edge lines
        cv2.line(frame, (crop_x, 0), (crop_x, fh), (255, 100, 0), 2)
        cv2.line(frame, (crop_x + crop_w, 0), (crop_x + crop_w, fh), (255, 100, 0), 2)

        # Segment boundary flash: white bar across top
        if i in flash_set:
            cv2.rectangle(frame, (0, 0), (fw, 10), (255, 255, 255), -1)

        # Segment info overlay (black shadow + white text)
        label = f"Seg {seg_idx + 1} | {mode} | {t:.1f}s"
        cv2.putText(frame, label, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(frame, label, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze subject tracking and output debug video + scene breakdown.")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--jump-threshold", type=float, default=150.0,
        help="Crop-x pixel change to trigger a segment break (default: 150)")
    parser.add_argument("--min-segment", type=float, default=2.0,
        help="Minimum segment duration in seconds (default: 2.0)")
    parser.add_argument("--audio", action="store_true",
        help="Mux original audio into the debug video")
    args = parser.parse_args()

    video_path = args.video_path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0.0
    crop_w = int(fh * 9 / 16)
    sample_every = 1 if total_frames < 10 else 3

    stem = Path(video_path).stem
    out_dir = OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PLAN]  Video     : {video_path}")
    print(f"[PLAN]  Size      : {fw}x{fh}  FPS: {fps:.1f}  Duration: {duration_s:.1f}s")
    print(f"[PLAN]  Frames    : {total_frames}  (sampling every {sample_every})")
    print(f"[PLAN]  Output dir: {out_dir}/")

    raw_xs, filled_modes, detection_mask, annotations, total_frames = _detect_all_frames(
        cap, total_frames, fw, fh, crop_w, sample_every
    )
    cap.release()

    smooth_xs = _smooth(_median_filter(raw_xs, kernel=7), sigma=3.0)
    dom_modes = _rolling_dominant_mode(filled_modes, window=11)
    segments = _build_segments(smooth_xs, dom_modes, fps, args.jump_threshold, args.min_segment)

    print(f"[PLAN]  Segments  : {len(segments)}")

    metrics = _compute_metrics(segments, smooth_xs, filled_modes, detection_mask, fps, sample_every)

    csv_path = out_dir / f"{stem}_scenes_planning.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCENE_FIELDS)
        writer.writeheader()
        writer.writerows(metrics)
    print(f"[PLAN]  CSV       : {csv_path}")

    json_path = out_dir / f"{stem}_scenes_planning.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[PLAN]  JSON      : {json_path}")

    debug_path = out_dir / f"{stem}_debug_planning.mp4"
    print(f"[PLAN]  Writing debug video...")
    _write_debug_video(
        video_path, total_frames, smooth_xs, filled_modes, annotations,
        segments, fps, fw, fh, crop_w, debug_path, audio=args.audio,
    )
    print(f"[PLAN]  Debug     : {debug_path}")
    print(f"[PLAN]  Done.")


if __name__ == "__main__":
    main()
