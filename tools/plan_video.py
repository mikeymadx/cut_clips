#!/usr/bin/env python3
"""Video planning tool: analyze subject tracking across a full video and output
a debug video + scene breakdown (CSV + JSON) to inform an edit/panning plan.

Usage:
    python tools/plan_video.py <video_path> [--jump-threshold 150] [--min-segment 2.0] [--audio]
                               [--max-subjects 3] [--all-subjects]
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.detect_mediapipe import detect_subject, get_face_boxes
from tools.detect_yolo import detect_bodies_tracked, reset_body_tracker
from tools.track_subject import _median_filter, _smooth

OUTPUT_DIR = Path(".tmp/planning")


def _face_overlaps_box(body_box, face_boxes):
    """Return True if any face box's center falls within body_box."""
    if len(face_boxes) == 0:
        return False
    x1, y1, x2, y2 = body_box
    for fb in face_boxes:
        fcx = (fb[0] + fb[2]) / 2.0
        fcy = (fb[1] + fb[3]) / 2.0
        if x1 <= fcx <= x2 and y1 <= fcy <= y2:
            return True
    return False


def _segment_track_metrics(track_obs, start_f, end_f):
    """Compute per-segment tracking importance metrics from accumulated track observations.

    Returns (num_tracks, focus_track_id, focus_importance, focus_has_face, focus_area_pct,
             tracks_detail) where tracks_detail is a list of all tracks ranked by importance.
    """
    seg_tracks = {}
    for tid, obs in track_obs.items():
        idx_in_seg = [j for j, fi in enumerate(obs['frame_indices']) if start_f <= fi < end_f]
        if not idx_in_seg:
            continue
        areas = [obs['areas'][j] for j in idx_in_seg]
        center_dists = [obs['center_dists'][j] for j in idx_in_seg]
        face_flags = [obs['face_flags'][j] for j in idx_in_seg]

        seg_len = end_f - start_f
        age_norm = len(idx_in_seg) / max(seg_len, 1)
        avg_area = float(np.mean(areas))
        avg_center_score = 1.0 - float(np.mean(center_dists))
        has_face = any(face_flags)

        importance = (avg_area * 0.35 + float(has_face) * 0.25
                      + age_norm * 0.20 + avg_center_score * 0.15)
        seg_tracks[tid] = {
            'importance': importance,
            'has_face': has_face,
            'avg_area': avg_area,
        }

    if not seg_tracks:
        return 0, -1, 0.0, 0, 0.0, []

    ranked = sorted(seg_tracks.items(), key=lambda kv: kv[1]['importance'], reverse=True)
    tracks_detail = [
        {
            "id": int(tid),
            "importance": round(info['importance'], 3),
            "has_face": int(info['has_face']),
            "area_pct": round(info['avg_area'] * 100.0, 1),
        }
        for tid, info in ranked
    ]

    focus_tid, focus = ranked[0]
    return (
        len(seg_tracks),
        int(focus_tid),
        round(focus['importance'], 3),
        int(focus['has_face']),
        round(focus['avg_area'] * 100.0, 1),
        tracks_detail,
    )


def _detect_all_frames(cap, total_frames, fw, fh, crop_w, sample_every,
                       max_subjects=3, all_subjects=False):
    default_x = (fw - crop_w) // 2
    raw_xs = [None] * total_frames
    raw_modes = [None] * total_frames
    annotations = [None] * total_frames
    track_obs = {}

    frame_area = fw * fh
    cx_frame = fw / 2.0
    cy_frame = fh / 2.0
    max_dist = (cx_frame ** 2 + cy_frame ** 2) ** 0.5

    reset_body_tracker()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    actual_total = total_frames

    for i in range(total_frames):
        if i % 300 == 0 and i > 0:
            print(f"[PLAN]  Frame {i}/{total_frames}...")
        ret, frame = cap.read()
        if not ret:
            actual_total = i
            break

        # Body tracking on every frame for track continuity
        small = cv2.resize(frame, (fw // 2, fh // 2))
        body_boxes, body_track_ids = detect_bodies_tracked(
            small, full_scale=2.0,
            max_subjects=max(max_subjects, 5),
            all_subjects=all_subjects,
        )

        # Face boxes only on sampled frames (expensive)
        face_boxes = np.empty((0, 4), dtype=float)
        if i % sample_every == 0 and body_boxes is not None:
            face_boxes = get_face_boxes(frame, fw, fh)

        # Accumulate per-track observations
        if body_boxes is not None:
            for box, tid in zip(body_boxes, body_track_ids):
                if tid < 0:
                    continue
                x1, y1, x2, y2 = box
                area_norm = (x2 - x1) * (y2 - y1) / frame_area
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                center_dist_norm = ((cx - cx_frame) ** 2 + (cy - cy_frame) ** 2) ** 0.5 / max_dist
                has_face = _face_overlaps_box(box, face_boxes)
                if tid not in track_obs:
                    track_obs[tid] = {
                        'areas': [], 'center_dists': [], 'face_flags': [], 'frame_indices': [],
                    }
                track_obs[tid]['areas'].append(area_norm)
                track_obs[tid]['center_dists'].append(center_dist_norm)
                track_obs[tid]['face_flags'].append(has_face)
                track_obs[tid]['frame_indices'].append(i)

        # Existing crop_x trajectory detection on sampled frames
        if i % sample_every != 0:
            continue
        result = detect_subject(frame, fw, fh, crop_w,
                                max_subjects=max_subjects, all_subjects=all_subjects)
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

    return raw_xs, filled_modes, detection_mask, annotations, actual_total, track_obs


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


def _compute_metrics(segments, smooth_xs, filled_modes, detection_mask, fps, sample_every, track_obs):
    rows = []
    for seg_idx, (start_f, end_f, trigger) in enumerate(segments):
        xs = smooth_xs[start_f:end_f]
        seg_modes = filled_modes[start_f:end_f]
        sampled = [i for i in range(start_f, end_f) if i % sample_every == 0]
        detected = sum(1 for i in sampled if i < len(detection_mask) and detection_mask[i])
        detection_rate = detected / len(sampled) if sampled else 0.0
        mode_counts = Counter(seg_modes)
        dominant = mode_counts.most_common(1)[0][0] if mode_counts else "none"

        num_tracks, focus_track_id, focus_importance, focus_has_face, focus_area_pct, tracks_detail = \
            _segment_track_metrics(track_obs, start_f, end_f)

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
            "num_tracks": num_tracks,
            "focus_track_id": focus_track_id,
            "focus_importance": focus_importance,
            "focus_has_face": focus_has_face,
            "focus_area_pct": focus_area_pct,
            "tracks_detail": tracks_detail,
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

        # Detection boxes — selected highlighted, others labeled P1/P2/P3
        if last_annotation is not None:
            all_boxes, selected_idx, ann_mode = last_annotation
            track_color = (0, 255, 0) if ann_mode == "face" else (0, 255, 255)
            p_num = 1
            for bi, box in enumerate(all_boxes):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                if bi == selected_idx:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, 3)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)
                    lp = (x1 + 4, y1 + 34)
                    cv2.putText(frame, f"P{p_num}", lp, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
                    cv2.putText(frame, f"P{p_num}", lp, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                    p_num += 1

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
    parser.add_argument("--max-subjects", type=int, default=3,
        help="Max number of people to detect per frame (default: 3, max: 5). Ignored with --all-subjects.")
    parser.add_argument("--all-subjects", action="store_true",
        help="Detect and display every person in frame with no cap. Overrides --max-subjects.")
    parser.add_argument("--from-json", default=None, metavar="PATH",
        help="Use scene boundaries from an existing planning JSON instead of auto-segmentation. "
             "User-filled fields (label, subject_focus, description, pan_plan) are preserved.")
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

    max_subjects = min(args.max_subjects, 5)
    all_subjects = args.all_subjects

    stem = Path(video_path).stem
    out_dir = OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PLAN]  Video     : {video_path}")
    print(f"[PLAN]  Size      : {fw}x{fh}  FPS: {fps:.1f}  Duration: {duration_s:.1f}s")
    print(f"[PLAN]  Frames    : {total_frames}  (sampling every {sample_every})")
    print(f"[PLAN]  Subjects  : {'all' if all_subjects else max_subjects}")
    print(f"[PLAN]  Output dir: {out_dir}/")

    raw_xs, filled_modes, detection_mask, annotations, total_frames, track_obs = \
        _detect_all_frames(
            cap, total_frames, fw, fh, crop_w, sample_every,
            max_subjects=max_subjects, all_subjects=all_subjects,
        )
    cap.release()

    smooth_xs = _smooth(_median_filter(raw_xs, kernel=7), sigma=3.0)
    dom_modes = _rolling_dominant_mode(filled_modes, window=11)

    user_fields = None
    if args.from_json:
        with open(args.from_json) as f:
            source = json.load(f)
        segments = [
            (int(round(s["start_s"] * fps)), int(round(s["end_s"] * fps)), s["trigger"])
            for s in source
        ]
        user_fields = [
            {k: s.get(k, "") for k in ("label", "subject_focus", "description", "pan_plan")}
            for s in source
        ]
        print(f"[PLAN]  Segments  : {len(segments)} (from JSON)")
    else:
        segments = _build_segments(smooth_xs, dom_modes, fps, args.jump_threshold, args.min_segment)
        print(f"[PLAN]  Segments  : {len(segments)}")

    print(f"[PLAN]  Tracks    : {len(track_obs)}")

    metrics = _compute_metrics(
        segments, smooth_xs, filled_modes, detection_mask, fps, sample_every, track_obs,
    )

    if user_fields:
        for row, uf in zip(metrics, user_fields):
            row.update(uf)

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
