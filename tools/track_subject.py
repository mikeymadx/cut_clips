import subprocess
import sys
import numpy as np
import cv2
from tools.detect_mediapipe import detect_subject, get_face_boxes
from tools.detect_yolo import detect_bodies_tracked, reset_body_tracker

try:
    from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d
    _SCIPY = True
except ImportError:
    _SCIPY = False


def _draw_segment_overlay(frame, current_time, planning_segments):
    """Draw active planning segment info in the top-left corner."""
    active = None
    for (seg_num, seg_start, seg_end, hint) in planning_segments:
        if seg_start <= current_time < seg_end:
            active = (seg_num, seg_start, seg_end, hint)
            break
    if active is None:
        return
    seg_num, seg_start, seg_end, hint = active
    lines = [
        f"SEG {seg_num}  {seg_start:.2f}s -> {seg_end:.2f}s",
        f"t={current_time:.2f}s",
        f"hint: {hint or 'none'}",
    ]
    x, y = 16, 48
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        y += 44


def _hint_for_frame(t_seconds, planning_segments):
    """Return the subject_focus hint active at t_seconds, or empty string."""
    for (_, seg_start, seg_end, hint) in planning_segments:
        if seg_start <= t_seconds < seg_end:
            return hint or ""
    return ""


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


def _pick_focus_track(track_obs, total_frames):
    """Return the track ID with the highest composite importance score."""
    if not track_obs:
        return None
    scores = {}
    for tid, obs in track_obs.items():
        n = len(obs['areas'])
        if n == 0:
            continue
        avg_area = float(np.mean(obs['areas']))
        avg_center_score = 1.0 - float(np.mean(obs['center_dists']))
        age_norm = n / max(total_frames, 1)
        has_face = float(obs['has_face'])
        scores[tid] = avg_area * 0.35 + has_face * 0.25 + age_norm * 0.20 + avg_center_score * 0.15
    return max(scores, key=lambda t: scores[t]) if scores else None


def _detect_frames(cap, start_frame, end_frame, default_x, crop_w, sample_every=3,
                   subject_hint=None, max_subjects=3, all_subjects=False,
                   start_s=0.0, fps=30.0, planning_segments=None):
    """Run body tracking on every frame to identify the focus subject,
    then return per-frame crop_x values locked to that subject.

    Returns (per-frame crop_x list, per-frame annotations).
    """
    total = end_frame - start_frame
    raw_xs = [None] * total
    annotations = [None] * total
    _raw_annotations = [None] * total  # (boxes, track_ids, mode) before focus selection

    fw_ref = fh_ref = None
    frame_area = cx_frame = cy_frame = max_dist = None
    track_obs = {}

    reset_body_tracker()
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        if fw_ref is None:
            fw_ref, fh_ref = fw, fh
            frame_area = fw * fh
            cx_frame, cy_frame = fw / 2.0, fh / 2.0
            max_dist = (cx_frame ** 2 + cy_frame ** 2) ** 0.5

        small = cv2.resize(frame, (fw // 2, fh // 2))
        body_boxes, body_track_ids = detect_bodies_tracked(
            small, full_scale=2.0,
            max_subjects=max(max_subjects, 5),
            all_subjects=all_subjects,
        )

        face_boxes = np.empty((0, 4), dtype=float)
        if i % sample_every == 0 and body_boxes is not None:
            face_boxes = get_face_boxes(frame, fw, fh)

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
                        'areas': [], 'center_dists': [], 'has_face': False,
                        'xs': [], 'frame_indices': [],
                    }
                crop_x = float(np.clip(cx - crop_w / 2.0, 0, fw - crop_w))
                track_obs[tid]['areas'].append(area_norm)
                track_obs[tid]['center_dists'].append(center_dist_norm)
                track_obs[tid]['has_face'] = track_obs[tid]['has_face'] or has_face
                track_obs[tid]['xs'].append((i, crop_x))
                track_obs[tid]['frame_indices'].append(i)

            if i % sample_every == 0:
                mode = "face" if len(face_boxes) > 0 else "body"
                _raw_annotations[i] = (body_boxes.copy(), body_track_ids.copy(), mode)

    # Pick the focus track and build raw_xs from its crop positions
    focus_tid = _pick_focus_track(track_obs, total)

    if focus_tid is not None and focus_tid in track_obs:
        for frame_i, crop_x in track_obs[focus_tid]['xs']:
            raw_xs[frame_i] = crop_x

    # Build final annotations with correct selected_idx pointing to focus track
    for i in range(total):
        if _raw_annotations[i] is not None:
            boxes, tids, mode = _raw_annotations[i]
            focus_idxs = np.where(tids == focus_tid)[0] if focus_tid is not None else []
            selected_idx = int(focus_idxs[0]) if len(focus_idxs) > 0 else 0
            annotations[i] = (boxes, selected_idx, mode)

    # Fall back to detect_subject on sampled frames if tracking produced nothing
    if all(v is None for v in raw_xs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if i % sample_every != 0:
                continue
            fh, fw = frame.shape[:2]
            frame_hint = (
                _hint_for_frame(start_s + i / fps, planning_segments)
                if planning_segments else subject_hint
            )
            result = detect_subject(frame, fw, fh, crop_w,
                                    subject_hint=frame_hint,
                                    max_subjects=max_subjects,
                                    all_subjects=all_subjects)
            if result is not None:
                raw_xs[i], annotations[i] = result

    # Linear interpolation between known detections; edges use default_x / last known
    known_indices = [i for i, v in enumerate(raw_xs) if v is not None]

    if not known_indices:
        raw_xs = [default_x] * total
    else:
        for i in range(known_indices[0]):
            raw_xs[i] = default_x
        for i in range(known_indices[-1] + 1, total):
            raw_xs[i] = raw_xs[known_indices[-1]]
        for a, b in zip(known_indices, known_indices[1:]):
            if b - a <= 1:
                continue
            x_a, x_b = raw_xs[a], raw_xs[b]
            steps = b - a
            for j in range(1, steps):
                t = j / steps
                raw_xs[a + j] = x_a + t * (x_b - x_a)

    return raw_xs, annotations


def _median_filter(xs, kernel=7):
    """Rolling median to remove single-frame outlier detections."""
    if kernel <= 1:
        return list(xs)
    half = kernel // 2
    n = len(xs)
    result = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(float(np.median(xs[lo:hi])))
    return result


def _smooth(xs, alpha=0.5, sigma=3.0):
    """Gaussian smoothing (acausal, no lag). Falls back to EMA if scipy unavailable."""
    arr = np.array(xs, dtype=float)
    if _SCIPY:
        return list(_gaussian_filter1d(arr, sigma=sigma))
    smoothed = [arr[0]]
    for x in arr[1:]:
        smoothed.append(alpha * float(x) + (1 - alpha) * smoothed[-1])
    return smoothed


def detect_crop_trajectory(video_path, start_s, end_s, alpha=0.5, sigma=3.0, median_kernel=7,
                           subject_hint=None, max_subjects=3, all_subjects=False):
    """Return per-frame crop_x values tracking the focus subject."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_w = int(fh * 9 / 16)
    default_x = (fw - crop_w) // 2
    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)

    sample_every = 1 if (end_frame - start_frame) < 10 else 3
    raw_xs, _ = _detect_frames(cap, start_frame, end_frame, default_x, crop_w, sample_every,
                                subject_hint=subject_hint, max_subjects=max_subjects,
                                all_subjects=all_subjects, start_s=start_s, fps=fps)
    cap.release()
    return _smooth(_median_filter(raw_xs, kernel=median_kernel), alpha=alpha, sigma=sigma)


def write_tracked_vertical(video_path, start_s, end_s, output_path, alpha=0.5, sigma=3.0,
                           median_kernel=7, debug=False, subject_hint=None,
                           max_subjects=3, all_subjects=False, planning_segments=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_w = int(fh * 9 / 16)
    if crop_w >= fw:
        cap.release()
        print(f"[WARN] Source appears to already be vertical — using static center crop")
        _fallback_ffmpeg(video_path, start_s, end_s, output_path)
        return

    default_x = (fw - crop_w) // 2
    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)
    total_frames = end_frame - start_frame

    if subject_hint:
        print(f"[TRACK] Subject hint: '{subject_hint}'")
    print(f"[TRACK] Detecting subject in {total_frames} frames...")
    sample_every = 1 if total_frames < 10 else 3
    raw_xs, annotations = _detect_frames(cap, start_frame, end_frame, default_x, crop_w,
                                          sample_every, subject_hint=subject_hint,
                                          max_subjects=max_subjects, all_subjects=all_subjects,
                                          start_s=start_s, fps=fps,
                                          planning_segments=planning_segments)
    smooth_xs = _smooth(_median_filter(raw_xs, kernel=median_kernel), alpha=alpha, sigma=sigma)

    detected = sum(1 for x in raw_xs if x != default_x)
    if detected == 0:
        print(f"[WARN] No subject detected — falling back to center crop")

    fps_str = str(fps)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", "1080x1920", "-pix_fmt", "bgr24", "-r", fps_str,
        "-i", "pipe:0",
        "-ss", str(start_s), "-to", str(end_s),
        "-i", video_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-crf", "18",
        "-shortest",
        output_path
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    last_annotation = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i, crop_x in enumerate(smooth_xs):
        ret, frame = cap.read()
        if not ret:
            break

        if debug:
            if annotations[i] is not None:
                last_annotation = annotations[i]
            if last_annotation is not None:
                all_boxes, selected_idx, mode = last_annotation
                track_color = (0, 255, 0) if mode == "face" else (0, 255, 255)
                label = "FACE" if mode == "face" else "BODY"
                p_num = 1
                for bi, box in enumerate(all_boxes):
                    if bi == selected_idx:
                        cv2.rectangle(frame,
                            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                            track_color, 4)
                        cv2.putText(frame, label,
                            (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, track_color, 3)
                    else:
                        cv2.rectangle(frame,
                            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                            (180, 180, 180), 2)
                        lp = (int(box[0]) + 4, int(box[1]) + 34)
                        cv2.putText(frame, f"P{p_num}", lp, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
                        cv2.putText(frame, f"P{p_num}", lp, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
                        p_num += 1
            x = int(crop_x)
            cv2.line(frame, (x, 0), (x, fh), (255, 100, 0), 3)
            cv2.line(frame, (x + crop_w, 0), (x + crop_w, fh), (255, 100, 0), 3)
            if planning_segments:
                _draw_segment_overlay(frame, start_s + i / fps, planning_segments)

        x = int(crop_x)
        cropped = frame[:, x:x + crop_w]
        scaled = cv2.resize(cropped, (1080, 1920))
        proc.stdin.write(scaled.tobytes())

    proc.stdin.close()
    proc.wait()
    cap.release()


def write_debug_horizontal(video_path, start_s, end_s, output_path, subject_hint=None,
                           max_subjects=3, all_subjects=False, planning_segments=None):
    """Render a full-frame 1920x1080 debug video with detection boxes drawn."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w = int(fh * 9 / 16)

    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)
    total_frames = end_frame - start_frame
    sample_every = 1 if total_frames < 10 else 3

    print(f"[TRACK] Detecting subject in {total_frames} frames (horizontal debug)...")
    _, annotations = _detect_frames(cap, start_frame, end_frame, (fw - crop_w) // 2, crop_w,
                                    sample_every, subject_hint=subject_hint,
                                    max_subjects=max_subjects, all_subjects=all_subjects,
                                    start_s=start_s, fps=fps,
                                    planning_segments=planning_segments)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{fw}x{fh}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0",
        "-ss", str(start_s), "-to", str(end_s),
        "-i", video_path,
        "-map", "0:v", "-map", "1:a",
        "-vf", "scale=1920:1080",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-crf", "18",
        "-shortest", output_path
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    last_annotation = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if annotations[i] is not None:
            last_annotation = annotations[i]
        if last_annotation is not None:
            all_boxes, selected_idx, mode = last_annotation
            track_color = (0, 255, 0) if mode == "face" else (0, 255, 255)
            label = "FACE" if mode == "face" else "BODY"
            p_num = 1
            for bi, box in enumerate(all_boxes):
                if bi == selected_idx:
                    cv2.rectangle(frame,
                        (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                        track_color, 4)
                    cv2.putText(frame, label,
                        (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, track_color, 3)
                else:
                    cv2.rectangle(frame,
                        (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                        (180, 180, 180), 2)
                    lp = (int(box[0]) + 4, int(box[1]) + 34)
                    cv2.putText(frame, f"P{p_num}", lp, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
                    cv2.putText(frame, f"P{p_num}", lp, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
                    p_num += 1
        if planning_segments:
            _draw_segment_overlay(frame, start_s + i / fps, planning_segments)
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    cap.release()


def _fallback_ffmpeg(video_path, start_s, end_s, output_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ss", str(start_s), "-to", str(end_s),
        "-vf", "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-crf", "18",
        output_path
    ], check=True, stderr=subprocess.DEVNULL)
