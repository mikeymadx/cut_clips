import subprocess
import sys
import numpy as np
import cv2
from tools.detect_mediapipe import detect_subject

try:
    from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d
    _SCIPY = True
except ImportError:
    _SCIPY = False


def _detect_frames(cap, start_frame, end_frame, default_x, crop_w, sample_every=3):
    """Run subject detection on sampled frames.
    Returns (per-frame crop_x list, per-frame annotations)."""
    total = end_frame - start_frame
    raw_xs = [None] * total
    annotations = [None] * total

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        if i % sample_every != 0:
            continue

        fh, fw = frame.shape[:2]
        result = detect_subject(frame, fw, fh, crop_w)
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


def detect_crop_trajectory(video_path, start_s, end_s, alpha=0.5, sigma=3.0, median_kernel=7):
    """Return per-frame crop_x values tracking the subject's face (or body fallback)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_w = int(fh * 9 / 16)
    default_x = (fw - crop_w) // 2
    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)

    sample_every = 1 if (end_frame - start_frame) < 10 else 3
    raw_xs, _ = _detect_frames(cap, start_frame, end_frame, default_x, crop_w, sample_every)
    cap.release()
    return _smooth(_median_filter(raw_xs, kernel=median_kernel), alpha=alpha, sigma=sigma)


def write_tracked_vertical(video_path, start_s, end_s, output_path, alpha=0.5, sigma=3.0, median_kernel=7, debug=False):
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

    print(f"[TRACK] Detecting subject in {total_frames} frames...")
    sample_every = 1 if total_frames < 10 else 3
    raw_xs, annotations = _detect_frames(cap, start_frame, end_frame, default_x, crop_w, sample_every)
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
                all_boxes, selected_box, mode = last_annotation
                color = (0, 255, 0) if mode == "face" else (0, 255, 255)
                label = "FACE" if mode == "face" else "BODY"
                for box in all_boxes:
                    cv2.rectangle(frame,
                        (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                        (255, 255, 255), 2)
                cv2.rectangle(frame,
                    (int(selected_box[0]), int(selected_box[1])),
                    (int(selected_box[2]), int(selected_box[3])),
                    color, 4)
                cv2.putText(frame, label,
                    (int(selected_box[0]), int(selected_box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            x = int(crop_x)
            cv2.line(frame, (x, 0), (x, fh), (255, 100, 0), 3)
            cv2.line(frame, (x + crop_w, 0), (x + crop_w, fh), (255, 100, 0), 3)

        x = int(crop_x)
        cropped = frame[:, x:x + crop_w]
        scaled = cv2.resize(cropped, (1080, 1920))
        proc.stdin.write(scaled.tobytes())

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
