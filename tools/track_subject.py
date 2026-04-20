import subprocess
import sys
import numpy as np
import cv2
from ultralytics import YOLO

_body_model = None
_face_cascade = None


def _get_body_model():
    global _body_model
    if _body_model is None:
        _body_model = YOLO("yolov8n.pt")
    return _body_model


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def _detect_frames(cap, start_frame, end_frame, default_x, crop_w, sample_every=3):
    """Run face detection (falling back to body) on sampled frames.
    Returns (per-frame crop_x list, per-frame annotations)."""
    body_model = _get_body_model()
    face_cascade = _get_face_cascade()
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

        h, fw = frame.shape[:2]
        small = cv2.resize(frame, (fw // 2, h // 2))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Try face detection first (Haar cascade, results in half-res coords)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) > 0:
            # faces = (x, y, w, h) — pick largest, scale to full res
            areas = faces[:, 2] * faces[:, 3]
            x, y, w, h_f = faces[areas.argmax()] * 2
            selected = np.array([x, y, x + w, y + h_f], dtype=float)
            center_x = x + w / 2.0
            crop_x = float(np.clip(center_x - crop_w / 2, 0, fw - crop_w))
            raw_xs[i] = crop_x
            annotations[i] = (selected.reshape(1, 4), selected, "face")
            continue

        # Fall back to body detection
        body_results = body_model(small, classes=[0], verbose=False, imgsz=640)
        if body_results and body_results[0].boxes is not None and len(body_results[0].boxes):
            boxes = body_results[0].boxes.xyxy.cpu().numpy() * 2.0
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            selected = boxes[areas.argmax()]
            center_x = (selected[0] + selected[2]) / 2.0
            crop_x = float(np.clip(center_x - crop_w / 2, 0, fw - crop_w))
            raw_xs[i] = crop_x
            annotations[i] = (boxes, selected, "body")

    # Forward-fill None values; fall back to default_x
    last = default_x
    for i in range(total):
        if raw_xs[i] is not None:
            last = raw_xs[i]
        else:
            raw_xs[i] = last

    return raw_xs, annotations


def _smooth(xs, alpha=0.5):
    """Exponential moving average smoothing."""
    smoothed = [xs[0]]
    for x in xs[1:]:
        smoothed.append(alpha * x + (1 - alpha) * smoothed[-1])
    return smoothed


def detect_crop_trajectory(video_path, start_s, end_s, alpha=0.5):
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
    return _smooth(raw_xs, alpha=alpha)


def write_tracked_vertical(video_path, start_s, end_s, output_path, alpha=0.5, debug=False):
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
    smooth_xs = _smooth(raw_xs, alpha=alpha)

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
