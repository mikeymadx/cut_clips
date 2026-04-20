import numpy as np
import cv2
from ultralytics import YOLO
from tools.color_utils import pick_by_hint

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


def detect_bodies_tracked(small_frame, full_scale=2.0, max_subjects=5, all_subjects=False):
    """Run YOLOv8 body detection with ByteTrack tracking.

    small_frame: pre-resized frame passed to the model
    full_scale: multiply bbox coords by this to recover full-res coordinates
    Returns (boxes, track_ids) or (None, None) if no detections.
      boxes: np.ndarray (N, 4) xyxy at full resolution, sorted by area desc
      track_ids: np.ndarray (N,) int track IDs; -1 if unassigned by tracker
    """
    model = _get_body_model()
    results = model.track(small_frame, classes=[0], verbose=False, imgsz=640,
                          persist=True, tracker="bytetrack.yaml")
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, None

    boxes = results[0].boxes.xyxy.cpu().numpy() * full_scale
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    raw_ids = results[0].boxes.id
    track_ids = (raw_ids.cpu().numpy().astype(int)
                 if raw_ids is not None
                 else np.full(len(boxes), -1, dtype=int))

    sorted_idx = np.argsort(areas)[::-1]
    if not all_subjects:
        sorted_idx = sorted_idx[:min(max_subjects, 5)]

    return boxes[sorted_idx], track_ids[sorted_idx]


def reset_body_tracker():
    """Reset ByteTrack state. Call before processing a new video or clip."""
    global _body_model
    if (_body_model is not None
            and hasattr(_body_model, 'predictor')
            and _body_model.predictor is not None):
        _body_model.predictor = None


def detect_subject(frame, fw, fh, crop_w, subject_hint=None, max_subjects=3, all_subjects=False):
    """Detect subject using Haar cascade face detection with YOLOv8 body fallback.

    Returns (crop_x, annotation) or None if nothing detected.
    annotation = (all_boxes, selected_idx, mode_str)
      all_boxes: np.ndarray shape (N, 4) — all detected candidates, sorted by area desc
      selected_idx: int index into all_boxes for the tracked subject
      mode_str: 'face' or 'body'

    subject_hint: color name (e.g. 'yellow') — picks the candidate whose torso best
                  matches; falls back to largest-area if no good match.
    max_subjects: max candidates returned (default 3, clamped to 5). Ignored if all_subjects=True.
    all_subjects: if True, return every detected person with no cap.
    """
    face_cascade = _get_face_cascade()
    body_model = _get_body_model()

    small = cv2.resize(frame, (fw // 2, fh // 2))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) > 0:
        areas = faces[:, 2] * faces[:, 3]
        sorted_idx = np.argsort(areas)[::-1]
        if not all_subjects:
            sorted_idx = sorted_idx[:min(max_subjects, 5)]
        all_boxes = np.array([
            np.array([x * 2, y * 2, (x + w) * 2, (y + h) * 2], dtype=float)
            for x, y, w, h in faces[sorted_idx]
        ])

        best_idx = pick_by_hint(frame, all_boxes, subject_hint) if subject_hint and len(all_boxes) > 1 else 0
        selected = all_boxes[best_idx]
        center_x = (selected[0] + selected[2]) / 2.0
        crop_x = float(np.clip(center_x - crop_w / 2, 0, fw - crop_w))
        return crop_x, (all_boxes, best_idx, "face")

    # Fall back to body detection
    body_results = body_model(small, classes=[0], verbose=False, imgsz=640)
    if body_results and body_results[0].boxes is not None and len(body_results[0].boxes):
        boxes = body_results[0].boxes.xyxy.cpu().numpy() * 2.0
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_idx = np.argsort(areas)[::-1]
        if not all_subjects:
            sorted_idx = sorted_idx[:min(max_subjects, 5)]
        all_boxes = boxes[sorted_idx]

        best_idx = pick_by_hint(frame, all_boxes, subject_hint) if subject_hint and len(all_boxes) > 1 else 0
        selected = all_boxes[best_idx]
        center_x = (selected[0] + selected[2]) / 2.0
        crop_x = float(np.clip(center_x - crop_w / 2, 0, fw - crop_w))
        return crop_x, (all_boxes, best_idx, "body")

    return None
