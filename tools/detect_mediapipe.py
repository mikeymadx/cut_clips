import urllib.request
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tools.detect_yolo import _get_body_model
from tools.color_utils import pick_by_hint

_face_detector = None
_MODEL_PATH = Path(__file__).resolve().parent.parent / "blaze_face_short_range.tflite"
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        if not _MODEL_PATH.exists():
            print(f"[INFO] Downloading MediaPipe face model...")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print(f"[INFO] Model saved to {_MODEL_PATH.name}")
        base_options = mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH))
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.7
        )
        _face_detector = mp_vision.FaceDetector.create_from_options(options)
    return _face_detector


def get_face_boxes(frame, fw, fh):
    """Return all detected face bboxes as np.ndarray shape (N, 4) at full resolution.

    Returns empty array (shape (0, 4)) when no faces detected.
    """
    face_detector = _get_face_detector()
    small = cv2.resize(frame, (fw // 2, fh // 2))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_detector.detect(mp_image)
    if not result.detections:
        return np.empty((0, 4), dtype=float)
    boxes = []
    for det in result.detections:
        bb = det.bounding_box
        x, y = bb.origin_x * 2, bb.origin_y * 2
        w, h = bb.width * 2, bb.height * 2
        boxes.append([x, y, x + w, y + h])
    return np.array(boxes, dtype=float)


def detect_subject(frame, fw, fh, crop_w, subject_hint=None, max_subjects=3, all_subjects=False):
    """Detect subject using MediaPipe face detection with YOLOv8 body fallback.

    Returns (crop_x, annotation) or None if nothing detected.
    annotation = (all_boxes, selected_idx, mode_str)
      all_boxes: np.ndarray shape (N, 4) — all detected candidates, sorted by confidence
      selected_idx: int index into all_boxes for the tracked subject
      mode_str: 'face' or 'body'

    subject_hint: color name (e.g. 'yellow') — picks the candidate whose torso best
                  matches; falls back to highest-confidence if no good match.
    max_subjects: max candidates returned (default 3, clamped to 5). Ignored if all_subjects=True.
    all_subjects: if True, return every detected person with no cap.
    """
    face_detector = _get_face_detector()

    small = cv2.resize(frame, (fw // 2, fh // 2))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_detector.detect(mp_image)

    if result.detections:
        detections = sorted(result.detections, key=lambda d: d.categories[0].score, reverse=True)
        if not all_subjects:
            detections = detections[:min(max_subjects, 5)]

        all_boxes = []
        for det in detections:
            bb = det.bounding_box
            x, y = bb.origin_x * 2, bb.origin_y * 2
            w, h = bb.width * 2, bb.height * 2
            all_boxes.append(np.array([x, y, x + w, y + h], dtype=float))
        all_boxes = np.array(all_boxes)

        # Face boxes only cover the head — extend downward ~2x face height to reach the shirt
        # for color scoring only; crop centering still uses the original face boxes
        if subject_hint and len(all_boxes) > 1:
            extended = all_boxes.copy()
            for i, box in enumerate(extended):
                face_h = box[3] - box[1]
                extended[i][1] = box[3]                                   # top = bottom of face
                extended[i][3] = box[3] + face_h * 2.0                   # bottom = 2 face-heights below
            best_idx = pick_by_hint(frame, extended, subject_hint)
        else:
            best_idx = 0
        selected = all_boxes[best_idx]
        center_x = (selected[0] + selected[2]) / 2.0
        crop_x = float(np.clip(center_x - crop_w / 2, 0, fw - crop_w))
        return crop_x, (all_boxes, best_idx, "face")

    # Fall back to YOLOv8 body detection
    body_model = _get_body_model()
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
