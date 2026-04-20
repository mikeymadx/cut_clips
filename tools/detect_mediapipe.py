import urllib.request
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tools.detect_yolo import _get_body_model

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


def detect_subject(frame, fw, fh, crop_w):
    """Detect subject using MediaPipe face detection with YOLOv8 body fallback.
    Returns (crop_x, annotation) or None if nothing detected.
    annotation = (all_boxes, selected_box, mode_str)
    """
    face_detector = _get_face_detector()

    small = cv2.resize(frame, (fw // 2, fh // 2))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_detector.detect(mp_image)

    if result.detections:
        # Pick highest-confidence detection
        best = max(result.detections, key=lambda d: d.categories[0].score)
        bb = best.bounding_box  # pixel coords on the half-res image
        x = bb.origin_x * 2
        y = bb.origin_y * 2
        w = bb.width * 2
        h = bb.height * 2
        selected = np.array([x, y, x + w, y + h], dtype=float)
        center_x = x + w / 2.0
        crop_x = float(np.clip(center_x - crop_w / 2, 0, fw - crop_w))
        return crop_x, (selected.reshape(1, 4), selected, "face")

    # Fall back to YOLOv8 body detection
    body_model = _get_body_model()
    body_results = body_model(small, classes=[0], verbose=False, imgsz=640)
    if body_results and body_results[0].boxes is not None and len(body_results[0].boxes):
        boxes = body_results[0].boxes.xyxy.cpu().numpy() * 2.0
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        selected = boxes[areas.argmax()]
        center_x = (selected[0] + selected[2]) / 2.0
        crop_x = float(np.clip(center_x - crop_w / 2, 0, fw - crop_w))
        return crop_x, (boxes, selected, "body")

    return None
