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


def detect_subject(frame, fw, fh, crop_w):
    """Detect subject using Haar cascade face detection with YOLOv8 body fallback.
    Returns (crop_x, annotation) or None if nothing detected.
    annotation = (all_boxes, selected_box, mode_str)
    """
    face_cascade = _get_face_cascade()
    body_model = _get_body_model()

    small = cv2.resize(frame, (fw // 2, fh // 2))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Try face detection first (Haar cascade, results in half-res coords)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) > 0:
        areas = faces[:, 2] * faces[:, 3]
        x, y, w, h_f = faces[areas.argmax()] * 2
        selected = np.array([x, y, x + w, y + h_f], dtype=float)
        center_x = x + w / 2.0
        crop_x = float(np.clip(center_x - crop_w / 2, 0, fw - crop_w))
        return crop_x, (selected.reshape(1, 4), selected, "face")

    # Fall back to body detection
    body_results = body_model(small, classes=[0], verbose=False, imgsz=640)
    if body_results and body_results[0].boxes is not None and len(body_results[0].boxes):
        boxes = body_results[0].boxes.xyxy.cpu().numpy() * 2.0
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        selected = boxes[areas.argmax()]
        center_x = (selected[0] + selected[2]) / 2.0
        crop_x = float(np.clip(center_x - crop_w / 2, 0, fw - crop_w))
        return crop_x, (boxes, selected, "body")

    return None
