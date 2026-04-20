import numpy as np
import cv2

# OpenCV HSV scale: H=0-180, S=0-255, V=0-255
NAMED_COLORS = {
    "yellow":  [(20, 40)],
    "orange":  [(10, 20)],
    "red":     [(0, 10), (160, 180)],
    "green":   [(40, 80)],
    "cyan":    [(80, 100)],
    "blue":    [(100, 130)],
    "purple":  [(130, 160)],
    "pink":    [(155, 175)],
    "white":   None,
    "black":   None,
}


def score_color(frame, bbox, color_name):
    """Score how well the torso region of bbox matches color_name. Returns 0.0–1.0.

    Samples the middle 40–75% of the bounding box height (skipping head/face)
    and computes what fraction of pixels match the target color in HSV space.
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    bh = y2 - y1
    ty1 = y1 + int(bh * 0.40)
    ty2 = y1 + int(bh * 0.75)
    if ty2 <= ty1 or x2 <= x1:
        return 0.0
    roi = frame[ty1:ty2, x1:x2]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    total = H.size
    name = color_name.lower().strip()
    if name == "white":
        return float(np.sum(S < 50) / total)
    if name == "black":
        return float(np.sum(V < 50) / total)
    ranges = NAMED_COLORS.get(name)
    if ranges is None:
        return 0.0
    sat_mask = S > 40
    match = np.zeros(H.shape, dtype=bool)
    for (h_lo, h_hi) in ranges:
        match |= (H >= h_lo) & (H <= h_hi)
    return float(np.sum(match & sat_mask) / total)


def pick_by_hint(frame, all_boxes, subject_hint):
    """Return the index in all_boxes whose torso best matches subject_hint color.
    Falls back to index 0 if no box scores above the minimum threshold.
    """
    scores = [score_color(frame, box, subject_hint) for box in all_boxes]
    best_idx = int(np.argmax(scores))
    return best_idx if scores[best_idx] >= 0.05 else 0
