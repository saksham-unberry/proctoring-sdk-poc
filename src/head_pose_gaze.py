"""
head_pose_gaze.py
=================
Head pose and gaze estimation using MediaPipe Face Landmarker.

No OpenAI / Gemini / Claude APIs used.

What this module provides
--------------------------
1. Head pose  — pitch (up/down), yaw (left/right), roll (tilt)
   Source: facial_transformation_matrixes from Face Landmarker
   → 4x4 camera-space matrix → decompose to Euler angles

2. Gaze direction — where the eyes are pointing (left / right / center)
   Source: iris landmarks 468-477 relative to eye corner landmarks
   → normalized offset → gaze_x ∈ [-1, 1]  (left to right)
   → normalized offset → gaze_y ∈ [-1, 1]  (up to down)

3. Gaze-toward-extra-person signal
   Given the extra person's horizontal position in the frame,
   check if the candidate's yaw + gaze_x point in that direction.

Model download
--------------
Face Landmarker model (~29 MB):
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Limitations
-----------
- Head pose from a 2D webcam image is estimated, not measured.
  Small errors (±5–10°) are normal.
- Gaze estimation from landmarks is coarse — it tells you
  "looking left/right/center" but not the exact gaze target.
- A single frame is never conclusive. Use temporal counts.

Future improvement
------------------
OpenFace provides more accurate AU-based gaze vectors at the cost of
a heavier external dependency. Recommended if landmark-based gaze is
too noisy for the production threshold.
"""

import math
import urllib.request
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────────────────────────────────────────────────────────
# Model download
# ─────────────────────────────────────────────────────────────────────────────

LANDMARKER_MODEL_PATH = Path("models/face_landmarker.task")
LANDMARKER_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def ensure_landmarker_model():
    """Download Face Landmarker model if not already present."""
    if not LANDMARKER_MODEL_PATH.exists():
        print(f"Downloading Face Landmarker model → {LANDMARKER_MODEL_PATH} (~29 MB)...")
        urllib.request.urlretrieve(LANDMARKER_MODEL_URL, LANDMARKER_MODEL_PATH)
        print("  ✓ Downloaded")
    else:
        print(f"  ✓ Face Landmarker model already present: {LANDMARKER_MODEL_PATH}")


def create_face_landmarker(
    num_faces: int = 2,
    model_path: Path = LANDMARKER_MODEL_PATH,
) -> mp_vision.FaceLandmarker:
    """
    Create a reusable FaceLandmarker detector.

    num_faces: maximum number of faces to detect (set to 2 for
               candidate + one potential extra person's face)
    """
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(model_path)
        ),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,  # needed for head pose
        num_faces=num_faces,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


# ─────────────────────────────────────────────────────────────────────────────
# Head pose from transformation matrix
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_matrix_to_euler(R: np.ndarray) -> tuple:
    """
    Decompose a 3×3 rotation matrix to Euler angles (pitch, yaw, roll) in degrees.

    Convention (camera-space, MediaPipe):
      pitch > 0  → head tilted downward
      yaw   > 0  → head turned to the RIGHT (from viewer's perspective)
      roll  > 0  → head tilted counterclockwise
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0.0

    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def get_head_pose(transformation_matrix) -> dict:
    """
    Extract pitch / yaw / roll from a 4×4 facial transformation matrix.

    Returns dict with 'pitch', 'yaw', 'roll' in degrees.
    """
    mat = np.array(transformation_matrix)
    R   = mat[:3, :3]
    pitch, yaw, roll = _rotation_matrix_to_euler(R)
    return {
        "pitch": round(pitch, 1),
        "yaw":   round(yaw, 1),
        "roll":  round(roll, 1),
    }


def interpret_head_pose(pose: dict) -> dict:
    """
    Convert raw angles to human-readable direction strings.

    Thresholds are intentionally conservative to avoid false positives
    from normal posture variations.
    """
    yaw   = pose["yaw"]
    pitch = pose["pitch"]

    # Horizontal direction
    if yaw > 20:
        h_dir = "looking_right"
    elif yaw < -20:
        h_dir = "looking_left"
    else:
        h_dir = "facing_camera"

    # Vertical direction
    if pitch > 15:
        v_dir = "looking_down"
    elif pitch < -15:
        v_dir = "looking_up"
    else:
        v_dir = "level"

    return {
        "horizontal":    h_dir,
        "vertical":      v_dir,
        "raw_yaw":       pose["yaw"],
        "raw_pitch":     pose["pitch"],
        "raw_roll":      pose["roll"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gaze from iris landmarks
# ─────────────────────────────────────────────────────────────────────────────

# Face Landmarker landmark indices
# Eye corners (normalized 0–1 in image space)
_LEFT_EYE_OUTER  = 33
_LEFT_EYE_INNER  = 133
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263

# Iris centers (5 points each)
_LEFT_IRIS  = list(range(468, 473))   # 468–472
_RIGHT_IRIS = list(range(473, 478))   # 473–477


def _landmark_to_px(lm, image_w: int, image_h: int) -> tuple:
    return lm.x * image_w, lm.y * image_h


def _iris_center(landmarks, indices, image_w, image_h) -> tuple:
    """Average position of iris landmark points in pixel space."""
    xs = [landmarks[i].x * image_w for i in indices]
    ys = [landmarks[i].y * image_h for i in indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def landmarks_bbox(landmarks, image_w: int, image_h: int) -> list:
    """Return a pixel bounding box from normalized face landmarks."""
    xs = [landmark.x * image_w for landmark in landmarks]
    ys = [landmark.y * image_h for landmark in landmarks]
    return [
        round(max(0.0, min(xs)), 1),
        round(max(0.0, min(ys)), 1),
        round(min(float(image_w), max(xs)), 1),
        round(min(float(image_h), max(ys)), 1),
    ]


def get_gaze(landmarks, image_w: int, image_h: int) -> dict:
    """
    Estimate gaze direction from iris position relative to eye corners.

    gaze_x: -1.0 (far left) … 0.0 (center) … +1.0 (far right)
    gaze_y: -1.0 (far up)   … 0.0 (center) … +1.0 (far down)

    Returns a dict with gaze_x, gaze_y, and a string label.
    """
    try:
        # ── Left eye ──────────────────────────────────────────────────────
        l_outer_x, _ = _landmark_to_px(landmarks[_LEFT_EYE_OUTER],  image_w, image_h)
        l_inner_x, _ = _landmark_to_px(landmarks[_LEFT_EYE_INNER],  image_w, image_h)
        l_iris_x, l_iris_y = _iris_center(landmarks, _LEFT_IRIS, image_w, image_h)

        l_eye_width = abs(l_inner_x - l_outer_x)
        l_eye_mid_x = (l_outer_x + l_inner_x) / 2
        l_gaze_x = (l_iris_x - l_eye_mid_x) / (l_eye_width / 2 + 1e-6)

        # ── Right eye ─────────────────────────────────────────────────────
        r_inner_x, _ = _landmark_to_px(landmarks[_RIGHT_EYE_INNER], image_w, image_h)
        r_outer_x, _ = _landmark_to_px(landmarks[_RIGHT_EYE_OUTER], image_w, image_h)
        r_iris_x, r_iris_y = _iris_center(landmarks, _RIGHT_IRIS, image_w, image_h)

        r_eye_width = abs(r_outer_x - r_inner_x)
        r_eye_mid_x = (r_outer_x + r_inner_x) / 2
        r_gaze_x = (r_iris_x - r_eye_mid_x) / (r_eye_width / 2 + 1e-6)

        # ── Average both eyes ─────────────────────────────────────────────
        gaze_x = (l_gaze_x + r_gaze_x) / 2

        # Vertical: use left iris y relative to eye mid
        l_top_y    = landmarks[_LEFT_EYE_OUTER].y * image_h
        l_bottom_y = landmarks[133].y * image_h   # approximate bottom
        l_eye_h    = abs(l_bottom_y - l_top_y) + 1e-6
        l_mid_y    = (l_top_y + l_bottom_y) / 2
        gaze_y = (l_iris_y - l_mid_y) / (l_eye_h / 2)

        # ── Label ─────────────────────────────────────────────────────────
        gaze_x_c = round(gaze_x, 3)
        gaze_y_c = round(gaze_y, 3)

        if gaze_x_c > 0.25:
            label = "gaze_right"
        elif gaze_x_c < -0.25:
            label = "gaze_left"
        else:
            label = "gaze_center"

        return {
            "gaze_x":       gaze_x_c,
            "gaze_y":       gaze_y_c,
            "gaze_label":   label,
            "available":    True,
        }

    except (IndexError, AttributeError):
        return {
            "gaze_x":     0.0,
            "gaze_y":     0.0,
            "gaze_label": "unavailable",
            "available":  False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main per-image function
# ─────────────────────────────────────────────────────────────────────────────

def analyse_head_pose_and_gaze(
    image_bgr: np.ndarray,
    landmarker: mp_vision.FaceLandmarker,
) -> list:
    """
    Run Face Landmarker on an image and return head pose + gaze for each face.

    Returns a list of dicts (one per detected face), each with:
      - face_index
      - head_pose  (pitch, yaw, roll + direction strings)
      - gaze       (gaze_x, gaze_y, gaze_label)
      - landmark_count
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    result = landmarker.detect(mp_image)

    face_results = []

    if not result.face_landmarks:
        return face_results

    for i, (lms, mat) in enumerate(zip(
        result.face_landmarks,
        result.facial_transformation_matrixes or []
    )):
        raw_pose   = get_head_pose(mat)
        pose_interp = interpret_head_pose(raw_pose)
        gaze       = get_gaze(lms, w, h)

        face_results.append({
            "face_index":      i,
            "face_bbox":       landmarks_bbox(lms, w, h),
            "head_pose":       pose_interp,
            "gaze":            gaze,
            "landmark_count":  len(lms),
        })

    return face_results


# ─────────────────────────────────────────────────────────────────────────────
# Engagement signal: is the candidate looking toward the extra person?
# ─────────────────────────────────────────────────────────────────────────────

# Thresholds
_YAW_TOWARD_THRESHOLD   = 20.0   # degrees — candidate yaw must exceed this
_GAZE_TOWARD_THRESHOLD  = 0.25   # normalised gaze_x magnitude

def is_looking_toward_extra(
    candidate_pose: dict,
    extra_person_horizontal_zone: str,
    yaw_threshold: float = _YAW_TOWARD_THRESHOLD,
    gaze_threshold: float = _GAZE_TOWARD_THRESHOLD,
) -> dict:
    """
    Determine if the candidate is looking toward the extra person's side.

    extra_person_horizontal_zone: one of
        "near_candidate_left", "near_candidate_right", "near_candidate_center",
        "background_left", "background_right", "background_center"

    Returns:
        looking_toward  : bool
        confidence      : "high" / "medium" / "low" / "none"
        reason          : explanation string
    """
    yaw       = candidate_pose["head_pose"]["raw_yaw"]
    gaze_x    = candidate_pose["gaze"]["gaze_x"]
    gaze_lbl  = candidate_pose["gaze"]["gaze_label"]
    h_dir     = candidate_pose["head_pose"]["horizontal"]
    gaze_avail = candidate_pose["gaze"]["available"]

    extra_side = "right" if "right" in extra_person_horizontal_zone else \
                 "left"  if "left"  in extra_person_horizontal_zone else "center"

    # Does head yaw point toward extra person?
    if extra_side == "right":
        yaw_toward  = yaw > yaw_threshold
        gaze_toward = gaze_x > gaze_threshold
    elif extra_side == "left":
        yaw_toward  = yaw < -yaw_threshold
        gaze_toward = gaze_x < -gaze_threshold
    else:
        # Extra person in center — facing camera is neutral, not conclusive
        yaw_toward  = abs(yaw)  < 10
        gaze_toward = abs(gaze_x) < 0.15

    if yaw_toward and gaze_avail and gaze_toward:
        return {
            "looking_toward": True,
            "confidence":     "high",
            "reason": (
                f"Head yaw ({yaw:.1f}°) and gaze ({gaze_lbl}) both point toward "
                f"extra person on the {extra_side}. Strong engagement signal."
            ),
        }
    elif yaw_toward:
        return {
            "looking_toward": True,
            "confidence":     "medium",
            "reason": (
                f"Head yaw ({yaw:.1f}°) points toward extra person on the {extra_side}. "
                f"Gaze {'confirms' if gaze_toward else 'does not confirm'} direction."
            ),
        }
    else:
        return {
            "looking_toward": False,
            "confidence":     "none",
            "reason": (
                f"Head yaw ({yaw:.1f}°) and gaze ({gaze_lbl}) do not point "
                f"toward extra person on the {extra_side}. Candidate faces camera."
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_head_pose_gaze(image_bgr: np.ndarray, face_results: list) -> np.ndarray:
    """
    Draw head pose axes and gaze direction arrow on the image.

    Green  = face_index 0 (assumed candidate)
    Cyan   = face_index 1+ (extra person's face if detected)
    """
    h, w = image_bgr.shape[:2]

    for face in face_results:
        color  = (0, 220, 0) if face["face_index"] == 0 else (220, 220, 0)
        pose   = face["head_pose"]
        gaze   = face["gaze"]
        yaw    = pose["raw_yaw"]
        pitch  = pose["raw_pitch"]
        gaze_x = gaze["gaze_x"]

        # Rough face center estimate (top-middle of image per face index offset)
        cx = w // 2
        cy = h // 3 + face["face_index"] * (h // 4)

        # Draw yaw arrow
        arrow_len = 60
        end_x = int(cx + arrow_len * math.sin(math.radians(yaw)))
        end_y = int(cy - arrow_len * math.sin(math.radians(pitch)))
        cv2.arrowedLine(image_bgr, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)

        # Draw gaze indicator
        gaze_end_x = int(cx + 40 * gaze_x)
        cv2.circle(image_bgr, (gaze_end_x, cy + 20), 5, (255, 100, 0), -1)

        # Text label
        label = (
            f"F{face['face_index']} yaw={yaw:.0f}° "
            f"pitch={pitch:.0f}° "
            f"gaze={gaze['gaze_label']}"
        )
        cv2.putText(image_bgr, label,
                    (10, 25 + face["face_index"] * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image_bgr
