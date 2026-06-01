"""
extra_person_risk.py
====================
Contextual risk assessment for extra persons detected in a proctoring frame.

Key design decisions
--------------------
- A single extra person is NOT automatically treated as cheating.
- Risk is a signal, not a verdict.
- Single-image analysis produces positional / proximity signals only.
- Temporal analysis (across frames) is needed for stronger conclusions.
- No OpenAI / Gemini / Claude APIs are used anywhere in this file.

Risk flag naming convention
----------------------------
  EXTRA_PERSON_PRESENT_LOW                — background, far, brief
  EXTRA_PERSON_CONTEXTUAL_RISK_MEDIUM     — moderate proximity or duration
  POSSIBLE_HUMAN_ASSISTANCE_HIGH          — close + sustained + other signals

Limitations
-----------
  Single-image: Cannot confirm whether the extra person is engaging with
  the candidate. Positional signals only.

  Temporal: Can estimate sustained presence and proximity, but cannot
  confirm audio communication, shared materials, or intent.

Head-pose / gaze (future)
--------------------------
  MediaPipe Face Landmarker (mp.tasks.vision.FaceLandmarker) can provide
  468 3D landmarks and derived head-pose angles (pitch, yaw, roll).
  If the candidate's yaw repeatedly points toward the extra person's
  horizontal position, that is a stronger engagement signal.

  OpenFace (https://github.com/TadasBaltrusaitis/OpenFace) provides more
  accurate gaze vectors (where the eyes are actually looking) and AU
  (Action Unit) intensities. It requires a separate installation.

  Both should be added as an optional enrichment step after the base
  YOLO + rule logic produces a MEDIUM or HIGH flag.
"""

import math
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# 1.  BBOX GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bbox_area(xyxy: list) -> float:
    """Return pixel area of a bounding box given as [x1, y1, x2, y2]."""
    return max(0.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))


def bbox_center(xyxy: list) -> tuple:
    """Return (cx, cy) of a bounding box given as [x1, y1, x2, y2]."""
    return ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)


def normalized_distance(xyxy_a: list, xyxy_b: list, image_w: int, image_h: int) -> float:
    """
    Euclidean distance between the centers of two bounding boxes,
    normalized by the image diagonal.
    0.0  = same position
    1.0  = opposite corners
    """
    cx_a, cy_a = bbox_center(xyxy_a)
    cx_b, cy_b = bbox_center(xyxy_b)
    dist = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
    diagonal = math.sqrt(image_w ** 2 + image_h ** 2)
    return round(dist / diagonal, 4) if diagonal > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MAIN CANDIDATE IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def identify_main_candidate(person_boxes: list, image_w: int, image_h: int) -> Optional[dict]:
    """
    Identify the main candidate from a list of person bounding boxes.

    Strategy
    --------
    Score each person by a weighted combination of:
      - Normalised area  (70 %) — person closest to camera is usually largest
      - Horizontal centrality (30 %) — candidate is usually centre-frame

    person_boxes: list of dicts with keys 'bbox' ([x1,y1,x2,y2]) and 'confidence'
    Returns the box dict with the highest score, or None if list is empty.
    """
    if not person_boxes:
        return None

    img_area = image_w * image_h
    img_cx   = image_w / 2.0
    best, best_score = None, -1.0

    for box in person_boxes:
        area      = bbox_area(box["bbox"])
        area_norm = area / img_area                          # 0 → 1

        cx, _     = bbox_center(box["bbox"])
        # centrality: 1.0 = perfectly centered, 0.0 = at horizontal edge
        centrality = 1.0 - abs(cx - img_cx) / img_cx

        score = 0.70 * area_norm + 0.30 * centrality
        if score > best_score:
            best_score = score
            best       = box

    return best


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EXTRA PERSON POSITION CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_extra_person_position(
    candidate_xyxy: list,
    extra_xyxy: list,
    image_w: int,
    image_h: int,
) -> str:
    """
    Classify where the extra person is relative to the candidate and the frame.

    Returns one of:
      "near_candidate_left"    — close to candidate, on the left
      "near_candidate_right"   — close to candidate, on the right
      "near_candidate_center"  — close to candidate, roughly same horizontal zone
      "background_left"        — small/far, left third of frame
      "background_right"       — small/far, right third of frame
      "background_center"      — small/far, centre of frame

    A person is considered "background" when their bbox area is less than 35 %
    of the candidate's bbox area (i.e. they appear noticeably smaller / further).
    """
    cand_area  = bbox_area(candidate_xyxy)
    extra_area = bbox_area(extra_xyxy)
    size_ratio = extra_area / cand_area if cand_area > 0 else 0.0

    extra_cx, _ = bbox_center(extra_xyxy)

    # Horizontal third
    if extra_cx < image_w * 0.33:
        h_zone = "left"
    elif extra_cx > image_w * 0.66:
        h_zone = "right"
    else:
        h_zone = "center"

    if size_ratio < 0.35:
        return f"background_{h_zone}"
    else:
        return f"near_candidate_{h_zone}"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SINGLE-IMAGE RISK ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

# Thresholds — easy to tune without touching logic
_PROXIMITY_THRESHOLD_HIGH   = 0.22  # normalised distance ≤ this → HIGH risk
_PROXIMITY_THRESHOLD_MEDIUM = 0.40  # normalised distance ≤ this → MEDIUM risk
_SIZE_RATIO_HIGH            = 0.55  # extra person at least 55 % as large as candidate → HIGH
_SIZE_RATIO_MEDIUM          = 0.30  # extra person at least 30 % as large as candidate → MEDIUM

_LIMITATION_SINGLE_IMAGE = (
    "Single-image analysis: engagement between the candidate and the extra "
    "person CANNOT be confirmed from one frame. This is a positional/proximity "
    "signal only. Temporal analysis across multiple frames (with optional "
    "head-pose and audio signals) is required for a stronger risk assessment."
)


def _threshold(config: Optional[dict], key: str, default: float) -> float:
    return float(config[key]) if config and key in config else default


def assess_single_image_risk(
    candidate_box: dict,
    extra_boxes: list,
    image_w: int,
    image_h: int,
    config: Optional[dict] = None,
) -> dict:
    """
    Assess contextual risk for extra persons detected in a single image.

    Parameters
    ----------
    candidate_box : dict with 'bbox' and 'confidence'
    extra_boxes   : list of dicts with 'bbox' and 'confidence'
    image_w, image_h : frame dimensions in pixels

    Returns
    -------
    dict with overall_risk_level, overall_risk_flag, overall_reason,
    per-person details, and a limitation statement.
    """
    if not extra_boxes:
        return {
            "extra_person_present":    False,
            "extra_person_count":      0,
            "overall_risk_level":      "NONE",
            "overall_risk_flag":       None,
            "overall_reason":          "No extra person detected.",
            "extra_person_details":    [],
            "limitation":              None,
        }

    _RISK_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    details     = []

    for extra in extra_boxes:
        cand_xyxy  = candidate_box["bbox"]
        extra_xyxy = extra["bbox"]

        cand_area  = bbox_area(cand_xyxy)
        extra_area = bbox_area(extra_xyxy)
        size_ratio = extra_area / cand_area if cand_area > 0 else 0.0

        ignore_size_ratio = _threshold(config, "ignore_size_ratio", 0.10)
        if size_ratio < ignore_size_ratio:
            continue  # ignore extremely small background noise/false detections

        dist       = normalized_distance(cand_xyxy, extra_xyxy, image_w, image_h)
        position   = classify_extra_person_position(cand_xyxy, extra_xyxy, image_w, image_h)

        # ── Risk decision ──────────────────────────────────────────────────
        proximity_high = _threshold(config, "proximity_high", _PROXIMITY_THRESHOLD_HIGH)
        proximity_medium = _threshold(config, "proximity_medium", _PROXIMITY_THRESHOLD_MEDIUM)
        size_ratio_high = _threshold(config, "size_ratio_high", _SIZE_RATIO_HIGH)
        size_ratio_medium = _threshold(config, "size_ratio_medium", _SIZE_RATIO_MEDIUM)

        if dist <= proximity_high or size_ratio >= size_ratio_high:
            risk = "HIGH"
            flag = "POSSIBLE_HUMAN_ASSISTANCE_HIGH"
            reason = (
                f"Extra person is physically close to the candidate "
                f"(normalised distance: {dist}, position: {position}, "
                f"relative size: {size_ratio:.2f}). "
                "Proximity suggests possible interaction. "
                "Temporal and audio analysis needed to confirm engagement."
            )

        elif dist <= proximity_medium or size_ratio >= size_ratio_medium:
            risk = "MEDIUM"
            flag = "EXTRA_PERSON_CONTEXTUAL_RISK_MEDIUM"
            reason = (
                f"Extra person detected at {position} with moderate proximity "
                f"(normalised distance: {dist}, relative size: {size_ratio:.2f}). "
                "Context is ambiguous from a single frame. "
                "Multiple frames or audio data needed for further assessment."
            )

        else:
            risk = "LOW"
            flag = "EXTRA_PERSON_PRESENT_LOW"
            reason = (
                f"Extra person appears in the {position} area, "
                f"far from the candidate (normalised distance: {dist}, "
                f"relative size: {size_ratio:.2f}). "
                "Likely a background presence with no engagement signal."
            )

        details.append({
            "extra_confidence":    extra.get("confidence", None),
            "extra_bbox":          extra_xyxy,
            "position":            position,
            "normalised_distance": dist,
            "size_ratio":          round(size_ratio, 3),
            "risk_level":          risk,
            "risk_flag":           flag,
            "reason":              reason,
        })

    if not details:
        return {
            "extra_person_present":    False,
            "extra_person_count":      0,
            "overall_risk_level":      "NONE",
            "overall_risk_flag":       None,
            "overall_reason":          "No extra person detected.",
            "extra_person_details":    [],
            "limitation":              None,
        }

    # Overall = worst case across all extra persons
    worst = max(details, key=lambda d: _RISK_ORDER[d["risk_level"]])

    return {
        "extra_person_present":    True,
        "extra_person_count":      len(extra_boxes),
        "candidate_bbox":          candidate_box["bbox"],
        "candidate_confidence":    candidate_box.get("confidence"),
        "extra_person_details":    details,
        "overall_risk_level":      worst["risk_level"],
        "overall_risk_flag":       worst["risk_flag"],
        "overall_reason":          worst["reason"],
        "limitation":              _LIMITATION_SINGLE_IMAGE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TEMPORAL TRACKER  (video / sequential frames)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalExtraPersonTracker:
    """
    Tracks extra person presence across sequential frames (video or
    time-series screenshots from a proctoring session).

    Usage
    -----
    tracker = TemporalExtraPersonTracker(fps=1)

    for frame_result in frame_results:          # single-image risk dicts
        tracker.add_frame(frame_result)

    summary = tracker.analyze()
    print(summary["risk_flag"], summary["reason"])

    Additional signals to wire in (future)
    ---------------------------------------
    - head_pose_toward_extra (bool)  — from MediaPipe Face Landmarker yaw
    - audio_activity (bool)          — from Silero VAD / WebRTC VAD
    - other_flag_present (bool)      — PHONE_DETECTED / BROWSER_SWITCH etc.
    When those are passed to analyze(), they can push MEDIUM → HIGH.
    """

    # Duration thresholds
    _SUSTAINED_PRESENCE_RATIO    = 0.30   # extra person in ≥ 30 % of frames
    _SUSTAINED_CLOSE_RATIO       = 0.20   # close proximity in ≥ 20 % of frames
    _HIGH_PRESENCE_RATIO         = 0.50   # extra person in ≥ 50 % of frames
    _HIGH_CLOSE_RATIO            = 0.30   # close proximity in ≥ 30 % of frames
    _CLOSE_DISTANCE_THRESHOLD    = 0.25   # normalised distance

    def __init__(self, fps: float = 1.0, config: Optional[dict] = None):
        self.fps           = fps
        self.frame_history = []           # list of {"timestamp": float, "result": dict}
        self.config        = config or {}

    def add_frame(self, single_image_result: dict, timestamp_seconds: float = None):
        """
        Add a frame's single-image risk result to the tracker.

        single_image_result: output of assess_single_image_risk()
        timestamp_seconds  : pass explicitly, or leave None to auto-increment by 1/fps
        """
        if timestamp_seconds is None:
            timestamp_seconds = len(self.frame_history) / self.fps

        self.frame_history.append({
            "timestamp": timestamp_seconds,
            "result":    single_image_result,
        })

    def analyze(
        self,
        head_pose_toward_extra_count: int = 0,  # frames where candidate gaze → extra person
        audio_activity_count:         int = 0,  # frames where voice/audio detected
        other_risk_flag_count:        int = 0,  # frames where another risk flag co-occurs
    ) -> dict:
        """
        Analyse temporal patterns and return an escalated risk assessment.

        Optional enrichment signals (pass counts; 0 = not available):
          head_pose_toward_extra_count — from MediaPipe Face Landmarker
          audio_activity_count         — from Silero / WebRTC VAD
          other_risk_flag_count        — PHONE_DETECTED, BROWSER_SWITCH etc.
        """
        if not self.frame_history:
            return {
                "risk_level": "NONE",
                "risk_flag":  None,
                "reason":     "No frames have been added to the tracker.",
            }

        total_frames = len(self.frame_history)

        frames_with_extra = [
            f for f in self.frame_history
            if f["result"].get("extra_person_present")
        ]

        if not frames_with_extra:
            return {
                "risk_level":             "NONE",
                "risk_flag":              None,
                "reason":                 "Extra person not detected in any frame.",
                "total_frames":           total_frames,
                "frames_with_extra":      0,
                "presence_ratio":         0.0,
            }

        presence_ratio  = len(frames_with_extra) / total_frames

        # Frames where any extra person was within close distance
        close_frames = [
            f for f in frames_with_extra
            if any(
                d["normalised_distance"] <= _threshold(
                    self.config, "close_distance", self._CLOSE_DISTANCE_THRESHOLD
                )
                for d in f["result"].get("extra_person_details", [])
            )
        ]
        close_ratio = len(close_frames) / total_frames

        # Duration
        t_start = self.frame_history[0]["timestamp"]
        t_end   = self.frame_history[-1]["timestamp"]
        session_duration    = t_end - t_start if t_end > t_start else 1.0
        presence_seconds    = presence_ratio * session_duration
        close_seconds       = close_ratio    * session_duration

        # Optional enrichment ratios
        head_pose_ratio  = head_pose_toward_extra_count / total_frames
        audio_ratio      = audio_activity_count         / total_frames
        other_flag_ratio = other_risk_flag_count        / total_frames

        # ── Risk escalation ────────────────────────────────────────────────
        corroborating_high = (
            presence_ratio >= _threshold(
                self.config, "presence_medium_ratio", self._SUSTAINED_PRESENCE_RATIO
            )
            and (
                head_pose_ratio > _threshold(self.config, "head_pose_ratio", 0.2)
                or audio_ratio > _threshold(self.config, "audio_ratio", 0.2)
                or other_flag_ratio > _threshold(self.config, "other_flag_ratio", 0.15)
            )
        )
        sustained_nearby_presence = (
            presence_ratio >= _threshold(self.config, "presence_high_ratio", self._HIGH_PRESENCE_RATIO)
            and close_ratio >= _threshold(self.config, "close_high_ratio", self._HIGH_CLOSE_RATIO)
        )

        if corroborating_high:
            risk = "HIGH"
            flag = "POSSIBLE_HUMAN_ASSISTANCE_HIGH"
            reason = (
                f"Extra person present in {presence_ratio*100:.0f}% of frames "
                f"({presence_seconds:.0f}s), with close proximity in "
                f"{close_ratio*100:.0f}% of frames ({close_seconds:.0f}s). "
            )
            if head_pose_ratio > _threshold(self.config, "head_pose_ratio", 0.2):
                reason += f"Candidate head/gaze toward extra person in {head_pose_ratio*100:.0f}% of frames. "
            if audio_ratio > _threshold(self.config, "audio_ratio", 0.2):
                reason += f"Audio/voice activity detected in {audio_ratio*100:.0f}% of frames. "
            if other_flag_ratio > _threshold(self.config, "other_flag_ratio", 0.15):
                reason += f"Co-occurring risk flags (phone/browser/object) in {other_flag_ratio*100:.0f}% of frames. "
            reason += "Sustained presence + proximity + corroborating signals indicate elevated risk."

        elif (
            sustained_nearby_presence
            or
            presence_ratio >= _threshold(
                self.config, "presence_medium_ratio", self._SUSTAINED_PRESENCE_RATIO
            )
            or close_ratio >= _threshold(
                self.config, "close_medium_ratio", self._SUSTAINED_CLOSE_RATIO
            )
        ):
            risk = "MEDIUM"
            flag = "EXTRA_PERSON_CONTEXTUAL_RISK_MEDIUM"
            reason = (
                f"Extra person present in {presence_ratio*100:.0f}% of frames "
                f"({presence_seconds:.0f}s). "
                f"Close proximity detected in {close_ratio*100:.0f}% of frames. "
                "Sustained presence warrants review, but assistance is not inferred "
                "without corroborating engagement, audio, or object signals. "
                "Add head-pose and audio signals to refine the assessment."
            )

        else:
            risk = "LOW"
            flag = "EXTRA_PERSON_PRESENT_LOW"
            reason = (
                f"Extra person appeared briefly ({presence_seconds:.0f}s, "
                f"{presence_ratio*100:.0f}% of session), "
                f"with minimal close-proximity frames ({close_ratio*100:.0f}%). "
                "Likely a background presence with no engagement evidence."
            )

        return {
            "risk_level":                   risk,
            "risk_flag":                    flag,
            "reason":                       reason,
            "total_frames":                 total_frames,
            "frames_with_extra_person":     len(frames_with_extra),
            "presence_ratio":               round(presence_ratio, 3),
            "presence_seconds":             round(presence_seconds, 1),
            "close_proximity_frames":       len(close_frames),
            "close_ratio":                  round(close_ratio, 3),
            "close_seconds":                round(close_seconds, 1),
            "head_pose_toward_extra_ratio": round(head_pose_ratio, 3),
            "audio_activity_ratio":         round(audio_ratio, 3),
            "other_flag_ratio":             round(other_flag_ratio, 3),
            "limitation": (
                "Temporal analysis: engagement is inferred from duration, proximity, "
                "and optional corroborating signals. It is not a definitive proof "
                "of cheating. Human review of flagged intervals is recommended."
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CONVENIENCE: annotate image with extra-person risk overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_extra_person_risk(image_bgr, risk_result: dict):
    """
    Draw risk overlay on a BGR image (NumPy array) in-place.

    Candidate box  → green
    Extra LOW      → yellow
    Extra MEDIUM   → orange
    Extra HIGH     → red
    """
    import cv2

    _COLOR_MAP = {
        "LOW":    (0, 255, 255),    # yellow
        "MEDIUM": (0, 165, 255),    # orange
        "HIGH":   (0, 0, 255),      # red
        "NONE":   (200, 200, 200),  # grey
    }

    if risk_result.get("candidate_bbox"):
        x1, y1, x2, y2 = [int(v) for v in risk_result["candidate_bbox"]]
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(image_bgr, "CANDIDATE", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    for detail in risk_result.get("extra_person_details", []):
        color = _COLOR_MAP.get(detail["risk_level"], (200, 200, 200))
        x1, y1, x2, y2 = [int(v) for v in detail["extra_bbox"]]
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{detail['risk_flag']} {detail['normalised_distance']:.2f}"
        cv2.putText(image_bgr, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Overall risk banner
    overall = risk_result.get("overall_risk_level", "NONE")
    banner_color = _COLOR_MAP.get(overall, (200, 200, 200))
    h, w = image_bgr.shape[:2]
    cv2.rectangle(image_bgr, (0, h - 30), (w, h), (30, 30, 30), -1)
    cv2.putText(image_bgr,
                f"EXTRA PERSON RISK: {overall}  |  {risk_result.get('overall_risk_flag', '')}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, banner_color, 1)

    return image_bgr
