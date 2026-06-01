"""MediaPipe face evidence adapters."""

from __future__ import annotations

from pathlib import Path

from evidence import FaceEvidence
from geometry import bbox_area, bbox_iou


def _detector_bbox(detector_box) -> list[float]:
    return [
        float(detector_box.origin_x),
        float(detector_box.origin_y),
        float(detector_box.origin_x + detector_box.width),
        float(detector_box.origin_y + detector_box.height),
    ]


def create_face_detector(model_path: Path, confidence: float):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=confidence,
    )
    return mp_vision.FaceDetector.create_from_options(options)


def detect_faces(image_bgr, detector) -> list[FaceEvidence]:
    import cv2
    import mediapipe as mp

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)
    faces = []
    for detection in result.detections or []:
        score = detection.categories[0].score if detection.categories else None
        faces.append(
            FaceEvidence(
                confidence=round(float(score), 3) if score is not None else None,
                bbox=[round(value, 1) for value in _detector_bbox(detection.bounding_box)],
            )
        )
    return faces


def merge_landmarker_faces(
    detector_faces: list[FaceEvidence],
    landmarker_faces: list[dict],
) -> list[FaceEvidence]:
    merged = list(detector_faces)
    for landmarker_face in landmarker_faces:
        bbox = landmarker_face["face_bbox"]
        best_index = max(
            range(len(merged)),
            key=lambda index: bbox_iou(merged[index].bbox, bbox),
            default=None,
        )
        if best_index is not None and bbox_iou(merged[best_index].bbox, bbox) >= 0.20:
            face = merged[best_index]
            merged[best_index] = FaceEvidence(
                confidence=face.confidence,
                bbox=face.bbox,
                head_pose=landmarker_face["head_pose"],
                gaze=landmarker_face["gaze"],
                source="face_detector+landmarker",
            )
            continue
        merged.append(
            FaceEvidence(
                confidence=None,
                bbox=bbox,
                head_pose=landmarker_face["head_pose"],
                gaze=landmarker_face["gaze"],
                source="landmarker",
            )
        )
    return sorted(merged, key=lambda face: bbox_area(face.bbox), reverse=True)
