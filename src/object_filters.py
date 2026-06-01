"""Post-detection object filters that use face context."""

from __future__ import annotations

from evidence import Detection, FaceEvidence
from geometry import box_contains_center


def suppress_face_phone_false_positives(
    objects: list[Detection],
    faces: list[FaceEvidence],
    config: dict[str, float],
) -> tuple[list[Detection], list[dict]]:
    """Suppress weak phone boxes centered inside a face bbox."""
    min_confidence = config["face_phone_min_confidence"]
    accepted = []
    suppressed = []
    for obj in objects:
        centered_in_face = obj.label == "cell_phone" and any(
            box_contains_center(face.bbox, obj.bbox) for face in faces
        )
        if centered_in_face and obj.confidence < min_confidence:
            suppressed.append(
                {
                    "object": obj.to_dict(),
                    "reason": "weak cell_phone detection centered inside face bbox",
                }
            )
            continue
        accepted.append(obj)
    return accepted, suppressed
