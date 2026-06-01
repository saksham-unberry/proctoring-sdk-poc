"""Candidate and candidate-face selection."""

from __future__ import annotations

from evidence import Detection, FaceEvidence
from geometry import bbox_area, bbox_center, box_contains_center


def select_candidate(
    people: list[Detection],
    faces: list[FaceEvidence],
    image_width: int,
    image_height: int,
) -> tuple[Detection | None, FaceEvidence | None]:
    """Prefer a face-associated central candidate, then fall back to person geometry."""
    if not people:
        return None, None

    scored: list[tuple[float, Detection, FaceEvidence | None]] = []
    image_area = max(1, image_width * image_height)
    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0

    for person in people:
        person_cx, person_cy = bbox_center(person.bbox)
        centrality_x = 1.0 - min(1.0, abs(person_cx - image_center_x) / max(1.0, image_center_x))
        centrality_y = 1.0 - min(1.0, abs(person_cy - image_center_y) / max(1.0, image_center_y))
        size_score = min(1.0, bbox_area(person.bbox) / image_area)

        associated_faces = [
            face for face in faces
            if box_contains_center(person.bbox, face.bbox, margin=8.0)
        ]
        candidate_face = max(associated_faces, key=lambda face: bbox_area(face.bbox), default=None)
        face_score = 1.0 if candidate_face else 0.0
        score = 0.45 * size_score + 0.25 * centrality_x + 0.10 * centrality_y + 0.20 * face_score
        scored.append((score, person, candidate_face))

    _, candidate, candidate_face = max(scored, key=lambda item: item[0])
    return candidate, candidate_face
