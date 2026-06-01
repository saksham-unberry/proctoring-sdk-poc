"""Bounding-box geometry used across association and detection fusion."""

from __future__ import annotations


def bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = bbox_area([x1, y1, x2, y2])
    union = bbox_area(first) + bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def box_contains_center(container: list[float], box: list[float], margin: float = 0.0) -> bool:
    cx, cy = bbox_center(box)
    return (
        container[0] - margin <= cx <= container[2] + margin
        and container[1] - margin <= cy <= container[3] + margin
    )
