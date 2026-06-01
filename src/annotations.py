"""Annotated image output for accuracy runs."""

from __future__ import annotations

from pathlib import Path

from evidence import FrameEvidence


RISK_COLORS = {
    "NONE": (120, 220, 120),
    "LOW": (0, 220, 255),
    "MEDIUM": (0, 165, 255),
    "HIGH": (0, 0, 255),
}

OBJECT_COLORS = {
    "person": (255, 190, 70),
    "cell_phone": (20, 20, 255),
    "book": (40, 180, 255),
    "laptop": (255, 80, 180),
    "monitor": (255, 0, 180),
    "keyboard": (180, 180, 255),
    "mouse": (180, 180, 255),
}


def _box_tuple(box: list[float]) -> tuple[int, int, int, int]:
    return tuple(int(round(value)) for value in box)


def _label(image_bgr, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    import cv2

    x, y = origin
    y = max(16, y)
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(image_bgr, (x, y - height - baseline - 4), (x + width + 5, y + 3), (20, 20, 20), -1)
    cv2.putText(image_bgr, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)


def annotate_frame(image_bgr, evidence: FrameEvidence, decision) -> object:
    import cv2

    annotated = image_bgr.copy()
    for obj in evidence.objects:
        color = OBJECT_COLORS.get(obj.label, (220, 220, 220))
        x1, y1, x2, y2 = _box_tuple(obj.bbox)
        thickness = 3 if obj == evidence.candidate else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        suffix = " candidate" if obj == evidence.candidate else ""
        _label(annotated, f"{obj.label}{suffix} {obj.confidence:.2f}", (x1, y1 - 5), color)

    for face in evidence.faces:
        x1, y1, x2, y2 = _box_tuple(face.bbox)
        color = (0, 240, 120) if face == evidence.candidate_face else (80, 255, 80)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        _label(annotated, "candidate face" if face == evidence.candidate_face else "face", (x1, y2 + 14), color)

    for detail in decision.extra_person.get("extra_person_details", []):
        x1, y1, x2, y2 = _box_tuple(detail["extra_bbox"])
        color = RISK_COLORS.get(detail["risk_level"], (220, 220, 220))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        _label(annotated, f"extra {detail['risk_level'].lower()}", (x1, y2 + 28), color)

    banner_color = RISK_COLORS.get(decision.level, (220, 220, 220))
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 44), (24, 24, 24), -1)
    cv2.putText(
        annotated,
        f"risk {decision.level}: {decision.flag or 'clean'}",
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        banner_color,
        1,
    )
    flags = ", ".join(decision.base_flags) if decision.base_flags else "no base flags"
    cv2.putText(
        annotated,
        flags[:90],
        (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (235, 235, 235),
        1,
    )
    return annotated


def save_annotated_frame(image_bgr, evidence: FrameEvidence, decision, output_dir: Path) -> Path:
    import cv2

    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    path = annotated_dir / f"{Path(evidence.image).stem}_annotated.jpg"
    cv2.imwrite(str(path), annotate_frame(image_bgr, evidence, decision))
    return path


def clear_annotated_frames(output_dir: Path) -> None:
    annotated_dir = output_dir / "annotated"
    if not annotated_dir.exists():
        return
    for path in annotated_dir.glob("*_annotated.jpg"):
        path.unlink()
