"""Structured evidence shared by detector, policy, report, and evaluation code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: list[float]
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FaceEvidence:
    confidence: float | None
    bbox: list[float]
    head_pose: dict[str, Any] | None = None
    gaze: dict[str, Any] | None = None
    source: str = "face_detector"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameEvidence:
    image: str
    width: int
    height: int
    objects: list[Detection] = field(default_factory=list)
    faces: list[FaceEvidence] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    suppressed_objects: list[dict[str, Any]] = field(default_factory=list)
    candidate: Detection | None = None
    candidate_face: FaceEvidence | None = None

    def objects_for(self, label: str) -> list[Detection]:
        return [obj for obj in self.objects if obj.label == label]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "size": {"width": self.width, "height": self.height},
            "objects": [obj.to_dict() for obj in self.objects],
            "faces": [face.to_dict() for face in self.faces],
            "quality": self.quality,
            "suppressed_objects": self.suppressed_objects,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "candidate_face": self.candidate_face.to_dict() if self.candidate_face else None,
        }
