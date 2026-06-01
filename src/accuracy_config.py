"""Configuration loading for the repeatable accuracy pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


COCO_CLASSES = {
    0: "person",
    67: "cell_phone",
    73: "book",
    63: "laptop",
    62: "monitor",
    66: "keyboard",
    64: "mouse",
}


@dataclass(frozen=True)
class PipelineConfig:
    image_dir: Path
    output_dir: Path
    face_detector_model: Path
    face_landmarker_model: Path
    object_detection_family: str
    yolo_models: tuple[str, ...]
    yolo_image_size: int
    yolo_global_confidence: float
    yolo_fusion_iou: float
    edge_crop_enabled: bool
    edge_crop_ratio: float
    edge_crop_min_confidence: float
    edge_crop_labels: tuple[str, ...]
    class_thresholds: dict[str, float]
    face_confidence: float
    landmarker_faces: int
    quality: dict[str, float]
    extra_person: dict[str, float]
    engagement: dict[str, float]
    object_filters: dict[str, float]
    temporal: dict[str, float]
    gpt_review: dict[str, float | str | bool]

    @property
    def yolo_class_ids(self) -> list[int]:
        configured = set(self.class_thresholds)
        return [class_id for class_id, label in COCO_CLASSES.items() if label in configured]


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    paths = raw["paths"]
    object_detection = raw["object_detection"]
    face_detection = raw["face_detection"]

    return PipelineConfig(
        image_dir=Path(paths["image_dir"]),
        output_dir=Path(paths["output_dir"]),
        face_detector_model=Path(paths["face_detector_model"]),
        face_landmarker_model=Path(paths["face_landmarker_model"]),
        object_detection_family=str(object_detection.get("family", "yolo")).lower(),
        yolo_models=tuple(object_detection["models"]),
        yolo_image_size=int(object_detection["image_size"]),
        yolo_global_confidence=float(object_detection["global_confidence"]),
        yolo_fusion_iou=float(object_detection["fusion_iou"]),
        edge_crop_enabled=bool(object_detection.get("edge_crop_enabled", False)),
        edge_crop_ratio=float(object_detection.get("edge_crop_ratio", 0.0)),
        edge_crop_min_confidence=float(object_detection.get("edge_crop_min_confidence", 0.0)),
        edge_crop_labels=tuple(object_detection.get("edge_crop_labels", [])),
        class_thresholds={
            label: float(threshold)
            for label, threshold in object_detection["class_thresholds"].items()
        },
        face_confidence=float(face_detection["confidence"]),
        landmarker_faces=int(face_detection["landmarker_faces"]),
        quality={key: float(value) for key, value in raw["quality"].items()},
        extra_person={key: float(value) for key, value in raw["extra_person"].items()},
        engagement={key: float(value) for key, value in raw["engagement"].items()},
        object_filters={key: float(value) for key, value in raw["object_filters"].items()},
        temporal={key: float(value) for key, value in raw["temporal"].items()},
        gpt_review=dict(raw.get("gpt_review", {})),
    )
