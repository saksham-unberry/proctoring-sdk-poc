"""Ultralytics object detection backends with cross-model fusion."""

from __future__ import annotations

from evidence import Detection
from geometry import bbox_iou


def _class_ids_for_labels(labels: tuple[str, ...] | list[str]) -> list[int]:
    from accuracy_config import COCO_CLASSES

    configured = set(labels)
    return [class_id for class_id, label in COCO_CLASSES.items() if label in configured]


def _edge_crops(width: int, height: int, ratio: float) -> list[tuple[str, tuple[int, int, int, int]]]:
    crop_w = max(1, min(width, round(width * ratio)))
    crop_h = max(1, min(height, round(height * ratio)))
    return [
        ("edge_left", (0, 0, crop_w, height)),
        ("edge_right", (width - crop_w, 0, width, height)),
        ("edge_bottom", (0, height - crop_h, width, height)),
    ]


def _map_crop_bbox(bbox: list[float], crop: tuple[int, int, int, int]) -> list[float]:
    x1, y1, _, _ = crop
    return [bbox[0] + x1, bbox[1] + y1, bbox[2] + x1, bbox[3] + y1]


def fuse_detections(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    fused: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        match_index = next(
            (
                index for index, kept in enumerate(fused)
                if kept.label == detection.label
                and bbox_iou(kept.bbox, detection.bbox) >= iou_threshold
            ),
            None,
        )
        if match_index is None:
            fused.append(detection)
            continue

        kept = fused[match_index]
        sources = tuple(dict.fromkeys((*kept.sources, *detection.sources)))
        if detection.confidence > kept.confidence:
            fused[match_index] = Detection(
                label=detection.label,
                confidence=detection.confidence,
                bbox=detection.bbox,
                sources=sources,
            )
        else:
            fused[match_index] = Detection(
                label=kept.label,
                confidence=kept.confidence,
                bbox=kept.bbox,
                sources=sources,
            )
    return fused


def _model_class(family: str):
    if family == "yolo":
        from ultralytics import YOLO

        return YOLO
    if family == "rtdetr":
        from ultralytics import RTDETR

        return RTDETR
    raise ValueError(f"Unsupported object detection family: {family}")


class UltralyticsEnsembleDetector:
    def __init__(self, config):
        self.config = config
        model_class = _model_class(config.object_detection_family)
        self.models = [(model_name, model_class(model_name)) for model_name in config.yolo_models]

    def detect(self, image_path) -> list[Detection]:
        import cv2
        from pathlib import Path

        detections: list[Detection] = []
        resolved_image_path = Path(image_path).resolve()
        edge_image = None
        edge_crops: list[tuple[str, tuple[int, int, int, int]]] = []
        edge_class_ids: list[int] = []
        edge_thresholds: dict[str, float] = {}

        if self.config.edge_crop_enabled:
            edge_class_ids = _class_ids_for_labels(self.config.edge_crop_labels)
            if edge_class_ids:
                edge_image = cv2.imread(str(resolved_image_path))
            if edge_image is not None:
                height, width = edge_image.shape[:2]
                edge_crops = _edge_crops(width, height, self.config.edge_crop_ratio)
                edge_thresholds = dict(self.config.class_thresholds)
                for label in self.config.edge_crop_labels:
                    edge_thresholds[label] = min(
                        edge_thresholds.get(label, 1.0),
                        self.config.edge_crop_min_confidence,
                    )

        for model_name, model in self.models:
            detections.extend(
                self._detect_source(
                    model=model,
                    source=str(resolved_image_path),
                    source_name=model_name,
                    class_ids=self.config.yolo_class_ids,
                    thresholds=self.config.class_thresholds,
                    offset=None,
                )
            )

            if edge_image is None:
                continue
            for crop_name, crop in edge_crops:
                x1, y1, x2, y2 = crop
                crop_image = edge_image[y1:y2, x1:x2]
                detections.extend(
                    self._detect_source(
                        model=model,
                        source=crop_image,
                        source_name=f"{model_name}:{crop_name}",
                        class_ids=edge_class_ids,
                        thresholds=edge_thresholds,
                        offset=crop,
                    )
                )
        return fuse_detections(detections, self.config.yolo_fusion_iou)

    def _detect_source(
        self,
        model,
        source,
        source_name: str,
        class_ids: list[int],
        thresholds: dict[str, float],
        offset: tuple[int, int, int, int] | None,
    ) -> list[Detection]:
        from accuracy_config import COCO_CLASSES

        detections: list[Detection] = []
        results = model(
            source,
            conf=self.config.yolo_global_confidence,
            imgsz=self.config.yolo_image_size,
            classes=class_ids,
            verbose=False,
        )
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = COCO_CLASSES.get(class_id)
                confidence = float(box.conf[0])
                if label is None or confidence < thresholds.get(label, 1.0):
                    continue
                bbox = [round(value, 1) for value in box.xyxy[0].tolist()]
                if offset is not None:
                    bbox = [round(value, 1) for value in _map_crop_bbox(bbox, offset)]
                detections.append(
                    Detection(
                        label=label,
                        confidence=round(confidence, 3),
                        bbox=bbox,
                        sources=(source_name,),
                    )
                )
        return detections


YoloEnsembleDetector = UltralyticsEnsembleDetector
