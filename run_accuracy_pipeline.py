"""Run the extracted accuracy pipeline on image frames."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from accuracy_config import load_config
from annotations import clear_annotated_frames, save_annotated_frame
from candidate_association import select_candidate
from evidence import FrameEvidence
from evaluation_metrics import evaluate_frame_report
from extra_person_risk import TemporalExtraPersonTracker
from face_detector import create_face_detector, detect_faces, merge_landmarker_faces
from gpt_review_selector import write_gpt_review_manifest
from head_pose_gaze import analyse_head_pose_and_gaze, create_face_landmarker
from object_detector import UltralyticsEnsembleDetector
from object_filters import suppress_face_phone_false_positives
from quality_checks import assess_quality
from reporting import write_reports
from risk_policy import decide_frame_risk


def _image_paths(image_dir: Path) -> list[Path]:
    paths = [
        *image_dir.glob("*.png"),
        *image_dir.glob("*.jpg"),
        *image_dir.glob("*.jpeg"),
    ]
    return sorted(paths, key=lambda path: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)])


def _require_file(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def run(config_path: Path, labels_path: Path | None = None) -> dict:
    import cv2

    config = load_config(config_path)
    _require_file(config.face_detector_model, "MediaPipe face detector model")
    _require_file(config.face_landmarker_model, "MediaPipe face landmarker model")
    detector = UltralyticsEnsembleDetector(config)
    face_detector = create_face_detector(config.face_detector_model, config.face_confidence)
    landmarker = create_face_landmarker(
        num_faces=config.landmarker_faces,
        model_path=config.face_landmarker_model,
    )
    clear_annotated_frames(config.output_dir)

    records = []
    for image_path in _image_paths(config.image_dir):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        height, width = image_bgr.shape[:2]
        objects = detector.detect(image_path)
        detector_faces = detect_faces(image_bgr, face_detector)
        landmarker_faces = analyse_head_pose_and_gaze(image_bgr, landmarker)
        faces = merge_landmarker_faces(detector_faces, landmarker_faces)
        objects, suppressed_objects = suppress_face_phone_false_positives(
            objects,
            faces,
            config.object_filters,
        )
        candidate, candidate_face = select_candidate(
            [obj for obj in objects if obj.label == "person"],
            faces,
            width,
            height,
        )
        evidence = FrameEvidence(
            image=image_path.name,
            width=width,
            height=height,
            objects=objects,
            faces=faces,
            quality=assess_quality(image_bgr, config.quality),
            suppressed_objects=suppressed_objects,
            candidate=candidate,
            candidate_face=candidate_face,
        )
        decision = decide_frame_risk(evidence, config)
        annotated_path = save_annotated_frame(image_bgr, evidence, decision, config.output_dir)
        records.append(
            {
                "evidence": evidence.to_dict(),
                "image_path": str(image_path.resolve()),
                "risk": decision.to_dict(),
                "annotated_path": str(annotated_path),
            }
        )

    json_path, csv_path = write_reports(records, config.output_dir)
    tracker = TemporalExtraPersonTracker(fps=config.temporal["fps"], config=config.temporal)
    head_pose_count = 0
    corroborating_count = 0
    for record in records:
        risk = record["risk"]
        tracker.add_frame(risk["extra_person"])
        head_pose_count += any(signal["looking_toward"] for signal in risk["engagement"])
        corroborating_count += bool(
            {"PHONE_DETECTED", "BOOK_DETECTED", "LAPTOP_DETECTED", "MONITOR_DETECTED"}
            & set(risk["base_flags"])
        )
    temporal = tracker.analyze(
        head_pose_toward_extra_count=head_pose_count,
        other_risk_flag_count=corroborating_count,
    )
    temporal_path = config.output_dir / "reports" / "temporal_summary.json"
    temporal_path.write_text(json.dumps(temporal, indent=2), encoding="utf-8")
    gpt_manifest_path, gpt_manifest = write_gpt_review_manifest(
        records,
        config.output_dir,
        config.gpt_review,
    )
    summary = {
        "frames": len(records),
        "json_report": str(json_path),
        "csv_report": str(csv_path),
        "annotated_dir": str(config.output_dir / "annotated"),
        "temporal_report": str(temporal_path),
        "temporal_risk_level": temporal["risk_level"],
        "gpt_review_manifest": str(gpt_manifest_path),
        "gpt_review_selected_frames": gpt_manifest["selected_count"],
        "gpt_review_estimated_frame_reduction_percent": gpt_manifest[
            "estimated_frame_reduction_percent"
        ],
    }
    if labels_path:
        metrics = evaluate_frame_report(csv_path, labels_path)
        metrics_path = config.output_dir / "reports" / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary["metrics_report"] = str(metrics_path)
        summary["metrics"] = metrics
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/accuracy.toml", type=Path)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.labels), indent=2))


if __name__ == "__main__":
    main()
