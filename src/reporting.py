"""Stable JSON and CSV reports for accuracy runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


CSV_COLUMNS = [
    "image",
    "width",
    "height",
    "person_count",
    "face_count",
    "cell_phone_count",
    "book_count",
    "laptop_count",
    "monitor_count",
    "suppressed_object_count",
    "candidate_present",
    "candidate_face_present",
    "lighting_status",
    "blur_status",
    "blocked_status",
    "quality_degraded",
    "base_flags",
    "extra_person_risk",
    "risk_level",
    "risk_flag",
    "risk_reason",
    "annotated_path",
]


def frame_row(record: dict) -> dict:
    evidence = record["evidence"]
    risk = record["risk"]
    objects = evidence["objects"]
    labels = [obj["label"] for obj in objects]
    return {
        "image": evidence["image"],
        "width": evidence["size"]["width"],
        "height": evidence["size"]["height"],
        "person_count": labels.count("person"),
        "face_count": len(evidence["faces"]),
        "cell_phone_count": labels.count("cell_phone"),
        "book_count": labels.count("book"),
        "laptop_count": labels.count("laptop"),
        "monitor_count": labels.count("monitor"),
        "suppressed_object_count": len(evidence.get("suppressed_objects", [])),
        "candidate_present": evidence["candidate"] is not None,
        "candidate_face_present": evidence["candidate_face"] is not None,
        "lighting_status": evidence["quality"].get("lighting_status"),
        "blur_status": evidence["quality"].get("blur_status"),
        "blocked_status": evidence["quality"].get("blocked_status"),
        "quality_degraded": evidence["quality"].get("degraded"),
        "base_flags": ";".join(risk["base_flags"]),
        "extra_person_risk": risk["extra_person"]["overall_risk_level"],
        "risk_level": risk["level"],
        "risk_flag": risk["flag"],
        "risk_reason": risk["reason"],
        "annotated_path": record.get("annotated_path"),
    }


def write_reports(records: list[dict], output_dir: Path) -> tuple[Path, Path]:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "frame_evidence.json"
    csv_path = reports_dir / "frame_report.csv"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(frame_row(record) for record in records)
    return json_path, csv_path
