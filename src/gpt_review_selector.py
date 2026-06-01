"""Select the smallest useful set of frames for expensive GPT review."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RISK_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
DEFAULT_ALWAYS_INCLUDE_FLAGS = {
    "PHONE_DETECTED",
    "CORROBORATED_SUSPICION",
    "SUSPICIOUS_REVIEW",
    "NO_PERSON",
    "FACE_MISSING",
    "CAMERA_BLOCKED",
}


def _risk_value(level: str | None) -> int:
    return RISK_ORDER.get(str(level or "NONE").upper(), 0)


def _frame_number(image_name: str, fallback: int) -> int:
    numbers = re.findall(r"\d+", image_name)
    return int(numbers[-1]) if numbers else fallback


def _configured_flags(value: Any) -> set[str]:
    if value is None:
        return DEFAULT_ALWAYS_INCLUDE_FLAGS
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item) for item in value}


def _signal_key(record: dict[str, Any]) -> str:
    risk = record["risk"]
    return str(risk.get("flag") or risk.get("level") or "NONE")


def _object_counts(evidence: dict[str, Any]) -> Counter:
    return Counter(obj["label"] for obj in evidence.get("objects", []))


def _max_confidence(evidence: dict[str, Any], label: str) -> float:
    confidences = [
        float(obj.get("confidence", 0.0))
        for obj in evidence.get("objects", [])
        if obj.get("label") == label
    ]
    return max(confidences, default=0.0)


def _prompt_summary(record: dict[str, Any]) -> dict[str, Any]:
    evidence = record["evidence"]
    risk = record["risk"]
    counts = _object_counts(evidence)
    quality = evidence.get("quality", {})
    return {
        "image": evidence["image"],
        "risk_level": risk["level"],
        "risk_flag": risk["flag"],
        "risk_reason": risk["reason"],
        "base_flags": risk["base_flags"],
        "objects": dict(counts),
        "face_count": len(evidence.get("faces", [])),
        "extra_person_risk": risk.get("extra_person", {}).get("overall_risk_level"),
        "engagement": risk.get("engagement", []),
        "quality": {
            "lighting_status": quality.get("lighting_status"),
            "blur_status": quality.get("blur_status"),
            "blocked_status": quality.get("blocked_status"),
        },
    }


def score_frame_for_gpt(record: dict[str, Any]) -> tuple[int, list[str]]:
    """Return an explainable priority score for sending a frame to GPT."""
    risk = record["risk"]
    evidence = record["evidence"]
    level = risk.get("level")
    score = _risk_value(level) * 40
    reasons: list[str] = []

    if level in {"MEDIUM", "HIGH"}:
        reasons.append(f"{level.lower()} local risk")

    flag = risk.get("flag")
    if flag:
        reasons.append(str(flag).lower())
    if flag == "PHONE_DETECTED":
        score += 35
        phone_confidence = _max_confidence(evidence, "cell_phone")
        score += int(phone_confidence * 20)
        reasons.append(f"phone confidence {phone_confidence:.2f}")
    elif flag in {"CORROBORATED_SUSPICION", "SUSPICIOUS_REVIEW"}:
        score += 25

    base_flags = set(risk.get("base_flags", []))
    critical_base_flags = base_flags & DEFAULT_ALWAYS_INCLUDE_FLAGS
    if critical_base_flags:
        score += 20
        reasons.extend(flag.lower() for flag in sorted(critical_base_flags))

    extra = risk.get("extra_person", {})
    extra_level = extra.get("overall_risk_level")
    if extra_level in {"MEDIUM", "HIGH"}:
        score += 15 if extra_level == "MEDIUM" else 25
        reasons.append(f"{extra_level.lower()} extra-person context")

    if any(signal.get("looking_toward") for signal in risk.get("engagement", [])):
        score += 25
        reasons.append("candidate looking toward extra person")

    quality = evidence.get("quality", {})
    if quality.get("blocked_status") == "CAMERA_BLOCKED":
        score += 20
        reasons.append("camera blocked")

    return score, list(dict.fromkeys(reasons))


def select_gpt_review_frames(
    records: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact manifest for costly multimodal review.

    The selector keeps locally clean frames out of GPT by default, prioritizes
    strong/corroborated evidence, and caps repeated frames with the same signal.
    """
    config = config or {}
    enabled = bool(config.get("enabled", True))
    min_level = str(config.get("min_risk_level", "MEDIUM")).upper()
    min_score = int(config.get("min_score", 80))
    max_frames = int(config.get("max_frames", 12))
    max_per_signal = int(config.get("max_per_signal", 4))
    min_gap = int(config.get("min_gap_frames", 2))
    always_include_flags = _configured_flags(config.get("always_include_flags"))

    skipped_by_level = Counter(record["risk"].get("level", "NONE") for record in records)
    if not enabled or max_frames <= 0:
        return {
            "enabled": enabled,
            "total_frames": len(records),
            "selected_count": 0,
            "estimated_frame_reduction_percent": 100.0 if records else 0.0,
            "selected_frames": [],
            "skipped_by_level": dict(skipped_by_level),
        }

    candidates = []
    for index, record in enumerate(records):
        risk = record["risk"]
        score, reasons = score_frame_for_gpt(record)
        level = str(risk.get("level", "NONE")).upper()
        flag = risk.get("flag")
        eligible = (
            _risk_value(level) >= _risk_value(min_level)
            or score >= min_score
            or flag in always_include_flags
        )
        if eligible:
            candidates.append(
                {
                    "index": index,
                    "frame_number": _frame_number(record["evidence"]["image"], index),
                    "score": score,
                    "reasons": reasons,
                    "record": record,
                    "signal_key": _signal_key(record),
                }
            )

    candidates.sort(key=lambda item: (-item["score"], item["frame_number"]))
    selected = []
    signal_counts: defaultdict[str, int] = defaultdict(int)
    last_frame_for_signal: dict[str, int] = {}

    for item in candidates:
        if len(selected) >= max_frames:
            break
        key = item["signal_key"]
        if signal_counts[key] >= max_per_signal:
            continue
        previous = last_frame_for_signal.get(key)
        if previous is not None and abs(item["frame_number"] - previous) < min_gap:
            continue
        selected.append(item)
        signal_counts[key] += 1
        last_frame_for_signal[key] = item["frame_number"]

    selected.sort(key=lambda item: item["frame_number"])
    selected_frames = []
    for item in selected:
        record = item["record"]
        selected_frames.append(
            {
                "image": record["evidence"]["image"],
                "image_path": record.get("image_path"),
                "annotated_path": record.get("annotated_path"),
                "priority_score": item["score"],
                "selection_reasons": item["reasons"],
                "prompt_summary": _prompt_summary(record),
            }
        )
        skipped_by_level[record["risk"].get("level", "NONE")] -= 1

    selected_count = len(selected_frames)
    reduction = 100.0
    if records:
        reduction = round((1 - selected_count / len(records)) * 100, 2)

    return {
        "enabled": enabled,
        "total_frames": len(records),
        "selected_count": selected_count,
        "estimated_frame_reduction_percent": reduction,
        "selected_frames": selected_frames,
        "skipped_by_level": {key: value for key, value in skipped_by_level.items() if value > 0},
    }


def write_gpt_review_manifest(
    records: list[dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    manifest = select_gpt_review_frames(records, config)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = reports_dir / "gpt_review_manifest.json"
    import json

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, manifest
