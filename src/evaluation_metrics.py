"""Metrics for labeled frame-level accuracy checks."""

from __future__ import annotations

import csv
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "y"}


def _as_bool(value: str | bool | None) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return value.strip().lower() in TRUE_VALUES


def binary_metrics(expected: list[bool], predicted: list[bool]) -> dict[str, float | int]:
    tp = sum(exp and pred for exp, pred in zip(expected, predicted))
    tn = sum(not exp and not pred for exp, pred in zip(expected, predicted))
    fp = sum(not exp and pred for exp, pred in zip(expected, predicted))
    fn = sum(exp and not pred for exp, pred in zip(expected, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_frame_report(report_csv: str | Path, labels_csv: str | Path) -> dict[str, dict]:
    with Path(report_csv).open(newline="", encoding="utf-8") as handle:
        predictions = {row["image"]: row for row in csv.DictReader(handle)}
    with Path(labels_csv).open(newline="", encoding="utf-8") as handle:
        labels = [row for row in csv.DictReader(handle) if row["image"] in predictions]

    tasks = {
        "candidate_present": lambda row: _as_bool(row["candidate_present"]),
        "phone_present": lambda row: int(row["cell_phone_count"]) > 0,
        "book_present": lambda row: int(row["book_count"]) > 0,
        "laptop_present": lambda row: int(row["laptop_count"]) > 0,
        "monitor_present": lambda row: int(row["monitor_count"]) > 0,
        "extra_person_present": lambda row: int(row["person_count"]) > 1,
    }
    metrics = {}
    for label_column, prediction_fn in tasks.items():
        expected = []
        predicted = []
        for label in labels:
            expected_value = _as_bool(label.get(label_column))
            if expected_value is None:
                continue
            expected.append(expected_value)
            predicted.append(bool(prediction_fn(predictions[label["image"]])))
        if expected:
            metrics[label_column] = binary_metrics(expected, predicted)
    return metrics
