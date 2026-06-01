import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from candidate_association import select_candidate
from evaluation_metrics import binary_metrics
from evidence import Detection, FaceEvidence, FrameEvidence
from gpt_review_selector import select_gpt_review_frames
from object_detector import _edge_crops, _map_crop_bbox, _model_class, fuse_detections
from object_filters import suppress_face_phone_false_positives
from reporting import frame_row
from risk_policy import decide_frame_risk


class AccuracyPipelineTests(unittest.TestCase):
    def test_candidate_prefers_person_with_face_association(self):
        people = [
            Detection("person", 0.95, [0, 300, 430, 640]),
            Detection("person", 0.80, [180, 120, 390, 620]),
        ]
        face = FaceEvidence(0.90, [245, 180, 325, 280])

        candidate, candidate_face = select_candidate(people, [face], 480, 640)

        self.assertEqual(candidate, people[1])
        self.assertEqual(candidate_face, face)

    def test_fusion_keeps_detection_sources_without_double_counting(self):
        detections = [
            Detection("person", 0.80, [10, 10, 100, 200], ("a.pt",)),
            Detection("person", 0.90, [12, 12, 102, 202], ("b.pt",)),
            Detection("cell_phone", 0.50, [220, 220, 250, 270], ("b.pt",)),
        ]

        fused = fuse_detections(detections, 0.5)

        self.assertEqual(len(fused), 2)
        person = next(item for item in fused if item.label == "person")
        self.assertEqual(person.sources, ("b.pt", "a.pt"))
        self.assertEqual(person.confidence, 0.90)

    def test_rtdetr_backend_is_available(self):
        self.assertEqual(_model_class("rtdetr").__name__, "RTDETR")

    def test_edge_crops_cover_partial_phone_regions(self):
        crops = dict(_edge_crops(1000, 800, 0.35))

        self.assertEqual(crops["edge_left"], (0, 0, 350, 800))
        self.assertEqual(crops["edge_right"], (650, 0, 1000, 800))
        self.assertEqual(crops["edge_bottom"], (0, 520, 1000, 800))
        self.assertEqual(
            _map_crop_bbox([5, 100, 120, 250], crops["edge_bottom"]),
            [5, 620, 120, 770],
        )

    def test_phone_is_critical_in_policy(self):
        config = SimpleNamespace(extra_person={}, engagement={"yaw_toward_degrees": 20.0, "gaze_toward": 0.25})
        person = Detection("person", 0.90, [80, 50, 380, 630])
        evidence = FrameEvidence(
            image="frame.png",
            width=480,
            height=640,
            objects=[person, Detection("cell_phone", 0.55, [210, 320, 260, 410])],
            faces=[FaceEvidence(0.88, [170, 100, 300, 260])],
            quality={"degraded": False},
            candidate=person,
        )

        decision = decide_frame_risk(evidence, config)

        self.assertEqual(decision.level, "HIGH")
        self.assertEqual(decision.flag, "PHONE_DETECTED")

    def test_unengaged_extra_person_is_low_public_presence_signal(self):
        config = SimpleNamespace(extra_person={}, engagement={"yaw_toward_degrees": 20.0, "gaze_toward": 0.25})
        candidate = Detection("person", 0.95, [100, 100, 420, 630])
        extra = Detection("person", 0.90, [10, 180, 180, 600])
        evidence = FrameEvidence(
            image="public-place.png",
            width=480,
            height=640,
            objects=[candidate, extra],
            faces=[FaceEvidence(0.90, [160, 150, 360, 390])],
            quality={"degraded": False},
            candidate=candidate,
        )

        decision = decide_frame_risk(evidence, config)

        self.assertEqual(decision.level, "LOW")
        self.assertEqual(decision.flag, "PUBLIC_PRESENCE_OR_QUALITY_SIGNAL")
        self.assertIn("without candidate engagement", decision.reason)

    def test_quality_only_frame_is_not_public_presence(self):
        config = SimpleNamespace(extra_person={}, engagement={"yaw_toward_degrees": 20.0, "gaze_toward": 0.25})
        candidate = Detection("person", 0.95, [100, 100, 420, 630])
        evidence = FrameEvidence(
            image="solo-blurry.png",
            width=480,
            height=640,
            objects=[candidate],
            faces=[FaceEvidence(0.90, [160, 150, 360, 390])],
            quality={"blur_status": "BLURRY", "degraded": True},
            candidate=candidate,
        )

        decision = decide_frame_risk(evidence, config)

        self.assertEqual(decision.level, "LOW")
        self.assertEqual(decision.flag, "QUALITY_SIGNAL")
        self.assertIn("blurry", decision.reason)

    def test_weak_phone_centered_in_face_is_suppressed(self):
        phone = Detection("cell_phone", 0.50, [200, 535, 275, 565], ("yolo11l.pt",))
        person = Detection("person", 0.90, [0, 250, 440, 640])
        face = FaceEvidence(0.85, [145, 380, 390, 625])

        accepted, suppressed = suppress_face_phone_false_positives(
            [person, phone],
            [face],
            {"face_phone_min_confidence": 0.75},
        )

        self.assertEqual(accepted, [person])
        self.assertEqual(suppressed[0]["object"]["label"], "cell_phone")
        self.assertIn("inside face", suppressed[0]["reason"])

    def test_binary_metrics_surface_false_negatives(self):
        metrics = binary_metrics([True, True, False, False], [True, False, True, False])

        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["f1"], 0.5)

    def test_report_row_keeps_annotated_image_path(self):
        row = frame_row(
            {
                "evidence": {
                    "image": "frame.png",
                    "size": {"width": 1, "height": 1},
                    "objects": [],
                    "faces": [],
                    "quality": {},
                    "candidate": None,
                    "candidate_face": None,
                },
                "risk": {
                    "base_flags": [],
                    "extra_person": {"overall_risk_level": "NONE"},
                    "level": "NONE",
                    "flag": None,
                    "reason": "clean",
                },
                "annotated_path": "outputs/frame_annotated.jpg",
            }
        )

        self.assertEqual(row["annotated_path"], "outputs/frame_annotated.jpg")

    def test_gpt_review_selector_skips_clean_frames(self):
        records = [
            {
                "evidence": {
                    "image": "ss_1.png",
                    "size": {"width": 480, "height": 640},
                    "objects": [{"label": "person"}],
                    "faces": [{}],
                    "quality": {"lighting_status": "GOOD_LIGHT", "blur_status": "SHARP"},
                },
                "risk": {
                    "level": "NONE",
                    "flag": None,
                    "reason": "clean",
                    "base_flags": [],
                    "extra_person": {"overall_risk_level": "NONE"},
                    "engagement": [],
                },
                "annotated_path": "outputs/annotated/ss_1.png",
            },
            {
                "evidence": {
                    "image": "ss_2.png",
                    "size": {"width": 480, "height": 640},
                    "objects": [{"label": "person"}, {"label": "cell_phone"}],
                    "faces": [{}],
                    "quality": {"lighting_status": "GOOD_LIGHT", "blur_status": "SHARP"},
                },
                "risk": {
                    "level": "HIGH",
                    "flag": "PHONE_DETECTED",
                    "reason": "phone",
                    "base_flags": ["PHONE_DETECTED"],
                    "extra_person": {"overall_risk_level": "NONE"},
                    "engagement": [],
                },
                "annotated_path": "outputs/annotated/ss_2.png",
            },
        ]

        manifest = select_gpt_review_frames(records, {"max_frames": 12})

        self.assertEqual(manifest["selected_count"], 1)
        self.assertEqual(manifest["selected_frames"][0]["image"], "ss_2.png")
        self.assertEqual(manifest["estimated_frame_reduction_percent"], 50.0)

    def test_gpt_review_selector_caps_repeated_signals(self):
        records = []
        for index in range(6):
            records.append(
                {
                    "evidence": {
                        "image": f"ss_{index}.png",
                        "size": {"width": 480, "height": 640},
                        "objects": [{"label": "person"}, {"label": "cell_phone"}],
                        "faces": [{}],
                        "quality": {},
                    },
                    "risk": {
                        "level": "HIGH",
                        "flag": "PHONE_DETECTED",
                        "reason": "phone",
                        "base_flags": ["PHONE_DETECTED"],
                        "extra_person": {"overall_risk_level": "NONE"},
                        "engagement": [],
                    },
                    "annotated_path": f"outputs/annotated/ss_{index}.png",
                }
            )

        manifest = select_gpt_review_frames(
            records,
            {"max_frames": 10, "max_per_signal": 2, "min_gap_frames": 1},
        )

        self.assertEqual(manifest["selected_count"], 2)


if __name__ == "__main__":
    unittest.main()
