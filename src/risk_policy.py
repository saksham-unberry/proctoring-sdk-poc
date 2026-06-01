"""Risk policy separated from detector output."""

from __future__ import annotations

from dataclasses import dataclass

from evidence import FrameEvidence
from extra_person_risk import assess_single_image_risk
from head_pose_gaze import is_looking_toward_extra


@dataclass(frozen=True)
class RiskDecision:
    level: str
    flag: str | None
    reason: str
    base_flags: list[str]
    extra_person: dict
    engagement: list[dict]
    review_required: bool

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "flag": self.flag,
            "reason": self.reason,
            "base_flags": self.base_flags,
            "extra_person": self.extra_person,
            "engagement": self.engagement,
            "review_required": self.review_required,
        }


def base_flags(evidence: FrameEvidence) -> list[str]:
    labels = {obj.label for obj in evidence.objects}
    people = evidence.objects_for("person")
    flags = []
    if "cell_phone" in labels:
        flags.append("PHONE_DETECTED")
    if len(people) > 1:
        flags.append("MULTIPLE_PEOPLE")
    if not people:
        flags.append("NO_PERSON")
    if "book" in labels:
        flags.append("BOOK_DETECTED")
    if "laptop" in labels:
        flags.append("LAPTOP_DETECTED")
    if "monitor" in labels:
        flags.append("MONITOR_DETECTED")
    if len(evidence.faces) > 1:
        flags.append("MULTIPLE_FACES")
    if not evidence.faces:
        flags.append("FACE_MISSING")
    if evidence.quality.get("lighting_status") == "LOW_LIGHT":
        flags.append("LOW_LIGHT")
    if evidence.quality.get("lighting_status") == "OVEREXPOSED":
        flags.append("OVEREXPOSED")
    if evidence.quality.get("blur_status") == "BLURRY":
        flags.append("BLURRY")
    if evidence.quality.get("blocked_status") == "CAMERA_BLOCKED":
        flags.append("CAMERA_BLOCKED")
    return flags


def _extra_risk(evidence: FrameEvidence, config) -> dict:
    people = evidence.objects_for("person")
    if evidence.candidate is None:
        return {
            "extra_person_present": False,
            "extra_person_count": max(0, len(people) - 1),
            "overall_risk_level": "NONE",
            "overall_risk_flag": "NO_CANDIDATE",
            "overall_reason": "Candidate could not be associated from frame evidence.",
            "extra_person_details": [],
            "limitation": None,
        }
    extras = [person for person in people if person != evidence.candidate]
    return assess_single_image_risk(
        evidence.candidate.to_dict(),
        [person.to_dict() for person in extras],
        evidence.width,
        evidence.height,
        config.extra_person,
    )


def _engagement(evidence: FrameEvidence, extra_risk: dict, config) -> list[dict]:
    face = evidence.candidate_face
    if not face or not face.head_pose or not face.gaze:
        return []
    pose = {"head_pose": face.head_pose, "gaze": face.gaze}
    return [
        is_looking_toward_extra(
            pose,
            detail["position"],
            yaw_threshold=config.engagement["yaw_toward_degrees"],
            gaze_threshold=config.engagement["gaze_toward"],
        )
        for detail in extra_risk.get("extra_person_details", [])
    ]


def decide_frame_risk(evidence: FrameEvidence, config) -> RiskDecision:
    flags = base_flags(evidence)
    extra_risk = _extra_risk(evidence, config)
    engagement = _engagement(evidence, extra_risk, config)
    looking_toward_extra = any(signal["looking_toward"] for signal in engagement)

    suspicious = []
    contextual = []
    quality_context = []
    public_presence = []
    if "PHONE_DETECTED" in flags:
        return RiskDecision(
            level="HIGH",
            flag="PHONE_DETECTED",
            reason="Critical object evidence: phone detected.",
            base_flags=flags,
            extra_person=extra_risk,
            engagement=engagement,
            review_required=True,
        )
    if looking_toward_extra:
        suspicious.append("candidate looking toward extra person")
        if extra_risk["overall_risk_level"] in {"MEDIUM", "HIGH"}:
            suspicious.append(f"engaged extra person {extra_risk['overall_risk_level'].lower()}")

    contextual.extend(
        flag.lower().replace("_detected", "").replace("_", " ")
        for flag in flags
        if flag in {"BOOK_DETECTED", "LAPTOP_DETECTED", "MONITOR_DETECTED", "FACE_MISSING", "NO_PERSON"}
    )
    quality_context.extend(
        flag.lower().replace("_", " ")
        for flag in flags
        if flag in {"LOW_LIGHT", "OVEREXPOSED", "BLURRY", "CAMERA_BLOCKED"}
    )
    if "MULTIPLE_FACES" in flags:
        public_presence.append("multiple visible faces")
    if extra_risk["overall_risk_level"] in {"LOW", "MEDIUM", "HIGH"} and not looking_toward_extra:
        public_presence.append(
            "extra person visible without candidate engagement"
        )

    if len(suspicious) >= 2 or suspicious and contextual:
        level, flag = "HIGH", "CORROBORATED_SUSPICION"
        reason = "Corroborated suspicious evidence: " + ", ".join(suspicious + contextual) + "."
    elif suspicious:
        level, flag = "MEDIUM", "SUSPICIOUS_REVIEW"
        reason = "Suspicious evidence needs review: " + ", ".join(suspicious) + "."
    elif len(contextual) >= 2:
        level, flag = "MEDIUM", "MULTIPLE_CONTEXTUAL_SIGNALS"
        reason = "Multiple contextual signals: " + ", ".join(contextual) + "."
    elif contextual:
        level, flag = "LOW", "CONTEXTUAL_SIGNAL"
        reason = "Contextual signal: " + ", ".join(contextual) + "."
    elif public_presence:
        level, flag = "LOW", "PUBLIC_PRESENCE_OR_QUALITY_SIGNAL"
        context = public_presence + quality_context
        reason = "Public-presence context: " + ", ".join(context) + "."
    elif quality_context:
        level, flag = "LOW", "QUALITY_SIGNAL"
        reason = "Frame quality signal: " + ", ".join(quality_context) + "."
    else:
        level, flag, reason = "NONE", None, "No configured risk evidence."

    if evidence.quality.get("degraded") and level == "NONE":
        level, flag = "LOW", "DEGRADED_FRAME"
        reason = "Frame quality is degraded; absence of other evidence is less reliable."

    return RiskDecision(
        level=level,
        flag=flag,
        reason=reason,
        base_flags=flags,
        extra_person=extra_risk,
        engagement=engagement,
        review_required=level in {"MEDIUM", "HIGH"},
    )
