"""OpenCV quality checks used before risk policy."""

from __future__ import annotations


def assess_quality(image_bgr, config: dict[str, float]) -> dict[str, float | str | bool]:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    std_dev = float(np.std(gray))

    if brightness < config["low_light_threshold"]:
        lighting_status = "LOW_LIGHT"
    elif brightness > config["overexposed_threshold"]:
        lighting_status = "OVEREXPOSED"
    else:
        lighting_status = "GOOD_LIGHT"

    blur_status = "BLURRY" if blur_score < config["blur_threshold"] else "SHARP"
    blocked = brightness < config["blocked_brightness"] and std_dev < config["blocked_std"]
    degraded = blocked or blur_status == "BLURRY" or lighting_status != "GOOD_LIGHT"

    return {
        "lighting_status": lighting_status,
        "brightness_score": round(brightness, 2),
        "blur_status": blur_status,
        "blur_score": round(blur_score, 2),
        "blocked_status": "CAMERA_BLOCKED" if blocked else "NOT_BLOCKED",
        "degraded": degraded,
    }
