# Proctoring SDK POC: Progress & Next Steps

This document outlines the progress made on the Proctoring SDK Proof of Concept (POC). We built a fully local, privacy-first computer vision pipeline that evaluates cheating risk without relying on expensive, slow, or black-box LLM APIs.

---

## 1. What Has Been Implemented So Far

We have developed three core layers of visual analysis, which are then combined into a single intelligent risk score.

### A. Environment & Quality Checks (OpenCV)
Before checking for cheating, we ensure the camera feed is actually usable.
- **Lighting Analysis:** Flags if the room is too dark (`LOW_LIGHT`) or completely blown out (`OVEREXPOSED`).
- **Focus Check:** Flags if the camera is severely out of focus (`BLURRY`).
- **Obstruction Check:** Detects if the candidate covered the webcam (`CAMERA_BLOCKED`).

### B. Object & Face Detection (YOLO + MediaPipe)
We use highly efficient local AI models to understand what is in the frame.
- **Restricted Objects:** YOLO detects forbidden items like `Cell Phones`, `Books`, and `Additional Monitors`.
- **Face Presence:** MediaPipe tracks if the candidate's face is actually visible, or if they walked away (`FACE_MISSING`).
- **Multiple Faces:** Flags if a second face appears in the webcam view.

### C. Contextual Extra Person Analysis (Advanced)
A major problem with simple proctoring is that a second person in the background (like a roommate walking by) triggers an automatic failure. We built a smarter system that understands context:
- **Proximity:** Measures exactly how close the extra person is to the candidate.
- **Head Pose & Gaze:** Uses 3D facial landmarks to calculate the candidate's head angle (yaw/pitch) and eye direction. We can detect if the candidate is actually *looking* at the extra person.

### D. The Unified Risk Aggregator
Instead of arbitrary point scoring, we built a human-like escalation engine that categorizes signals to prevent false positives:
- **CRITICAL (Auto-HIGH):** E.g., Phone detected on camera.
- **SUSPICIOUS:** E.g., Candidate looking directly at an extra person who is standing nearby. (Triggers MEDIUM, or HIGH if combined with other signals).
- **CONTEXTUAL:** E.g., A book on the desk. (Only triggers MEDIUM if combined with something else, otherwise LOW).
- **AMBIENT:** E.g., A tiny blurry person far in the background. (Only triggers LOW).

---

## 2. Core Technologies Used (The Libraries)

We deliberately chose fast, local libraries to ensure privacy, low latency, and zero ongoing API costs.

- **OpenCV (`opencv-python`)**: Used for raw image processing. It analyzes pixels directly to measure brightness (lighting checks) and Laplacian variance (blur detection).
- **YOLO11 (`ultralytics`)**: A state-of-the-art, high-speed Object Detection model. We use it to draw bounding boxes around restricted objects (phones, laptops, books) and to find the exact location of all people in the room.
- **MediaPipe (`mediapipe`)**: A lightweight machine learning framework by Google. We use its `FaceDetector` to quickly count faces, and its `FaceLandmarker` to map 478 3D points on the candidate's face. This provides the mathematical matrix needed to calculate head pose (yaw/pitch) and iris position (gaze).
- **Pandas (`pandas`)**: Used for data aggregation, structuring our risk reports into CSV/JSON formats, and merging signals from different models.

---

## 3. System Flow Architecture

How does a single frame get processed from start to finish?

1. **Input Stage:** The webcam captures a frame and passes it to the pipeline as a raw image array.
2. **Quality Gate (OpenCV):** The image is checked for extreme darkness, overexposure, or blur. If it fails severely, the frame is flagged immediately.
3. **Detection Layer (YOLO + MediaPipe):**
   - YOLO scans for restricted items and maps out all people.
   - MediaPipe scans for faces to ensure the candidate is present.
4. **Contextual Analysis (Python Logic):**
   - We identify the "Main Candidate" (the largest, most central person).
   - We calculate the physical distance between the candidate and any "Extra Persons".
   - We extract the candidate's head pose and gaze to see where they are looking.
5. **Risk Aggregator (The Brain):**
   - All signals (objects, faces, distances, gaze) are fed into the Category-Based Escalation engine.
   - The engine outputs a final `LOW`, `MEDIUM`, or `HIGH` risk score with a human-readable reason.
6. **Output Stage:** The result is saved to our JSON/CSV reports and an annotated image is generated for manual review.

---

## 4. Where Is The Current System Lacking? (Limitations)

While the POC is highly effective for a static image, there are known limitations we must address for a production-ready video system.

1. **Gaze Tracking is Coarse:** 
   Our current gaze detection uses MediaPipe iris tracking. It can reliably tell if a candidate is looking far left or far right, but it is not accurate enough to tell if they are looking at the corner of their screen vs off-screen.
2. **2D Cameras Lack Depth:** 
   We estimate the physical distance between people using bounding box sizes and 2D distances. A poster on a wall could temporarily trick the system into thinking a person is in the room.
3. **No Audio Context:** 
   Currently, we are deaf. A candidate could be receiving audio answers via a bluetooth earpiece, or talking to someone off-camera, and this visual-only system would not catch it.
4. **Single-Frame Limitations:** 
   This POC was run on static screenshots. A single frame where a candidate glances away is not proof of cheating; we need to measure *how long* they looked away.

---

## 5. What Can Be Done Next (The Roadmap)

To turn this POC into a robust, production-ready Proctoring SDK, we recommend the following next steps:

### A. Implement Audio Analysis (Voice Activity Detection)
We need to add a lightweight, local audio model (like Silero VAD) to the pipeline.
- **Goal:** Detect if human speech is happening in the room.
- **Why:** Audio activity combined with an "extra person" visual flag creates a near-perfect indicator of active cheating.

### B. Activate the Temporal Tracker (Video Analysis)
The logic for video analysis is already written in our code, we just need to feed it a live stream instead of static images.
- **Goal:** Track signals over time (e.g., "Candidate looked away for 10 seconds").
- **Why:** This eliminates false positives caused by brief glances or someone quickly walking past an open door in the background.

### C. Upgrade to OpenFace for Precision Gaze
If we need strict gaze tracking (e.g., ensuring they aren't reading off a second monitor), we should integrate OpenFace.
- **Goal:** Highly accurate 3D gaze vectors and Action Units.
- **Why:** It tells us exactly where the eyes are pointing in 3D space, far exceeding MediaPipe's capabilities.

### D. Fine-Tune YOLO (If Necessary)
If our testing reveals that YOLO is missing certain specific items (like specific types of earpieces or smartwatches), we can fine-tune the model.
- **Goal:** Train the YOLO model on a custom dataset of proctoring-specific images.
- **Why:** Base YOLO is good, but a fine-tuned model becomes exceptionally accurate for our specific use case.
