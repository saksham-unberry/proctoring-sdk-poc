# Proctoring SDK POC

Proof-of-concept work for frame-level online proctoring risk checks. The repository is organized into two intentionally separate parts:

1. **Accuracy pipeline**
   This is the production-style, script-driven implementation. It lives in `src/`, is configured through `configs/*.toml`, and is run through `run_accuracy_pipeline.py`. Use this path when you want repeatable accuracy runs, reports, metrics, tests, or a cleaner base for SDK/API work.

2. **Notebook POCs**
   These are exploratory Jupyter notebooks kept in `notebooks/`. They are not the main pipeline entry point. Use them when you want to inspect the original experiments, try ideas interactively, or review the earlier POC flow step by step.

The notebooks and the accuracy pipeline may use similar concepts, models, images, and output names, but they should be treated as separate workflows. Changes in a notebook do not automatically change the extracted accuracy pipeline, and running the CLI pipeline does not require running the notebooks.

## Project Structure

```text
.
|-- configs/                 # TOML configs for the extracted accuracy pipeline
|-- data/                    # Labels and small structured input data
|-- images/                  # Local sample input frames (ignored by Git)
|-- models/                  # Local model weights and MediaPipe assets (ignored by Git)
|-- notebooks/               # Separate notebook POCs and exploratory experiments
|-- src/                     # Extracted accuracy pipeline modules
|-- tests/                   # Unit tests for the accuracy pipeline
|-- run_accuracy_pipeline.py # CLI entry point for full frame processing
`-- requirements.txt         # Python dependencies
```

Generated outputs are written under `outputs/` and are intentionally ignored.

## Which Workflow Should I Use?

Use the **accuracy pipeline** when you need:

- repeatable command-line runs
- TOML-based configuration
- generated CSV/JSON reports
- annotated output frames
- label-based evaluation
- unit-testable Python modules

Use the **notebooks** when you need:

- interactive exploration
- visual debugging
- experimentation with thresholds/models
- a walkthrough of the earlier POC logic
- one-off analysis that does not need to be part of the extracted pipeline yet

In short: `run_accuracy_pipeline.py` is the main structured pipeline; `notebooks/` contains separate exploratory POCs.

## SDKs And Libraries Used

### Accuracy Pipeline

The extracted accuracy pipeline uses the dependencies below through `run_accuracy_pipeline.py`, `src/`, and `configs/`.

| SDK / Library | Where used | Purpose |
| --- | --- | --- |
| `ultralytics` | `src/object_detector.py` | Runs YOLO and RT-DETR object detection models for people, phones, books, laptops, monitors, keyboards, and mice. |
| `mediapipe` | `src/face_detector.py`, `src/head_pose_gaze.py` | Detects faces, runs FaceDetector/FaceLandmarker, estimates head pose, and provides gaze-related signals. |
| `opencv-python` / `cv2` | `run_accuracy_pipeline.py`, `src/quality_checks.py`, `src/annotations.py` | Reads frames, converts color formats, checks blur/brightness/camera blockage, draws boxes, and writes annotated images. |
| `numpy` | `src/quality_checks.py`, `src/head_pose_gaze.py` | Performs image statistics and numeric calculations for quality, pose, and gaze logic. |
| Python standard library | `src/accuracy_config.py`, `src/reporting.py`, `run_accuracy_pipeline.py` | Loads TOML configs, parses CLI arguments, handles paths, and writes CSV/JSON reports. |

The accuracy pipeline is configured through:

- `configs/accuracy.toml` for YOLO-based runs.
- `configs/rtdetr_accuracy.toml` for RT-DETR-based runs.

### Notebook POCs

The notebooks use a similar computer-vision stack, but they are separate exploratory workflows.

| SDK / Library | Where used | Purpose |
| --- | --- | --- |
| `ultralytics` / `YOLO` | `notebooks/proctoring_sdk_poc.ipynb`, `notebooks/extra_person_risk_poc.ipynb` | Runs interactive YOLO experiments and compares object detections. |
| `mediapipe` | `notebooks/proctoring_sdk_poc.ipynb`, `notebooks/extra_person_risk_poc.ipynb` through `src` imports | Runs face detection, face landmarking, head pose, and gaze experiments. |
| `opencv-python` / `cv2` | `notebooks/proctoring_sdk_poc.ipynb`, `notebooks/extra_person_risk_poc.ipynb` | Loads images, converts color spaces, checks image quality, draws annotations, and saves visual outputs. |
| `numpy` | Notebook image and math cells | Handles arrays and numeric calculations. |
| `pandas` | Notebook report/analysis cells | Builds tables, reads and writes CSVs, merges reports, and summarizes results. |
| `matplotlib` | Notebook visualization cells | Creates charts and visual inspection plots. |
| Python standard library | Notebook cells using `json`, `pathlib`, `urllib.request` | Reads/writes JSON reports, handles paths, and downloads local MediaPipe model files when needed. |

There is no active OpenAI, Gemini, or Claude API call in the current code. `src/gpt_review_selector.py` only creates a manifest of suspicious frames that could later be sent for multimodal review.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place required local model files in `models/`:

```text
models/
|-- face_detector.tflite
|-- face_landmarker.task
|-- yolo11l.pt
|-- yolo12x.pt
`-- rtdetr-l.pt
```

## Accuracy Pipeline Usage

The accuracy pipeline is the code-first path. It reads frames from `images/`, uses settings from `configs/`, writes generated reports/annotated frames to `outputs/`, and can optionally compare results against labels in `data/labels/`.

YOLO ensemble accuracy pipeline:

```bash
python run_accuracy_pipeline.py --config configs/accuracy.toml --labels data/labels/frame_labels_template.csv
```

RT-DETR accuracy pipeline:

```bash
python run_accuracy_pipeline.py --config configs/rtdetr_accuracy.toml --labels data/labels/frame_labels_template.csv
```

## Notebook POC Usage

The notebooks are kept together in `notebooks/`. They are separate from the extracted accuracy pipeline and should be opened/run as independent Jupyter experiments:

```text
notebooks/
|-- proctoring_sdk_poc.ipynb
|-- extra_person_risk_poc.ipynb
`-- final_risk_aggregator_poc.ipynb
```

Some notebook cells may expect local assets such as sample images, model files, or generated reports to exist relative to the notebook working directory. That is notebook-specific behavior. The CLI accuracy pipeline uses `configs/*.toml` instead and does not depend on notebook state.

## Test

```bash
python -m unittest discover tests
```
