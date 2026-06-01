# Accuracy Pipeline Changes

## Goal

The POC now has a repeatable accuracy path outside the notebooks. The goal is
not to claim that a model is accurate before ground truth exists. The goal is to
make detection, risk policy, temporal aggregation, reporting, and evaluation
measurable so accuracy work can be done deliberately.

## What Changed

### 1. Accuracy config is versioned

`configs/accuracy.toml` holds YOLO models, image size, per-class confidence
thresholds, face thresholds, image-quality thresholds, extra-person thresholds,
gaze thresholds, and temporal thresholds.

Why:

- Thresholds were split between notebooks and Python modules.
- A threshold change needs to be reproducible when validation metrics change.
- The accuracy config currently uses both available large YOLO weights and a
  larger `960` image size to favor recall for small restricted objects.

### 2. Detector output is separated from risk judgement

The new evidence path is:

```txt
detectors -> frame evidence -> candidate association -> risk policy -> reports
```

Core files:

- `src/evidence.py`
- `src/object_detector.py`
- `src/face_detector.py`
- `src/quality_checks.py`
- `src/risk_policy.py`

Why:

- A detector should report objects, boxes, confidence, source model, faces, and
  quality evidence.
- Risk escalation should happen in one policy layer so false positives can be
  reviewed and tested separately from detector recall.

### 3. YOLO results can be fused across models

`src/object_detector.py` runs the configured YOLO models and fuses overlapping
same-class detections using IoU. Each fused detection retains model sources.

Why:

- The repo already has `yolo12x.pt` and `yolo11l.pt`.
- Using both models can improve recall when a restricted object is missed by
  one model, while fusion prevents double-counting obvious overlaps.
- Precision and recall still need to be measured on labels before choosing a
  production config.

### 4. Candidate selection is face-aware

`src/candidate_association.py` prefers a candidate person box that has an
associated face and remains central/large in the frame.

`src/head_pose_gaze.py` now exposes a landmark-derived face bbox so landmarker
head pose and gaze can be linked back to person evidence.

Why:

- The previous main-candidate heuristic used only person bbox area and horizontal
  centrality.
- Candidate identity errors corrupt extra-person proximity, gaze-to-extra
  checks, and final risk.

### 5. Quality is preserved as evidence

`src/quality_checks.py` returns existing blur, light, and blocked-camera checks
plus a `degraded` marker.

Why:

- A degraded frame is not just a risk flag. It also means absence of object
  evidence is weaker.
- The policy now emits a degraded-frame signal when a poor-quality frame would
  otherwise look clean.

### 6. Face-context phone suppression is auditable

`src/object_filters.py` suppresses weak `cell_phone` boxes whose center lies
inside a detected face bbox. Suppressed boxes are kept in JSON evidence under
`suppressed_objects` instead of becoming accepted phone evidence.

Why:

- Close-up mouth, beard, and eyeglass detail can look like a COCO phone to YOLO.
- In `ss_182.png`, the hand-area phone candidate is outside the candidate face,
  while the mouth-area phone candidate is a weaker box centered inside the face.
- A face-centered phone box can still be retained when it exceeds the stricter
  threshold in `configs/accuracy.toml`.

### 7. Reports use a stable schema

`src/reporting.py` writes:

- `outputs/accuracy_run/reports/frame_evidence.json`
- `outputs/accuracy_run/reports/frame_report.csv`
- `outputs/accuracy_run/reports/temporal_summary.json`
- `outputs/accuracy_run/annotated/*_annotated.jpg`

Why:

- The old notebook CSV writer still looked for `yolov8n.pt` and `yolov8s.pt`
  after the notebook switched to `yolo11l.pt` and `yolo12x.pt`.
- That produced `0` person counts in CSV rows while JSON detections and risk
  flags showed multiple people.
- The new CSV schema uses stable signal names such as `person_count` and
  `cell_phone_count`, independent of configured YOLO weight names.
- The temporal summary consumes image files in natural filename order. For
  production, pass real timestamped frames from one session rather than mixing
  unrelated screenshots.

### 8. Label-driven metrics are available

`data/labels/frame_labels_template.csv` is the label template.

`src/evaluation_metrics.py` computes binary precision, recall, F1, TP, TN, FP,
and FN for candidate presence, extra-person presence, phone, book, laptop, and
monitor presence when labels are supplied.

Why:

- The old `evaluation_table.csv` has blank `Expected`, `Correct?`, and `Notes`
  fields, so accuracy tuning could not be trusted yet.
- A threshold or model comparison without labels is an output comparison, not
  an accuracy comparison.

### 9. Public-place presence is not assistance by itself

The frame policy now keeps unengaged extra people and multiple visible faces as
low public-presence context instead of suspicious evidence. A single frame needs
candidate engagement or stronger object evidence before extra-person context can
escalate risk.

The temporal tracker also no longer labels sustained nearby presence as `HIGH`
without corroborating head pose/gaze, audio, or other risk evidence.

Why:

- Assessments may be taken in public or shared places.
- A background person, passerby, or nearby worker is not evidence that the
  candidate is being helped.
- Visual presence should remain visible for review without collapsing into an
  assistance accusation.

### 10. Accuracy runs clear stale annotated images

The runner clears prior `*_annotated.jpg` files before writing a new annotated
batch. Reports were already overwritten per run, but old annotated frames could
otherwise remain in the folder and be mistaken for frames from the current run.

Quality-only frames now use `QUALITY_SIGNAL`; `PUBLIC_PRESENCE...` is reserved
for frames with extra-person or multiple-face context.

### 11. RT-DETR can run as a benchmark profile

`configs/rtdetr_accuracy.toml` runs the same evidence and risk pipeline with
pretrained Ultralytics RT-DETR object detections and writes to
`outputs/rtdetr_accuracy_run`.

Why:

- Off-the-shelf detector choice should be compared on the same images before
  deciding that domain fine-tuning is necessary.
- Keeping YOLO and RT-DETR in separate output folders makes visual comparison
  possible without overwriting the baseline.

## How To Run

```powershell
.\.venv\Scripts\python.exe run_accuracy_pipeline.py
```

To benchmark the pretrained RT-DETR profile on the same `images` folder:

```powershell
.\.venv\Scripts\python.exe run_accuracy_pipeline.py --config configs\rtdetr_accuracy.toml
```

After filling a label CSV based on the template:

```powershell
.\.venv\Scripts\python.exe run_accuracy_pipeline.py --labels data\labels\frame_labels.csv
```

The labeled run writes `metrics.json` with measurable precision and recall.

## Before And After

| Area | Previous POC path | New accuracy path |
| --- | --- | --- |
| Main pipeline | Three notebook flows and CSV merges | One repeatable runner |
| Thresholds | Notebook/module constants | Versioned TOML config |
| Object evidence | Per-notebook YOLO result handling | Fused configurable YOLO evidence |
| Candidate choice | Largest/central person heuristic | Face-aware person association |
| Gaze link | First landmarker face assumed candidate | Landmarker bbox merged with face/person evidence |
| Reports | Model-name-specific CSV columns | Stable signal columns |
| Processed images | Annotated output split across notebook folders | One annotated image folder for the accuracy run |
| Temporal risk | Notebook demo used placeholder close distances | Runner uses real frame extra-person details |
| Evaluation | Blank manual evaluation columns | Label template and metric computation |

## Previous Output Comparison

The existing `outputs/reports/report.csv` row for `ss_107.png` reports
`n_person_count=0` and `s_person_count=0`. The existing
`outputs/reports/report.json` for the same frame contains three person
detections for both configured YOLO models, and the same CSV row includes
`MULTIPLE_PEOPLE`.

That is a reporting mismatch, not a detector result. The new report avoids this
class of mismatch by counting fused evidence labels instead of searching for old
weight-specific columns.

## RT-DETR Benchmark Comparison

The pretrained RT-DETR profile was run on the current 40-image `images` batch
and compared against the default YOLO accuracy profile through the same face
association, risk policy, reports, and annotated-image writer.

| Output group | YOLO profile | RT-DETR profile |
| --- | ---: | ---: |
| `HIGH PHONE_DETECTED` | 10 | 11 |
| `LOW QUALITY_SIGNAL` | 26 | 19 |
| `LOW PUBLIC_PRESENCE_OR_QUALITY_SIGNAL` | 3 | 2 |
| `LOW CONTEXTUAL_SIGNAL` | 0 | 7 |
| `MEDIUM MULTIPLE_CONTEXTUAL_SIGNALS` | 1 | 1 |

This is an output comparison, not a measured precision/recall result, because
the batch does not yet have frame labels. The visual review still found useful
differences:

- RT-DETR detected the same small wall patch as `cell_phone` in `ss_182.png`,
  `ss_212.png`, and `ss_242.png`. Those look like false phone evidence.
- YOLO detected obvious large phones in `ss_392.png` and `ss_512.png` that
  RT-DETR missed.
- RT-DETR reduced some YOLO person false positives around a close phone crop in
  `ss_452.png` and `ss_602.png`.
- Both profiles still need better blur calibration on this screenshot batch and
  guardrails for tiny background person/face context.

Current recommendation: keep the YOLO profile as the default pretrained path,
keep RT-DETR available for labeled comparison, and spend the next no-training
accuracy effort on policy/geometry filters plus labeled validation before
deciding on fine-tuning.

## What Is Still Needed For Best Accuracy

1. Fill a labeled validation set from real proctoring frames.
2. Keep a held-out test set that is not used while tuning thresholds.
3. Measure false positives separately for `HIGH` risk because that has the
   highest product cost.
4. Add real video clips and session labels so temporal thresholds are tuned on
   duration, not only static screenshots.
5. Fine-tune YOLO only after labeled metrics show repeatable misses for
   proctoring-specific objects or poses.

## Files Added Or Changed

- `configs/accuracy.toml`
- `configs/rtdetr_accuracy.toml`
- `data/labels/frame_labels_template.csv`
- `run_accuracy_pipeline.py`
- `src/accuracy_config.py`
- `src/annotations.py`
- `src/evidence.py`
- `src/geometry.py`
- `src/object_detector.py`
- `src/object_filters.py`
- `src/face_detector.py`
- `src/candidate_association.py`
- `src/quality_checks.py`
- `src/risk_policy.py`
- `src/reporting.py`
- `src/evaluation_metrics.py`
- `src/head_pose_gaze.py`
- `src/extra_person_risk.py`
- `tests/test_accuracy_pipeline.py`
- `requirements.txt`
