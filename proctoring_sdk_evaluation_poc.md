# Non-LLM Proctoring SDK Evaluation POC

## 1. Current Goal

The immediate goal is not to build the full proctoring product.

The immediate goal is to test whether we can replace expensive LLM-based proctoring checks with cheaper specialized SDKs/models.

This POC should test SDKs/libraries on the sample images provided by the manager.

The main idea:

```txt
Use pre-trained models first.
Do not train immediately.
Check what works.
Tune thresholds and image sizes.
Only fine-tune/train if the base models consistently fail.
```

---

## 2. What We Are Avoiding

We are avoiding expensive hosted LLM or multimodal API-based checks such as:

```txt
OpenAI vision models
Gemini vision models
Claude vision models
LLM-based reasoning
LLM-generated cheating judgement
```

The problem with those systems:

```txt
High API cost
Hard to scale for many candidates
Non-deterministic outputs
Possible inconsistent judgement
Difficult to tune precisely
Harder to debug
```

---

## 3. What We Are Allowed To Use

This is not a "no AI" system.

We are allowed to use specialized CV/audio/ML models and SDKs such as:

```txt
YOLO
MediaPipe
OpenCV
ONNX Runtime
TensorFlow.js
Ultralytics
WebRTC VAD
Silero VAD
Classical CV rules
Rule engines
```

For the current image-based POC, the first three tools to test are:

```txt
YOLO       -> object/person/phone/book/laptop/monitor detection
MediaPipe  -> face detection, face count, landmarks
OpenCV     -> image quality checks such as blur, low light, blocked camera
```

---

## 4. Recommended First Approach

Start with a simple `.ipynb` notebook.

The notebook should:

```txt
1. Load all sample images from a folder
2. Run YOLO on every image
3. Run MediaPipe face detection on every image
4. Run OpenCV quality checks on every image
5. Save annotated images
6. Save a JSON/CSV report
7. Compare expected vs actual results manually
```

This should be done before any training.

---

## 5. Why We Should Not Train Immediately

Training should not be the first step.

Reasons:

```txt
Training requires a labeled dataset
Images need bounding box annotations
Data preparation takes time
Model training needs experimentation
Training may be unnecessary if base models work well
Bad dataset quality can make model worse
```

First, we need to know where the base models fail.

Example:

```txt
If YOLO already detects phone/person/book correctly, training is not needed.
If YOLO misses only small phones, try threshold/image-size tuning first.
If YOLO consistently misses phones/papers/headphones, then fine-tuning is justified.
```

---

## 6. First POC Detection Scope

For the initial image-based POC, test these detections:

### Object Detection

Using YOLO:

```txt
person
multiple people
cell phone
book
laptop
monitor / tv
keyboard
mouse
headphones, if model supports it
```

### Face Detection

Using MediaPipe:

```txt
face present
face missing
face count
multiple faces
face bounding box
face confidence
```

### Image Quality

Using OpenCV:

```txt
low light
overexposure
blur
camera blocked
low resolution
poor visibility
```

---

## 7. Recommended Notebook Structure

Use a notebook like:

```txt
proctoring_sdk_poc.ipynb
```

Suggested sections:

```txt
1. Install dependencies
2. Import libraries
3. Define image folder paths
4. Load sample images
5. YOLO object detection
6. MediaPipe face detection
7. OpenCV image quality checks
8. Combine results
9. Save annotated images
10. Save JSON/CSV report
11. Manual observations
12. Final recommendation
```

---

## 8. Suggested Project Folder Structure

```txt
proctoring-sdk-poc/
  images/
    sample_1.jpg
    sample_2.jpg
    sample_3.jpg

  outputs/
    annotated/
      sample_1_yolo.jpg
      sample_1_combined.jpg

    reports/
      report.json
      report.csv

  notebooks/
    proctoring_sdk_poc.ipynb

  src/
    yolo_detection.py
    mediapipe_face.py
    image_quality.py
    report_utils.py

  requirements.txt
  README.md
```

For the first version, everything can be inside the notebook.

Later, if the notebook becomes messy, move logic into Python files.

---

## 9. Recommended Dependencies

For local notebook testing:

```bash
pip install ultralytics
pip install opencv-python
pip install mediapipe
pip install pandas
pip install matplotlib
pip install pillow
```

Optional:

```bash
pip install onnxruntime
pip install roboflow
```

If running on Google Colab:

```bash
!pip install ultralytics opencv-python mediapipe pandas matplotlib pillow
```

---

## 10. YOLO Testing Plan

### Purpose

YOLO should be used for object detection.

It should detect:

```txt
person
cell phone
book
laptop
monitor
keyboard
mouse
```

### First Model To Try

Start with:

```txt
YOLOv8n
```

Why:

```txt
Fast
Lightweight
Easy to run
Good for first benchmark
```

Then try:

```txt
YOLOv8s
```

Why:

```txt
Better accuracy than YOLOv8n
Still manageable for POC
```

### Basic YOLO Code

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("images/sample_1.jpg", conf=0.25, imgsz=640)

for result in results:
    print(result.boxes)
    result.save(filename="outputs/annotated/sample_1_yolo.jpg")
```

### Batch YOLO Test

```python
from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n.pt")

image_dir = Path("images")
output_dir = Path("outputs/annotated")
output_dir.mkdir(parents=True, exist_ok=True)

for image_path in image_dir.glob("*.*"):
    results = model(str(image_path), conf=0.25, imgsz=640)

    for result in results:
        save_path = output_dir / f"{image_path.stem}_yolo.jpg"
        result.save(filename=str(save_path))

        print("Image:", image_path.name)
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            print(label, conf)
```

---

## 11. YOLO Tuning Without Training

Before training, try these tuning options.

### 1. Confidence Threshold

Default confidence may miss small objects.

Try:

```python
results = model(image_path, conf=0.25)
```

If missing objects:

```python
results = model(image_path, conf=0.15)
```

But remember:

```txt
Lower threshold = more detections but more false positives
Higher threshold = fewer false positives but more missed detections
```

Recommended starting thresholds:

```txt
person: 0.50
cell phone: 0.25 - 0.35
book: 0.25 - 0.35
laptop: 0.40
monitor/tv: 0.40
```

### 2. Image Size

Small phones may be missed at low image size.

Try:

```python
results = model(image_path, conf=0.25, imgsz=960)
```

or:

```python
results = model(image_path, conf=0.25, imgsz=1280)
```

Tradeoff:

```txt
Higher imgsz = better small object detection
Higher imgsz = slower inference
```

### 3. Try Larger Model

Try:

```python
model = YOLO("yolov8s.pt")
```

If needed:

```python
model = YOLO("yolov8m.pt")
```

For POC, compare:

```txt
YOLOv8n vs YOLOv8s
```

Do not jump to large models immediately.

---

## 12. MediaPipe Face Detection Plan

### Purpose

MediaPipe should detect:

```txt
face present
face missing
face count
multiple faces
face bounding box
```

### Why MediaPipe

```txt
Fast
Reliable for face detection
Works without OpenAI/Gemini
Good for proctoring-style webcam images
Can later support landmarks/head pose
```

### Basic MediaPipe Face Detection

```python
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

image = cv2.imread("images/sample_1.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
) as face_detection:

    results = face_detection.process(image_rgb)

    faces = results.detections if results.detections else []
    print("Face count:", len(faces))
```

### MediaPipe Tuning

Try different confidence values:

```txt
0.3
0.5
0.7
```

Recommended:

```txt
Start with 0.5
Lower to 0.3 if faces are missed
Increase to 0.7 if false faces appear
```

### Face Output

For each image, store:

```txt
face_count
face_present
multiple_faces
face_confidence
face_bbox
```

---

## 13. OpenCV Image Quality Plan

### Purpose

OpenCV should help detect image quality issues.

This is important because object/face models fail when the image is poor.

Detect:

```txt
low light
overexposure
blur
camera blocked
poor visibility
```

### Brightness Check

```python
import cv2
import numpy as np

def check_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 50:
        return "LOW_LIGHT", brightness
    elif brightness > 220:
        return "OVEREXPOSED", brightness
    else:
        return "GOOD_LIGHT", brightness
```

### Blur Check

```python
def check_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur_score < 80:
        return "BLURRY", blur_score
    else:
        return "SHARP", blur_score
```

### Camera Blocked / Dark Frame Check

```python
def check_blocked(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    std_dev = np.std(gray)

    if brightness < 20 and std_dev < 10:
        return "CAMERA_BLOCKED"
    return "NOT_BLOCKED"
```

### Output

For each image:

```txt
brightness_score
lighting_status
blur_score
blur_status
blocked_status
```

---

## 14. Combined Report Format

The notebook should generate a combined report per image.

Example JSON:

```json
{
  "image": "sample_1.jpg",
  "yolo": {
    "objects": [
      {
        "label": "person",
        "confidence": 0.91
      },
      {
        "label": "cell phone",
        "confidence": 0.67
      }
    ],
    "person_count": 1,
    "phone_detected": true,
    "book_detected": false,
    "laptop_detected": false
  },
  "mediapipe": {
    "face_count": 1,
    "face_present": true,
    "multiple_faces": false
  },
  "opencv": {
    "lighting_status": "GOOD_LIGHT",
    "brightness_score": 132.4,
    "blur_status": "SHARP",
    "blur_score": 210.8,
    "blocked_status": "NOT_BLOCKED"
  },
  "risk_flags": [
    "PHONE_DETECTED"
  ]
}
```

---

## 15. Manual Evaluation Table

The final notebook should create a table like this:

```txt
Image | Expected | YOLO Result | MediaPipe Result | OpenCV Result | Correct? | Notes
```

Example:

```txt
sample_1.jpg | phone visible | phone detected 0.76 | 1 face | good quality | yes | good result
sample_2.jpg | normal | no phone | 1 face | good quality | yes | clean
sample_3.jpg | phone visible | missed phone | 1 face | low light | no | try lower threshold/imgsz
sample_4.jpg | no candidate | no person | 0 face | good quality | yes | correct
```

This table is important for explaining results to the manager.

---

## 16. Decision Tree After Initial Testing

### Case 1: Base Models Work Well

If YOLO + MediaPipe + OpenCV work well:

```txt
No training needed immediately.
Use pre-trained models.
Tune thresholds.
Add rule engine.
Move toward backend integration.
```

### Case 2: Some Objects Are Missed

If phone/book/paper is missed sometimes:

Try:

```txt
Lower confidence threshold
Increase image size
Try YOLOv8s instead of YOLOv8n
Improve image preprocessing
Use cropped regions
```

### Case 3: Repeated Failure

If the same kind of object is missed repeatedly:

Examples:

```txt
phones at edge of frame
small phones
paper/notes
earphones
second monitor
low-light objects
```

Then fine-tuning is justified.

### Case 4: Too Many False Positives

If model detects phone where there is no phone:

Try:

```txt
Increase confidence threshold
Use class-specific threshold
Require repeated detection in video
Ignore tiny bounding boxes
Add rule-based filtering
Fine-tune with negative examples
```

---

## 17. When Training Is Needed

Training/fine-tuning is needed only if base models are consistently weak on the company's real proctoring images.

Fine-tuning is useful for:

```txt
small phones
partially visible phones
phones near face
papers/notes
earphones/headphones
second screen
low-light webcam images
specific camera angles
specific cheating scenarios
```

Training is not needed if:

```txt
base model detects most required objects
errors can be fixed with threshold tuning
errors are caused by bad image quality
sample size is too small to conclude
```

---

## 18. Fine-Tuning Plan If Needed

If training becomes necessary, move to Google Colab.

### Important Note About TPU

YOLO training usually works better on GPU than TPU.

For Colab, prefer:

```txt
GPU runtime: T4 / L4 / A100 if available
```

TPU is more commonly useful for TensorFlow/JAX workflows.

Ultralytics YOLO/PyTorch training is usually GPU-based.

So in Colab:

```txt
Runtime → Change runtime type → GPU
```

not TPU, unless a specific TensorFlow model requires TPU.

---

## 19. Dataset Preparation For YOLO Fine-Tuning

You need labeled images.

### Required Dataset

Collect images for:

```txt
normal candidate
phone visible
phone in hand
phone on desk
phone partially visible
book visible
paper/notes visible
extra person visible
headphones/earphones
low light
blur
side face
candidate looking down
```

### Annotation Tools

Use one of:

```txt
Roboflow
CVAT
LabelImg
Label Studio
```

### YOLO Dataset Format

Folder structure:

```txt
dataset/
  images/
    train/
    val/

  labels/
    train/
    val/

  data.yaml
```

Example `data.yaml`:

```yaml
path: /content/dataset
train: images/train
val: images/val

names:
  0: person
  1: phone
  2: book
  3: paper
  4: headphones
  5: monitor
```

---

## 20. YOLO Training Code For Colab

Install:

```python
!pip install ultralytics
```

Train:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="/content/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

For better accuracy:

```python
model = YOLO("yolov8s.pt")
```

After training, the best model is usually saved at:

```txt
runs/detect/train/weights/best.pt
```

Then test:

```python
model = YOLO("runs/detect/train/weights/best.pt")
results = model("test_image.jpg", conf=0.25)
```

---

## 21. What To Tell The Manager

A good update would be:

```txt
I will first benchmark pre-trained YOLO, MediaPipe, and OpenCV on the provided proctoring images. The first goal is to check whether object detection, face detection, and image quality checks work without using OpenAI/Gemini-style APIs.

If the base models are not accurate enough, I will first tune confidence thresholds and image size. Only if the failures are consistent, I will prepare a fine-tuning plan using labeled proctoring images, likely on Google Colab GPU.
```

---

## 22. Final Recommended Workflow

Follow this order:

```txt
1. Create image folder with manager-provided samples
2. Create an IPython notebook
3. Install YOLO, MediaPipe, OpenCV
4. Run YOLOv8n on all images
5. Run YOLOv8s on all images
6. Run MediaPipe face detection
7. Run OpenCV quality checks
8. Save annotated outputs
9. Generate JSON/CSV report
10. Manually compare expected vs actual
11. Tune thresholds and image size
12. Decide if fine-tuning is needed
13. If needed, prepare labeled dataset
14. Train YOLO on Colab GPU
15. Compare base model vs tuned model vs custom trained model
```

---

## 23. Final Output Expected From POC

The final output should include:

```txt
Notebook
Annotated images
JSON report
CSV report
Manual evaluation table
Observations
SDK recommendation
Whether training is required or not
```

---

## 24. Final Recommendation

Start simple.

Do not train first.

Use:

```txt
YOLOv8n / YOLOv8s
MediaPipe Face Detection
OpenCV quality checks
```

Then evaluate on the provided images.

If accuracy is acceptable:

```txt
Use these libraries and move toward backend integration.
```

If accuracy is weak:

```txt
Tune threshold and image size first.
```

If still weak:

```txt
Fine-tune YOLO on company-specific proctoring images using Google Colab GPU.
```

This is the most practical and production-friendly approach.
