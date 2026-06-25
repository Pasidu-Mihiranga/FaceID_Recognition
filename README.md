# Face ID Recognition System

A production-ready, highly advanced biometric facial recognition and continuous learning system designed with state-of-the-art computer vision algorithms, PyTorch deep learning pipelines, and a modern Neumorphic (Soft UI) web interface.

This repository features advanced metric learning implementations, transfer learning/fine-tuning pipelines, convolutional liveness detection (anti-spoofing), face image quality assessment (FIQA), and rigorous offline biometric benchmarking.

---

## Core Capabilities and Architecture

The system has been refactored into a modular Python package structure conforming to industry-standard software engineering practices:

```
FaceID_Recognition/
├── data/                         # SQLite database, registered face embeddings, and cache
├── docs/                         # Detailed installation and changelog documentation
├── notebooks/                    # Jupyter research and evaluation notebooks
├── src/                          # Main source package
│   ├── core/                     # Central configuration management and custom exceptions
│   ├── database/                 # SQLite integration utilizing efficient NumPy array storage
│   ├── detection/                # Abstracted face detection engines (RetinaFace, MTCNN, OpenCV, Dlib)
│   ├── evaluation/               # Biometric performance metrics (FAR, FRR, EER, ROC/AUC)
│   ├── learning/                 # Continuous learning manager for online database adaptation
│   ├── liveness/                 # MobileNetV2 CNN and LBP-based face anti-spoofing (spoof detection)
│   ├── processing/               # Advanced preprocessing (CLAHE, LAB illumination correction, Augmentation)
│   ├── quality/                  # CNN-based Face Image Quality Assessment (FIQA)
│   ├── recognition/              # Feature extraction models (ArcFace, FaceNet, VGG-Face) and identity managers
│   └── web/                      # Flask-based web server and backend APIs
├── static/                       # Neumorphic CSS sheets, custom JS modules, and assets
├── templates/                    # Neumorphic user interface HTML templates
├── main.py                       # Main console entrypoint and systems integration
└── requirements.txt              # Dependency definitions
```

---

## Feature Details

### 1. Neumorphic Web Interface
The web interface has been entirely redesigned from the ground up using custom vanilla CSS to implement a tactile Neumorphic (Soft UI) design system:
- **Depth Physics**: Extruded cards, inset fields, and active-state button presses using dual-shadow offset configurations.
- **Modern Typography**: Reconfigured with Plus Jakarta Sans for titles and DM Sans for clean readability.
- **Enhanced UX**: Includes live camera video streaming overlayed with animated laser scanlines and interactive drag-and-drop image upload zones.

### 2. Deep Learning Recognizers
- **ArcFace (InsightFace)**: Employs additive angular margin loss backbones (Buffalo_L) running on ONNX Runtime for high-speed, state-of-the-art embedding generation.
- **DeepFace Wrappers**: Integrates FaceNet and VGG-Face configurations.
- **Robust Cropping Handlers**: Automatically manages full-frame and pre-cropped face inputs to avoid double-cropping errors during feature extraction.

### 3. Metric and Transfer Learning Frameworks
- **From-Scratch Metric Losses**: Native PyTorch implementations of Additive Angular Margin Loss (ArcFace Loss) and Triplet Margin Loss with online hard-negative/semi-hard mining logic (`src/training/metric_learning.py`).
- **Fine-Tuning Pipeline**: A multi-stage training framework (`src/training/fine_tuner.py`) that performs initial feature extraction head training on frozen backbones before unfreezing layers for end-to-end backpropagation.

### 4. Biometric Protection Systems
- **Liveness Detection**: A custom MobileNetV2 CNN classifier (`src/liveness/liveness_detector.py`) trained to identify presentation attacks (such as printed photos or digital screen replays), combined with texture analysis using Local Binary Patterns (LBP).
- **Face Image Quality Assessment (FIQA)**: A regression network (`src/quality/quality_assessor.py`) scoring faces on resolution, sharpness (Laplacian variance), contrast, and structure to filter out low-quality inputs before registration.

### 5. Efficient SQLite Database using NumPy
- Replaced vulnerable pickle serialization with direct binary serialization of NumPy arrays into the SQLite database.
- Tracks granular audit logs, biometric confidence levels, timestamps, face bounding boxes, and facial landmark coordinates.

---

## Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Pasidu-Mihiranga/FaceID_Recognition.git
   cd FaceID_Recognition
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Setup environment variables:
   Create a `.env` file in the root directory based on the `.env.example` file:
   ```env
   DETECTOR_TYPE=retinaface
   RECOGNITION_MODEL=arcface
   RECOGNITION_THRESHOLD=0.15
   LEARNING_THRESHOLD=0.7
   PORT=5000
   HOST=0.0.0.0
   ```

### Running the Web Application

To launch the web interface locally, run the script:
```bash
scripts/start_web.bat
```
Alternatively, execute the main entrypoint:
```bash
python main.py --web
```
Open `http://localhost:5000` in your web browser.

---

## Scientific Evaluation and Research Notebooks

The `notebooks/` directory contains complete validation workflows that analyze the system's performance:

1. **`01_model_comparison.ipynb`**: Ablation and latency comparison between ArcFace, FaceNet, and traditional recognition baselines.
2. **`02_threshold_analysis.ipynb`**: Evaluates Equal Error Rate (EER) by plotting False Accept Rate (FAR) vs. False Reject Rate (FRR) curves across different threshold scales.
3. **`03_augmentation_impact.ipynb`**: Analyzes the impact of data augmentation (rotations, brightness shifts, translations) on model accuracy.
4. **`04_liveness_detection.ipynb`**: Logs the training process and confusion matrix evaluations for the liveness classification network.
5. **`05_performance_benchmarks.ipynb`**: CPU vs. GPU profiling covering face detection, alignment, embedding extraction, and database lookup speeds.

---

## API Documentation

### Initializing the Face ID System

```python
from main import FaceIDSystem
import cv2

# Initialize FaceIDSystem
system = FaceIDSystem(
    detector_type='retinaface',
    recognition_model='arcface',
    recognition_threshold=0.15
)

# Register a new subject
success = system.register_person("path/to/image.jpg", "Thanoj")

# Recognize a face
image = cv2.imread("path/to/test.jpg")
name, confidence, face_info = system.recognize_person(image)

if name:
    print(f"Recognized: {name} (Confidence: {confidence:.2f})")
else:
    print("Unknown face detected")
```

### Key API Endpoints

- `POST /api/register` - Registers a person from an uploaded image.
- `POST /api/video_register` - Registers a person using a continuous video stream.
- `POST /api/recognize` - Identifies individuals from uploaded frames.
- `GET /api/stats` - Retreives system health metrics, total detections, database status, and CPU usage.
- `GET /api/persons` - Returns a JSON array of all registered individuals.
- `DELETE /api/person/<id>` - Deletes a registered profile and associated database records.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
