# Setup and Installation Guide

This guide describes how to set up the environment, install the dependencies, and configure the Face ID Recognition system.

---

## 📋 System Prerequisites

Before proceeding, ensure your development system meets the following requirements:
*   **Operating System**: Windows 10/11
*   **Python Runtime**: Python 3.14.x (64-bit)
*   **Permissions**: Administrative terminal capability for installing packages and opening ports.
*   **GPU Acceleration (Optional)**: CUDA Toolkit 12.x compatible GPU for faster model inference.

---

## 🎯 Step-by-Step Installation

### Step 1: Clone the Repository
Open a terminal (Command Prompt or PowerShell) and clone the repository:
```cmd
git clone https://github.com/Pasidu-Mihiranga/FaceID_Recognition.git
cd FaceID_Recognition
```

### Step 2: Set Up the Virtual Environment
Create a localized virtual environment named `faceid_env` using Python 3.14:
```cmd
python -m venv faceid_env
```
Activate the environment:
*   **Command Prompt (CMD)**:
    ```cmd
    faceid_env\Scripts\activate.bat
    ```
*   **PowerShell**:
    ```powershell
    .\faceid_env\Scripts\Activate.ps1
    ```

### Step 3: Install Required Dependencies
With the virtual environment active, install all required packages using `requirements.txt`:
```cmd
pip install -r requirements.txt
```
*Note: This will install major dependencies including PyTorch, Torchvision, OpenCV, ONNX Runtime, and Flask.*

### Step 4: Configure the Environment Variables
Create a file named `.env` in the root directory. You can use the `.env.example` file as a reference:
```env
# Central configuration variables
DETECTOR_TYPE=retinaface
RECOGNITION_MODEL=arcface
RECOGNITION_THRESHOLD=0.15
LEARNING_THRESHOLD=0.7
FLASK_PORT=5000
FLASK_HOST=0.0.0.0
FACEID_SECRET_KEY=generate_your_random_hex_string_here
```

---

## 🚀 System Verification & Startup

### Start the Neumorphic Web Dashboard
Double-click the batch file:
```cmd
scripts\start_web.bat
```
Or execute manually:
```cmd
python main.py --web
```
Access the dashboard at `http://localhost:5000` in your web browser.

### Start the Local Webcam Face Recognition Loop
Double-click the batch file:
```cmd
scripts\start_camera.bat
```
Or execute manually:
```cmd
python main.py --camera
```

---

## 🔧 Troubleshooting Common Setup Issues

### 1. NumPy/OpenCV DLL Load Errors
*   **Symptom**: `ImportError: DLL load failed while importing cv2`
*   **Resolution**: Install the visual studio runtime dependencies or reinstall OpenCV inside the active environment:
    ```cmd
    pip uninstall opencv-python
    pip install opencv-python
    ```

### 2. ONNX Runtime CUDA Execution Provider Warning
*   **Symptom**: `UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names`
*   **Meaning**: The system did not find a compatible NVIDIA CUDA GPU/Toolkit configuration, and fell back to the CPU execution provider. The system will continue to run correctly on the CPU.
*   **Resolution**: If you have an NVIDIA GPU, install the appropriate CUDA and cuDNN libraries matching the ONNX Runtime specification.

### 3. Port 5000 is Already in Use
*   **Symptom**: `OSError: [Errno 98] Address already in use` or Flask fails to start.
*   **Resolution**: Find and kill the process holding port 5000, or edit the `FLASK_PORT` environment variable inside your `.env` file to a different port (e.g. `5001`).
    To check on Windows Command Prompt:
    ```cmd
    netstat -ano | findstr :5000
    ```

### 4. Missing Liveness Model Weights
*   **Symptom**: Log warning states: `Liveness detection model weights not found at models\liveness\best_liveness.pth.`
*   **Resolution**: Ensure `best_liveness.pth` is saved inside the `models/liveness/` directory. If it is not present, the system will bypass liveness checking and continue to work as a standard facial recognition system.
