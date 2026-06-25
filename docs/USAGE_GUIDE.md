# User and Command-Line Interface Usage Guide

This guide explains how to operate the Face ID Recognition system, describing command-line options, neumorphic web user workflows, registration strategies, and resource closing protocols.

---

## 🖥️ Command-Line Interface (CLI)

The main entry point for the system is [main.py](file:///c:/Users/PMIHIR/Desktop/FaceID/FaceID_Recognition/main.py). It parses parameters to run in either web-server or interactive-camera recognition mode.

### Usage Syntax
```cmd
python main.py [OPTIONS]
```

### CLI Parameters
| Option | Argument | Description | Default |
|:---|:---|:---|:---|
| `--web` | *None* | Launches the Neumorphic Web Interface (Flask server). | *Disabled* |
| `--camera` | *None* | Runs the interactive opencv camera recognition loop in a desktop window. | *Disabled* |
| `--port` | `INT` | Specifies the port for the Flask web application (overridden by `.env`'s `FLASK_PORT`). | `5000` |

---

## 🏃 Running the Services

### 1. Neumorphic Web Dashboard (Recommended)
This launches a browser-accessible web application with a complete UI for registration, dashboard analytics, and diagnostic logs.
*   **Via Script**: Double-click `scripts\start_web.bat`.
*   **Via CLI**: `python main.py --web`

### 2. Interactive Webcam recognition window
This starts a local window capture stream showing frames from your webcam in real-time.
*   **Via Script**: Double-click `scripts\start_camera.bat`.
*   **Via CLI**: `python main.py --camera`

---

## 🎨 Web Interface Workflows

Once the Flask application is running at `http://localhost:5000`, the following features are available in the neumorphic navigation bar:

### 1. Subject Registration
Registers individuals by uploading a static image.
*   Navigate to **Register**.
*   Input the person's name in the tactile text field.
*   Drag and drop or select an image containing a clear face.
*   Click **Register Person**. If successful, the model extracts the embedding and registers the profile in the database.

### 2. Multi-Angle Video Registration
Registers a person from a short video clip (captures multi-angle faces automatically).
*   Navigate to **Video Register**.
*   Input the person's name.
*   Upload a video clip showing different angles of the face.
*   The system parses frames, selects the highest-quality face crops, and registers them.

### 3. Face Identification
Identifies individuals and performs anti-spoofing checks.
*   Navigate to **Recognize**.
*   Upload a photo.
*   The system detects the face and runs a liveness check. If a spoof attempt (e.g. screen photo) is detected, it will block recognition, display a red mask icon, and present a **SPOOF ATTEMPT REJECTED** warning.

### 4. System Analytics Dashboard
Monitors database state and performance statistics.
*   Navigate to **Dashboard**.
*   View total registered profiles, total enrolled images, overall match logs, average query execution latency, and log audits.

---

## 🚨 System Shutdown & Clean Up

When running python processes or camera hooks, proper resource cleanup prevents resource locks:

1.  **Shutdown Web Server**: Press `Ctrl + C` in the command line terminal running the server. This releases port bindings and commits final SQLite cache states.
2.  **Stop Camera Stream**: Press `q` or `ESC` in the OpenCV video window, or run `Ctrl + C` in the camera CLI terminal. This releases camera capture hardware locks.
