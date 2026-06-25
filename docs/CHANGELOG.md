# Development Changelog

This changelog records the development history, release versions, and phase implementations of the Face ID Recognition system.

---

## 🌟 Version 1.0.0 (Release) — Liveness Integration & Final Consolidation
*   **Release Date**: June 2026
*   **Summary**: Completed liveness verification integration and reorganized documentation.

### New Features & Changes
*   **Liveness Verification Gate**: Fully integrated the trained PyTorch MobileNetV2 CNN classifier (`models/liveness/best_liveness.pth`) into real-time camera loops and web recognition routes.
*   **Spoof Rejection Visuals**: The UI now draws a RED bounding box and a `SPOOF (XX.X%)` label in OpenCV, while the web browser recognize view displays warning alerts and a red mask icon when spoofing is detected.
*   **Documentation Reorganization**: Consolidated 16 scattered intermediate reports inside `docs/` into 4 clean, standardized guides (`SETUP_GUIDE.md`, `USAGE_GUIDE.md`, `ARCHITECTURE.md`, `CHANGELOG.md`).

---

## 🚀 Phase 5 & 6 — ML Foundation & Advanced Anti-Spoofing
*   **Development Period**: Mid 2026

### Metric & Transfer Learning
*   Created `src/training/metric_learning.py` featuring angular margin loss (ArcFace Loss) and triplet loss with online semi-hard mining.
*   Built a multi-stage ArcFace transfer learning fine-tuning system (`src/training/fine_tuner.py`).

### Offline Evaluation Framework
*   Implemented standardized biometric evaluations (`src/evaluation/metrics.py`) calculating False Accept Rate (FAR), False Reject Rate (FRR), Equal Error Rate (EER), and Area Under Curve (AUC).
*   Configured empirical benchmark notebook evaluations on the Labeled Faces in the Wild (LFW) dataset, yielding a verification EER of 1.00% at a threshold of 0.634.

### Biometric Verification Models
*   Trained custom CNN liveness detection backbones on NUAA photo datasets to differentiate human skin texture from printed ink/paper.
*   Integrated Face Image Quality Assessment (FIQA) gates to automatically reject blurry or poorly exposed photos during registration.

---

## 🎨 Phase 4.5 — Neumorphism (Soft UI) Dashboard Redesign
*   **Development Period**: Early 2026

### Front-End Overhaul
*   Replaced the generic Bootstrap library with a custom, lightweight CSS-variable design tokens layout (`tokens.css`, `components.css`, `layout.css`).
*   Implemented physical-depth dual shadow offsets to create embossed, tactile neumorphic cards, carved-in input fields, and smooth active press animations.
*   Redesigned registration pages, real-time video capture panels with laser-scan animations, and the dashboard metrics grid.

---

## 📁 Phase 1 & 2 — Re-Architecting & Code Cleanups
*   **Development Period**: Late 2025

### Code Integrity
*   Consolidated 3 separate script installers into one centralized `install.py`.
*   Removed vulnerable pickle serialization, replacing it with NumPy array binary BLOB storage.
*   Recreated `AdvancedFaceProcessor` to standardize rotation alignments.
*   Refactored the codebase into clean sub-packages under `src/`.
*   Secured queries using parameterized SQL inputs to prevent injection attacks.
*   Replaced hardcoded session authentication keys with dynamic OS environment variables.
*   Cleaned all source files by stripping emojis and formatting log prints.
