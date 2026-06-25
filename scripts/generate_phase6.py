import os
import json

def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content.strip() + '\n')

def create_notebook(path, title, cells_content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    cells = []
    
    # Title markdown cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n"]
    })
    
    for content in cells_content:
        cells.append({
            "cell_type": content["type"],
            "metadata": {},
            "source": [line + "\n" for line in content["source"].split('\n')],
            **({"outputs": [], "execution_count": None} if content["type"] == "code" else {})
        })
        
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

# 1. Liveness Detector
liveness_detector_content = """
\"\"\"
Custom CNN for face anti-spoofing / liveness detection.
Detects: printed photos, screen replays, masks.
\"\"\"

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2

class LivenessDetector(nn.Module):
    def __init__(self, num_classes=2): # 0: fake, 1: real
        super(LivenessDetector, self).__init__()
        
        # Lightweight backbone suitable for real-time mobile/webcam use
        self.backbone = models.mobilenet_v2(pretrained=True)
        
        # Replace the classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

    @staticmethod
    def extract_lbp_features(image_cv):
        \"\"\"
        Local Binary Pattern texture features.
        Printed photos lack the micro-texture depth of real skin.
        \"\"\"
        try:
            from skimage.feature import local_binary_pattern
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            return hist
        except ImportError:
            return None
"""

# 2. Train Liveness
train_liveness_content = """
\"\"\"
Training script for liveness detection model.
Uses binary cross entropy to classify real vs spoofed faces.
\"\"\"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
import os

from .liveness_detector import LivenessDetector

logger = logging.getLogger(__name__)

def train_liveness_model(train_loader, val_loader, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    logger.info(f"Training Liveness Model on {device}")
    
    model = LivenessDetector(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = correct / total
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("models/liveness", exist_ok=True)
            torch.save(model.state_dict(), "models/liveness/best_liveness.pth")
            logger.info("Saved new best model!")
            
    return model
"""

# 3. Quality Assessor
quality_assessor_content = """
\"\"\"
Face Image Quality Assessment (FIQA) model.
Predicts quality score (0.0-1.0) for a face image.
\"\"\"

import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np

class FaceQualityAssessor(nn.Module):
    def __init__(self):
        super(FaceQualityAssessor, self).__init__()
        # Using ResNet18 for fast regression
        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        
        # Regression head outputting a single quality score (0-1)
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.backbone(x)

    @staticmethod
    def compute_laplacian_variance(image_cv):
        \"\"\"
        Mathematical heuristic for sharpness/blur detection.
        Lower variance means the image is blurry.
        \"\"\"
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
        
    def should_accept(self, image_cv, min_sharpness=100.0, min_ai_quality=0.6):
        \"\"\"Gate function: reject low-quality faces at registration time\"\"\"
        # 1. Fast heuristic check
        sharpness = self.compute_laplacian_variance(image_cv)
        if sharpness < min_sharpness:
            return False, f"Image too blurry (Sharpness: {sharpness:.1f})"
            
        # 2. AI check (assuming preprocessing into tensor is done)
        # score = self.forward(tensor).item()
        # if score < min_ai_quality:
        #    return False, "AI determined image quality is too low"
            
        return True, "Quality acceptable"
"""

# 4. Train Quality
train_quality_content = """
\"\"\"
Training script for face quality model.
Self-supervised approach: Uses recognition similarity as the ground-truth quality proxy.
\"\"\"

import torch
import torch.nn as nn
import torch.optim as optim
import logging

from .quality_assessor import FaceQualityAssessor

logger = logging.getLogger(__name__)

def train_quality_model(train_loader, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    \"\"\"
    train_loader should yield (image, similarity_score_to_clean_reference)
    \"\"\"
    logger.info("Training Quality Assessment Regression Model")
    
    model = FaceQualityAssessor().to(device)
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, target_scores in train_loader:
            inputs, target_scores = inputs.to(device), target_scores.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, target_scores)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        logger.info(f"Epoch {epoch+1}/{epochs} | MSE Loss: {running_loss/len(train_loader):.4f}")
        
    return model
"""

base_dir = r"c:\Users\PMIHIR\Desktop\FaceID\FaceID_Recognition\src"

# Create source files
create_file(os.path.join(base_dir, "liveness", "__init__.py"), "")
create_file(os.path.join(base_dir, "liveness", "liveness_detector.py"), liveness_detector_content)
create_file(os.path.join(base_dir, "liveness", "train_liveness.py"), train_liveness_content)

create_file(os.path.join(base_dir, "quality", "__init__.py"), "")
create_file(os.path.join(base_dir, "quality", "quality_assessor.py"), quality_assessor_content)
create_file(os.path.join(base_dir, "quality", "train_quality.py"), train_quality_content)

# Create Notebooks
notebooks_dir = r"c:\Users\PMIHIR\Desktop\FaceID\FaceID_Recognition\notebooks"

nb_01 = [
    {"type": "markdown", "source": "This notebook demonstrates how to compare FaceNet, ArcFace, and VGG-Face on the LFW benchmark dataset."},
    {"type": "code", "source": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom src.evaluation.benchmarks import LFWBenchmark\n\nprint('Ready to run model comparisons.')"}
]

nb_02 = [
    {"type": "markdown", "source": "This notebook analyzes score distributions to find the optimal threshold for False Accept Rate (FAR) and False Reject Rate (FRR)."},
    {"type": "code", "source": "import numpy as np\nfrom src.evaluation.metrics import FaceRecognitionEvaluator\nfrom src.evaluation.visualizations import plot_far_frr\n\nprint('Ready for threshold analysis.')"}
]

nb_03 = [
    {"type": "markdown", "source": "This ablation study determines the impact of color and lighting data augmentation on the final ArcFace recognition accuracy."},
    {"type": "code", "source": "import pandas as pd\nimport matplotlib.pyplot as plt\n\nprint('Ready for augmentation impact analysis.')"}
]

nb_04 = [
    {"type": "markdown", "source": "Training and Evaluation pipeline for the MobileNetV2 anti-spoofing liveness detector."},
    {"type": "code", "source": "import torch\nfrom src.liveness.liveness_detector import LivenessDetector\n\nmodel = LivenessDetector()\nprint('Liveness model initialized:', model)"}
]

nb_05 = [
    {"type": "markdown", "source": "Latency and throughput benchmarks for CPU vs GPU inference times."},
    {"type": "code", "source": "import time\nimport torch\n\nprint('Ready to run performance benchmarks.')"}
]

create_notebook(os.path.join(notebooks_dir, "01_model_comparison.ipynb"), "Model Comparison: ArcFace vs FaceNet", nb_01)
create_notebook(os.path.join(notebooks_dir, "02_threshold_analysis.ipynb"), "Optimal Threshold Selection (FAR vs FRR)", nb_02)
create_notebook(os.path.join(notebooks_dir, "03_augmentation_impact.ipynb"), "Data Augmentation Ablation Study", nb_03)
create_notebook(os.path.join(notebooks_dir, "04_liveness_detection.ipynb"), "Training Liveness Detection (Anti-Spoofing)", nb_04)
create_notebook(os.path.join(notebooks_dir, "05_performance_benchmarks.ipynb"), "System Performance Benchmarks", nb_05)

print("Phase 6 Advanced ML files generated successfully.")
