"""
Face Image Quality Assessment (FIQA) model.
Predicts quality score (0.0-1.0) for a face image.
"""

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
        """
        Mathematical heuristic for sharpness/blur detection.
        Lower variance means the image is blurry.
        """
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
        
    def should_accept(self, image_cv, min_sharpness=100.0, min_ai_quality=0.6):
        """Gate function: reject low-quality faces at registration time"""
        # 1. Fast heuristic check
        sharpness = self.compute_laplacian_variance(image_cv)
        if sharpness < min_sharpness:
            return False, f"Image too blurry (Sharpness: {sharpness:.1f})"
            
        # 2. AI check (assuming preprocessing into tensor is done)
        # score = self.forward(tensor).item()
        # if score < min_ai_quality:
        #    return False, "AI determined image quality is too low"
            
        return True, "Quality acceptable"
