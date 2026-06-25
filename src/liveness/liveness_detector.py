"""
Custom CNN for face anti-spoofing / liveness detection.
Detects: printed photos, screen replays, masks.
"""

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
        """
        Local Binary Pattern texture features.
        Printed photos lack the micro-texture depth of real skin.
        """
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
