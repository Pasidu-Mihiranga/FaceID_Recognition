"""
Transfer learning pipeline for ArcFace fine-tuning.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .training_utils import FaceDataset, get_training_transforms, get_validation_transforms
from .metric_learning import ArcFaceLoss
import logging

logger = logging.getLogger(__name__)

class ArcFaceFineTuner:
    def __init__(self, num_classes, embedding_dim=512, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Initialize standard backbone (e.g., ResNet50 modified for face recognition)
        # Here we mock the backbone for demonstration, in production you'd load pre-trained weights
        import torchvision.models as models
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, embedding_dim)
        self.backbone = self.backbone.to(self.device)
        
        self.arcface_head = ArcFaceLoss(embedding_dim=embedding_dim, num_classes=num_classes).to(self.device)

    def train(self, data_dir, epochs=10, batch_size=32, lr=1e-3):
        logger.info(f"Starting ArcFace Fine-tuning on {self.device} for {epochs} epochs")
        
        dataset = FaceDataset(data_dir, transform=get_training_transforms())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
            
        optimizer = optim.Adam([
            {'params': self.backbone.parameters(), 'lr': lr * 0.1},
            {'params': self.arcface_head.parameters(), 'lr': lr}
        ])
        
        self.backbone.train()
        self.arcface_head.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                embeddings = self.backbone(images)
                loss = self.arcface_head(embeddings, labels)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
        
        return True

    def export_model(self, path):
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'head_state_dict': self.arcface_head.state_dict(),
            'embedding_dim': self.embedding_dim,
            'num_classes': self.num_classes
        }, path)
        logger.info(f"Model exported to {path}")
