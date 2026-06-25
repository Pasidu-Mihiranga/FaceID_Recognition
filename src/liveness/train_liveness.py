"""
Training script for liveness detection model.
Uses binary cross entropy to classify real vs spoofed faces.
"""

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
