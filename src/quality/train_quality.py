"""
Training script for face quality model.
Self-supervised approach: Uses recognition similarity as the ground-truth quality proxy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging

from .quality_assessor import FaceQualityAssessor

logger = logging.getLogger(__name__)

def train_quality_model(train_loader, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    train_loader should yield (image, similarity_score_to_clean_reference)
    """
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
