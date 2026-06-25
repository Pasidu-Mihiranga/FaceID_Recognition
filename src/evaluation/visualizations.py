"""
Publication-quality visualization of evaluation results.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def plot_roc_curve(fpr, tpr, auc_score, model_name, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_far_frr(thresholds, far, frr, eer_threshold, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, far, label='False Accept Rate (FAR)', color='red')
    plt.plot(thresholds, frr, label='False Reject Rate (FRR)', color='blue')
    plt.axvline(x=eer_threshold, color='gray', linestyle='--', label=f'EER Threshold ({eer_threshold:.3f})')
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.title('FAR vs FRR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_distributions(genuine_scores, impostor_scores, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(genuine_scores, color='green', label='Genuine', kde=True, stat="density", bins=50, alpha=0.5)
    sns.histplot(impostor_scores, color='red', label='Impostor', kde=True, stat="density", bins=50, alpha=0.5)
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.title('Genuine vs Impostor Score Distributions')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
