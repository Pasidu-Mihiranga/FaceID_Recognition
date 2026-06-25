"""
Comprehensive face recognition evaluation metrics.
Implements standard biometric evaluation protocols.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

class FaceRecognitionEvaluator:
    @staticmethod
    def compute_far_frr(genuine_scores, impostor_scores, thresholds):
        far = []
        frr = []
        for t in thresholds:
            far_val = np.sum(impostor_scores >= t) / len(impostor_scores) if len(impostor_scores) > 0 else 0
            frr_val = np.sum(genuine_scores < t) / len(genuine_scores) if len(genuine_scores) > 0 else 0
            far.append(far_val)
            frr.append(frr_val)
        return np.array(far), np.array(frr)

    @staticmethod
    def compute_eer(genuine_scores, impostor_scores):
        labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
        scores = np.concatenate([genuine_scores, impostor_scores])
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer = fpr[eer_idx]
        eer_threshold = thresholds[eer_idx]
        return eer, eer_threshold

    @staticmethod
    def compute_roc_curve(labels, scores):
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    @staticmethod
    def compute_rank_n_accuracy(query_embeddings, gallery_embeddings, query_labels, gallery_labels, ranks=[1, 5]):
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(query_embeddings, gallery_embeddings)
        
        accuracies = {r: 0.0 for r in ranks}
        for i, query_label in enumerate(query_labels):
            sorted_indices = np.argsort(-sim_matrix[i])
            for r in ranks:
                top_r_labels = gallery_labels[sorted_indices[:r]]
                if query_label in top_r_labels:
                    accuracies[r] += 1
                    
        for r in ranks:
            accuracies[r] /= len(query_labels)
            
        return accuracies
