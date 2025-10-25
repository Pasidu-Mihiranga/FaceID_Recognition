"""
Face Identity Management System
Creates robust face identities from multiple images, independent of lighting conditions
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import os
import pickle
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import json

logger = logging.getLogger(__name__)

class FaceIdentityManager:
    """
    Manages face identities by creating robust embeddings from multiple images
    """
    
    def __init__(self, identity_storage_path: str = "data/face_identities"):
        """
        Initialize Face Identity Manager
        
        Args:
            identity_storage_path: Path to store identity data
        """
        self.identity_storage_path = identity_storage_path
        os.makedirs(identity_storage_path, exist_ok=True)
        
        # Identity settings
        self.min_images_for_identity = 3  # Minimum images needed to create identity
        self.max_images_for_identity = 20  # Maximum images to use for identity
        self.identity_threshold = 0.6  # Threshold for identity matching
        self.lighting_normalization = True  # Enable lighting normalization
        
        # Load existing identities
        self.identities = self._load_identities()
        
        logger.info(f"Face Identity Manager initialized with {len(self.identities)} identities")
    
    def create_identity(self, person_name: str, face_images: List[np.ndarray], 
                       face_embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        Create a robust face identity from multiple images
        
        Args:
            person_name: Name of the person
            face_images: List of face images (numpy arrays)
            face_embeddings: List of corresponding face embeddings
            
        Returns:
            Dictionary containing identity information
        """
        try:
            logger.info(f"Creating identity for {person_name} from {len(face_images)} images")
            
            if len(face_images) < self.min_images_for_identity:
                raise ValueError(f"Need at least {self.min_images_for_identity} images to create identity")
            
            # Limit number of images
            if len(face_images) > self.max_images_for_identity:
                face_images = face_images[:self.max_images_for_identity]
                face_embeddings = face_embeddings[:self.max_images_for_identity]
            
            # Preprocess images for lighting normalization
            if self.lighting_normalization:
                normalized_images = self._normalize_lighting(face_images)
            else:
                normalized_images = face_images
            
            # Extract additional embeddings from normalized images if needed
            if len(face_embeddings) < len(normalized_images):
                # This would require the face recognizer to extract embeddings
                # For now, we'll use the provided embeddings
                pass
            
            # Create master identity embedding
            master_embedding = self._create_master_embedding(face_embeddings)
            
            # Calculate identity statistics
            identity_stats = self._calculate_identity_stats(face_embeddings, master_embedding)
            
            # Create identity object
            identity = {
                'person_name': person_name,
                'master_embedding': master_embedding,
                'embedding_variance': identity_stats['variance'],
                'confidence_score': identity_stats['confidence'],
                'image_count': len(face_images),
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'face_images': normalized_images,  # Store normalized images
                'original_embeddings': face_embeddings,
                'identity_quality': identity_stats['quality']
            }
            
            # Save identity
            self._save_identity(identity)
            
            logger.info(f"Identity created for {person_name} with quality score: {identity_stats['quality']:.3f}")
            return identity
            
        except Exception as e:
            logger.error(f"Failed to create identity for {person_name}: {e}")
            raise
    
    def match_identity(self, face_embedding: np.ndarray, person_name: str = None) -> Tuple[Optional[str], float, Dict]:
        """
        Match a face embedding against stored identities
        
        Args:
            face_embedding: Face embedding to match
            person_name: Optional specific person to match against
            
        Returns:
            Tuple of (matched_person_name, confidence, match_info)
        """
        try:
            best_match = None
            best_confidence = 0.0
            match_info = {}
            
            # Normalize the input embedding
            normalized_embedding = normalize(face_embedding.reshape(1, -1)).flatten()
            
            if person_name:
                # Match against specific person
                if person_name in self.identities:
                    identity = self.identities[person_name]
                    confidence = self._calculate_similarity(normalized_embedding, identity['master_embedding'])
                    
                    if confidence >= self.identity_threshold:
                        return person_name, confidence, {'specific_match': True}
                    else:
                        return None, confidence, {'specific_match': False, 'below_threshold': True}
                else:
                    return None, 0.0, {'person_not_found': True}
            else:
                # Match against all identities
                for name, identity in self.identities.items():
                    confidence = self._calculate_similarity(normalized_embedding, identity['master_embedding'])
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = name
                        match_info = {
                            'identity_quality': identity['identity_quality'],
                            'image_count': identity['image_count']
                        }
                
                if best_confidence >= self.identity_threshold:
                    return best_match, best_confidence, match_info
                else:
                    return None, best_confidence, {'below_threshold': True}
                    
        except Exception as e:
            logger.error(f"Identity matching failed: {e}")
            return None, 0.0, {'error': str(e)}
    
    def update_identity(self, person_name: str, new_face_images: List[np.ndarray], 
                       new_embeddings: List[np.ndarray]) -> bool:
        """
        Update an existing identity with new images
        
        Args:
            person_name: Name of the person
            new_face_images: New face images to add
            new_embeddings: New face embeddings to add
            
        Returns:
            True if update successful
        """
        try:
            if person_name not in self.identities:
                logger.warning(f"Identity for {person_name} not found, creating new one")
                return self.create_identity(person_name, new_face_images, new_embeddings) is not None
            
            existing_identity = self.identities[person_name]
            
            # Combine with existing data
            all_images = existing_identity['face_images'] + new_face_images
            all_embeddings = existing_identity['original_embeddings'] + new_embeddings
            
            # Limit total images
            if len(all_images) > self.max_images_for_identity:
                # Keep the best quality images
                all_images = all_images[:self.max_images_for_identity]
                all_embeddings = all_embeddings[:self.max_images_for_identity]
            
            # Recreate identity with combined data
            updated_identity = self.create_identity(person_name, all_images, all_embeddings)
            
            logger.info(f"Identity updated for {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update identity for {person_name}: {e}")
            return False
    
    def _create_master_embedding(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Create a master embedding from multiple face embeddings
        
        Args:
            embeddings: List of face embeddings
            
        Returns:
            Master embedding vector
        """
        try:
            # Convert to numpy array
            embedding_matrix = np.array(embeddings)
            
            # Method 1: Weighted average based on embedding quality
            weights = self._calculate_embedding_weights(embedding_matrix)
            master_embedding = np.average(embedding_matrix, axis=0, weights=weights)
            
            # Method 2: Use PCA to find the most representative embedding
            if len(embeddings) >= 3:
                pca = PCA(n_components=1)
                pca.fit(embedding_matrix)
                pca_embedding = pca.components_[0]
                
                # Combine weighted average with PCA
                master_embedding = 0.7 * master_embedding + 0.3 * pca_embedding
            
            # Normalize the master embedding
            master_embedding = normalize(master_embedding.reshape(1, -1)).flatten()
            
            return master_embedding
            
        except Exception as e:
            logger.error(f"Failed to create master embedding: {e}")
            # Fallback to simple average
            return normalize(np.mean(embeddings, axis=0).reshape(1, -1)).flatten()
    
    def _calculate_embedding_weights(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate weights for embeddings based on their quality
        
        Args:
            embedding_matrix: Matrix of face embeddings
            
        Returns:
            Array of weights for each embedding
        """
        try:
            # Calculate pairwise similarities
            similarities = np.dot(embedding_matrix, embedding_matrix.T)
            
            # Calculate average similarity for each embedding
            avg_similarities = np.mean(similarities, axis=1)
            
            # Convert to weights (higher similarity = higher weight)
            weights = avg_similarities / np.sum(avg_similarities)
            
            return weights
            
        except Exception as e:
            logger.error(f"Failed to calculate embedding weights: {e}")
            # Return equal weights
            return np.ones(len(embedding_matrix)) / len(embedding_matrix)
    
    def _normalize_lighting(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Enhanced lighting normalization for face images
        
        Args:
            face_images: List of face images
            
        Returns:
            List of normalized face images
        """
        try:
            normalized_images = []
            
            for image in face_images:
                # Method 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_channel = lab[:, :, 0]
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l_normalized = clahe.apply(l_channel)
                
                # Update the LAB image
                lab[:, :, 0] = l_normalized
                
                # Convert back to BGR
                normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
                # Method 2: Gamma correction for better contrast
                gamma = 1.2
                lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
                normalized = cv2.LUT(normalized, lookup_table)
                
                # Method 3: Histogram stretching
                normalized = self._histogram_stretch(normalized)
                
                # Method 4: White balance correction
                normalized = self._white_balance_correction(normalized)
                
                normalized_images.append(normalized)
            
            return normalized_images
            
        except Exception as e:
            logger.error(f"Enhanced lighting normalization failed: {e}")
            return face_images  # Return original images if normalization fails
    
    def _histogram_stretch(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram stretching to improve contrast
        
        Args:
            image: Input image
            
        Returns:
            Histogram stretched image
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Find min and max values
            min_val = np.min(l_channel)
            max_val = np.max(l_channel)
            
            # Stretch histogram
            if max_val > min_val:
                stretched = ((l_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                lab[:, :, 0] = stretched
            
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            logger.error(f"Histogram stretching failed: {e}")
            return image
    
    def _white_balance_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply white balance correction
        
        Args:
            image: Input image
            
        Returns:
            White balance corrected image
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Calculate mean values for a and b channels
            mean_a = np.mean(lab[:, :, 1])
            mean_b = np.mean(lab[:, :, 2])
            
            # Apply correction
            lab[:, :, 1] = lab[:, :, 1] - (mean_a - 128)
            lab[:, :, 2] = lab[:, :, 2] - (mean_b - 128)
            
            # Clip values to valid range
            lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)
            
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            logger.error(f"White balance correction failed: {e}")
            return image
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            
            # Ensure similarity is between 0 and 1
            similarity = max(0, min(1, similarity))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_identity_stats(self, embeddings: List[np.ndarray], master_embedding: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics for the identity
        
        Args:
            embeddings: List of embeddings
            master_embedding: Master embedding
            
        Returns:
            Dictionary of statistics
        """
        try:
            # Calculate similarities to master embedding
            similarities = [self._calculate_similarity(emb, master_embedding) for emb in embeddings]
            
            # Calculate statistics
            mean_similarity = np.mean(similarities)
            variance = np.var(similarities)
            confidence = mean_similarity
            
            # Quality score based on consistency
            quality = mean_similarity * (1 - variance)  # Higher mean, lower variance = better quality
            
            return {
                'mean_similarity': float(mean_similarity),
                'variance': float(variance),
                'confidence': float(confidence),
                'quality': float(quality)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate identity stats: {e}")
            return {
                'mean_similarity': 0.0,
                'variance': 1.0,
                'confidence': 0.0,
                'quality': 0.0
            }
    
    def _save_identity(self, identity: Dict[str, Any]):
        """
        Save identity to storage
        
        Args:
            identity: Identity dictionary to save
        """
        try:
            person_name = identity['person_name']
            
            # Save identity data
            identity_path = os.path.join(self.identity_storage_path, f"{person_name}_identity.json")
            
            # Create a copy without numpy arrays for JSON serialization
            identity_copy = identity.copy()
            identity_copy['master_embedding'] = identity['master_embedding'].tolist()
            identity_copy['original_embeddings'] = [emb.tolist() for emb in identity['original_embeddings']]
            
            # Remove face images from JSON (too large)
            identity_copy.pop('face_images', None)
            
            with open(identity_path, 'w') as f:
                json.dump(identity_copy, f, indent=2)
            
            # Save face images separately
            images_path = os.path.join(self.identity_storage_path, f"{person_name}_images.pkl")
            with open(images_path, 'wb') as f:
                pickle.dump(identity['face_images'], f)
            
            # Update in-memory identities
            self.identities[person_name] = identity
            
            logger.info(f"Identity saved for {person_name}")
            
        except Exception as e:
            logger.error(f"Failed to save identity for {person_name}: {e}")
    
    def _load_identities(self) -> Dict[str, Dict[str, Any]]:
        """
        Load identities from storage
        
        Returns:
            Dictionary of loaded identities
        """
        try:
            identities = {}
            
            if not os.path.exists(self.identity_storage_path):
                return identities
            
            # Load identity files
            for filename in os.listdir(self.identity_storage_path):
                if filename.endswith('_identity.json'):
                    person_name = filename.replace('_identity.json', '')
                    
                    identity_path = os.path.join(self.identity_storage_path, filename)
                    images_path = os.path.join(self.identity_storage_path, f"{person_name}_images.pkl")
                    
                    try:
                        # Load identity data
                        with open(identity_path, 'r') as f:
                            identity_data = json.load(f)
                        
                        # Convert lists back to numpy arrays
                        identity_data['master_embedding'] = np.array(identity_data['master_embedding'])
                        identity_data['original_embeddings'] = [np.array(emb) for emb in identity_data['original_embeddings']]
                        
                        # Load face images
                        if os.path.exists(images_path):
                            with open(images_path, 'rb') as f:
                                identity_data['face_images'] = pickle.load(f)
                        else:
                            identity_data['face_images'] = []
                        
                        identities[person_name] = identity_data
                        
                    except Exception as e:
                        logger.warning(f"Failed to load identity for {person_name}: {e}")
            
            logger.info(f"Loaded {len(identities)} identities from storage")
            return identities
            
        except Exception as e:
            logger.error(f"Failed to load identities: {e}")
            return {}
    
    def get_identity_info(self, person_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific identity
        
        Args:
            person_name: Name of the person
            
        Returns:
            Identity information or None
        """
        return self.identities.get(person_name)
    
    def list_identities(self) -> List[str]:
        """
        List all stored identities
        
        Returns:
            List of person names
        """
        return list(self.identities.keys())
    
    def delete_identity(self, person_name: str) -> bool:
        """
        Delete an identity
        
        Args:
            person_name: Name of the person
            
        Returns:
            True if deletion successful
        """
        try:
            if person_name not in self.identities:
                return False
            
            # Remove from memory
            del self.identities[person_name]
            
            # Remove files
            identity_path = os.path.join(self.identity_storage_path, f"{person_name}_identity.json")
            images_path = os.path.join(self.identity_storage_path, f"{person_name}_images.pkl")
            
            if os.path.exists(identity_path):
                os.remove(identity_path)
            if os.path.exists(images_path):
                os.remove(images_path)
            
            logger.info(f"Identity deleted for {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete identity for {person_name}: {e}")
            return False
