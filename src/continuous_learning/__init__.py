"""
Continuous Learning Module
Implements automatic model updates and learning from new face data
"""

import numpy as np
import cv2
import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque
import threading
import time
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContinuousLearningManager:
    """Manages continuous learning for face recognition system"""
    
    def __init__(self, recognition_manager, database_manager, 
                 learning_threshold: float = 0.7, 
                 min_samples_for_update: int = 3,
                 max_embeddings_per_person: int = 20):
        """
        Initialize continuous learning manager
        
        Args:
            recognition_manager: Face recognition manager instance
            database_manager: Database manager instance
            learning_threshold: Threshold for automatic learning
            min_samples_for_update: Minimum samples needed for model update
            max_embeddings_per_person: Maximum embeddings to keep per person
        """
        self.recognition_manager = recognition_manager
        self.database_manager = database_manager
        self.learning_threshold = learning_threshold
        self.min_samples_for_update = min_samples_for_update
        self.max_embeddings_per_person = max_embeddings_per_person
        
        # Learning queues for each person
        self.learning_queues = defaultdict(lambda: deque(maxlen=max_embeddings_per_person))
        
        # Learning statistics
        self.learning_stats = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'last_update_time': None,
            'persons_updated': set()
        }
        
        # Background learning thread
        self.learning_thread = None
        self.stop_learning = False
        
        # Start background learning
        self.start_background_learning()
    
    def process_recognition_result(self, face_image: np.ndarray, 
                                 person_name: Optional[str], 
                                 confidence: float) -> bool:
        """
        Process a recognition result for continuous learning
        
        Args:
            face_image: Detected face image
            person_name: Recognized person name (None if unknown)
            confidence: Recognition confidence
            
        Returns:
            True if learning was triggered, False otherwise
        """
        try:
            # Only learn from high-confidence recognitions
            if confidence < self.learning_threshold:
                return False
            
            if person_name:
                # Known person - add to learning queue
                embedding = self.recognition_manager.recognizer.extract_embedding(face_image)
                
                self.learning_queues[person_name].append({
                    'embedding': embedding,
                    'image': face_image,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                })
                
                logger.info(f"Added sample to learning queue for {person_name}")
                
                # Check if we have enough samples for learning
                if len(self.learning_queues[person_name]) >= self.min_samples_for_update:
                    return self._trigger_learning(person_name)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to process recognition result: {e}")
            return False
    
    def _trigger_learning(self, person_name: str) -> bool:
        """
        Trigger learning for a specific person
        
        Args:
            person_name: Name of the person to update
            
        Returns:
            True if learning was successful, False otherwise
        """
        try:
            logger.info(f"Triggering learning for {person_name}")
            
            # Get samples from queue
            samples = list(self.learning_queues[person_name])
            
            if len(samples) < self.min_samples_for_update:
                logger.warning(f"Not enough samples for {person_name}: {len(samples)}")
                return False
            
            # Extract embeddings
            embeddings = [sample['embedding'] for sample in samples]
            
            # Quality check - remove outliers
            cleaned_embeddings = self._remove_outliers(embeddings)
            
            if len(cleaned_embeddings) < self.min_samples_for_update:
                logger.warning(f"Not enough clean samples for {person_name}")
                return False
            
            # Update the recognition model
            success = self._update_person_model(person_name, cleaned_embeddings)
            
            if success:
                # Clear the learning queue
                self.learning_queues[person_name].clear()
                
                # Update statistics
                self.learning_stats['total_updates'] += 1
                self.learning_stats['successful_updates'] += 1
                self.learning_stats['last_update_time'] = datetime.now()
                self.learning_stats['persons_updated'].add(person_name)
                
                logger.info(f"Successfully updated model for {person_name}")
                return True
            else:
                self.learning_stats['failed_updates'] += 1
                logger.error(f"Failed to update model for {person_name}")
                return False
                
        except Exception as e:
            logger.error(f"Learning failed for {person_name}: {e}")
            self.learning_stats['failed_updates'] += 1
            return False
    
    def _remove_outliers(self, embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """
        Remove outlier embeddings using clustering
        
        Args:
            embeddings: List of face embeddings
            
        Returns:
            List of cleaned embeddings
        """
        try:
            if len(embeddings) < 3:
                return embeddings
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings)
            
            # Use DBSCAN clustering to identify outliers
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
            cluster_labels = clustering.fit_predict(embeddings_array)
            
            # Keep embeddings that are not outliers (label != -1)
            cleaned_embeddings = []
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Not an outlier
                    cleaned_embeddings.append(embeddings[i])
            
            logger.info(f"Removed {len(embeddings) - len(cleaned_embeddings)} outliers")
            return cleaned_embeddings
            
        except Exception as e:
            logger.error(f"Outlier removal failed: {e}")
            return embeddings
    
    def _update_person_model(self, person_name: str, new_embeddings: List[np.ndarray]) -> bool:
        """
        Update the recognition model with new embeddings for a person
        
        Args:
            person_name: Name of the person
            new_embeddings: New embeddings to add
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Get person from database
            person = self.database_manager.get_person_by_name(person_name)
            if not person:
                logger.error(f"Person {person_name} not found in database")
                return False
            
            # Get existing embeddings
            existing_embeddings = self.database_manager.get_person_embeddings(person['id'])
            
            # Combine with new embeddings
            all_embeddings = existing_embeddings + new_embeddings
            
            # Limit the number of embeddings
            if len(all_embeddings) > self.max_embeddings_per_person:
                # Keep the most recent embeddings
                all_embeddings = all_embeddings[-self.max_embeddings_per_person:]
            
            # Update the recognition manager's database
            self.recognition_manager.face_database[person_name] = all_embeddings
            self.recognition_manager.save_database()
            
            # Save new embeddings to database
            for embedding in new_embeddings:
                # Create a temporary image path for the embedding
                temp_image_path = f"data/learning_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                
                # Save embedding to database
                self.database_manager.add_face_image(
                    person_id=person['id'],
                    image_path=temp_image_path,
                    embedding=embedding,
                    confidence=1.0
                )
            
            logger.info(f"Updated {person_name} with {len(new_embeddings)} new embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update person model for {person_name}: {e}")
            return False
    
    def start_background_learning(self):
        """Start background learning thread"""
        if self.learning_thread is None or not self.learning_thread.is_alive():
            self.stop_learning = False
            self.learning_thread = threading.Thread(target=self._background_learning_loop)
            self.learning_thread.daemon = True
            self.learning_thread.start()
            logger.info("Background learning thread started")
    
    def stop_background_learning(self):
        """Stop background learning thread"""
        self.stop_learning = True
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5)
        logger.info("Background learning thread stopped")
    
    def _background_learning_loop(self):
        """Background learning loop"""
        while not self.stop_learning:
            try:
                # Check for persons with enough samples
                persons_to_update = []
                for person_name, queue in self.learning_queues.items():
                    if len(queue) >= self.min_samples_for_update:
                        persons_to_update.append(person_name)
                
                # Process updates
                for person_name in persons_to_update:
                    self._trigger_learning(person_name)
                
                # Sleep for a while before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Background learning error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'total_updates': self.learning_stats['total_updates'],
            'successful_updates': self.learning_stats['successful_updates'],
            'failed_updates': self.learning_stats['failed_updates'],
            'success_rate': (self.learning_stats['successful_updates'] / 
                           max(1, self.learning_stats['total_updates'])),
            'last_update_time': self.learning_stats['last_update_time'],
            'persons_updated': list(self.learning_stats['persons_updated']),
            'queue_sizes': {name: len(queue) for name, queue in self.learning_queues.items()}
        }
    
    def force_learning_update(self, person_name: str) -> bool:
        """
        Force a learning update for a specific person
        
        Args:
            person_name: Name of the person to update
            
        Returns:
            True if update was successful, False otherwise
        """
        if person_name in self.learning_queues and len(self.learning_queues[person_name]) > 0:
            return self._trigger_learning(person_name)
        else:
            logger.warning(f"No samples in queue for {person_name}")
            return False
    
    def clear_learning_queue(self, person_name: str):
        """Clear the learning queue for a specific person"""
        if person_name in self.learning_queues:
            self.learning_queues[person_name].clear()
            logger.info(f"Cleared learning queue for {person_name}")
    
    def get_person_learning_status(self, person_name: str) -> Dict[str, Any]:
        """Get learning status for a specific person"""
        queue = self.learning_queues.get(person_name, deque())
        
        return {
            'samples_in_queue': len(queue),
            'ready_for_update': len(queue) >= self.min_samples_for_update,
            'last_sample_time': queue[-1]['timestamp'] if queue else None,
            'average_confidence': np.mean([s['confidence'] for s in queue]) if queue else 0.0
        }
    
    def adaptive_threshold_learning(self, recognition_results: List[Dict]) -> float:
        """
        Adaptively adjust learning threshold based on recognition results
        
        Args:
            recognition_results: List of recent recognition results
            
        Returns:
            New learning threshold
        """
        try:
            if not recognition_results:
                return self.learning_threshold
            
            # Calculate statistics
            confidences = [r['confidence'] for r in recognition_results]
            mean_confidence = np.mean(confidences)
            std_confidence = np.std(confidences)
            
            # Adjust threshold based on performance
            if mean_confidence > 0.8 and std_confidence < 0.1:
                # High confidence, low variance - can be more selective
                new_threshold = min(0.9, self.learning_threshold + 0.05)
            elif mean_confidence < 0.6 or std_confidence > 0.2:
                # Low confidence or high variance - be more inclusive
                new_threshold = max(0.5, self.learning_threshold - 0.05)
            else:
                new_threshold = self.learning_threshold
            
            if abs(new_threshold - self.learning_threshold) > 0.01:
                logger.info(f"Adjusted learning threshold: {self.learning_threshold:.3f} -> {new_threshold:.3f}")
                self.learning_threshold = new_threshold
            
            return self.learning_threshold
            
        except Exception as e:
            logger.error(f"Adaptive threshold learning failed: {e}")
            return self.learning_threshold
    
    def export_learning_data(self, export_path: str):
        """Export learning data for analysis"""
        try:
            export_data = {
                'learning_stats': self.learning_stats,
                'learning_queues': {
                    name: [
                        {
                            'confidence': sample['confidence'],
                            'timestamp': sample['timestamp'].isoformat()
                        }
                        for sample in queue
                    ]
                    for name, queue in self.learning_queues.items()
                },
                'export_time': datetime.now().isoformat()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Learning data exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export learning data: {e}")

if __name__ == "__main__":
    # Test the continuous learning system
    print("Continuous Learning Module Test")
    
    # This would normally be initialized with actual managers
    # For testing, we'll just show the structure
    learning_manager = None  # Would be initialized with real managers
    
    print("Continuous learning module initialized successfully")
