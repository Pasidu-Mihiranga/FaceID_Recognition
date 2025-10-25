"""
External Dataset Training Module
Provides functions to train the Face ID system with external datasets
"""

import os
import cv2
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import logging
from main import FaceIDSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_from_folder_dataset(dataset_path: str, face_id_system: FaceIDSystem) -> Tuple[int, int]:
    """
    Train the system from a folder-based dataset
    
    Args:
        dataset_path: Path to the dataset folder
        face_id_system: Initialized FaceIDSystem instance
        
    Returns:
        Tuple of (successful_registrations, total_attempts)
    """
    successful_registrations = 0
    total_attempts = 0
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return 0, 0
    
    # Get all person folders
    person_folders = [f for f in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, f))]
    
    logger.info(f"Found {len(person_folders)} people to train")
    
    for person_folder in person_folders:
        person_path = os.path.join(dataset_path, person_folder)
        person_name = person_folder.replace('_', ' ').title()  # Convert folder name to person name
        
        # Get all images in the person's folder
        image_files = [f for f in os.listdir(person_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        logger.info(f"Training {person_name} with {len(image_files)} images...")
        
        person_success = 0
        
        for image_file in image_files:
            image_path = os.path.join(person_path, image_file)
            total_attempts += 1
            
            try:
                success = face_id_system.register_person(image_path, person_name)
                if success:
                    person_success += 1
                    successful_registrations += 1
                    logger.info(f"  ✅ {image_file}")
                else:
                    logger.warning(f"  ❌ {image_file} - No face detected")
            except Exception as e:
                logger.error(f"  ❌ {image_file} - Error: {e}")
        
        logger.info(f"  {person_name}: {person_success}/{len(image_files)} images registered")
    
    logger.info(f"Training Complete! {successful_registrations}/{total_attempts} images registered")
    return successful_registrations, total_attempts

def train_from_csv_dataset(csv_path: str, face_id_system: FaceIDSystem) -> int:
    """
    Train the system from a CSV dataset
    
    Args:
        csv_path: Path to the CSV file
        face_id_system: Initialized FaceIDSystem instance
        
    Returns:
        Number of successful registrations
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV file does not exist: {csv_path}")
        return 0
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return 0
    
    logger.info(f"Loading dataset from {csv_path}")
    logger.info(f"Found {len(df)} images to process")
    
    successful_registrations = 0
    
    for index, row in df.iterrows():
        image_path = row['image_path']
        person_name = row['person_name']
        
        # Check if image exists
        if not os.path.exists(image_path):
            logger.warning(f"❌ Image not found: {image_path}")
            continue
        
        try:
            success = face_id_system.register_person(image_path, person_name)
            if success:
                successful_registrations += 1
                logger.info(f"✅ {person_name}: {os.path.basename(image_path)}")
            else:
                logger.warning(f"❌ {person_name}: {os.path.basename(image_path)} - No face detected")
        except Exception as e:
            logger.error(f"❌ {person_name}: {os.path.basename(image_path)} - Error: {e}")
    
    logger.info(f"Training Complete! {successful_registrations}/{len(df)} images registered")
    return successful_registrations

def train_from_lfw_dataset(lfw_path: str, face_id_system: FaceIDSystem, max_people: int = 100) -> int:
    """
    Train from LFW dataset
    
    Args:
        lfw_path: Path to LFW dataset folder
        face_id_system: Initialized FaceIDSystem instance
        max_people: Maximum number of people to train
        
    Returns:
        Number of successful registrations
    """
    if not os.path.exists(lfw_path):
        logger.error(f"LFW path does not exist: {lfw_path}")
        return 0
    
    successful_registrations = 0
    people_processed = 0
    
    # LFW structure: lfw/name/image.jpg
    for person_folder in os.listdir(lfw_path):
        if people_processed >= max_people:
            break
            
        person_path = os.path.join(lfw_path, person_folder)
        if not os.path.isdir(person_path):
            continue
        
        person_name = person_folder.replace('_', ' ')
        people_processed += 1
        
        logger.info(f"Training {person_name}...")
        
        # Get first image for each person
        image_files = [f for f in os.listdir(person_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            image_path = os.path.join(person_path, image_files[0])
            
            try:
                success = face_id_system.register_person(image_path, person_name)
                if success:
                    successful_registrations += 1
                    logger.info(f"  ✅ {person_name}")
                else:
                    logger.warning(f"  ❌ {person_name} - No face detected")
            except Exception as e:
                logger.error(f"  ❌ {person_name} - Error: {e}")
    
    logger.info(f"LFW Training Complete! {successful_registrations}/{people_processed} people registered")
    return successful_registrations

def validate_dataset_image(image_path: str) -> Tuple[bool, int, int]:
    """
    Validate if an image is suitable for training
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (is_valid, face_count, face_size)
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False, 0, 0
        
        # Initialize face detector for validation
        face_id = FaceIDSystem()
        faces = face_id.face_detector.detect_faces(image)
        
        if not faces:
            return False, 0, 0
        
        if len(faces) > 1:
            return False, len(faces), 0
        
        # Check face size
        face_info = faces[0]
        x, y, w, h = face_info['bbox']
        face_size = w * h
        
        # Face should be at least 50x50 pixels
        if w < 50 or h < 50:
            return False, 1, face_size
        
        return True, 1, face_size
        
    except Exception as e:
        logger.error(f"Validation error for {image_path}: {e}")
        return False, 0, 0

def train_with_validation(dataset_path: str, face_id_system: FaceIDSystem, min_face_size: int = 2500) -> int:
    """
    Train with image validation
    
    Args:
        dataset_path: Path to dataset
        face_id_system: Initialized FaceIDSystem instance
        min_face_size: Minimum face size in pixels (50x50 = 2500)
        
    Returns:
        Number of successful registrations
    """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return 0
    
    successful_registrations = 0
    total_images = 0
    valid_images = 0
    
    # Get all image files recursively
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, file)
                total_images += 1
                
                # Validate image
                is_valid, face_count, face_size = validate_dataset_image(image_path)
                
                if not is_valid:
                    logger.warning(f"❌ Invalid: {file} (faces: {face_count}, size: {face_size})")
                    continue
                
                if face_size < min_face_size:
                    logger.warning(f"❌ Too small: {file} (size: {face_size})")
                    continue
                
                valid_images += 1
                
                # Extract person name from folder structure
                person_name = os.path.basename(root).replace('_', ' ').title()
                
                try:
                    success = face_id_system.register_person(image_path, person_name)
                    if success:
                        successful_registrations += 1
                        logger.info(f"✅ {person_name}: {file}")
                    else:
                        logger.warning(f"❌ Failed to register: {file}")
                except Exception as e:
                    logger.error(f"❌ Error: {file} - {e}")
    
    logger.info(f"Training Complete!")
    logger.info(f"Total images: {total_images}")
    logger.info(f"Valid images: {valid_images}")
    logger.info(f"Successfully registered: {successful_registrations}")
    
    return successful_registrations

def train_large_dataset_batch(dataset_path: str, face_id_system: FaceIDSystem, batch_size: int = 100) -> int:
    """
    Train large datasets in batches to manage memory
    
    Args:
        dataset_path: Path to dataset
        face_id_system: Initialized FaceIDSystem instance
        batch_size: Number of images to process before saving
        
    Returns:
        Number of successful registrations
    """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return 0
    
    successful_registrations = 0
    processed_images = 0
    
    # Collect all image paths
    all_images = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, file)
                person_name = os.path.basename(root).replace('_', ' ').title()
                all_images.append((image_path, person_name))
    
    logger.info(f"Found {len(all_images)} images to process")
    
    # Process in batches
    for i in range(0, len(all_images), batch_size):
        batch = all_images[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_images)-1)//batch_size + 1}")
        
        batch_success = 0
        
        for image_path, person_name in batch:
            processed_images += 1
            
            try:
                success = face_id_system.register_person(image_path, person_name)
                if success:
                    batch_success += 1
                    successful_registrations += 1
                    
                if processed_images % 10 == 0:
                    logger.info(f"  Processed {processed_images}/{len(all_images)}")
                    
            except Exception as e:
                logger.error(f"  Error: {os.path.basename(image_path)} - {e}")
        
        logger.info(f"  Batch complete: {batch_success}/{len(batch)} successful")
    
    logger.info(f"Large Dataset Training Complete!")
    logger.info(f"Successfully registered: {successful_registrations}/{processed_images} images")
    return successful_registrations

def create_dataset_csv(dataset_path: str, output_csv: str) -> bool:
    """
    Create a CSV file from a folder-based dataset
    
    Args:
        dataset_path: Path to the dataset folder
        output_csv: Path to output CSV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        data = []
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(root, file)
                    person_name = os.path.basename(root).replace('_', ' ').title()
                    data.append({'image_path': image_path, 'person_name': person_name})
        
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        
        logger.info(f"Created CSV with {len(data)} entries: {output_csv}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create CSV: {e}")
        return False

def get_dataset_statistics(dataset_path: str) -> Dict[str, Any]:
    """
    Get statistics about a dataset
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    if not os.path.exists(dataset_path):
        return {"error": "Dataset path does not exist"}
    
    stats = {
        "total_people": 0,
        "total_images": 0,
        "people_with_images": {},
        "image_formats": {},
        "total_size_mb": 0
    }
    
    for root, dirs, files in os.walk(dataset_path):
        if root == dataset_path:  # Skip root directory
            continue
            
        person_name = os.path.basename(root)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if image_files:
            stats["total_people"] += 1
            stats["people_with_images"][person_name] = len(image_files)
            stats["total_images"] += len(image_files)
            
            # Count image formats
            for file in image_files:
                ext = os.path.splitext(file)[1].lower()
                stats["image_formats"][ext] = stats["image_formats"].get(ext, 0) + 1
            
            # Calculate total size
            for file in image_files:
                file_path = os.path.join(root, file)
                try:
                    stats["total_size_mb"] += os.path.getsize(file_path) / (1024 * 1024)
                except:
                    pass
    
    return stats

if __name__ == "__main__":
    # Example usage
    print("External Dataset Training Module")
    print("This module provides functions to train Face ID system with external datasets")
    print("\nExample usage:")
    print("from external_dataset_trainer import train_from_folder_dataset")
    print("from main import FaceIDSystem")
    print("")
    print("face_id = FaceIDSystem()")
    print("success_count, total_count = train_from_folder_dataset('path/to/dataset', face_id)")
