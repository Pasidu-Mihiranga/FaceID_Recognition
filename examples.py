"""
Face ID System - Example Usage Script
Demonstrates how to use the Face ID System for various tasks
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from main import FaceIDSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_images():
    """Create sample images for testing"""
    logger.info("Creating sample images...")
    
    # Create sample directory
    sample_dir = Path("data/sample_images")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample face images (colored rectangles as placeholders)
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    
    for i, (color, name) in enumerate(zip(colors, names)):
        # Create a simple face-like image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:] = color
        
        # Add some face-like features
        cv2.circle(img, (80, 70), 10, (255, 255, 255), -1)  # Left eye
        cv2.circle(img, (120, 70), 10, (255, 255, 255), -1)  # Right eye
        cv2.circle(img, (100, 100), 5, (255, 255, 255), -1)  # Nose
        cv2.ellipse(img, (100, 130), (20, 10), 0, 0, 180, (255, 255, 255), 2)  # Mouth
        
        # Save image
        img_path = sample_dir / f"{name.lower()}.jpg"
        cv2.imwrite(str(img_path), img)
        logger.info(f"Created sample image: {img_path}")
    
    return sample_dir

def example_basic_usage():
    """Basic usage example"""
    logger.info("=== Basic Usage Example ===")
    
    # Initialize the system
    face_id = FaceIDSystem(
        detector_type='opencv',  # Use OpenCV for better compatibility
        recognition_model='arcface',
        recognition_threshold=0.6
    )
    
    # Create sample images
    sample_dir = create_sample_images()
    
    # Register persons
    logger.info("Registering persons...")
    for img_path in sample_dir.glob("*.jpg"):
        person_name = img_path.stem.capitalize()
        success = face_id.register_person(str(img_path), person_name)
        logger.info(f"Registered {person_name}: {'Success' if success else 'Failed'}")
    
    # Test recognition
    logger.info("Testing recognition...")
    for img_path in sample_dir.glob("*.jpg"):
        person_name = img_path.stem.capitalize()
        image = cv2.imread(str(img_path))
        
        recognized_name, confidence, face_info = face_id.recognize_face(image)
        
        logger.info(f"Image: {person_name}")
        logger.info(f"Recognized: {recognized_name}")
        logger.info(f"Confidence: {confidence:.3f}")
        logger.info("-" * 40)
    
    # Get system statistics
    stats = face_id.get_system_stats()
    logger.info("System Statistics:")
    logger.info(f"Total Persons: {stats['database_stats']['total_persons']}")
    logger.info(f"Total Images: {stats['database_stats']['total_images']}")
    
    # Cleanup
    face_id.cleanup_system()
    
    return face_id

def example_continuous_learning():
    """Continuous learning example"""
    logger.info("=== Continuous Learning Example ===")
    
    face_id = FaceIDSystem(
        detector_type='opencv',
        recognition_model='arcface',
        recognition_threshold=0.6
    )
    
    # Register a person
    sample_dir = Path("data/sample_images")
    if sample_dir.exists():
        alice_img = sample_dir / "alice.jpg"
        if alice_img.exists():
            face_id.register_person(str(alice_img), "Alice")
            logger.info("Registered Alice")
    
    # Simulate multiple recognitions (continuous learning)
    logger.info("Simulating continuous learning...")
    for i in range(5):
        # Create a slightly different image of Alice
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:] = (255, 0, 0)  # Red background
        
        # Add some variation
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add face features
        cv2.circle(img, (80, 70), 10, (255, 255, 255), -1)
        cv2.circle(img, (120, 70), 10, (255, 255, 255), -1)
        cv2.circle(img, (100, 100), 5, (255, 255, 255), -1)
        cv2.ellipse(img, (100, 130), (20, 10), 0, 0, 180, (255, 255, 255), 2)
        
        # Recognize
        recognized_name, confidence, face_info = face_id.recognize_face(img)
        
        logger.info(f"Recognition {i+1}: {recognized_name} (confidence: {confidence:.3f})")
        
        # Process for continuous learning
        if recognized_name and confidence > 0.7:
            face_id.learning_manager.process_recognition_result(img, recognized_name, confidence)
    
    # Check learning statistics
    learning_stats = face_id.learning_manager.get_learning_stats()
    logger.info("Learning Statistics:")
    logger.info(f"Total Updates: {learning_stats['total_updates']}")
    logger.info(f"Successful Updates: {learning_stats['successful_updates']}")
    
    face_id.cleanup_system()

def example_camera_integration():
    """Camera integration example"""
    logger.info("=== Camera Integration Example ===")
    
    face_id = FaceIDSystem(
        detector_type='opencv',
        recognition_model='arcface',
        recognition_threshold=0.6
    )
    
    # Register some persons first
    sample_dir = Path("data/sample_images")
    if sample_dir.exists():
        for img_path in sample_dir.glob("*.jpg"):
            person_name = img_path.stem.capitalize()
            face_id.register_person(str(img_path), person_name)
    
    logger.info("Starting camera recognition...")
    logger.info("Press 'q' in the camera window to quit")
    
    # Start camera recognition
    success = face_id.start_camera_recognition()
    if success:
        try:
            import time
            while face_id.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping camera recognition...")
            face_id.stop_camera_recognition()
    else:
        logger.error("Failed to start camera recognition")
    
    face_id.cleanup_system()

def example_web_interface():
    """Web interface example"""
    logger.info("=== Web Interface Example ===")
    
    face_id = FaceIDSystem(
        detector_type='opencv',
        recognition_model='arcface',
        recognition_threshold=0.6
    )
    
    # Register some sample persons
    sample_dir = Path("data/sample_images")
    if sample_dir.exists():
        for img_path in sample_dir.glob("*.jpg"):
            person_name = img_path.stem.capitalize()
            face_id.register_person(str(img_path), person_name)
    
    # Start web interface
    from src.web_interface import create_web_interface
    
    logger.info("Starting web interface on http://localhost:5000")
    web_interface = create_web_interface(face_id)
    
    try:
        web_interface.run(debug=False)
    except KeyboardInterrupt:
        logger.info("Stopping web interface...")
    
    face_id.cleanup_system()

def example_database_operations():
    """Database operations example"""
    logger.info("=== Database Operations Example ===")
    
    face_id = FaceIDSystem()
    
    # Get all persons
    persons = face_id.database.get_all_persons()
    logger.info(f"Total persons in database: {len(persons)}")
    
    for person in persons:
        logger.info(f"Person: {person['name']}")
        logger.info(f"  ID: {person['id']}")
        logger.info(f"  Images: {person['total_images']}")
        logger.info(f"  Created: {person['created_at']}")
        logger.info(f"  Last seen: {person['last_seen']}")
        
        # Get person's images
        images = face_id.database.get_person_images(person['id'])
        logger.info(f"  Image details: {len(images)} images")
        
        # Get person's embeddings
        embeddings = face_id.database.get_person_embeddings(person['id'])
        logger.info(f"  Embeddings: {len(embeddings)} vectors")
        logger.info("-" * 40)
    
    # Get recognition statistics
    stats = face_id.database.get_recognition_stats(days=7)
    logger.info("Recognition Statistics (7 days):")
    logger.info(f"  Total recognitions: {stats.get('total_recognitions', 0)}")
    logger.info(f"  Successful recognitions: {stats.get('successful_recognitions', 0)}")
    logger.info(f"  Unknown faces: {stats.get('unknown_faces', 0)}")
    logger.info(f"  Success rate: {stats.get('success_rate', 0) * 100:.1f}%")
    
    face_id.cleanup_system()

def main():
    """Run all examples"""
    print("Face ID System - Example Usage")
    print("=" * 50)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Continuous Learning", example_continuous_learning),
        ("Database Operations", example_database_operations),
        # ("Camera Integration", example_camera_integration),  # Commented out as it requires camera
        # ("Web Interface", example_web_interface),  # Commented out as it starts a server
    ]
    
    for name, example_func in examples:
        try:
            print(f"\nRunning {name} Example...")
            example_func()
            print(f"{name} Example completed successfully!")
        except Exception as e:
            print(f"{name} Example failed: {e}")
            logger.error(f"{name} example failed: {e}")
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    main()
