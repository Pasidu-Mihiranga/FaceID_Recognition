"""
Face ID System - Complete Implementation
Advanced face recognition system with continuous learning capabilities

This system integrates:
- Multiple face detection models (MTCNN, RetinaFace, OpenCV, Dlib)
- Multiple face recognition models (ArcFace, FaceNet, VGG-Face)
- SQLite database for face data management
- Continuous learning from new encounters
- Real-time camera integration
- Web interface for easy management

Usage:
    python main.py                    # Run the main system
    python main.py --web              # Start with web interface
    python main.py --camera           # Start camera recognition
    python main.py --register         # Register a person
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from main import FaceIDSystem
from src.web_interface import create_web_interface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_id_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the Face ID System"""
    parser = argparse.ArgumentParser(description='Face ID System - Advanced Face Recognition')
    parser.add_argument('--detector', default='opencv', 
                       choices=['mtcnn', 'retinaface', 'opencv', 'dlib'],
                       help='Face detector type (default: opencv)')
    parser.add_argument('--model', default='arcface',
                       choices=['arcface', 'facenet', 'vggface'],
                       help='Face recognition model (default: arcface)')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Recognition threshold (default: 0.6)')
    parser.add_argument('--web', action='store_true',
                       help='Start web interface')
    parser.add_argument('--camera', action='store_true',
                       help='Start camera recognition')
    parser.add_argument('--register', type=str,
                       help='Register a person with image path and name')
    parser.add_argument('--port', type=int, default=5000,
                       help='Web interface port (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Web interface host (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting Face ID System...")
        logger.info(f"Detector: {args.detector}, Model: {args.model}, Threshold: {args.threshold}")
        
        # Initialize the Face ID System
        face_id = FaceIDSystem(
            detector_type=args.detector,
            recognition_model=args.model,
            recognition_threshold=args.threshold
        )
        
        # Handle different modes
        if args.web:
            # Start web interface
            logger.info(f"Starting web interface on {args.host}:{args.port}")
            web_interface = create_web_interface(face_id, args.host, args.port)
            web_interface.run(debug=False)
            
        elif args.camera:
            # Start camera recognition
            logger.info("Starting camera recognition...")
            success = face_id.start_camera_recognition()
            if success:
                logger.info("Camera recognition started. Press 'q' to quit.")
                try:
                    import time
                    while face_id.is_running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Stopping camera recognition...")
                    face_id.stop_camera_recognition()
            else:
                logger.error("Failed to start camera recognition")
                
        elif args.register:
            # Register a person
            parts = args.register.split(',')
            if len(parts) != 2:
                logger.error("Invalid register format. Use: --register 'image_path,name'")
                return
            
            image_path, person_name = parts[0].strip(), parts[1].strip()
            
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return
            
            logger.info(f"Registering person: {person_name}")
            success = face_id.register_person(image_path, person_name)
            
            if success:
                logger.info(f"Successfully registered {person_name}")
            else:
                logger.error(f"Failed to register {person_name}")
                
        else:
            # Interactive mode
            interactive_mode(face_id)
            
    except Exception as e:
        logger.error(f"Face ID System failed: {e}")
        sys.exit(1)
    finally:
        if 'face_id' in locals():
            face_id.cleanup_system()

def interactive_mode(face_id):
    """Interactive command-line mode"""
    print("\n" + "="*60)
    print("Face ID System - Interactive Mode")
    print("="*60)
    print("Available commands:")
    print("1. register <image_path> <person_name> - Register a new person")
    print("2. recognize <image_path> - Recognize a face in an image")
    print("3. camera - Start camera recognition")
    print("4. stats - Show system statistics")
    print("5. web - Start web interface")
    print("6. quit - Exit the system")
    print("="*60)
    
    while True:
        try:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'quit' or command == 'exit':
                break
            elif command.startswith('register '):
                parts = command.split(' ', 2)
                if len(parts) == 3:
                    image_path, person_name = parts[1], parts[2]
                    if os.path.exists(image_path):
                        success = face_id.register_person(image_path, person_name)
                        print(f"Registration {'successful' if success else 'failed'}")
                    else:
                        print(f"Image file not found: {image_path}")
                else:
                    print("Usage: register <image_path> <person_name>")
                    
            elif command.startswith('recognize '):
                parts = command.split(' ', 1)
                if len(parts) == 2:
                    image_path = parts[1]
                    if os.path.exists(image_path):
                        import cv2
                        image = cv2.imread(image_path)
                        person_name, confidence, face_info = face_id.recognize_face(image)
                        
                        if person_name:
                            print(f"Recognized: {person_name} (confidence: {confidence:.3f})")
                        else:
                            print(f"Unknown face (confidence: {confidence:.3f})")
                    else:
                        print(f"Image file not found: {image_path}")
                else:
                    print("Usage: recognize <image_path>")
                    
            elif command == 'camera':
                print("Starting camera recognition...")
                success = face_id.start_camera_recognition()
                if success:
                    print("Camera recognition started. Press 'q' in the camera window to quit.")
                    try:
                        import time
                        while face_id.is_running:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("Stopping camera recognition...")
                        face_id.stop_camera_recognition()
                else:
                    print("Failed to start camera recognition")
                    
            elif command == 'stats':
                stats = face_id.get_system_stats()
                print("\nSystem Statistics:")
                print("-" * 40)
                
                if stats.get('database_stats'):
                    db_stats = stats['database_stats']
                    print(f"Registered Persons: {db_stats.get('total_persons', 0)}")
                    print(f"Total Images: {db_stats.get('total_images', 0)}")
                
                if stats.get('recent_stats'):
                    recent_stats = stats['recent_stats']
                    print(f"Recognitions (7 days): {recent_stats.get('total_recognitions', 0)}")
                    print(f"Success Rate: {recent_stats.get('success_rate', 0) * 100:.1f}%")
                
                if stats.get('learning_stats'):
                    learning_stats = stats['learning_stats']
                    print(f"Learning Updates: {learning_stats.get('total_updates', 0)}")
                    print(f"Learning Success Rate: {learning_stats.get('success_rate', 0) * 100:.1f}%")
                    
            elif command == 'web':
                print("Starting web interface...")
                web_interface = create_web_interface(face_id)
                web_interface.run(debug=False)
                
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
