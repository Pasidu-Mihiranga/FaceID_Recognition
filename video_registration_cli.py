#!/usr/bin/env python3
"""
Video Registration Command Line Interface
Simple interface for testing video registration
"""

import sys
import os
from main import FaceIDSystem
from video_registration import VideoRegistrationSystem

def main():
    """Main function for video registration CLI"""
    print("🎥 Video Registration System")
    print("=" * 50)
    
    # Initialize Face ID System
    print("Initializing Face ID System...")
    face_id = FaceIDSystem(
        detector_type='opencv',
        recognition_model='arcface',
        recognition_threshold=0.7
    )
    print("✅ Face ID System initialized")
    
    # Initialize Video Registration System
    print("Initializing Video Registration System...")
    video_reg = VideoRegistrationSystem(face_id)
    print("✅ Video Registration System initialized")
    
    while True:
        print("\n" + "=" * 50)
        print("Video Registration Menu:")
        print("1. Start Video Registration")
        print("2. Process Video File")
        print("3. View Registered Persons")
        print("4. Test Recognition")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Start video registration
            person_name = input("Enter person name: ").strip()
            if person_name:
                print(f"\n🎥 Starting video registration for {person_name}")
                print("Follow the on-screen instructions...")
                
                success = video_reg.start_video_registration(person_name)
                
                if success:
                    print(f"✅ Video registration successful for {person_name}")
                else:
                    print(f"❌ Video registration failed for {person_name}")
            else:
                print("❌ Person name is required")
        
        elif choice == '2':
            # Process video file
            video_path = input("Enter video file path: ").strip()
            person_name = input("Enter person name: ").strip()
            
            if video_path and person_name:
                if os.path.exists(video_path):
                    print(f"\n🎥 Processing video file: {video_path}")
                    success = video_reg.process_video_file(video_path, person_name)
                    
                    if success:
                        print(f"✅ Video processing successful for {person_name}")
                    else:
                        print(f"❌ Video processing failed for {person_name}")
                else:
                    print(f"❌ Video file not found: {video_path}")
            else:
                print("❌ Both video path and person name are required")
        
        elif choice == '3':
            # View registered persons
            print("\n📋 Registered Persons:")
            stats = face_id.get_system_stats()
            
            if stats and 'database_stats' in stats:
                persons = stats['database_stats'].get('persons', [])
                if persons:
                    for person in persons:
                        print(f"  • {person['name']}: {person['images']} images")
                else:
                    print("  No persons registered yet")
            else:
                print("  Error loading registered persons")
        
        elif choice == '4':
            # Test recognition
            print("\n🔍 Testing Recognition:")
            print("Starting camera for recognition test...")
            print("Press 'q' to quit recognition")
            
            success = face_id.start_camera_recognition()
            if success:
                print("✅ Recognition test started")
            else:
                print("❌ Failed to start recognition test")
        
        elif choice == '5':
            # Exit
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")
    
    # Cleanup
    video_reg.cleanup()
    face_id.cleanup_system()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
