#!/usr/bin/env python3
"""
Compatible Installation Script for Face ID System
Handles Python 3.14 compatibility issues
"""

import sys
import subprocess
import os
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Installing: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"SUCCESS: {description}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check Python version and provide recommendations"""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 14:
        print("WARNING: Python 3.14+ detected!")
        print("Many ML libraries don't have pre-built wheels for Python 3.14+")
        print("Consider using Python 3.11 or 3.12 for better compatibility")
        return False
    return True

def install_basic_dependencies():
    """Install basic dependencies that work with Python 3.14"""
    basic_packages = [
        "flask>=2.0.0",
        "flask-cors>=3.0.0", 
        "pillow>=8.0.0",
        "requests>=2.27.1",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.30.0",
        "gdown>=3.10.1"
    ]
    
    print("\nInstalling basic dependencies...")
    for package in basic_packages:
        success = run_command(f"pip install {package}", package)
        if not success:
            print(f"Failed to install {package}, continuing...")

def install_opencv_alternative():
    """Try alternative OpenCV installation methods"""
    print("\nTrying alternative OpenCV installation methods...")
    
    # Method 1: Try opencv-python-headless (lighter version)
    success = run_command("pip install opencv-python-headless", "OpenCV Headless")
    if success:
        return True
    
    # Method 2: Try specific version
    success = run_command("pip install opencv-python==4.8.1.78", "OpenCV Specific Version")
    if success:
        return True
    
    # Method 3: Try from conda-forge if conda is available
    try:
        subprocess.run("conda --version", check=True, capture_output=True)
        print("Conda detected, trying conda installation...")
        success = run_command("conda install -c conda-forge opencv", "OpenCV via Conda")
        if success:
            return True
    except:
        pass
    
    print("All OpenCV installation methods failed")
    return False

def install_tensorflow_alternative():
    """Try alternative TensorFlow installation methods"""
    print("\nTrying alternative TensorFlow installation methods...")
    
    # Method 1: Try TensorFlow CPU only
    success = run_command("pip install tensorflow-cpu", "TensorFlow CPU")
    if success:
        return True
    
    # Method 2: Try specific version
    success = run_command("pip install tensorflow==2.13.0", "TensorFlow 2.13.0")
    if success:
        return True
    
    # Method 3: Try nightly build
    success = run_command("pip install tf-nightly", "TensorFlow Nightly")
    if success:
        return True
    
    print("All TensorFlow installation methods failed")
    return False

def create_compatible_requirements():
    """Create a compatible requirements file"""
    compatible_reqs = """# Compatible requirements for Python 3.14
# Basic dependencies (should work)
flask>=2.0.0
flask-cors>=3.0.0
pillow>=8.0.0
requests>=2.27.1
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.30.0
gdown>=3.10.1

# Optional dependencies (may fail on Python 3.14)
# Uncomment these if you have Visual Studio Build Tools installed
# opencv-python>=4.5.0
# tensorflow>=2.10.0
# deepface>=0.0.70

# Alternative lightweight options
opencv-python-headless>=4.5.0
tensorflow-cpu>=2.10.0
"""
    
    with open("requirements_compatible.txt", "w") as f:
        f.write(compatible_reqs)
    
    print("Created requirements_compatible.txt with compatible packages")

def install_visual_studio_tools():
    """Provide instructions for installing Visual Studio Build Tools"""
    print("\n" + "="*60)
    print("VISUAL STUDIO BUILD TOOLS REQUIRED")
    print("="*60)
    print("To install packages that require compilation (like OpenCV, TensorFlow):")
    print("1. Download Visual Studio Build Tools from:")
    print("   https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022")
    print("2. Install with 'C++ build tools' workload")
    print("3. Restart your terminal/command prompt")
    print("4. Run this script again")
    print("="*60)

def main():
    """Main installation process"""
    print("Face ID System - Compatible Installation Script")
    print("="*60)
    
    # Check Python version
    is_compatible = check_python_version()
    
    # Install basic dependencies
    install_basic_dependencies()
    
    # Try OpenCV alternatives
    opencv_success = install_opencv_alternative()
    
    # Try TensorFlow alternatives
    tf_success = install_tensorflow_alternative()
    
    # Create compatible requirements file
    create_compatible_requirements()
    
    # Summary
    print("\n" + "="*60)
    print("INSTALLATION SUMMARY")
    print("="*60)
    print(f"Python Version: {sys.version}")
    print(f"OpenCV Installation: {'SUCCESS' if opencv_success else 'FAILED'}")
    print(f"TensorFlow Installation: {'SUCCESS' if tf_success else 'FAILED'}")
    
    if not opencv_success or not tf_success:
        print("\nRECOMMENDATIONS:")
        print("1. Use the minimal system: python minimal_face_id.py")
        print("2. Install Visual Studio Build Tools for full functionality")
        print("3. Consider using Python 3.11 or 3.12 for better compatibility")
        print("4. Use the compatible requirements: pip install -r requirements_compatible.txt")
        
        if not opencv_success:
            install_visual_studio_tools()
    else:
        print("\nSUCCESS: All dependencies installed!")
        print("You can now use the full Face ID System")

if __name__ == "__main__":
    main()
