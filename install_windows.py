"""
Face ID System - Windows Installation Script
Handles installation of dependencies with Windows-specific considerations
"""

import subprocess
import sys
import os
import platform

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("X Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"OK Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_core_dependencies():
    """Install core dependencies that should work on Windows"""
    print("Installing core dependencies...")
    
    core_packages = [
        "numpy>=1.21.0,<1.25.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "scikit-learn>=1.0.0",
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "imutils>=0.5.0"
    ]
    
    failed_packages = []
    
    for package in core_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"OK {package} installed successfully")
        else:
            print(f"FAILED to install {package}")
            failed_packages.append(package)
    
    return failed_packages

def install_tensorflow():
    """Install TensorFlow with Windows compatibility"""
    print("Installing TensorFlow...")
    
    # Try different TensorFlow versions
    tf_versions = [
        "tensorflow>=2.10.0,<2.16.0",
        "tensorflow-cpu>=2.10.0,<2.16.0",
        "tensorflow==2.10.0"
    ]
    
    for tf_version in tf_versions:
        print(f"Trying {tf_version}...")
        if install_package(tf_version):
            print(f"OK TensorFlow installed successfully")
            return True
    
    print("FAILED TensorFlow installation")
    print("You may need to install Visual C++ Redistributable")
    print("Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    return False

def install_pytorch():
    """Install PyTorch with Windows compatibility"""
    print("Installing PyTorch...")
    
    # Try CPU-only version first (more compatible)
    pytorch_packages = [
        "torch torchvision --index-url https://download.pytorch.org/whl/cpu",
        "torch>=1.12.0 torchvision>=0.13.0 --index-url https://download.pytorch.org/whl/cpu",
        "torch>=1.12.0 torchvision>=0.13.0"
    ]
    
    for package in pytorch_packages:
        print(f"Trying PyTorch installation...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + package.split())
            print("✓ PyTorch installed successfully")
            return True
        except subprocess.CalledProcessError:
            continue
    
    print("✗ PyTorch installation failed")
    return False

def install_face_recognition_libraries():
    """Install face recognition libraries (optional)"""
    print("Installing face recognition libraries...")
    
    face_libs = [
        ("mtcnn", "mtcnn>=0.1.0"),
        ("deepface", "deepface>=0.0.70"),
        ("face-recognition", "face-recognition>=1.3.0")
    ]
    
    installed = []
    
    for lib_name, package in face_libs:
        print(f"Installing {lib_name}...")
        if install_package(package):
            print(f"✓ {lib_name} installed successfully")
            installed.append(lib_name)
        else:
            print(f"⚠ {lib_name} installation failed (optional)")
    
    return installed

def install_dlib():
    """Install dlib (can be tricky on Windows)"""
    print("Installing dlib...")
    
    # Try different approaches for dlib
    dlib_attempts = [
        "dlib>=19.22.0",
        "dlib==19.24.2",
        "dlib"
    ]
    
    for attempt in dlib_attempts:
        print(f"Trying dlib installation...")
        if install_package(attempt):
            print("✓ dlib installed successfully")
            return True
    
    print("⚠ dlib installation failed")
    print("You may need to install CMake and Visual Studio Build Tools")
    print("Or use conda: conda install -c conda-forge dlib")
    return False

def test_imports():
    """Test if critical imports work"""
    print("Testing imports...")
    
    critical_imports = [
        ("numpy", "import numpy as np"),
        ("opencv", "import cv2"),
        ("flask", "import flask"),
        ("sklearn", "from sklearn.cluster import DBSCAN")
    ]
    
    optional_imports = [
        ("tensorflow", "import tensorflow as tf"),
        ("torch", "import torch"),
        ("mtcnn", "import mtcnn"),
        ("deepface", "import deepface"),
        ("dlib", "import dlib")
    ]
    
    critical_success = 0
    optional_success = 0
    
    for name, import_code in critical_imports:
        try:
            exec(import_code)
            print(f"✓ {name} import successful")
            critical_success += 1
        except ImportError:
            print(f"✗ {name} import failed")
    
    for name, import_code in optional_imports:
        try:
            exec(import_code)
            print(f"✓ {name} import successful")
            optional_success += 1
        except ImportError:
            print(f"⚠ {name} import failed (optional)")
    
    return critical_success == len(critical_imports), optional_success

def main():
    """Main installation process"""
    print("Face ID System - Windows Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check if we're on Windows
    if platform.system() != "Windows":
        print("This script is optimized for Windows. Proceeding anyway...")
    
    # Install dependencies
    print("\n1. Installing core dependencies...")
    failed_core = install_core_dependencies()
    
    print("\n2. Installing TensorFlow...")
    tf_success = install_tensorflow()
    
    print("\n3. Installing PyTorch...")
    torch_success = install_pytorch()
    
    print("\n4. Installing face recognition libraries...")
    face_libs = install_face_recognition_libraries()
    
    print("\n5. Installing dlib...")
    dlib_success = install_dlib()
    
    print("\n6. Testing imports...")
    critical_ok, optional_count = test_imports()
    
    # Summary
    print("\n" + "=" * 50)
    print("Installation Summary:")
    print(f"Core dependencies: {'✓' if not failed_core else '✗'}")
    print(f"TensorFlow: {'✓' if tf_success else '✗'}")
    print(f"PyTorch: {'✓' if torch_success else '✗'}")
    print(f"Face recognition libs: {len(face_libs)}/{4} installed")
    print(f"Dlib: {'✓' if dlib_success else '✗'}")
    print(f"Critical imports: {'✓' if critical_ok else '✗'}")
    print(f"Optional imports: {optional_count}/5 successful")
    
    if critical_ok:
        print("\n🎉 Core installation successful!")
        print("You can now run the Face ID System with basic functionality.")
        
        if optional_count >= 3:
            print("✓ Most optional libraries installed - full functionality available!")
        else:
            print("⚠ Some optional libraries missing - limited functionality.")
            print("You can install them later as needed.")
        
        print("\nNext steps:")
        print("1. Test the system: python test_system.py")
        print("2. Run examples: python examples.py")
        print("3. Start web interface: python face_id_system.py --web")
        
        return True
    else:
        print("\n❌ Installation failed!")
        print("Please check the error messages above and try again.")
        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
