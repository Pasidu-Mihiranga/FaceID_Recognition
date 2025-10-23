"""
Face ID System - Simple Windows Installation
Basic installation script without Unicode characters
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main installation process"""
    print("Face ID System - Windows Installation")
    print("=" * 50)
    
    # Check Python version
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"OK Python version: {version.major}.{version.minor}.{version.micro}")
    
    # Core packages that should work on Windows
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
    
    print("\nInstalling core dependencies...")
    failed_packages = []
    
    for package in core_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"OK {package} installed successfully")
        else:
            print(f"FAILED to install {package}")
            failed_packages.append(package)
    
    # Try TensorFlow
    print("\nInstalling TensorFlow...")
    tf_versions = [
        "tensorflow>=2.10.0,<2.16.0",
        "tensorflow-cpu>=2.10.0,<2.16.0"
    ]
    
    tf_success = False
    for tf_version in tf_versions:
        print(f"Trying {tf_version}...")
        if install_package(tf_version):
            print(f"OK TensorFlow installed successfully")
            tf_success = True
            break
    
    if not tf_success:
        print("WARNING: TensorFlow installation failed")
        print("You may need Visual C++ Redistributable")
    
    # Try PyTorch
    print("\nInstalling PyTorch...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "--index-url", 
            "https://download.pytorch.org/whl/cpu"
        ])
        print("OK PyTorch installed successfully")
        torch_success = True
    except subprocess.CalledProcessError:
        print("WARNING: PyTorch installation failed")
        torch_success = False
    
    # Test critical imports
    print("\nTesting imports...")
    critical_imports = [
        ("numpy", "import numpy as np"),
        ("opencv", "import cv2"),
        ("flask", "import flask"),
        ("sklearn", "from sklearn.cluster import DBSCAN")
    ]
    
    critical_success = 0
    for name, import_code in critical_imports:
        try:
            exec(import_code)
            print(f"OK {name} import successful")
            critical_success += 1
        except ImportError:
            print(f"FAILED {name} import")
    
    # Summary
    print("\n" + "=" * 50)
    print("Installation Summary:")
    print(f"Core dependencies: {len(core_packages) - len(failed_packages)}/{len(core_packages)} installed")
    print(f"TensorFlow: {'OK' if tf_success else 'FAILED'}")
    print(f"PyTorch: {'OK' if torch_success else 'FAILED'}")
    print(f"Critical imports: {critical_success}/{len(critical_imports)} successful")
    
    if critical_success == len(critical_imports):
        print("\nSUCCESS: Core installation completed!")
        print("You can now run the Face ID System with basic functionality.")
        print("\nNext steps:")
        print("1. Test the system: python test_system.py")
        print("2. Run examples: python examples.py")
        print("3. Start web interface: python face_id_system.py --web")
        return True
    else:
        print("\nERROR: Installation failed!")
        print("Please check the error messages above and try again.")
        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
