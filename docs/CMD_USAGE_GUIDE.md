# 🖥️ **CMD Usage Guide for Face ID System**

## 🚀 **Quick Start with CMD:**

### **Method 1: Use Batch Files (EASIEST)**
```cmd
# Start Web Interface
start_web.bat

# Start Camera Recognition  
start_camera.bat
```

### **Method 2: Manual CMD Commands**
```cmd
# Open CMD and navigate to project
cd C:\Users\PMIHIR\Desktop\FaceID\FID

# Activate Python 3.13 virtual environment
faceid_env\Scripts\activate.bat

# Verify Python version
python --version

# Test OpenCV
python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Start Web Interface
python face_id_system.py --web

# OR Start Camera Recognition
python face_id_system.py --camera
```

## 🔧 **CMD vs PowerShell Differences:**

| Command | PowerShell | CMD |
|---------|------------|-----|
| **Activate Environment** | `faceid_env\Scripts\activate` | `faceid_env\Scripts\activate.bat` |
| **Chain Commands** | `cmd1 && cmd2` | `cmd1 & cmd2` |
| **File Search** | `Get-ChildItem` | `dir` |
| **Process Check** | `Get-Process` | `tasklist` |

## 🎯 **Why CMD Works Better:**

1. **✅ Simpler Syntax** - No complex PowerShell parsing
2. **✅ Better Compatibility** - Works with all Python scripts
3. **✅ Batch File Support** - Easy automation
4. **✅ No Execution Policy** - No PowerShell restrictions

## 🚨 **Common Issues Fixed:**

### **Issue 1: NumPy Compatibility**
```cmd
# Problem: Using Python 3.14 instead of 3.13
# Solution: Always activate virtual environment first
faceid_env\Scripts\activate.bat
```

### **Issue 2: PowerShell Syntax Errors**
```cmd
# Problem: && not supported in PowerShell
# Solution: Use CMD or separate commands
cmd1
cmd2
```

### **Issue 3: Path Issues**
```cmd
# Problem: Wrong working directory
# Solution: Always cd to project folder first
cd C:\Users\PMIHIR\Desktop\FaceID\FID
```

## 📋 **Complete CMD Workflow:**

### **Step 1: Open CMD**
- Press `Win + R`
- Type `cmd`
- Press Enter

### **Step 2: Navigate to Project**
```cmd
cd C:\Users\PMIHIR\Desktop\FaceID\FID
```

### **Step 3: Activate Environment**
```cmd
faceid_env\Scripts\activate.bat
```

### **Step 4: Verify Setup**
```cmd
python --version
python -c "import cv2; print('OpenCV working!')"
```

### **Step 5: Start System**
```cmd
# For Web Interface
python face_id_system.py --web

# For Camera Recognition
python face_id_system.py --camera

# For Testing
python test_system.py
```

## 🎉 **Success Indicators:**

- ✅ **Python Version:** `Python 3.13.9`
- ✅ **OpenCV:** `OpenCV version: 4.12.0`
- ✅ **Web Server:** `Running on http://localhost:5000`
- ✅ **Camera:** `Camera initialized successfully`

## 🔧 **Troubleshooting:**

### **If OpenCV Fails:**
```cmd
# Reinstall OpenCV in virtual environment
faceid_env\Scripts\activate.bat
pip uninstall opencv-python
pip install opencv-python==4.8.1.78
```

### **If TensorFlow Fails:**
```cmd
# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### **If Web Server Won't Start:**
```cmd
# Check if port 5000 is free
netstat -an | findstr :5000
```

## 🎯 **Your System is Ready!**

**Use CMD with the batch files for the easiest experience:**
- `start_web.bat` - Web interface
- `start_camera.bat` - Camera recognition

**Both batch files automatically:**
- ✅ Activate Python 3.13 environment
- ✅ Test system components
- ✅ Start the appropriate service
- ✅ Show helpful status messages
