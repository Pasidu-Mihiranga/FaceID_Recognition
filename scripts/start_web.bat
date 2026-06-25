@echo off
echo Starting Face ID System with Python 3.13...
echo.

REM Activate virtual environment
call faceid_env\Scripts\activate.bat

REM Check Python version
echo Current Python version:
python --version
echo.

REM Test OpenCV
echo Testing OpenCV...
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
echo.

REM Start web interface
echo Starting web interface...
echo Access at: http://localhost:5000
echo Press Ctrl+C to stop
echo.
python face_id_system.py --web

pause
