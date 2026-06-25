@echo off
echo Starting Face ID System Camera Recognition...
echo.

REM Activate virtual environment
call faceid_env\Scripts\activate.bat

REM Check Python version
echo Current Python version:
python --version
echo.

REM Test system
echo Testing system...
python src/tools/test_system.py
echo.

REM Start camera recognition
echo Starting camera recognition...
echo Press Ctrl+C to stop
echo.
python main.py --camera

pause
