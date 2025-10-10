@echo off
echo ============================================
echo   Face Recognition Attendance System
echo ============================================
echo.
echo Activating virtual environment...
call .\venv_attendance\Scripts\activate.bat

echo.
echo Virtual environment activated!
echo Python version:
python --version
echo.
echo To run the application:
echo   python app.py
echo.
echo To deactivate the environment:
echo   deactivate
echo.
echo ============================================

cmd /k
