@echo off
setlocal

echo ======================================
echo Starting Gold Price Predictor services
echo ======================================

REM Uncomment the next line if you use a Python virtual environment
REM call venv\Scripts\activate

echo.
echo Launching FastAPI backend...
start cmd /k "uvicorn backend.main:app --reload"

timeout /t 5 >nul

echo.
echo Launching React frontend...
cd frontend\react-app
start cmd /k "npm start"
cd ..\..

echo.
echo Backend listening on http://127.0.0.1:8000
echo Frontend available at http://localhost:3000
echo Press any key to close this launcher window.
pause >nul

