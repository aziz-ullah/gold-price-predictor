@echo off
echo Testing local Docker deployment...
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker first.
    echo Visit: https://docs.docker.com/desktop/install/windows-install/
    pause
    exit /b 1
)

echo ✅ Docker is installed
echo.
echo Building and starting services...
echo.

REM Build and start services
docker-compose up --build -d

if %errorlevel% neq 0 (
    echo ❌ Failed to start services. Check Docker logs.
    pause
    exit /b 1
)

echo ✅ Services started successfully!
echo.
echo Testing backend health...
timeout /t 10 /nobreak >nul

REM Test backend health
curl -s http://localhost:8000/ >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend is running at http://localhost:8000
) else (
    echo ⚠️  Backend might still be starting. Check logs with: docker-compose logs backend
)

echo.
echo Testing frontend...
curl -s http://localhost:3000/ >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Frontend is running at http://localhost:3000
) else (
    echo ⚠️  Frontend might still be starting. Check logs with: docker-compose logs frontend
)

echo.
echo Deployment test complete!
echo.
echo To stop services: docker-compose down
echo To view logs: docker-compose logs -f
echo.
pause