@echo off
echo Starting Audio Watermark Lab...
echo.

echo Starting Backend Server (Flask)...
cd backend
start "Backend Server" cmd /k "python app.py"
cd ..

echo.
echo Starting Frontend Server (Vite)...
start "Frontend Server" cmd /k "npm run dev"

echo.
echo Both servers are starting...
echo Backend will be available at: http://localhost:5000
echo Frontend will be available at: http://localhost:8080
echo.
echo Please wait a moment for both servers to fully start.
echo.
pause 