@echo off
echo ===================================================
echo     Starting FinAgent Multi-Agent System...
echo ===================================================

echo [1/2] Starting FastAPI Backend on port 8000...
start cmd /k "title FastAPI Backend && uvicorn main:app --host 127.0.0.1 --port 8000"

echo [2/2] Starting Streamlit UI...
timeout /t 3 /nobreak >nul
start cmd /k "title Streamlit Frontend && streamlit run streamlit_app.py"

echo.
echo Both services have been started in separate windows!
echo Close those windows to stop the application.
