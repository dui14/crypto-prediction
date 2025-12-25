@echo off
echo Starting Crypto Prediction API...
echo.

cd /d %~dp0

REM Activate conda environment if exists
call conda activate crypto 2>nul

REM Install requirements if needed
pip install -r requirements.txt -q

REM Start the server
echo Starting server at http://localhost:8000
echo API documentation at http://localhost:8000/docs
echo.
python prediction_api.py
