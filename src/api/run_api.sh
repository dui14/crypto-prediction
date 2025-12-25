#!/bin/bash
echo "Starting Crypto Prediction API..."
echo ""

cd "$(dirname "$0")"

# Activate conda environment if exists
source activate crypto 2>/dev/null || true

# Install requirements if needed
pip install -r requirements.txt -q

# Start the server
echo "Starting server at http://localhost:8000"
echo "API documentation at http://localhost:8000/docs"
echo ""
python prediction_api.py
