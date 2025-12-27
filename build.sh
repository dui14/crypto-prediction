set -e

echo "Installing Python dependencies..."
pip install -r requirements.txt gdown

echo "Downloading models from Google Drive..."
mkdir -p models
gdown "https://drive.google.com/uc?id=16KuTw-8UOTBbS_isgixSIzd8k8UEDcgu" -O models.zip --quiet
echo "Downloaded models.zip"

echo "Extracting models..."
unzip -q models.zip -d models/
rm models.zip

echo "✅ All models ready!"
echo "Starting API server..."