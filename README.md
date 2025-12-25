# Crypto Price Prediction Dashboard

Dashboard dự đoán giá tiền điện tử sử dụng 5 mô hình Machine Learning (LightGBM, XGBoost, CatBoost, LSTM, GRU) với giao diện React hiện đại.

## Features

### 📊 Dữ liệu & Lấy dữ liệu
- **Real-time Data**: Lấy giá crypto live từ Binance API
- **Flexible Intervals**: Hỗ trợ 5-phút hoặc 15-phút timeframe
- **Historical Data**: Lấy dữ liệu lịch sử từ Binance (theo ngày/3 ngày)
- **WebSocket Support**: Cập nhật giá real-time tối ưu

### 🤖 Machine Learning Models
- **LightGBM**: 60 features (gradient boosting)
- **XGBoost**: 220 features (extreme gradient boosting)
- **CatBoost**: 220 features (categorical boosting)
- **LSTM**: 33 returns-based features (deep learning)
- **GRU**: 33 returns-based features (deep learning)

### 📈 Dự đoán & Trực quan hóa
- **Interactive Charts**: Đường giá thực (xanh) vs dự đoán (đỏ)
- **Multi-step Prediction**: Dự đoán 5m, 15m, 1h, 4h, 1d, 3d
- **Performance Metrics**: MAE, RMSE, direction accuracy
- **Trend Detection**: Phát hiện xu hướng Up/Down/Neutral
- **Batch Predictions**: Dự đoán lô cho dữ liệu lịch sử

### 🎨 Giao diện
- **Responsive Design**: UI hiện đại với Tailwind CSS
- **Language Support**: Hỗ trợ Tiếng Việt & English
- **Real-time Status**: Chỉ báo kết nối API
- **Configurable**: Chọn symbol, interval, model, horizon từ UI

## Tech Stack

- **Frontend**: React 18 + TypeScript
- **Backend API**: FastAPI + Python
- **Charts**: Recharts
- **Styling**: Tailwind CSS + PostCSS
- **HTTP**: Axios
- **ML Libraries**: LightGBM, XGBoost, CatBoost, TensorFlow/Keras
- **Date Handling**: date-fns

## Installation

### 1️⃣ Chuẩn bị môi trường

**Yêu cầu**:
- Python 3.9+
- Node.js 16+
- Git

### 2️⃣ Clone & Cài đặt

```bash
# Clone repository
git clone https://github.com/dui14/crypto-prediction.git

# Cài đặt Python dependencies
pip install -r requirements.txt

# Cài đặt Frontend dependencies
cd crypto-dashboard
npm install
```

### 3️⃣ Khởi động

**Terminal 1 - Backend API**:
```bash
cd src/api
python prediction_api.py
# hoặc Windows: run_api.bat
# hoặc Linux/Mac: ./run_api.sh
```
Backend chạy tại: `http://localhost:8000`

**Terminal 2 - Frontend React**:
```bash
cd crypto-dashboard
npm start
```
Frontend chạy tại: `http://localhost:3000`

## Sử dụng

### Giao diện Chính

1. **Chọn cấu hình**:
   - Symbol: BTC/USDT, ETH/USDT, v.v
   - Interval: 5m hoặc 15m
   - Model: LightGBM, XGBoost, CatBoost, LSTM, GRU
   - Horizon: 5m, 15m, 1h, 4h, 1d, 3d

2. **Bắt đầu Dự đoán**:
   - Nhấn "Start Prediction"
   - Lấy dữ liệu lịch sử từ Binance
   - Hiển thị chart với giá thực (xanh) + dự đoán (đỏ)

3. **Giám sát Hiệu suất**:
   - MAE: Sai số tuyệt đối trung bình
   - RMSE: Root Mean Squared Error
   - Direction Accuracy: % dự đoán chiều đúng
   - Trend: Xu hướng phát hiện (Up/Down/Neutral)

4. **Dừng & Quay lại**:
   - Click "Stop Prediction" để dừng
   - Click "Back to Realtime" quay về live mode

### API Endpoints

Xem chi tiết tại [src/api/README.md](src/api/README.md)

**Endpoints chính**:
- `GET /health` - Health check
- `GET /models` - Liệt kê các model có sẵn
- `POST /predict` - Dự đoán 1 candle
- `POST /predict/batch` - Dự đoán lô (historical)
- `POST /predict/future` - Dự đoán đa bước (horizon 5m-3d)

Swagger docs: `http://localhost:8000/docs`

## Project Structure

```
crypto/
├── src/
│   └── api/                    # Backend FastAPI
│       ├── prediction_api.py   # API server chính
│       ├── requirements.txt    # Python dependencies
│       ├── run_api.bat/sh      # Scripts khởi động
│       └── README.md           # API documentation
│
├── crypto-dashboard/           # Frontend React
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.tsx       # Container chính
│   │   │   ├── ControlPanel.tsx    # Selectors (symbol/interval/model)
│   │   │   ├── PriceChart.tsx      # Chart hiển thị
│   │   │   └── MetricsPanel.tsx    # Performance metrics
│   │   ├── services/
│   │   │   ├── binanceService.ts   # Binance API + WebSocket
│   │   │   ├── predictionService.ts # Gọi backend API
│   │   │   └── utils.ts            # Helper functions
│   │   ├── hooks/
│   │   │   └── usePrediction.ts    # Main prediction logic
│   │   ├── contexts/
│   │   │   └── LanguageContext.tsx # i18n (VI/EN)
│   │   ├── types/                  # TypeScript interfaces
│   │   └── App.tsx
│   ├── package.json
│   ├── tailwind.config.js      # Styling config
│   └── tsconfig.json           # TS config
│
├── models/                     # Trained model files
│   ├── lightgbm/
│   │   ├── lightgbm_model_gpu.txt      (60 features)
│   │   └── feature_names.json
│   ├── xgboost/
│   │   ├── xgboost_model_gpu_full.ubj  (220 features)
│   │   └── feature_names_gpu_full.json
│   ├── catboost/
│   │   ├── catboost_model_gpu_full.cbm (220 features)
│   │   └── feature_names_gpu_full.json
│   ├── lstm/
│   │   ├── lstm_model_v2_best.h5       (33 returns-based features)
│   │   ├── lstm_scaler.pkl
│   │   └── lstm_lookback.pkl
│   └── gru/
│       ├── gru_model_v2_best.h5        (33 returns-based features)
│       ├── gru_scaler.pkl
│       └── gru_lookback.pkl
│
├── data/                       # Data files
│   ├── csv/                    # Historical CSV data
│   ├── historical/             # Backup historical
│   └── live/                   # Live data cache
│
├── train/                      # Training scripts
│   └── train_lstm_v2.py
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## ML Models Chi tiết

### 1. LightGBM (60 features)
- **Type**: Gradient Boosting
- **Features**: 60 (price, MA, volatility, RSI, MACD, BB, volume, lags, time)
- **Output**: Predicted price (absolute)
- **Use case**: Nhanh, nhẹ, tốt cho real-time

### 2. XGBoost (220 features)
- **Type**: Extreme Gradient Boosting
- **Features**: 220 (comprehensive: OHLC, MAs, ATR, Stoch, MFI, OBV, lags, rolling stats)
- **Output**: Predicted price (absolute)
- **Use case**: Chính xác cao, yêu cầu tính toán lớn

### 3. CatBoost (220 features)
- **Type**: Categorical Boosting
- **Features**: 220 (giống XGBoost)
- **Output**: Predicted price (absolute)
- **Use case**: Xử lý categorical features tốt

### 4. LSTM (33 returns-based features, lookback=60)
- **Type**: Deep Learning (LSTM 128→64→32 layers)
- **Features**: 33 (returns-based, scale-independent)
- **Input**: (60, 33) - 60 timesteps x 33 features
- **Output**: Predicted return (%), không phải price
- **Use case**: Học long-term dependencies, trend

### 5. GRU (33 returns-based features, lookback=60)
- **Type**: Deep Learning (GRU 128→64→32 layers)
- **Features**: 33 (returns-based, scale-independent)
- **Input**: (60, 33) - 60 timesteps x 33 features
- **Output**: Predicted return (%), không phải price
- **Use case**: Nhanh hơn LSTM, tương tự hiệu suất

## Features Calculated

API tự động tính toán từ candle data:

### Tree Models (LightGBM/XGBoost/CatBoost)
- **Basic** (8): OHLCV, returns, spreads, price_range
- **Moving Averages** (24): SMA, EMA, WMA (periods: 5,10,15,20,30,50,100,200)
- **Volatility** (15): STD, VAR, ATR (periods: 5,10,20,30,50)
- **Technical Indicators** (63): RSI, MACD, Bollinger Bands, Stochastic, Williams %R, CCI, ROC, Momentum
- **Volume** (18): Volume SMA/EMA, ratios, VWAP, OBV, MFI
- **Lag Features** (54): Close, volume, returns (various lags)
- **Rolling Statistics** (24): Max, Min, Mean, Median, Skew, Kurtosis
- **Time Features** (17): Hour, day, month + cyclical encoding

### Deep Learning (LSTM/GRU)
- **Returns** (7): returns, returns_2/3/5/10/20, log_returns
- **Volatility** (3): volatility_5/10/20
- **RSI** (3): rsi_7/14/21
- **MACD** (3): macd_pct, signal, histogram (as % of price)
- **Bollinger** (2): bb_position, bb_width_pct
- **Stochastic** (2): stoch_k, stoch_d
- **Volume** (2): volume_ratio, volume_change
- **Candle** (5): range_pct, body_pct, candle_position, upper/lower_shadow_pct
- **Momentum** (3): momentum_5/10/20
- **Price Position** (3): price_sma10/20/50_ratio

## Metrics Explanation

- **MAE (Mean Absolute Error)**: Sai số tuyệt đối trung bình giữa giá dự đoán vs thực
  - `MAE = mean(|predicted - actual|)`

- **RMSE (Root Mean Squared Error)**: Căn bậc 2 của bình phương sai số trung bình
  - `RMSE = sqrt(mean((predicted - actual)^2))`
  - Phạt penalties outliers nặng hơn MAE

- **Direction Accuracy**: % lần dự đoán chiều giá đúng (Up/Down)
  - `% times: sign(predicted_return) == sign(actual_return)`

- **Trend**: Phát hiện xu hướng từ predicted prices
  - **Up**: Giá dự đoán tăng
  - **Down**: Giá dự đoán giảm
  - **Neutral**: Giá dự đoán ổn định

## Configuration

### Supported Cryptocurrencies
Bất kỳ token nào trên Binance, ví dụ:
- BTCUSDT, ETHUSDT, BNBUSDT (major)
- ADAUSDT, SOLUSDT, DOTUSDT, LINKUSDT, MATICUSDT (altcoins)
- Hoặc token custom khác

### Time Intervals
- **5m**: 5-minute candles (120 candles = 10 hours history)
- **15m**: 15-minute candles (120 candles = 30 hours history)

### Models
- LightGBM, XGBoost, CatBoost, LSTM, GRU

### Prediction Horizons
- **5m**: Dự đoán 5 phút sau
- **15m**: Dự đoán 15 phút sau
- **1h**: Dự đoán 1 giờ sau
- **4h**: Dự đoán 4 giờ sau
- **1d**: Dự đoán 1 ngày sau
- **3d**: Dự đoán 3 ngày sau

## Troubleshooting

### Backend không khởi động
```bash
# Check Python version
python --version  # Phải >= 3.9

# Check dependencies
pip list | findstr "fastapi lightgbm xgboost"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Frontend lỗi kết nối API
```bash
# Kiểm tra API đang chạy
curl http://localhost:8000/health

# Kiểm tra port 8000 available
netstat -ano | findstr ":8000"  # Windows
lsof -i :8000  # Mac/Linux
```

### Model không load
```bash
# Check model files exist
ls models/lightgbm/
ls models/xgboost/
ls models/catboost/
ls models/lstm/
ls models/gru/
```

### Binance API rate limit
- API tự động throttle nếu quá 1200 requests/minute
- Hạn chế fetch dữ liệu, dùng WebSocket cho real-time
