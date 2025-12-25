# Crypto Prediction API

FastAPI backend server dự đoán giá crypto từ 5 mô hình ML đã train.

**API Version**: 2.2.0

## Kiến trúc Model

```
models/
├── lightgbm/
│   ├── lightgbm_model_gpu.txt           # Model file
│   └── feature_names.json               # 60 features
│
├── xgboost/
│   ├── xgboost_model_gpu_full.ubj       # Model file (binary)
│   ├── xgboost_model_gpu_full.pkl       # Backup (pickle)
│   └── feature_names_gpu_full.json      # 220 features
│
├── catboost/
│   ├── catboost_model_gpu_full.cbm      # Model file
│   ├── cat_features_gpu_full.json       # Categorical feature indices
│   └── feature_names_gpu_full.json      # 220 features
│
├── lstm/
│   ├── lstm_model_v2_best.h5            # Keras model (v2)
│   ├── lstm_scaler.pkl                  # StandardScaler (33 features)
│   ├── lstm_lookback.pkl                # Lookback window (60)
│   └── lstm_model.h5                    # Fallback (old model)
│
└── gru/
    ├── gru_model_v2_best.h5             # Keras model (v2)
    ├── gru_scaler.pkl                   # StandardScaler (33 features)
    ├── gru_lookback.pkl                 # Lookback window (60)
    └── gru_model_gpu_full.h5            # Fallback (old model)
```

## Installation

```bash
cd src/api
pip install -r requirements.txt
```

**Requirements**:
- Python 3.9+
- FastAPI 0.104+
- LightGBM, XGBoost, CatBoost
- TensorFlow 2.15+ (cho LSTM/GRU)
- scikit-learn, pandas, numpy

## Running the Server

### Windows
```bash
run_api.bat
```

### Linux/macOS
```bash
chmod +x run_api.sh
./run_api.sh
```

### Manual (All platforms)
```bash
python prediction_api.py
```

Server sẽ chạy tại: **http://localhost:8000**

API Documentation (Swagger): **http://localhost:8000/docs**

## API Endpoints

### Health & Status

#### `GET /`
Health check cơ bản
```json
{
  "message": "Crypto Prediction API",
  "version": "2.2.0",
  "models_dir": "...",
  "models": ["LightGBM", "XGBoost", "CatBoost", "LSTM", "GRU"]
}
```

#### `GET /health`
Health check với timestamp
```json
{
  "status": "healthy",
  "timestamp": "2025-12-25T10:30:00.000000"
}
```

#### `GET /models`
Liệt kê các model có sẵn
```json
{
  "available_models": ["LightGBM", "XGBoost", "CatBoost", "LSTM", "GRU"]
}
```

### Prediction Endpoints

#### `POST /predict` - Single Prediction
Dự đoán 1 giá trị cho candle cuối cùng

**Request body**:
```json
{
  "candles": [
    {
      "timestamp": 1700000000000,
      "open": 37000.0,
      "high": 37100.0,
      "low": 36900.0,
      "close": 37050.0,
      "volume": 1000.0,
      "trades": 500,
      "vwap": 37020.0,
      "turnover": 37050000.0
    }
  ],
  "model_type": "LightGBM"
}
```

**Response**:
```json
{
  "prediction": 37150.25,
  "model_type": "LightGBM",
  "timestamp": 1700000000000,
  "features_used": 60
}
```

**Requirements**:
- Tối thiểu 60 candles
- `model_type`: LightGBM, XGBoost, CatBoost, LSTM, hoặc GRU

---

#### `POST /predict/batch` - Batch Predictions
Dự đoán lô cho dữ liệu lịch sử (historical backtest)

**Request body**:
```json
{
  "candles": [...],
  "model_type": "XGBoost",
  "count": 50
}
```

**Response**:
```json
{
  "predictions": [37100.5, 37150.2, 37080.1, ...],
  "model_type": "XGBoost",
  "count": 50
}
```

**Lưu ý**:
- Chỉ hoạt động với tree models (LightGBM, XGBoost, CatBoost)
- LSTM/GRU tạm thời disabled (Keras version issue)
- Trả lại `count` số dự đoán

---

#### `POST /predict/future` - Multi-step Future Prediction
Dự đoán đa bước cho các horizon khác nhau (5m, 15m, 1h, 4h, 1d, 3d)

**Request body**:
```json
{
  "candles": [...],
  "model_type": "LightGBM",
  "horizon": "1d",
  "interval": "5m"
}
```

**Response**:
```json
{
  "predictions": [
    {"timestamp": 1700000300000, "price": 37050.5, "step": 1},
    {"timestamp": 1700000600000, "price": 37080.2, "step": 2},
    ...
  ],
  "model_type": "LightGBM",
  "horizon": "1d",
  "current_price": 37050.0,
  "target_timestamp": 1700086400000
}
```

**Parameters**:
- `horizon`: 5m, 15m, 1h, 4h, 1d, hoặc 3d
- `interval`: 5m (mặc định) hoặc 15m (matching data interval)

## Features & Feature Calculation

API tự động tính toán các features từ candle data.

### Tree Models (LightGBM - 60 features / XGBoost, CatBoost - 220 features)

#### LightGBM Features (60)

| Category | Count | Details |
|----------|-------|---------|
| **Basic** | 9 | open, high, low, close, volume, trades, vwap, turnover, returns, high_low, close_open |
| **Moving Averages** | 12 | SMA_5/10/20/50, STD_5/10/20/50, EMA_5/10/20/50 |
| **RSI** | 3 | rsi_5, rsi_10, rsi_20 |
| **MACD** | 3 | macd, macd_signal, macd_diff |
| **Bollinger Bands** | 5 | bb_middle, bb_std, bb_upper, bb_lower, bb_width |
| **Volume** | 4 | volume_sma_10, volume_sma_20, volume_ratio, vwap_ratio |
| **Lag Features** | 18 | close_lag_1-20, volume_lag_1-20, returns_lag_1-20 |
| **Time** | 4 | hour, day_of_week, day_of_month, month |
| **Total** | **60** | |

#### XGBoost/CatBoost Features (220)

Ngoài tất cả 60 features của LightGBM, thêm:

| Category | Count | Details |
|----------|-------|---------|
| **Extended MAs** | 12 | WMA, thêm periods (100, 200) |
| **Advanced Volatility** | 10 | ATR, extended STD/VAR |
| **Extended RSI** | 2 | rsi_14, rsi_30 |
| **Bollinger Extended** | 18 | BB cho 3 periods (10,20,30) với upper/lower/width/position |
| **Stochastic** | 8 | stoch_k/d cho 4 periods (5,10,14,20) |
| **Williams %R** | 3 | williams_r_10/14/20 |
| **CCI** | 3 | cci_10/20/30 |
| **ROC & Momentum** | 8 | roc & momentum cho 4 periods |
| **Extended Volume** | 12 | Volume SMA/EMA/ratios/std cho periods extended |
| **VWAP Extended** | 4 | vwap_ratio, vwap_diff, vwap_sma_10/20 |
| **OBV** | 3 | obv, obv_sma_10, obv_sma_20 |
| **MFI** | 3 | mfi_10, mfi_14, mfi_20 |
| **Extended Lags** | 27 | OHLC lags (9 periods), volume/trades lags (6), returns lags |
| **Rolling Stats** | 24 | Max, Min, Mean, Median, Skew, Kurtosis (4 periods) |
| **Time Features Extended** | 11 | Thêm cyclical encoding (sin/cos), is_month_start/end, etc |
| **Total** | **220** | |

### Deep Learning Models (LSTM/GRU - 33 features)

**Đặc điểm**: 
- Features là **returns-based** (scale-independent)
- Input shape: **(60, 33)** - 60 timesteps × 33 features
- Output: **Predicted return (%)**, không phải price
- Final price = current_price × (1 + predicted_return)

#### 33 Returns-based Features

| Category | Count | Details |
|----------|-------|---------|
| **Returns** | 7 | returns, returns_2/3/5/10/20, log_returns |
| **Volatility** | 3 | volatility_5/10/20 |
| **RSI** | 3 | rsi_7/14/21 |
| **MACD %** | 3 | macd_pct, macd_signal_pct, macd_hist_pct |
| **Bollinger** | 2 | bb_position, bb_width_pct |
| **Stochastic** | 2 | stoch_k, stoch_d |
| **Volume** | 2 | volume_ratio, volume_change |
| **Candle** | 5 | range_pct, body_pct, candle_position, upper_shadow_pct, lower_shadow_pct |
| **Momentum** | 3 | momentum_5/10/20 |
| **Price Position** | 3 | price_sma10/20/50_ratio |
| **Total** | **33** | |

## Important Notes

### Data Requirements
- **Minimum candles**: 60 (để đủ tính features + sequence cho LSTM/GRU)
- **Recommended**: 120+ (cho batch prediction, rolling stats)
- **Binance API rate limit**: 1200 requests/minute

### Model-specific

**Tree Models (LightGBM, XGBoost, CatBoost)**:
- Dự đoán giá **tuyệt đối** (absolute price)
- Áp dụng price correction để chuẩn hóa

**LSTM/GRU v2**:
- Dự đoán **return (%)**, không phải price
- Final price = current_price × (1 + predicted_return)
- Sử dụng 60 timesteps lookback
- Features được scale bằng StandardScaler

### Price Correction Method
- **Method**: "offset" - tính offset trung bình từ calibration window
- Giúp điều chỉnh dự đoán model để match giá current

### WebSocket vs HTTP
- HTTP: `/predict`, `/predict/batch`, `/predict/future`
- WebSocket: (tương lai) real-time streaming predictions
