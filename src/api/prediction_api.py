"""
FastAPI Backend for Crypto Price Prediction
Serves predictions from trained models: LightGBM, XGBoost, CatBoost
LSTM/GRU temporarily disabled due to Keras version incompatibility
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Crypto Prediction API", version="2.1.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

print(f"📁 Base directory: {BASE_DIR}")
print(f"📁 Models directory: {MODELS_DIR}")

# Global cache
models_cache: Dict[str, Any] = {}
feature_names_cache: Dict[str, List[str]] = {}

# Price correction: Models were trained on historical data with different price levels
# Method: "relative" - use predicted change relative to input prices
#         "offset" - adjust by average offset between predicted and actual in the batch
PRICE_CORRECTION_METHOD = "offset"

# Horizon configs (in minutes)
HORIZON_CONFIG = {
    "5m": 5,           # 5 minutes
    "15m": 15,         # 15 minutes  
    "1h": 60,          # 60 minutes = 1 hour
    "4h": 4 * 60,      # 240 minutes = 4 hours
    "1d": 24 * 60,     # 1440 minutes = 1 day
    "3d": 3 * 24 * 60  # 4320 minutes = 3 days
}


class CandleData(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: Optional[int] = 0
    vwap: Optional[float] = None
    turnover: Optional[float] = None


class PredictionRequest(BaseModel):
    candles: List[CandleData]
    model_type: str


class PredictionResponse(BaseModel):
    prediction: float
    model_type: str
    timestamp: int
    features_used: int


class BatchPredictionRequest(BaseModel):
    candles: List[CandleData]
    model_type: str
    count: int = 50


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    model_type: str
    count: int


class FuturePredictionRequest(BaseModel):
    """Request for multi-step future prediction"""
    candles: List[CandleData]
    model_type: str
    horizon: str  # "4h", "1d", "3d"
    interval: str = "5m"  # Data interval (5m or 15m)


class FuturePredictionResponse(BaseModel):
    """Response for future predictions"""
    predictions: List[Dict[str, Any]]  # [{timestamp, price, step}, ...]
    model_type: str
    horizon: str
    current_price: float
    target_timestamp: int


class RealtimePredictionRequest(BaseModel):
    """Request for realtime trendline prediction"""
    candles: List[CandleData]
    model_type: str
    steps: int = 10  # Number of future steps to predict


class RealtimePredictionResponse(BaseModel):
    """Response for realtime trendline"""
    current_price: float
    predictions: List[Dict[str, Any]]  # [{timestamp, price}, ...]
    model_type: str
    trend: str  # "up", "down", "neutral"
    confidence: float


# ==================== EXACT FEATURE LISTS ====================

# LightGBM: 60 features
LIGHTGBM_FEATURES = [
    'open', 'high', 'low', 'close', 'trades', 'volume', 'vwap', 'turnover', 'returns', 'high_low', 'close_open',
    'sma_5', 'std_5', 'ema_5', 'sma_10', 'std_10', 'ema_10', 'sma_20', 'std_20', 'ema_20', 'sma_50', 'std_50', 'ema_50',
    'rsi_5', 'rsi_10', 'rsi_20', 'macd', 'macd_signal', 'macd_diff',
    'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'bb_width',
    'volume_sma_10', 'volume_sma_20', 'volume_ratio', 'vwap_ratio',
    'close_lag_1', 'volume_lag_1', 'returns_lag_1',
    'close_lag_2', 'volume_lag_2', 'returns_lag_2',
    'close_lag_3', 'volume_lag_3', 'returns_lag_3',
    'close_lag_5', 'volume_lag_5', 'returns_lag_5',
    'close_lag_10', 'volume_lag_10', 'returns_lag_10',
    'close_lag_20', 'volume_lag_20', 'returns_lag_20',
    'hour', 'day_of_week', 'day_of_month', 'month'
]

# LSTM/GRU v2: 33 returns-based features (scale-independent)
LSTM_GRU_V2_FEATURES = [
    # Returns (7)
    'returns', 'returns_2', 'returns_3', 'returns_5', 'returns_10', 'returns_20', 'log_returns',
    # Volatility (3)
    'volatility_5', 'volatility_10', 'volatility_20',
    # RSI (3)
    'rsi_7', 'rsi_14', 'rsi_21',
    # MACD (3)
    'macd_pct', 'macd_signal_pct', 'macd_hist_pct',
    # Bollinger (2)
    'bb_position', 'bb_width_pct',
    # Stochastic (2)
    'stoch_k', 'stoch_d',
    # Volume (2)
    'volume_ratio', 'volume_change',
    # Candle (5)
    'range_pct', 'body_pct', 'candle_position', 'upper_shadow_pct', 'lower_shadow_pct',
    # Momentum (3)
    'momentum_5', 'momentum_10', 'momentum_20',
    # Price position (3)
    'price_sma10_ratio', 'price_sma20_ratio', 'price_sma50_ratio'
]

# XGBoost/CatBoost: 220 features (loaded from json)


# ==================== MODEL LOADERS ====================

def load_lightgbm():
    """Load LightGBM model"""
    import lightgbm as lgb
    
    if "LightGBM" in models_cache:
        return models_cache["LightGBM"], LIGHTGBM_FEATURES
    
    model_path = os.path.join(MODELS_DIR, "lightgbm", "lightgbm_model_gpu.txt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LightGBM model not found at {model_path}")
    
    model = lgb.Booster(model_file=model_path)
    models_cache["LightGBM"] = model
    
    print(f"✅ LightGBM loaded: {len(LIGHTGBM_FEATURES)} features")
    return model, LIGHTGBM_FEATURES


def load_xgboost():
    """Load XGBoost model"""
    import xgboost as xgb
    
    if "XGBoost" in models_cache:
        return models_cache["XGBoost"], feature_names_cache["XGBoost"]
    
    # Load feature names first
    feature_path = os.path.join(MODELS_DIR, "xgboost", "feature_names_gpu_full.json")
    with open(feature_path, 'r') as f:
        feature_names = json.load(f)
    
    # Try pkl first
    pkl_path = os.path.join(MODELS_DIR, "xgboost", "xgboost_model_gpu_full.pkl")
    ubj_path = os.path.join(MODELS_DIR, "xgboost", "xgboost_model_gpu_full.ubj")
    
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
    elif os.path.exists(ubj_path):
        model = xgb.Booster()
        model.load_model(ubj_path)
    else:
        raise FileNotFoundError("XGBoost model not found")
    
    models_cache["XGBoost"] = model
    feature_names_cache["XGBoost"] = feature_names
    
    print(f"✅ XGBoost loaded: {len(feature_names)} features")
    return model, feature_names


def load_catboost():
    """Load CatBoost model"""
    from catboost import CatBoostRegressor
    
    if "CatBoost" in models_cache:
        return models_cache["CatBoost"], feature_names_cache["CatBoost"]
    
    model_path = os.path.join(MODELS_DIR, "catboost", "catboost_model_gpu_full.cbm")
    feature_path = os.path.join(MODELS_DIR, "catboost", "feature_names_gpu_full.json")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CatBoost model not found at {model_path}")
    
    model = CatBoostRegressor()
    model.load_model(model_path)
    
    with open(feature_path, 'r') as f:
        feature_names = json.load(f)
    
    models_cache["CatBoost"] = model
    feature_names_cache["CatBoost"] = feature_names
    
    print(f"✅ CatBoost loaded: {len(feature_names)} features")
    return model, feature_names


def load_lstm():
    """Load LSTM v2 model with scaler and lookback
    
    LSTM v2 uses 33 returns-based features and predicts returns (not prices).
    Input shape: (batch, 60, 33)
    Output: predicted return for next candle
    """
    if "LSTM" in models_cache:
        return models_cache["LSTM"], models_cache["LSTM_scaler"], models_cache["LSTM_lookback"]
    
    # LSTM v2 model paths
    model_path = os.path.join(MODELS_DIR, "lstm", "lstm_model_v2_best.h5")
    scaler_path = os.path.join(MODELS_DIR, "lstm", "lstm_scaler.pkl")
    lookback_path = os.path.join(MODELS_DIR, "lstm", "lstm_lookback.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model not found at {model_path}")
    
    # Build model architecture manually (compatible with Keras 3.x)
    # LSTM v2 architecture: LSTM(128)->Dropout->BN->LSTM(64)->Dropout->BN->LSTM(32)->Dropout->Dense(16)->Dense(1)
    try:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
        
        model = Sequential([
            Input(shape=(60, 33)),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        # Load weights from h5 file
        model.load_weights(model_path)
        print(f"✅ LSTM model loaded with weights from {model_path}")
    except Exception as e:
        print(f"⚠️ LSTM load failed: {e}")
        raise FileNotFoundError(f"Cannot load LSTM model: {e}")
    
    # Load scaler (trained on 33 features)
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = None
        print("⚠️ LSTM scaler not found, will use MinMax scaling")
    
    # Load lookback
    if os.path.exists(lookback_path):
        with open(lookback_path, 'rb') as f:
            lookback = pickle.load(f)
    else:
        lookback = 60
    
    models_cache["LSTM"] = model
    models_cache["LSTM_scaler"] = scaler
    models_cache["LSTM_lookback"] = lookback
    
    print(f"✅ LSTM v2 loaded: lookback={lookback}, features=33 (returns-based)")
    return model, scaler, lookback


def load_gru():
    """Load GRU v2 model with scaler and lookback
    
    GRU v2 uses 33 returns-based features (same as LSTM v2) and predicts returns.
    Input shape: (batch, 60, 33)
    Output: predicted return for next candle
    """
    if "GRU" in models_cache:
        return models_cache["GRU"], models_cache["GRU_scaler"], models_cache["GRU_lookback"]
    
    # GRU v2 model paths
    model_path = os.path.join(MODELS_DIR, "gru", "gru_model_v2_best.h5")
    scaler_path = os.path.join(MODELS_DIR, "gru", "gru_scaler.pkl")  # v2 scaler
    lookback_path = os.path.join(MODELS_DIR, "gru", "gru_lookback.pkl")  # v2 lookback
    
    # Fallback to old scaler paths if v2 not found
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(MODELS_DIR, "gru", "scaler_gpu_full.pkl")
    if not os.path.exists(lookback_path):
        lookback_path = os.path.join(MODELS_DIR, "gru", "lookback_gpu_full.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GRU model not found at {model_path}")
    
    # Build model architecture manually (compatible with Keras 3.x)
    # GRU v2 architecture (same structure as LSTM v2)
    try:
        from keras.models import Sequential
        from keras.layers import GRU, Dense, Dropout, BatchNormalization, Input
        
        model = Sequential([
            Input(shape=(60, 33)),
            GRU(128, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        # Load weights from h5 file
        model.load_weights(model_path)
        print(f"✅ GRU model loaded with weights from {model_path}")
    except Exception as e:
        print(f"⚠️ GRU load failed: {e}")
        raise FileNotFoundError(f"Cannot load GRU model: {e}")
    
    # Load scaler (trained on 33 features)
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = None
        print("⚠️ GRU scaler not found, will use MinMax scaling")
    
    # Load lookback
    if os.path.exists(lookback_path):
        with open(lookback_path, 'rb') as f:
            lookback = pickle.load(f)
    else:
        lookback = 60
    
    models_cache["GRU"] = model
    models_cache["GRU_scaler"] = scaler
    models_cache["GRU_lookback"] = lookback
    
    print(f"✅ GRU v2 loaded: lookback={lookback}, features=33 (returns-based)")
    return model, scaler, lookback


# ==================== FEATURE CALCULATION ====================

def safe_div(a, b, fill=0):
    """Safe division avoiding inf/nan"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(b != 0, a / b, fill)
        result = np.where(np.isfinite(result), result, fill)
    return result


def calculate_lstm_gru_v2_features(candles: List[CandleData]) -> pd.DataFrame:
    """
    Calculate 33 returns-based features for LSTM/GRU v2 models.
    These features are scale-independent (percentages/ratios).
    Returns DataFrame with all candles (for sequence creation).
    """
    df = pd.DataFrame([c.model_dump() for c in candles])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # ===== Returns (7 features) =====
    df['returns'] = df['close'].pct_change().fillna(0)
    df['returns_2'] = df['close'].pct_change(2).fillna(0)
    df['returns_3'] = df['close'].pct_change(3).fillna(0)
    df['returns_5'] = df['close'].pct_change(5).fillna(0)
    df['returns_10'] = df['close'].pct_change(10).fillna(0)
    df['returns_20'] = df['close'].pct_change(20).fillna(0)
    
    # Log returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1).fillna(df['close'])).fillna(0)
    
    # ===== Volatility (3 features) =====
    df['volatility_5'] = df['returns'].rolling(5, min_periods=1).std().fillna(0)
    df['volatility_10'] = df['returns'].rolling(10, min_periods=1).std().fillna(0)
    df['volatility_20'] = df['returns'].rolling(20, min_periods=1).std().fillna(0)
    
    # ===== RSI (3 features) =====
    for period in [7, 14, 21]:
        delta = df['close'].diff().fillna(0)
        gain = delta.where(delta > 0, 0).rolling(period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()
        rs = safe_div(gain.values, loss.values, fill=1)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # ===== MACD as percentage of price (3 features) =====
    ema12 = df['close'].ewm(span=12, min_periods=1).mean()
    ema26 = df['close'].ewm(span=26, min_periods=1).mean()
    df['macd_pct'] = safe_div((ema12 - ema26).values, df['close'].values, fill=0)
    df['macd_signal_pct'] = df['macd_pct'].ewm(span=9, min_periods=1).mean()
    df['macd_hist_pct'] = df['macd_pct'] - df['macd_signal_pct']
    
    # ===== Bollinger Band (2 features) =====
    sma20 = df['close'].rolling(20, min_periods=1).mean()
    std20 = df['close'].rolling(20, min_periods=1).std().fillna(0)
    bb_lower = sma20 - 2 * std20
    bb_range = 4 * std20
    df['bb_position'] = safe_div((df['close'] - bb_lower).values, bb_range.values, fill=0.5)
    df['bb_width_pct'] = safe_div(bb_range.values, sma20.values, fill=0)
    
    # ===== Stochastic (2 features) =====
    low14 = df['low'].rolling(14, min_periods=1).min()
    high14 = df['high'].rolling(14, min_periods=1).max()
    df['stoch_k'] = 100 * safe_div((df['close'] - low14).values, (high14 - low14).values, fill=50)
    df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=1).mean()
    
    # ===== Volume (2 features) =====
    vol_sma20 = df['volume'].rolling(20, min_periods=1).mean()
    df['volume_ratio'] = safe_div(df['volume'].values, vol_sma20.values, fill=1)
    df['volume_change'] = df['volume'].pct_change().fillna(0)
    
    # ===== Candle features (5 features) =====
    df['range_pct'] = safe_div((df['high'] - df['low']).values, df['close'].values, fill=0)
    df['body_pct'] = safe_div(np.abs(df['close'] - df['open']).values, df['close'].values, fill=0)
    df['candle_position'] = safe_div((df['close'] - df['low']).values, (df['high'] - df['low']).values, fill=0.5)
    df['upper_shadow_pct'] = safe_div(
        (df['high'] - df[['open', 'close']].max(axis=1)).values,
        df['close'].values,
        fill=0
    )
    df['lower_shadow_pct'] = safe_div(
        (df[['open', 'close']].min(axis=1) - df['low']).values,
        df['close'].values,
        fill=0
    )
    
    # ===== Momentum (3 features) =====
    df['momentum_5'] = df['close'] / df['close'].shift(5).fillna(df['close']) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10).fillna(df['close']) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20).fillna(df['close']) - 1
    
    # ===== Price position relative to MAs (3 features) =====
    df['price_sma10_ratio'] = df['close'] / df['close'].rolling(10, min_periods=1).mean() - 1
    df['price_sma20_ratio'] = df['close'] / df['close'].rolling(20, min_periods=1).mean() - 1
    df['price_sma50_ratio'] = df['close'] / df['close'].rolling(50, min_periods=1).mean() - 1
    
    # Fill any remaining NaN
    df = df.fillna(0)
    
    # Clip extreme values
    for col in LSTM_GRU_V2_FEATURES:
        if col in df.columns:
            df[col] = np.clip(df[col], -10, 10)
    
    return df


def calculate_lightgbm_features(candles: List[CandleData]) -> np.ndarray:
    """
    Calculate exactly 60 features for LightGBM in correct order.
    """
    df = pd.DataFrame([c.model_dump() for c in candles])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Fill missing vwap/turnover
    if df['vwap'].isna().all() or (df['vwap'] == 0).all():
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
    if df['turnover'].isna().all() or (df['turnover'] == 0).all():
        df['turnover'] = df['close'] * df['volume']
    
    # Basic features
    df['returns'] = df['close'].pct_change().fillna(0)
    df['high_low'] = df['high'] - df['low']
    df['close_open'] = df['close'] - df['open']
    
    # Moving averages (5, 10, 20, 50)
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
        df[f'std_{period}'] = df['close'].rolling(window=period, min_periods=1).std().fillna(0)
        df[f'ema_{period}'] = df['close'].ewm(span=period, min_periods=1).mean()
    
    # RSI (5, 10, 20)
    for period in [5, 10, 20]:
        delta = df['close'].diff().fillna(0)
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = safe_div(gain.values, loss.values, fill=1)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12, min_periods=1).mean()
    ema26 = df['close'].ewm(span=26, min_periods=1).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands (20-period)
    sma20 = df['close'].rolling(window=20, min_periods=1).mean()
    std20 = df['close'].rolling(window=20, min_periods=1).std().fillna(0)
    df['bb_middle'] = sma20
    df['bb_std'] = std20
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_width'] = safe_div((df['bb_upper'] - df['bb_lower']).values, sma20.values, fill=0)
    
    # Volume features
    df['volume_sma_10'] = df['volume'].rolling(window=10, min_periods=1).mean()
    df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_ratio'] = safe_div(df['volume'].values, df['volume_sma_10'].values, fill=1)
    df['vwap_ratio'] = safe_div(df['close'].values, df['vwap'].values, fill=1)
    
    # Lag features
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag).fillna(method='bfill')
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag).fillna(method='bfill')
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag).fillna(0)
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    
    # Extract last row with exact feature order
    result = np.array([df[f].iloc[-1] for f in LIGHTGBM_FEATURES], dtype=np.float64)
    
    # Replace any remaining nan/inf
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result.reshape(1, -1)


def calculate_full_features(candles: List[CandleData], feature_names: List[str]) -> np.ndarray:
    """
    Calculate 220 features for XGBoost/CatBoost in correct order.
    """
    df = pd.DataFrame([c.model_dump() for c in candles])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Fill missing
    if df['vwap'].isna().all() or (df['vwap'] == 0).all():
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
    if df['turnover'].isna().all() or (df['turnover'] == 0).all():
        df['turnover'] = df['close'] * df['volume']
    if df['trades'].isna().all():
        df['trades'] = 0
    
    # === Basic features (16) ===
    df['returns'] = df['close'].pct_change().fillna(0)
    df['log_returns'] = np.log(safe_div(df['close'].values, df['close'].shift(1).fillna(df['close']).values, fill=1))
    df['high_low'] = df['high'] - df['low']
    df['close_open'] = df['close'] - df['open']
    df['high_close'] = df['high'] - df['close']
    df['low_close'] = df['low'] - df['close']
    df['price_range'] = safe_div((df['high'] - df['low']).values, df['close'].values, fill=0)
    df['price_change'] = df['close'].diff().fillna(0)
    
    # === Moving Averages (8 periods x 3 types = 24) ===
    for period in [5, 10, 15, 20, 30, 50, 100, 200]:
        p = min(period, len(df))
        df[f'sma_{period}'] = df['close'].rolling(window=p, min_periods=1).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=p, min_periods=1).mean()
        # WMA
        def wma(x):
            weights = np.arange(1, len(x) + 1)
            return np.dot(x, weights) / weights.sum()
        df[f'wma_{period}'] = df['close'].rolling(window=p, min_periods=1).apply(wma, raw=True)
    
    # === Volatility (5 periods x 3 types = 15) ===
    for period in [5, 10, 20, 30, 50]:
        p = min(period, len(df))
        df[f'std_{period}'] = df['close'].rolling(window=p, min_periods=1).std().fillna(0)
        df[f'var_{period}'] = df['close'].rolling(window=p, min_periods=1).var().fillna(0)
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1).fillna(df['close']))
        low_close = abs(df['low'] - df['close'].shift(1).fillna(df['close']))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{period}'] = tr.rolling(window=p, min_periods=1).mean()
    
    # === RSI (5 periods) ===
    for period in [5, 10, 14, 20, 30]:
        p = min(period, len(df))
        delta = df['close'].diff().fillna(0)
        gain = delta.where(delta > 0, 0).rolling(window=p, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=p, min_periods=1).mean()
        rs = safe_div(gain.values, loss.values, fill=1)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # === MACD (4 features) ===
    ema12 = df['close'].ewm(span=12, min_periods=1).mean()
    ema26 = df['close'].ewm(span=26, min_periods=1).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['macd_histogram'] = df['macd_diff']
    
    # === Bollinger Bands (3 periods x 6 = 18) ===
    for period in [10, 20, 30]:
        p = min(period, len(df))
        sma = df['close'].rolling(window=p, min_periods=1).mean()
        std = df['close'].rolling(window=p, min_periods=1).std().fillna(0)
        df[f'bb_middle_{period}'] = sma
        df[f'bb_std_{period}'] = std
        df[f'bb_upper_{period}'] = sma + 2 * std
        df[f'bb_lower_{period}'] = sma - 2 * std
        width = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
        df[f'bb_width_{period}'] = safe_div(width.values, sma.values, fill=0)
        df[f'bb_position_{period}'] = safe_div(
            (df['close'] - df[f'bb_lower_{period}']).values,
            (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']).values,
            fill=0.5
        )
    
    # === Stochastic (4 periods x 2 = 8) ===
    for period in [5, 10, 14, 20]:
        p = min(period, len(df))
        low_min = df['low'].rolling(window=p, min_periods=1).min()
        high_max = df['high'].rolling(window=p, min_periods=1).max()
        df[f'stoch_k_{period}'] = 100 * safe_div(
            (df['close'] - low_min).values,
            (high_max - low_min).values,
            fill=50
        )
        df[f'stoch_d_{period}'] = pd.Series(df[f'stoch_k_{period}']).rolling(window=3, min_periods=1).mean()
    
    # === Williams %R (3 periods) ===
    for period in [10, 14, 20]:
        p = min(period, len(df))
        high_max = df['high'].rolling(window=p, min_periods=1).max()
        low_min = df['low'].rolling(window=p, min_periods=1).min()
        df[f'williams_r_{period}'] = -100 * safe_div(
            (high_max - df['close']).values,
            (high_max - low_min).values,
            fill=-50
        )
    
    # === CCI (3 periods) ===
    for period in [10, 20, 30]:
        p = min(period, len(df))
        typical = (df['high'] + df['low'] + df['close']) / 3
        sma = typical.rolling(window=p, min_periods=1).mean()
        mad = typical.rolling(window=p, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        ).fillna(1)
        df[f'cci_{period}'] = safe_div((typical - sma).values, (0.015 * mad).values, fill=0)
    
    # === ROC and Momentum (4 periods x 2 = 8) ===
    for period in [5, 10, 20, 30]:
        p = min(period, len(df))
        shifted = df['close'].shift(p).fillna(df['close'])
        df[f'roc_{period}'] = safe_div((df['close'] - shifted).values, shifted.values, fill=0) * 100
        df[f'momentum_{period}'] = df['close'] - shifted
    
    # === Volume features (18) ===
    for period in [5, 10, 20, 50]:
        p = min(period, len(df))
        df[f'volume_sma_{period}'] = df['volume'].rolling(window=p, min_periods=1).mean()
    
    df['volume_ema_10'] = df['volume'].ewm(span=10, min_periods=1).mean()
    df['volume_ema_20'] = df['volume'].ewm(span=20, min_periods=1).mean()
    
    for period in [5, 10, 20]:
        p = min(period, len(df))
        vol_sma = df['volume'].rolling(window=p, min_periods=1).mean()
        df[f'volume_ratio_{period}'] = safe_div(df['volume'].values, vol_sma.values, fill=1)
    
    df['volume_std_10'] = df['volume'].rolling(window=10, min_periods=1).std().fillna(0)
    df['volume_std_20'] = df['volume'].rolling(window=20, min_periods=1).std().fillna(0)
    
    # VWAP features
    df['vwap_ratio'] = safe_div(df['close'].values, df['vwap'].values, fill=1)
    df['vwap_diff'] = df['close'] - df['vwap']
    df['vwap_sma_10'] = df['vwap'].rolling(window=10, min_periods=1).mean()
    df['vwap_sma_20'] = df['vwap'].rolling(window=20, min_periods=1).mean()
    
    # OBV
    obv_sign = np.sign(df['close'].diff().fillna(0))
    df['obv'] = (obv_sign * df['volume']).cumsum()
    df['obv_sma_10'] = df['obv'].rolling(window=10, min_periods=1).mean()
    df['obv_sma_20'] = df['obv'].rolling(window=20, min_periods=1).mean()
    
    # MFI (3 periods)
    for period in [10, 14, 20]:
        p = min(period, len(df))
        typical = (df['high'] + df['low'] + df['close']) / 3
        mf = typical * df['volume']
        positive_mf = mf.where(typical > typical.shift(1).fillna(typical), 0).rolling(window=p, min_periods=1).sum()
        negative_mf = mf.where(typical < typical.shift(1).fillna(typical), 0).rolling(window=p, min_periods=1).sum()
        mfr = safe_div(positive_mf.values, negative_mf.values, fill=1)
        df[f'mfi_{period}'] = 100 - (100 / (1 + mfr))
    
    # === Lag features OHLC (9 lags x 4 = 36) ===
    for lag in [1, 2, 3, 5, 10, 15, 20, 30, 50]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag).fillna(method='bfill')
        df[f'open_lag_{lag}'] = df['open'].shift(lag).fillna(method='bfill')
        df[f'high_lag_{lag}'] = df['high'].shift(lag).fillna(method='bfill')
        df[f'low_lag_{lag}'] = df['low'].shift(lag).fillna(method='bfill')
    
    # Lag features volume/trades (6 lags x 2 = 12)
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag).fillna(method='bfill')
        df[f'trades_lag_{lag}'] = df['trades'].shift(lag).fillna(0)
    
    # Returns lag (6 lags)
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag).fillna(0)
    
    # === Rolling statistics (4 periods x 6 = 24) ===
    for period in [5, 10, 20, 50]:
        p = min(period, len(df))
        df[f'close_max_{period}'] = df['close'].rolling(window=p, min_periods=1).max()
        df[f'close_min_{period}'] = df['close'].rolling(window=p, min_periods=1).min()
        df[f'close_mean_{period}'] = df['close'].rolling(window=p, min_periods=1).mean()
        df[f'close_median_{period}'] = df['close'].rolling(window=p, min_periods=1).median()
        df[f'close_skew_{period}'] = df['close'].rolling(window=p, min_periods=1).skew().fillna(0)
        df[f'close_kurt_{period}'] = df['close'].rolling(window=p, min_periods=1).kurt().fillna(0)
    
    # === Time features (17) ===
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['timestamp'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['timestamp'].dt.is_quarter_end.astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Extract last row with exact feature order
    result = []
    for f in feature_names:
        if f in df.columns:
            result.append(df[f].iloc[-1])
        else:
            result.append(0.0)  # Missing feature
    
    result = np.array(result, dtype=np.float64)
    
    # Replace any remaining nan/inf
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result.reshape(1, -1)


# ==================== PREDICTION FUNCTIONS ====================

def predict_lightgbm(candles: List[CandleData]) -> float:
    """Predict using LightGBM"""
    model, feature_names = load_lightgbm()
    X = calculate_lightgbm_features(candles)
    prediction = model.predict(X)[0]
    return float(prediction)


def predict_xgboost(candles: List[CandleData]) -> float:
    """Predict using XGBoost"""
    import xgboost as xgb
    
    model, feature_names = load_xgboost()
    X = calculate_full_features(candles, feature_names)
    
    # Check if model is xgb.Booster (needs DMatrix) or sklearn wrapper
    if isinstance(model, xgb.Booster):
        # Booster needs DMatrix
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        prediction = model.predict(dmatrix)[0]
    else:
        # sklearn-style model (XGBRegressor)
        prediction = model.predict(X)[0]
    
    return float(prediction)


def predict_catboost(candles: List[CandleData]) -> float:
    """Predict using CatBoost"""
    from catboost import Pool
    
    model, feature_names = load_catboost()
    X = calculate_full_features(candles, feature_names)
    
    # Load cat_features indices
    cat_features_path = os.path.join(MODELS_DIR, "catboost", "cat_features_gpu_full.json")
    cat_features_idx = []
    if os.path.exists(cat_features_path):
        with open(cat_features_path, 'r') as f:
            cat_info = json.load(f)
            cat_features_idx = cat_info.get('cat_features_idx', [])
    
    # Convert categorical columns to int for CatBoost
    if cat_features_idx:
        X_df = pd.DataFrame(X, columns=feature_names)
        cat_feature_names = [feature_names[i] for i in cat_features_idx if i < len(feature_names)]
        for col in cat_feature_names:
            if col in X_df.columns:
                X_df[col] = X_df[col].astype(int)
        
        # Create Pool with categorical features
        pool = Pool(X_df, cat_features=cat_feature_names)
        prediction = model.predict(pool)
    else:
        prediction = model.predict(X)
    
    if isinstance(prediction, np.ndarray):
        return float(prediction[0])
    return float(prediction)


def predict_lstm(candles: List[CandleData]) -> float:
    """Predict using LSTM v2 model
    
    LSTM v2 uses 33 returns-based features and predicts the return for next candle.
    Input: sequence of 60 timesteps x 33 features
    Output: predicted return (percentage change)
    
    Final price = current_price * (1 + predicted_return)
    """
    from sklearn.preprocessing import MinMaxScaler
    
    model, scaler, lookback = load_lstm()
    current_price = candles[-1].close
    
    # Calculate 33 returns-based features for all candles
    df = calculate_lstm_gru_v2_features(candles)
    
    # Extract feature values in correct order
    feature_data = df[LSTM_GRU_V2_FEATURES].values.astype(np.float32)
    
    # Get last `lookback` rows for sequence
    if len(feature_data) >= lookback:
        X_seq = feature_data[-lookback:]
    else:
        # Pad with first row if not enough data
        padding_needed = lookback - len(feature_data)
        padding = np.tile(feature_data[0], (padding_needed, 1))
        X_seq = np.vstack([padding, feature_data])
    
    # Scale features
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X_seq)
        except Exception as e:
            print(f"⚠️ LSTM scaler error: {e}, using MinMax scaling")
            temp_scaler = MinMaxScaler()
            X_scaled = temp_scaler.fit_transform(X_seq)
    else:
        temp_scaler = MinMaxScaler()
        X_scaled = temp_scaler.fit_transform(X_seq)
    
    # Reshape for LSTM: (batch_size, timesteps, features)
    X = X_scaled.reshape(1, lookback, len(LSTM_GRU_V2_FEATURES))
    
    # Predict - model outputs predicted return (not price!)
    predicted_return = model.predict(X, verbose=0)[0][0]
    
    # Clip extreme predictions
    predicted_return = np.clip(predicted_return, -0.05, 0.05)  # Max ±5% change
    
    # Calculate predicted price
    prediction = current_price * (1 + predicted_return)
    
    print(f"🔮 LSTM v2: return={predicted_return:.6f}, current=${current_price:.2f}, pred=${prediction:.2f}")
    return float(prediction)


def predict_gru(candles: List[CandleData]) -> float:
    """Predict using GRU v2 model
    
    GRU v2 uses 33 returns-based features (same as LSTM v2) and predicts returns.
    Input: sequence of 60 timesteps x 33 features
    Output: predicted return (percentage change)
    
    Final price = current_price * (1 + predicted_return)
    """
    from sklearn.preprocessing import MinMaxScaler
    
    model, scaler, lookback = load_gru()
    current_price = candles[-1].close
    
    # Calculate 33 returns-based features for all candles
    df = calculate_lstm_gru_v2_features(candles)
    
    # Extract feature values in correct order
    feature_data = df[LSTM_GRU_V2_FEATURES].values.astype(np.float32)
    
    # Get last `lookback` rows for sequence
    if len(feature_data) >= lookback:
        X_seq = feature_data[-lookback:]
    else:
        # Pad with first row if not enough data
        padding_needed = lookback - len(feature_data)
        padding = np.tile(feature_data[0], (padding_needed, 1))
        X_seq = np.vstack([padding, feature_data])
    
    # Scale features
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X_seq)
        except Exception as e:
            print(f"⚠️ GRU scaler error: {e}, using MinMax scaling")
            temp_scaler = MinMaxScaler()
            X_scaled = temp_scaler.fit_transform(X_seq)
    else:
        temp_scaler = MinMaxScaler()
        X_scaled = temp_scaler.fit_transform(X_seq)
    
    # Reshape for GRU: (batch_size, timesteps, features)
    X = X_scaled.reshape(1, lookback, len(LSTM_GRU_V2_FEATURES))
    
    # Predict - model outputs predicted return (not price!)
    predicted_return = model.predict(X, verbose=0)[0][0]
    
    # Clip extreme predictions
    predicted_return = np.clip(predicted_return, -0.05, 0.05)  # Max ±5% change
    
    # Calculate predicted price
    prediction = current_price * (1 + predicted_return)
    
    print(f"🔮 GRU v2: return={predicted_return:.6f}, current=${current_price:.2f}, pred=${prediction:.2f}")
    return float(prediction)


# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "message": "Crypto Prediction API",
        "version": "2.2.0",
        "models_dir": MODELS_DIR,
        "models": ["LightGBM", "XGBoost", "CatBoost", "LSTM", "GRU"]
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/models")
async def list_models():
    """List available models by checking files"""
    available = []
    
    checks = {
        "LightGBM": os.path.join(MODELS_DIR, "lightgbm", "lightgbm_model_gpu.txt"),
        "XGBoost": os.path.join(MODELS_DIR, "xgboost", "xgboost_model_gpu_full.ubj"),
        "CatBoost": os.path.join(MODELS_DIR, "catboost", "catboost_model_gpu_full.cbm"),
        "LSTM": os.path.join(MODELS_DIR, "lstm", "lstm_model_final.h5"),
        "GRU": os.path.join(MODELS_DIR, "gru", "gru_model_gpu_full.h5"),
    }
    
    for name, path in checks.items():
        if os.path.exists(path):
            available.append(name)
        elif name == "XGBoost" and os.path.exists(os.path.join(MODELS_DIR, "xgboost", "xgboost_model_gpu_full.pkl")):
            available.append(name)
        elif name == "LSTM" and os.path.exists(os.path.join(MODELS_DIR, "lstm", "lstm_model.h5")):
            available.append(name)
    
    return {
        "available_models": available
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction"""
    try:
        model_type = request.model_type
        candles = request.candles
        
        if len(candles) < 60:
            raise HTTPException(status_code=400, detail="Need at least 60 candles for prediction")
        
        current_price = candles[-1].close
        
        if model_type == "LightGBM":
            raw_prediction = predict_lightgbm(candles)
            features_used = 60
        elif model_type == "XGBoost":
            raw_prediction = predict_xgboost(candles)
            features_used = 220
        elif model_type == "CatBoost":
            raw_prediction = predict_catboost(candles)
            features_used = 220
        elif model_type == "LSTM":
            raw_prediction = predict_lstm(candles)
            features_used = 1  # Close price sequence
        elif model_type == "GRU":
            raw_prediction = predict_gru(candles)
            features_used = 221
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_type}")
        
        # Apply price correction
        if PRICE_CORRECTION_METHOD == "relative":
            # Model predicts some price P based on features from price C
            # We compute the ratio and apply to current price
            # This assumes model learned relative patterns
            input_close = candles[-2].close if len(candles) > 1 else current_price
            if input_close > 0:
                ratio = raw_prediction / input_close
                # Limit extreme ratios
                ratio = max(0.95, min(1.05, ratio))
                prediction = current_price * ratio
            else:
                prediction = raw_prediction
        elif PRICE_CORRECTION_METHOD == "offset":
            # Calculate average offset from last N candles to calibrate
            # We predict on earlier subsets and compare to actual
            calibration_n = 10
            offsets = []
            for j in range(calibration_n):
                if len(candles) > 60 + j:
                    cal_subset = candles[:-(j+1)] if j > 0 else candles[:-1]
                    if len(cal_subset) >= 60:
                        if model_type == "LightGBM":
                            cal_pred = predict_lightgbm(cal_subset)
                        elif model_type == "XGBoost":
                            cal_pred = predict_xgboost(cal_subset)
                        elif model_type == "CatBoost":
                            cal_pred = predict_catboost(cal_subset)
                        elif model_type == "LSTM":
                            cal_pred = predict_lstm(cal_subset)
                        elif model_type == "GRU":
                            cal_pred = predict_gru(cal_subset)
                        else:
                            cal_pred = cal_subset[-1].close
                        
                        actual = candles[-(j+1)].close if j > 0 else candles[-1].close
                        offsets.append(actual - cal_pred)
            
            if offsets:
                avg_offset = sum(offsets) / len(offsets)
                prediction = raw_prediction + avg_offset
            else:
                prediction = raw_prediction
        else:
            prediction = raw_prediction
        
        return PredictionResponse(
            prediction=prediction,
            model_type=model_type,
            timestamp=candles[-1].timestamp,
            features_used=features_used
        )
    
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions"""
    try:
        model_type = request.model_type
        candles = request.candles
        count = min(request.count, len(candles) - 100)
        
        if count <= 0:
            raise HTTPException(status_code=400, detail="Not enough candles for batch prediction")
        
        if model_type in ["LSTM", "GRU"]:
            raise HTTPException(
                status_code=503,
                detail=f"{model_type} model temporarily disabled due to Keras version incompatibility"
            )
        
        predictions = []
        
        for i in range(count):
            end_idx = len(candles) - count + i + 1
            subset = candles[:end_idx]
            
            if len(subset) < 60:
                continue
            
            current_price = subset[-1].close
            prev_price = subset[-2].close if len(subset) > 1 else current_price
            
            if model_type == "LightGBM":
                raw_pred = predict_lightgbm(subset)
            elif model_type == "XGBoost":
                raw_pred = predict_xgboost(subset)
            elif model_type == "CatBoost":
                raw_pred = predict_catboost(subset)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown model: {model_type}")
            
            # Apply price correction
            if PRICE_CORRECTION_METHOD == "relative":
                if prev_price > 0:
                    ratio = raw_pred / prev_price
                    ratio = max(0.95, min(1.05, ratio))
                    pred = current_price * ratio
                else:
                    pred = raw_pred
            elif PRICE_CORRECTION_METHOD == "offset":
                offset = current_price - raw_pred
                pred = raw_pred + offset
            else:
                pred = raw_pred
            
            predictions.append(pred)
        
        return BatchPredictionResponse(
            predictions=predictions,
            model_type=model_type,
            count=len(predictions)
        )
    
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/future", response_model=FuturePredictionResponse)
async def predict_future(request: FuturePredictionRequest):
    """
    Predict future prices for a given horizon.
    - For tree models (LightGBM, XGBoost, CatBoost): Uses iterative multi-step prediction
    - For LSTM/GRU: Uses single prediction + trend extrapolation (since these models
      are trained for single-step and don't work well iteratively)
    """
    try:
        model_type = request.model_type
        candles = list(request.candles)
        horizon = request.horizon
        interval = request.interval
        
        if len(candles) < 60:
            raise HTTPException(status_code=400, detail="Need at least 60 candles for prediction")
        
        if horizon not in HORIZON_CONFIG:
            raise HTTPException(status_code=400, detail=f"Invalid horizon: {horizon}. Use '5m', '15m', '1h', '4h', '1d', or '3d'")
        
        # Calculate number of steps based on interval
        interval_minutes = 5 if interval == "5m" else 15
        horizon_minutes = HORIZON_CONFIG[horizon]
        num_steps = horizon_minutes // interval_minutes
        
        # Limit steps to prevent too long predictions
        max_steps = min(num_steps, 288)  # Max 288 steps (24h for 5m interval)
        
        current_price = candles[-1].close
        last_timestamp = candles[-1].timestamp
        
        predictions = []
        
        # =============== LSTM/GRU: Trend-based approach ===============
        # These models don't work well with iterative prediction
        # Instead, we get a single directional signal and extrapolate
        if model_type in ["LSTM", "GRU"]:
            # Get single prediction to determine direction/magnitude
            if model_type == "LSTM":
                single_pred = predict_lstm(candles)
            else:
                single_pred = predict_gru(candles)
            
            # Calculate the predicted change per step
            price_change = single_pred - current_price
            pct_change_per_step = price_change / current_price / max(max_steps, 1)
            
            # Also calculate recent momentum from actual data
            recent_prices = [c.close for c in candles[-20:]]
            if len(recent_prices) >= 2:
                recent_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] / len(recent_prices)
            else:
                recent_momentum = 0
            
            # Combine model signal (30%) with momentum (70%) for more realistic predictions
            # Since LSTM/GRU models have poor accuracy, we weight momentum more
            combined_change_per_step = 0.3 * pct_change_per_step + 0.7 * recent_momentum
            
            # Limit per-step change to reasonable bounds
            max_step_change = 0.002  # Max 0.2% per step
            combined_change_per_step = np.clip(combined_change_per_step, -max_step_change, max_step_change)
            
            # Add some mean reversion - as we go further, trend weakens
            decay_factor = 0.995  # Slight decay per step
            
            # Generate predictions with decaying trend
            cumulative_change = 0
            for step in range(max_steps):
                # Calculate change for this step with decay
                step_change = combined_change_per_step * (decay_factor ** step)
                cumulative_change += step_change
                
                # Apply cumulative change to get predicted price
                predicted_price = current_price * (1 + cumulative_change)
                
                # Add some realistic noise based on volatility
                recent_volatility = np.std(recent_prices) / np.mean(recent_prices) if recent_prices else 0
                noise = np.random.normal(0, recent_volatility * 0.1) * current_price
                predicted_price += noise
                
                step_timestamp = last_timestamp + (step + 1) * interval_minutes * 60 * 1000
                
                predictions.append({
                    "step": step + 1,
                    "timestamp": step_timestamp,
                    "price": float(predicted_price),
                    "minutes_ahead": (step + 1) * interval_minutes
                })
            
            print(f"📊 {model_type} trend: single_pred=${single_pred:.2f}, momentum={recent_momentum:.6f}, combined_change/step={combined_change_per_step:.6f}")
        
        # =============== Tree models: Iterative approach ===============
        else:
            working_candles = candles.copy()
            
            # Get calibration offset for tree models
            calibration_offset = 0
            if len(candles) >= 70:
                offsets = []
                for j in range(5):
                    cal_subset = candles[:-(j+1)] if j > 0 else candles[:-1]
                    if len(cal_subset) >= 60:
                        if model_type == "LightGBM":
                            cal_pred = predict_lightgbm(cal_subset)
                        elif model_type == "XGBoost":
                            cal_pred = predict_xgboost(cal_subset)
                        elif model_type == "CatBoost":
                            cal_pred = predict_catboost(cal_subset)
                        else:
                            cal_pred = cal_subset[-1].close
                        
                        actual = candles[-(j+1)].close if j > 0 else candles[-1].close
                        offsets.append(actual - cal_pred)
                
                if offsets:
                    calibration_offset = sum(offsets) / len(offsets)
            
            # Iterative multi-step prediction for tree models
            for step in range(max_steps):
                # Predict next candle
                if model_type == "LightGBM":
                    raw_pred = predict_lightgbm(working_candles)
                elif model_type == "XGBoost":
                    raw_pred = predict_xgboost(working_candles)
                elif model_type == "CatBoost":
                    raw_pred = predict_catboost(working_candles)
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown model: {model_type}")
                
                # Apply calibration
                predicted_price = raw_pred + calibration_offset
                
                # Calculate timestamp for this prediction
                step_timestamp = last_timestamp + (step + 1) * interval_minutes * 60 * 1000
                
                predictions.append({
                    "step": step + 1,
                    "timestamp": step_timestamp,
                    "price": float(predicted_price),
                    "minutes_ahead": (step + 1) * interval_minutes
                })
                
                # Create synthetic candle for next iteration
                recent_volatility = np.std([c.close for c in working_candles[-20:]])
                noise = np.random.normal(0, recent_volatility * 0.05) if recent_volatility > 0 else 0
                
                synthetic_candle = CandleData(
                    timestamp=step_timestamp,
                    open=working_candles[-1].close,
                    high=max(working_candles[-1].close, predicted_price + abs(noise)),
                    low=min(working_candles[-1].close, predicted_price - abs(noise)),
                    close=predicted_price,
                    volume=np.mean([c.volume for c in working_candles[-10:]]),
                    trades=int(np.mean([c.trades or 0 for c in working_candles[-10:]])),
                    vwap=predicted_price,
                    turnover=predicted_price * np.mean([c.volume for c in working_candles[-10:]])
                )
                
                working_candles.append(synthetic_candle)
                # Keep only last 300 candles to avoid memory issues
                if len(working_candles) > 300:
                    working_candles = working_candles[-300:]
        
        # Calculate target timestamp
        target_timestamp = last_timestamp + horizon_minutes * 60 * 1000
        
        return FuturePredictionResponse(
            predictions=predictions,
            model_type=model_type,
            horizon=horizon,
            current_price=current_price,
            target_timestamp=target_timestamp
        )
    
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/realtime", response_model=RealtimePredictionResponse)
async def predict_realtime(request: RealtimePredictionRequest):
    """
    Generate realtime trendline predictions.
    Returns multiple short-term predictions for smooth trendline display.
    """
    try:
        model_type = request.model_type
        candles = list(request.candles)
        steps = min(request.steps, 30)  # Max 30 steps for realtime
        
        if len(candles) < 60:
            raise HTTPException(status_code=400, detail="Need at least 60 candles for prediction")
        
        if model_type in ["LSTM", "GRU"]:
            raise HTTPException(
                status_code=503,
                detail=f"{model_type} model temporarily disabled due to Keras version incompatibility"
            )
        
        current_price = candles[-1].close
        last_timestamp = candles[-1].timestamp
        
        # Get calibration offset
        calibration_offset = 0
        if len(candles) >= 70:
            offsets = []
            for j in range(5):
                cal_subset = candles[:-(j+1)] if j > 0 else candles[:-1]
                if len(cal_subset) >= 60:
                    if model_type == "LightGBM":
                        cal_pred = predict_lightgbm(cal_subset)
                    elif model_type == "XGBoost":
                        cal_pred = predict_xgboost(cal_subset)
                    elif model_type == "CatBoost":
                        cal_pred = predict_catboost(cal_subset)
                    else:
                        cal_pred = cal_subset[-1].close
                    
                    actual = candles[-(j+1)].close if j > 0 else candles[-1].close
                    offsets.append(actual - cal_pred)
            
            if offsets:
                calibration_offset = sum(offsets) / len(offsets)
        
        predictions = []
        working_candles = candles.copy()
        
        # Generate short-term predictions (each step = 1 minute)
        for step in range(steps):
            if model_type == "LightGBM":
                raw_pred = predict_lightgbm(working_candles)
            elif model_type == "XGBoost":
                raw_pred = predict_xgboost(working_candles)
            elif model_type == "CatBoost":
                raw_pred = predict_catboost(working_candles)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown model: {model_type}")
            
            predicted_price = raw_pred + calibration_offset
            step_timestamp = last_timestamp + (step + 1) * 60 * 1000  # 1 minute per step
            
            predictions.append({
                "timestamp": step_timestamp,
                "price": predicted_price
            })
            
            # Update working candles with interpolated values
            if step < steps - 1:
                synthetic_candle = CandleData(
                    timestamp=step_timestamp,
                    open=working_candles[-1].close,
                    high=max(working_candles[-1].close, predicted_price),
                    low=min(working_candles[-1].close, predicted_price),
                    close=predicted_price,
                    volume=np.mean([c.volume for c in working_candles[-5:]]),
                    trades=int(np.mean([c.trades or 0 for c in working_candles[-5:]])),
                    vwap=predicted_price,
                    turnover=predicted_price * np.mean([c.volume for c in working_candles[-5:]])
                )
                working_candles.append(synthetic_candle)
                if len(working_candles) > 250:
                    working_candles = working_candles[-250:]
        
        # Determine trend
        if len(predictions) >= 2:
            price_change = predictions[-1]["price"] - current_price
            change_pct = (price_change / current_price) * 100
            
            if change_pct > 0.1:
                trend = "up"
            elif change_pct < -0.1:
                trend = "down"
            else:
                trend = "neutral"
            
            # Simple confidence based on consistency
            price_diffs = [predictions[i+1]["price"] - predictions[i]["price"] for i in range(len(predictions)-1)]
            same_direction = sum(1 for d in price_diffs if (d > 0) == (price_change > 0))
            confidence = same_direction / len(price_diffs) if price_diffs else 0.5
        else:
            trend = "neutral"
            confidence = 0.5
        
        return RealtimePredictionResponse(
            current_price=current_price,
            predictions=predictions,
            model_type=model_type,
            trend=trend,
            confidence=confidence
        )
    
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🚀 Starting Crypto Prediction API v2.1...")
    print(f"📁 Models directory: {MODELS_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
