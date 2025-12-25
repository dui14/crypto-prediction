"""
LSTM V2 - Scale-Independent Training
Predicts percentage returns instead of absolute prices
This makes the model work regardless of current BTC price level

Run: python train_lstm_v2.py
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datetime import datetime

# ==================== CONFIG ====================
LOOKBACK = 60          # 60 candles = 5 hours of 5m data
EPOCHS = 100
BATCH_SIZE = 128
DATA_PATH = '../data/csv/BTCUSDT_5m_2020_to_now_618k_rows.csv'
OUT_DIR = '../models/lstm'

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("LSTM V2 - Returns-Based Training (Scale-Independent)")
print("=" * 70)

# GPU Check
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs available: {gpus}")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✓ GPU memory growth enabled")
    print(f"✅ Training will use: GPU ({len(gpus)} device(s))")
else:
    print("⚠️  No GPU detected")
    print("⚠️  Training will use: CPU (slower)")
    print("   Tip: Install CUDA + cuDNN for GPU acceleration")

# ==================== 1. LOAD DATA ====================
print("\n[1/8] Loading data...")
df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"   Loaded {len(df):,} rows")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")

# ==================== 2. CREATE FEATURES ====================
print("\n[2/8] Creating return-based features...")

# Price returns (scale-independent)
df['returns'] = df['close'].pct_change()
df['returns_2'] = df['close'].pct_change(2)
df['returns_3'] = df['close'].pct_change(3)
df['returns_5'] = df['close'].pct_change(5)
df['returns_10'] = df['close'].pct_change(10)
df['returns_20'] = df['close'].pct_change(20)

# Log returns
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

# Volatility (rolling std of returns)
df['volatility_5'] = df['returns'].rolling(5).std()
df['volatility_10'] = df['returns'].rolling(10).std()
df['volatility_20'] = df['returns'].rolling(20).std()

# RSI (already 0-100 scale)
for period in [7, 14, 21]:
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

# MACD as percentage of price
ema12 = df['close'].ewm(span=12).mean()
ema26 = df['close'].ewm(span=26).mean()
df['macd_pct'] = (ema12 - ema26) / df['close']
df['macd_signal_pct'] = df['macd_pct'].ewm(span=9).mean()
df['macd_hist_pct'] = df['macd_pct'] - df['macd_signal_pct']

# Bollinger Band position (0-1 scale)
sma20 = df['close'].rolling(20).mean()
std20 = df['close'].rolling(20).std()
df['bb_position'] = (df['close'] - (sma20 - 2*std20)) / (4*std20 + 1e-10)
df['bb_width_pct'] = (4*std20) / sma20

# Stochastic Oscillator
low14 = df['low'].rolling(14).min()
high14 = df['high'].rolling(14).max()
df['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14 + 1e-10)
df['stoch_d'] = df['stoch_k'].rolling(3).mean()

# Volume features (ratios are scale-independent)
df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
df['volume_change'] = df['volume'].pct_change()

# Candle features (percentages)
df['range_pct'] = (df['high'] - df['low']) / df['close']
df['body_pct'] = abs(df['close'] - df['open']) / df['close']
df['candle_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
df['upper_shadow_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
df['lower_shadow_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

# Momentum
df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

# Price position relative to moving averages
df['price_sma10_ratio'] = df['close'] / df['close'].rolling(10).mean() - 1
df['price_sma20_ratio'] = df['close'] / df['close'].rolling(20).mean() - 1
df['price_sma50_ratio'] = df['close'] / df['close'].rolling(50).mean() - 1

# Target: next candle's return
df['target'] = df['returns'].shift(-1)

# Drop NaN rows
df = df.dropna().reset_index(drop=True)
print(f"   After feature engineering: {len(df):,} rows")

# ==================== 3. SELECT FEATURES ====================
feature_cols = [
    # Returns
    'returns', 'returns_2', 'returns_3', 'returns_5', 'returns_10', 'returns_20',
    'log_returns',
    # Volatility
    'volatility_5', 'volatility_10', 'volatility_20',
    # RSI
    'rsi_7', 'rsi_14', 'rsi_21',
    # MACD
    'macd_pct', 'macd_signal_pct', 'macd_hist_pct',
    # Bollinger
    'bb_position', 'bb_width_pct',
    # Stochastic
    'stoch_k', 'stoch_d',
    # Volume
    'volume_ratio', 'volume_change',
    # Candle
    'range_pct', 'body_pct', 'candle_position', 'upper_shadow_pct', 'lower_shadow_pct',
    # Momentum
    'momentum_5', 'momentum_10', 'momentum_20',
    # Price position
    'price_sma10_ratio', 'price_sma20_ratio', 'price_sma50_ratio'
]

print(f"\n[3/8] Selected {len(feature_cols)} features")

# ==================== 4. PREPARE DATA ====================
print("\n[4/8] Preparing sequences...")

X_data = df[feature_cols].values.astype(np.float32)
y_data = df['target'].values.astype(np.float32)

# Clip extreme values to prevent outliers
X_data = np.clip(X_data, -10, 10)

# Scale features to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_data)

# Create sequences
X, y = [], []
for i in range(LOOKBACK, len(X_scaled)):
    X.append(X_scaled[i-LOOKBACK:i])
    y.append(y_data[i])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")

# ==================== 5. SPLIT DATA ====================
print("\n[5/8] Splitting data (time-series safe)...")

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"   Train: {len(X_train):,} samples")
print(f"   Test:  {len(X_test):,} samples")

# ==================== 6. BUILD MODEL ====================
print("\n[6/8] Building LSTM model...")

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(LOOKBACK, len(feature_cols))),
    Dropout(0.2),
    BatchNormalization(),
    
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    BatchNormalization(),
    
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    
    Dense(16, activation='relu'),
    Dense(1)  # Output: predicted return (can be negative)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ==================== 7. TRAIN ====================
print("\n[7/8] Training...")

# Log device being used
if tf.config.list_physical_devices('GPU'):
    print("   🚀 Using device: GPU")
else:
    print("   💻 Using device: CPU")
print()

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(OUT_DIR, 'lstm_model_v2_best.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=False,  # Important for time series
    verbose=1
)

# ==================== 8. EVALUATE ====================
print("\n[8/8] Evaluating...")

y_pred_train = model.predict(X_train, verbose=0).flatten()
y_pred_test = model.predict(X_test, verbose=0).flatten()

# Metrics on returns
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

# Direction accuracy (most important for trading)
train_dir_acc = np.mean(np.sign(y_train) == np.sign(y_pred_train))
test_dir_acc = np.mean(np.sign(y_test) == np.sign(y_pred_test))

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\nTrain Metrics:")
print(f"   MAE:  {train_mae:.6f}")
print(f"   RMSE: {train_rmse:.6f}")
print(f"   Direction Accuracy: {train_dir_acc:.2%}")

print(f"\nTest Metrics:")
print(f"   MAE:  {test_mae:.6f}")
print(f"   RMSE: {test_rmse:.6f}")
print(f"   R²:   {test_r2:.4f}")
print(f"   Direction Accuracy: {test_dir_acc:.2%}")

# ==================== SAVE ====================
print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

# Save model
model_path = os.path.join(OUT_DIR, 'lstm_model_v2.h5')
model.save(model_path)
print(f"✓ Model saved: {model_path}")

# Save scaler
scaler_path = os.path.join(OUT_DIR, 'lstm_scaler_v2.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved: {scaler_path}")

# Save feature names
features_path = os.path.join(OUT_DIR, 'lstm_features_v2.json')
with open(features_path, 'w') as f:
    json.dump(feature_cols, f, indent=2)
print(f"✓ Features saved: {features_path}")

# Save config
config = {
    'version': 'v2',
    'lookback': LOOKBACK,
    'n_features': len(feature_cols),
    'features': feature_cols,
    'trained_at': datetime.now().isoformat(),
    'data_file': DATA_PATH,
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'epochs_trained': len(history.history['loss']),
    'metrics': {
        'train_mae': float(train_mae),
        'train_rmse': float(train_rmse),
        'train_direction_accuracy': float(train_dir_acc),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2),
        'test_direction_accuracy': float(test_dir_acc)
    }
}

config_path = os.path.join(OUT_DIR, 'lstm_config_v2.json')
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"✓ Config saved: {config_path}")

# Save training history
history_path = os.path.join(OUT_DIR, 'lstm_history_v2.json')
with open(history_path, 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
print(f"✓ History saved: {history_path}")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETED!")
print("=" * 70)
print(f"\nModel predicts RETURNS (not absolute prices)")
print(f"To get predicted price: current_price * (1 + predicted_return)")
print(f"\nDirection accuracy: {test_dir_acc:.2%} (>50% is profitable)")
