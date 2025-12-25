export interface ChartDataPoint {
  timestamp: number;
  actual?: number;
  predicted?: number;
  // Future predictions for different horizons
  predicted5m?: number;
  predicted15m?: number;
  predicted1h?: number;
  predicted4h?: number;
  predicted1d?: number;
  predicted3d?: number;
  // Realtime trendline prediction
  predictedTrendline?: number;
  time: string;
}

export interface FuturePrediction {
  horizon: PredictionHorizon;
  timestamp: number;  // Target timestamp
  price: number;      // Predicted price
  confidence?: number;
}

export interface Metrics {
  mae: number;
  rmse: number;
  directionAccuracy: number;
  lastUpdated: Date;
}

export interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export type TimeInterval = '5m' | '15m';

export type ModelType = 'LightGBM' | 'XGBoost' | 'LSTM' | 'GRU' | 'CatBoost';

// Prediction horizons - 5 minutes to 3 days
export type PredictionHorizon = '5m' | '15m' | '1h' | '4h' | '1d' | '3d';

// View modes
export type ViewMode = 'realtime' | 'prediction';

// Prediction state for real-time trendline
export interface PredictionState {
  isRunning: boolean;
  model: ModelType;
  horizon: PredictionHorizon;
  futurePredictions: FuturePrediction[];
  lastUpdate: Date | null;
}