import { Candle, ModelType, PredictionHorizon, FuturePrediction } from '../types';

// API endpoint - can be configured via environment variable
const API_BASE_URL = process.env.REACT_APP_PREDICTION_API_URL || 'http://localhost:8000';

export type ApiStatus = 'connected' | 'disconnected' | 'checking';

interface CandleApiData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  trades?: number;
  vwap?: number;
  turnover?: number;
}

interface PredictionResponse {
  prediction: number;
  model_type: string;
  timestamp: number;
  features_used: number;
}

interface BatchPredictionResponse {
  predictions: number[];
  model_type: string;
  count: number;
}

interface FuturePredictionResponse {
  predictions: Array<{
    step: number;
    timestamp: number;
    price: number;
    minutes_ahead: number;
  }>;
  model_type: string;
  horizon: string;
  current_price: number;
  target_timestamp: number;
}

interface RealtimePredictionResponse {
  current_price: number;
  predictions: Array<{
    timestamp: number;
    price: number;
  }>;
  model_type: string;
  trend: 'up' | 'down' | 'neutral';
  confidence: number;
}

// Event listeners for API status changes
type StatusChangeListener = (status: ApiStatus) => void;

export class PredictionService {
  private apiAvailable: boolean | null = null;
  private lastApiCheck: number = 0;
  private readonly API_CHECK_INTERVAL = 5000; // Check API availability every 5 seconds
  private statusListeners: StatusChangeListener[] = [];
  private currentStatus: ApiStatus = 'checking';

  /**
   * Subscribe to API status changes
   */
  onStatusChange(listener: StatusChangeListener): () => void {
    this.statusListeners.push(listener);
    // Immediately notify with current status
    listener(this.currentStatus);
    // Return unsubscribe function
    return () => {
      this.statusListeners = this.statusListeners.filter(l => l !== listener);
    };
  }

  /**
   * Notify all listeners of status change
   */
  private notifyStatusChange(status: ApiStatus): void {
    if (this.currentStatus !== status) {
      this.currentStatus = status;
      this.statusListeners.forEach(listener => listener(status));
    }
  }

  /**
   * Get current API status
   */
  getStatus(): ApiStatus {
    return this.currentStatus;
  }

  /**
   * Check if the prediction API is available
   */
  async checkApiAvailability(): Promise<boolean> {
    const now = Date.now();
    
    // Use cached result if checked recently
    if (this.apiAvailable !== null && (now - this.lastApiCheck) < this.API_CHECK_INTERVAL) {
      return this.apiAvailable;
    }

    this.notifyStatusChange('checking');

    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(3000), // 3 second timeout
      });
      this.apiAvailable = response.ok;
      this.notifyStatusChange(response.ok ? 'connected' : 'disconnected');
    } catch {
      this.apiAvailable = false;
      this.notifyStatusChange('disconnected');
    }
    
    this.lastApiCheck = now;
    return this.apiAvailable;
  }

  /**
   * Convert candles to API format
   */
  private candlesToApiFormat(candles: Candle[]): CandleApiData[] {
    return candles.map(c => ({
      timestamp: c.timestamp,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
      volume: c.volume,
      trades: 0,
      vwap: (c.high + c.low + c.close) / 3,
      turnover: c.close * c.volume,
    }));
  }

  /**
   * Make a single prediction using the trained model
   * Returns null if API is not available
   */
  async predict(candles: Candle[], modelType: ModelType): Promise<number | null> {
    if (candles.length < 20) {
      throw new Error('Not enough data for prediction');
    }

    // Check if API is available
    const apiAvailable = await this.checkApiAvailability();

    if (!apiAvailable) {
      console.log(`❌ API not available - cannot predict with ${modelType}`);
      return null;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          candles: this.candlesToApiFormat(candles.slice(-250)), // Send last 250 candles for feature calculation
          model_type: modelType,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data: PredictionResponse = await response.json();
      console.log(`🤖 ${modelType} prediction (API):`, data.prediction, `using ${data.features_used} features`);
      return data.prediction;
    } catch (error) {
      console.error('API prediction failed:', error);
      this.apiAvailable = false;
      this.notifyStatusChange('disconnected');
      return null;
    }
  }

  /**
   * Batch predictions for historical data
   * Returns null if API is not available
   */
  async predictBatch(
    candles: Candle[],
    modelType: ModelType,
    count: number
  ): Promise<number[] | null> {
    // Check if API is available
    const apiAvailable = await this.checkApiAvailability();

    if (!apiAvailable) {
      console.log(`❌ API not available - cannot batch predict with ${modelType}`);
      return null;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/predict/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          candles: this.candlesToApiFormat(candles),
          model_type: modelType,
          count: count,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data: BatchPredictionResponse = await response.json();
      console.log(`🤖 ${modelType} batch predictions (API):`, data.count, 'predictions');
      return data.predictions;
    } catch (error) {
      console.error('API batch prediction failed:', error);
      this.apiAvailable = false;
      this.notifyStatusChange('disconnected');
      return null;
    }
  }

  /**
   * Get list of available models from API
   */
  async getAvailableModels(): Promise<{ available: string[]; disabled: string[] }> {
    try {
      const response = await fetch(`${API_BASE_URL}/models`);
      if (response.ok) {
        const data = await response.json();
        return {
          available: data.available_models || [],
          disabled: data.disabled_models || [],
        };
      }
    } catch {
      // API not available
    }
    return {
      available: ['LightGBM', 'XGBoost', 'CatBoost'],
      disabled: ['LSTM', 'GRU'],
    };
  }

  /**
   * Predict future prices for a given horizon (4h, 1d, 3d)
   */
  async predictFuture(
    candles: Candle[],
    modelType: ModelType,
    horizon: PredictionHorizon,
    interval: '5m' | '15m' = '5m'
  ): Promise<FuturePrediction[] | null> {
    const apiAvailable = await this.checkApiAvailability();
    if (!apiAvailable) {
      console.log(`❌ API not available - cannot predict future with ${modelType}`);
      return null;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/predict/future`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          candles: this.candlesToApiFormat(candles.slice(-300)),
          model_type: modelType,
          horizon: horizon,
          interval: interval,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data: FuturePredictionResponse = await response.json();
      console.log(`🤖 ${modelType} future predictions (${horizon}):`, data.predictions.length, 'steps');
      
      return data.predictions.map(p => ({
        horizon: horizon,
        timestamp: p.timestamp,
        price: p.price,
      }));
    } catch (error) {
      console.error('API future prediction failed:', error);
      this.apiAvailable = false;
      this.notifyStatusChange('disconnected');
      return null;
    }
  }

  /**
   * Get realtime trendline predictions (updates every ~1 minute)
   */
  async predictRealtime(
    candles: Candle[],
    modelType: ModelType,
    steps: number = 10
  ): Promise<{
    currentPrice: number;
    predictions: Array<{ timestamp: number; price: number }>;
    trend: 'up' | 'down' | 'neutral';
    confidence: number;
  } | null> {
    const apiAvailable = await this.checkApiAvailability();
    if (!apiAvailable) {
      console.log(`❌ API not available - cannot get realtime predictions with ${modelType}`);
      return null;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/predict/realtime`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          candles: this.candlesToApiFormat(candles.slice(-250)),
          model_type: modelType,
          steps: steps,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data: RealtimePredictionResponse = await response.json();
      console.log(`🤖 ${modelType} realtime trendline: trend=${data.trend}, confidence=${data.confidence.toFixed(2)}`);
      
      return {
        currentPrice: data.current_price,
        predictions: data.predictions,
        trend: data.trend,
        confidence: data.confidence,
      };
    } catch (error) {
      console.error('API realtime prediction failed:', error);
      this.apiAvailable = false;
      this.notifyStatusChange('disconnected');
      return null;
    }
  }

  /**
   * Force refresh API availability check
   */
  refreshApiStatus(): void {
    this.apiAvailable = null;
    this.lastApiCheck = 0;
    this.checkApiAvailability();
  }
}

export const predictionService = new PredictionService();
