import { useState, useEffect, useCallback, useRef } from 'react';
import { binanceService } from '../services/binanceService';
import { predictionService, ApiStatus } from '../services/predictionService';
import {
  ChartDataPoint,
  TimeInterval,
  ModelType,
  ViewMode,
  Candle,
  PredictionHorizon,
  FuturePrediction,
  Metrics,
} from '../types';
import {
  formatTime,
} from '../services/utils';

// Prediction update interval (1 minute)
const PREDICTION_UPDATE_INTERVAL = 60 * 1000;

export const usePrediction = () => {
  // Settings
  const [symbol, setSymbol] = useState<string>('BTCUSDT');
  const [interval, setInterval] = useState<TimeInterval>('5m');
  const [model, setModel] = useState<ModelType>('LightGBM');
  const [horizon, setHorizon] = useState<PredictionHorizon>('4h');
  
  // View mode: 'realtime' = live price only, 'prediction' = static comparison with model
  const [viewMode, setViewMode] = useState<ViewMode>('realtime');
  
  // States
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [futurePredictions, setFuturePredictions] = useState<FuturePrediction[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking');
  const [trend, setTrend] = useState<'up' | 'down' | 'neutral' | null>(null);

  // Refs
  const historicalCandles = useRef<Candle[]>([]);
  const wsConnected = useRef<boolean>(false);
  const predictionIntervalRef = useRef<number | null>(null);

  // ==================== REALTIME MODE ====================
  
  // Handle realtime price updates (only actual price, no prediction)
  const handleRealtimePriceUpdate = useCallback((price: number) => {
    setCurrentPrice(price);
    
    // Update chart with only actual prices
    if (historicalCandles.current.length > 0) {
      const lastCandle = historicalCandles.current[historicalCandles.current.length - 1];
      lastCandle.close = price;
      lastCandle.high = Math.max(lastCandle.high, price);
      lastCandle.low = Math.min(lastCandle.low, price);

      setChartData(prevData => {
        const newData = [...prevData];
        const now = Date.now();
        const timeStr = formatTime(now);

        if (newData.length > 0) {
          const lastPoint = newData[newData.length - 1];
          const timeDiff = now - lastPoint.timestamp;

          if (timeDiff > 60000) {
            // Add new point every minute
            newData.push({
              timestamp: now,
              actual: price,
              time: timeStr,
            });
            if (newData.length > 100) {
              newData.shift();
            }
          } else {
            // Update last point
            newData[newData.length - 1] = {
              ...lastPoint,
              actual: price,
              time: timeStr,
            };
          }
        }
        return newData;
      });
    }
  }, []);

  // Start realtime mode - show live prices without prediction
  const startRealtime = useCallback(async () => {
    try {
      console.log('🔄 Starting realtime mode for', symbol, interval);
      setError(null);
      setIsLoading(true);
      setMetrics(null);
      setFuturePredictions([]);
      setTrend(null);
      
      // Fetch historical candles
      const candles = await binanceService.getKlines(symbol, interval, 100);
      console.log('✅ Fetched', candles.length, 'candles');
      historicalCandles.current = candles;
      
      // Create chart data with only actual prices (no predictions)
      const newChartData: ChartDataPoint[] = candles.slice(-50).map(candle => ({
        timestamp: candle.timestamp,
        actual: candle.close,
        time: formatTime(candle.timestamp),
      }));
      
      setChartData(newChartData);
      setCurrentPrice(candles[candles.length - 1].close);
      setIsLoading(false);
      
      // Subscribe to realtime updates
      binanceService.subscribeToPrice(symbol, handleRealtimePriceUpdate);
      wsConnected.current = true;
      
      console.log('✅ Realtime mode started - showing live prices');
    } catch (err) {
      console.error('❌ Error starting realtime:', err);
      setError('Failed to load market data. Please try again.');
      setIsLoading(false);
    }
  }, [symbol, interval, handleRealtimePriceUpdate]);

  // Stop realtime updates
  const stopRealtime = useCallback(() => {
    if (wsConnected.current) {
      binanceService.unsubscribe();
      wsConnected.current = false;
      console.log('⏹️ Realtime stopped');
    }
  }, []);

  // PREDICTION MODE 

  // Fetch predictions from API (core function)
  const fetchPredictions = useCallback(async () => {
    try {
      console.log('🔮 Fetching predictions with', model, 'for', horizon);
      
      // Fetch fresh candles
      const candles = await binanceService.getKlines(symbol, interval, 300);
      historicalCandles.current = candles;
      
      // Get current price - this will be frozen
      const latestPrice = candles[candles.length - 1].close;
      setCurrentPrice(latestPrice);
      
      // Get future predictions from API
      console.log('📡 Calling API /predict/future...');
      const predictions = await predictionService.predictFuture(
        candles, 
        model, 
        horizon, 
        interval
      );
      
      console.log('📡 API response:', predictions?.length, 'predictions');
      
      if (predictions && predictions.length > 0) {
        setFuturePredictions(predictions);
        
        // Determine trend
        const lastPrediction = predictions[predictions.length - 1];
        const priceChange = ((lastPrediction.price - latestPrice) / latestPrice) * 100;
        
        if (priceChange > 0.5) {
          setTrend('up');
        } else if (priceChange < -0.5) {
          setTrend('down');
        } else {
          setTrend('neutral');
        }
        
        console.log(`📊 Trend determined, Change: ${priceChange.toFixed(2)}%`);
      } else {
        console.log('⚠️ No predictions returned from API');
      }
      
      // Update chart data with historical prices (frozen at this point)
      const newChartData: ChartDataPoint[] = candles.slice(-50).map(candle => ({
        timestamp: candle.timestamp,
        actual: candle.close,
        time: formatTime(candle.timestamp),
      }));
      
      setChartData(newChartData);
      return true;
      
    } catch (err) {
      console.error('❌ Error fetching predictions:', err);
      return false;
    }
  }, [symbol, interval, model, horizon]);

  // Start prediction mode
  const startPrediction = useCallback(async () => {
    try {
      console.log('🔮 Starting prediction mode with model:', model, 'horizon:', horizon);
      setError(null);
      setIsLoading(true);
      
      // Stop realtime updates first - freeze the chart
      stopRealtime();
      
      // Check API status
      const apiAvailable = await predictionService.checkApiAvailability();
      if (!apiAvailable) {
        setError('⚠️ Prediction API is offline. Please start the API server first.');
        setIsLoading(false);
        return;
      }
      
      // Change to prediction view mode
      setViewMode('prediction');
      
      // Initial prediction - fetch immediately
      const success = await fetchPredictions();
      
      if (!success) {
        setError('⚠️ Failed to get predictions from API.');
        setIsLoading(false);
        return;
      }
      
      // Now set isPredicting for periodic updates
      setIsPredicting(true);
      setIsLoading(false);
      
      // Start periodic updates (every 1 minute) - but don't update actual prices
      predictionIntervalRef.current = window.setInterval(async () => {
        console.log('🔄 Periodic prediction update...');
        await fetchPredictions();
      }, PREDICTION_UPDATE_INTERVAL);
      
      console.log('✅ Prediction mode started - chart frozen, trendline updating every 1 minute');
    } catch (err) {
      console.error('❌ Error starting prediction:', err);
      setError('Failed to start prediction. Please try again.');
      setIsLoading(false);
      setIsPredicting(false);
    }
  }, [model, horizon, fetchPredictions, stopRealtime]);

  // Stop prediction mode
  const stopPrediction = useCallback(() => {
    console.log('⏹️ Stopping prediction mode');
    setIsPredicting(false);
    setTrend(null);
    setFuturePredictions([]);
    
    if (predictionIntervalRef.current !== null) {
      window.clearInterval(predictionIntervalRef.current);
      predictionIntervalRef.current = null;
    }
  }, []);

  // Switch back to realtime mode
  const backToRealtime = useCallback(() => {
    console.log('🔄 Switching back to realtime mode');
    stopPrediction();
    setViewMode('realtime');
    setMetrics(null);
    startRealtime();
  }, [stopPrediction, startRealtime]);

  //  EFFECTS ====================

  // Subscribe to API status changes
  useEffect(() => {
    const unsubscribe = predictionService.onStatusChange((status) => {
      setApiStatus(status);
    });
    
    // Initial check
    predictionService.checkApiAvailability();
    
    return unsubscribe;
  }, []);

  // Start realtime on mount
  useEffect(() => {
    startRealtime();
    
    return () => {
      stopRealtime();
      if (predictionIntervalRef.current !== null) {
        window.clearInterval(predictionIntervalRef.current);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // When symbol or interval changes, restart
  useEffect(() => {
    if (viewMode === 'realtime') {
      stopRealtime();
      startRealtime();
    } else if (isPredicting) {
      // Restart prediction with new symbol/interval
      fetchPredictions();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, interval]);

  // Update predictions when model or horizon changes while predicting
  useEffect(() => {
    if (isPredicting) {
      fetchPredictions();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, horizon]);

  return {
    // Settings
    symbol,
    interval,
    model,
    horizon,
    viewMode,
    
    // States
    isLoading,
    isPredicting,
    chartData,
    futurePredictions,
    metrics,
    currentPrice,
    error,
    apiStatus,
    trend,
    
    // Setters
    setSymbol,
    setInterval,
    setModel,
    setHorizon,
    
    // Actions
    startPrediction,
    stopPrediction,
    backToRealtime,
  };
};
