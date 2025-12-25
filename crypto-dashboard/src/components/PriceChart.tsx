import React from 'react';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from 'recharts';
import { ChartDataPoint, FuturePrediction, PredictionHorizon } from '../types';
import { useLanguage } from '../contexts/LanguageContext';

interface PriceChartProps {
  data: ChartDataPoint[];
  futurePredictions: FuturePrediction[];
  isRunning: boolean;
  isPredicting: boolean;
  currentPrice: number | null;
  horizon: PredictionHorizon;
  trend: 'up' | 'down' | 'neutral' | null;
}

export const PriceChart: React.FC<PriceChartProps> = ({ 
  data, 
  futurePredictions, 
  isRunning, 
  isPredicting,
  currentPrice,
  horizon,
  trend 
}) => {
  const { t, language } = useLanguage();

  // Dynamic horizon labels based on language from translations
  const HORIZON_LABELS: Record<PredictionHorizon, string> = {
    '5m': t.horizon5m,
    '15m': t.horizon15m,
    '1h': t.horizon1h,
    '4h': t.horizon4h,
    '1d': t.horizon1d,
    '3d': t.horizon3d,
  };

  // Helper function - defined before useMemo that uses it
  const formatFutureTime = (timestamp: number) => {
    const date = new Date(timestamp);
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    const day = date.getDate();
    const month = date.getMonth() + 1;
    return `${day}/${month} ${hours}:${minutes}`;
  };

  // Debug: Log received props
  React.useEffect(() => {
    console.log('📊 PriceChart props:', {
      dataLength: data?.length,
      futurePredictionsLength: futurePredictions?.length,
      isPredicting,
      currentPrice,
      horizon,
      trend
    });
    if (futurePredictions?.length > 0) {
      console.log('📊 First future prediction:', futurePredictions[0]);
      console.log('📊 Last future prediction:', futurePredictions[futurePredictions.length - 1]);
    }
  }, [data, futurePredictions, isPredicting, currentPrice, horizon, trend]);

  // Combine historical data with future predictions
  const combinedData = React.useMemo(() => {
    if (!data || data.length === 0) return [];
    
    // Historical data
    const historicalData = data.map(d => ({
      ...d,
      type: 'historical',
    }));
    
    // Future predictions
    if (futurePredictions && futurePredictions.length > 0 && isPredicting) {
      console.log('📊 Adding future predictions to chart:', futurePredictions.length);
      const futureData = futurePredictions.map(p => ({
        timestamp: p.timestamp,
        actual: undefined,
        predicted: p.price,
        predictedTrendline: p.price,
        time: formatFutureTime(p.timestamp),
        type: 'future',
      }));
      
      return [...historicalData, ...futureData];
    }
    
    return historicalData;
  }, [data, futurePredictions, isPredicting]);

  // Calculate min/max for Y axis
  const yDomain = React.useMemo(() => {
    if (combinedData.length === 0) return ['auto', 'auto'];
    
    let min = Infinity;
    let max = -Infinity;
    
    combinedData.forEach(d => {
      if (d.actual !== undefined) {
        min = Math.min(min, d.actual);
        max = Math.max(max, d.actual);
      }
      if (d.predicted !== undefined) {
        min = Math.min(min, d.predicted);
        max = Math.max(max, d.predicted);
      }
      if (d.predictedTrendline !== undefined) {
        min = Math.min(min, d.predictedTrendline);
        max = Math.max(max, d.predictedTrendline);
      }
    });
    
    // Add 2% padding
    const padding = (max - min) * 0.02;
    return [min - padding, max + padding];
  }, [combinedData]);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const dataPoint = payload[0].payload;
      const isFuture = dataPoint.type === 'future';
      
      return (
        <div className={`p-4 rounded-lg shadow-lg border ${
          isFuture ? 'bg-purple-50 border-purple-200' : 'bg-white border-gray-200'
        }`}>
          <p className="text-sm text-gray-600 mb-2">
            {isFuture ? t.predictionSummary + ': ' : ''}{dataPoint.time}
          </p>
          {payload.map((entry: any, index: number) => (
            entry.value !== undefined && (
              <p
                key={index}
                className="text-sm font-semibold"
                style={{ color: entry.color }}
              >
                {entry.name}: ${entry.value?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </p>
            )
          ))}
          {isFuture && (
            <p className="text-xs text-purple-600 mt-2">
              {t.predictionSummary} {HORIZON_LABELS[horizon]}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  // Find the index where future predictions start
  const futureStartIndex = combinedData.findIndex(d => d.type === 'future');

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-gray-800">
          {t.liveMarketAnalysis}
        </h2>
        <div className="flex items-center gap-3">
          {isPredicting && (
            <div className={`flex items-center px-3 py-1 rounded-full text-sm font-medium ${
              trend === 'up' ? 'bg-green-100 text-green-700' :
              trend === 'down' ? 'bg-red-100 text-red-700' :
              'bg-gray-100 text-gray-700'
            }`}>
              {trend === 'up' ? '📈' : trend === 'down' ? '📉' : '➡️'}
              <span className="ml-1">
                {trend === 'up' ? t.trendUp : trend === 'down' ? t.trendDown : t.trendNeutral}
              </span>
            </div>
          )}
          {isRunning && (
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse mr-2"></div>
              <span className="text-sm text-green-600 font-medium">
                {isPredicting ? t.predicting : t.comparingPrices}
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="text-sm text-gray-500 mb-4 flex flex-wrap gap-4">
        <span className="inline-flex items-center">
          <span className="w-3 h-3 bg-green-700 rounded-full mr-2"></span>
          {t.actualPrice}
        </span>
        <span className="inline-flex items-center">
          <span className="w-3 h-3 bg-red-500 rounded-full mr-2"></span>
          {t.predictedPrice}
        </span>
        {isPredicting && (
          <span className="inline-flex items-center">
            <span className="w-3 h-3 bg-purple-500 rounded-full mr-2"></span>
            {t.predictionSummary} ({HORIZON_LABELS[horizon]})
          </span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={500}>
        <ComposedChart
          data={combinedData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <defs>
            <linearGradient id="futurePredictionGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="time"
            stroke="#9ca3af"
            style={{ fontSize: '11px' }}
            tick={{ fontSize: 10 }}
            interval="preserveStartEnd"
          />
          <YAxis
            stroke="#9ca3af"
            style={{ fontSize: '12px' }}
            domain={yDomain as [number, number]}
            tickFormatter={(value) => `$${value.toLocaleString()}`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: '14px' }}
            iconType="line"
          />
          
          {/* Current price reference line */}
          {currentPrice && (
            <ReferenceLine
              y={currentPrice}
              stroke="#3b82f6"
              strokeDasharray="5 5"
              label={{ 
                value: `$${currentPrice.toLocaleString()}`, 
                position: 'right',
                fill: '#3b82f6',
                fontSize: 12
              }}
            />
          )}
          
          {/* Vertical line separating historical from future */}
          {futureStartIndex > 0 && (
            <ReferenceLine
              x={combinedData[futureStartIndex - 1]?.time}
              stroke="#8b5cf6"
              strokeDasharray="3 3"
              label={{ 
                value: t.now, 
                position: 'top',
                fill: '#8b5cf6',
                fontSize: 11
              }}
            />
          )}
          
          {/* Actual price line */}
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#047857"
            strokeWidth={2}
            dot={false}
            name={t.actual}
            connectNulls
            isAnimationActive={true}
            animationDuration={300}
          />
          
          {/* Historical predicted line */}
          <Line
            type="monotone"
            dataKey="predicted"
            stroke="#ef4444"
            strokeWidth={2}
            dot={false}
            name={t.predicted}
            connectNulls
            isAnimationActive={true}
            animationDuration={300}
          />
          
          {/* Future prediction trendline */}
          {isPredicting && (
            <>
              <Area
                type="monotone"
                dataKey="predictedTrendline"
                stroke="none"
                fill="url(#futurePredictionGradient)"
                name={t.futureAreaName}
                isAnimationActive={true}
                animationDuration={500}
              />
              <Line
                type="monotone"
                dataKey="predictedTrendline"
                stroke="#8b5cf6"
                strokeWidth={3}
                strokeDasharray="5 5"
                dot={{ fill: '#8b5cf6', r: 3 }}
                name={`${t.predictionSummary} ${HORIZON_LABELS[horizon]}`}
                connectNulls
                isAnimationActive={true}
                animationDuration={500}
              />
            </>
          )}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Future prediction summary */}
      {isPredicting && futurePredictions.length > 0 && currentPrice && (
        <div className="mt-4 p-4 bg-purple-50 rounded-lg">
          <h3 className="text-sm font-semibold text-purple-800 mb-2">
            {t.predictionSummary} {HORIZON_LABELS[horizon]}
          </h3>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-600">{t.currentPriceLabel}</span>
              <span className="ml-2 font-semibold">${currentPrice.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-gray-600">{t.predictedPriceLabel}</span>
              <span className={`ml-2 font-semibold ${
                futurePredictions[futurePredictions.length - 1]?.price > currentPrice 
                  ? 'text-green-600' 
                  : 'text-red-600'
              }`}>
                ${futurePredictions[futurePredictions.length - 1]?.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </span>
            </div>
            <div>
              <span className="text-gray-600">{t.changeLabel}</span>
              <span className={`ml-2 font-semibold ${
                futurePredictions[futurePredictions.length - 1]?.price > currentPrice 
                  ? 'text-green-600' 
                  : 'text-red-600'
              }`}>
                {(((futurePredictions[futurePredictions.length - 1]?.price - currentPrice) / currentPrice) * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      )}

      {data.length === 0 && (
        <div className="mt-8 text-center">
          <p className="text-gray-400 text-lg">
            {t.startToSeeChart}
          </p>
        </div>
      )}
    </div>
  );
};