import React from 'react';
import { ModelType, TimeInterval, ViewMode, PredictionHorizon } from '../types';
import { useLanguage } from '../contexts/LanguageContext';
import { ApiStatus } from '../services/predictionService';

interface ControlPanelProps {
  symbol: string;
  interval: TimeInterval;
  model: ModelType;
  horizon: PredictionHorizon;
  viewMode: ViewMode;
  isLoading: boolean;
  isPredicting: boolean;
  apiStatus: ApiStatus;
  trend: 'up' | 'down' | 'neutral' | null;
  onSymbolChange: (symbol: string) => void;
  onIntervalChange: (interval: TimeInterval) => void;
  onModelChange: (model: ModelType) => void;
  onHorizonChange: (horizon: PredictionHorizon) => void;
  onStartPrediction: () => void;
  onStopPrediction: () => void;
  onBackToRealtime: () => void;
}

const POPULAR_SYMBOLS = [
  'BTCUSDT',
  'ETHUSDT',
  'BNBUSDT',
  'ADAUSDT',
  'SOLUSDT',
  'DOTUSDT',
  'MATICUSDT',
  'LINKUSDT',
];

const MODELS: { name: ModelType; enabled: boolean; note?: string }[] = [
  { name: 'LightGBM', enabled: true },
  { name: 'XGBoost', enabled: true },
  { name: 'CatBoost', enabled: true },
  { name: 'LSTM', enabled: true },
  { name: 'GRU', enabled: true },
];

// HORIZONS will be dynamically built in component using translations

export const ControlPanel: React.FC<ControlPanelProps> = ({
  symbol,
  interval,
  model,
  horizon,
  viewMode,
  isLoading,
  isPredicting,
  apiStatus,
  trend,
  onSymbolChange,
  onIntervalChange,
  onModelChange,
  onHorizonChange,
  onStartPrediction,
  onStopPrediction,
  onBackToRealtime,
}) => {
  const { t } = useLanguage();
  const isPredictionMode = viewMode === 'prediction';

  // Dynamically build HORIZONS with translations
  const HORIZONS = [
    { value: '5m' as PredictionHorizon, label: t.horizon5m, description: t.horizonDesc5m },
    { value: '15m' as PredictionHorizon, label: t.horizon15m, description: t.horizonDesc15m },
    { value: '1h' as PredictionHorizon, label: t.horizon1h, description: t.horizonDesc1h },
    { value: '4h' as PredictionHorizon, label: t.horizon4h, description: t.horizonDesc4h },
    { value: '1d' as PredictionHorizon, label: t.horizon1d, description: t.horizonDesc1d },
    { value: '3d' as PredictionHorizon, label: t.horizon3d, description: t.horizonDesc3d },
  ];

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-gray-800">
          {t.configPanel}
        </h2>
        {/* View Mode Badge */}
        <div className="flex items-center gap-3">
          {isPredicting && trend && (
            <div className={`px-3 py-1 rounded-full text-sm font-medium flex items-center gap-1 ${
              trend === 'up' ? 'bg-green-100 text-green-700' : 
              trend === 'down' ? 'bg-red-100 text-red-700' : 
              'bg-gray-100 text-gray-700'
            }`}>
              {trend === 'up' ? '📈 ' + t.trendUp : 
               trend === 'down' ? '📉 ' + t.trendDown : 
               '➡️ ' + t.trendNeutral}
            </div>
          )}
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            isPredictionMode 
              ? 'bg-purple-100 text-purple-700' 
              : 'bg-green-100 text-green-700'
          }`}>
            {isPredicting ? t.predicting : isPredictionMode ? t.predictionMode : t.liveMode}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
        {/* Token Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            💰 {t.token}
          </label>
          <select
            value={symbol}
            onChange={(e) => onSymbolChange(e.target.value)}
            disabled={isPredicting || isLoading}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
          >
            {POPULAR_SYMBOLS.map((sym) => (
              <option key={sym} value={sym}>
                {sym.replace('USDT', '/USDT')}
              </option>
            ))}
          </select>
        </div>

        {/* Interval Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            ⏱️ {t.interval}
          </label>
          <select
            value={interval}
            onChange={(e) => onIntervalChange(e.target.value as TimeInterval)}
            disabled={isPredicting || isLoading}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
          >
            <option value="5m">5 {t.minutes}</option>
            <option value="15m">15 {t.minutes}</option>
          </select>
        </div>

        {/* Model Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            🤖 {t.model}
          </label>
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value as ModelType)}
            disabled={isPredicting || isLoading}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
          >
            {MODELS.map((m) => (
              <option 
                key={m.name} 
                value={m.name}
                disabled={!m.enabled}
              >
                {m.name} {!m.enabled && t.disabledModel}
              </option>
            ))}
          </select>
          {MODELS.find(m => m.name === model && !m.enabled) && (
            <p className="text-xs text-red-500 mt-1">
              {t.modelDisabledWarning}
            </p>
          )}
        </div>

        {/* Prediction Horizon Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            🎯 {t.predictionForLabel}
          </label>
          <select
            value={horizon}
            onChange={(e) => onHorizonChange(e.target.value as PredictionHorizon)}
            disabled={isPredicting || isLoading}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
          >
            {HORIZONS.map((h) => (
              <option key={h.value} value={h.value}>
                {h.label}
              </option>
            ))}
          </select>
        </div>

        {/* Start/Stop Prediction Button */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            {t.actions}
          </label>
          {!isPredicting ? (
            <button
              onClick={onStartPrediction}
              disabled={isLoading || apiStatus !== 'connected'}
              className={`w-full px-4 py-2 font-semibold rounded-lg transition-colors duration-200 flex items-center justify-center ${
                isLoading || apiStatus !== 'connected'
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-purple-600 text-white hover:bg-purple-700'
              }`}
              title={apiStatus !== 'connected' ? t.apiOfflineWarning : ''}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  {t.modelLoading}
                </>
              ) : (
                <>
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  {t.startPrediction}
                </>
              )}
            </button>
          ) : (
            <button
              onClick={onStopPrediction}
              className="w-full px-4 py-2 font-semibold rounded-lg transition-colors duration-200 flex items-center justify-center bg-red-600 text-white hover:bg-red-700"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
              </svg>
              {t.stopPrediction}
            </button>
          )}
        </div>

        {/* Back to Realtime Button */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            &nbsp;
          </label>
          <button
            onClick={onBackToRealtime}
            disabled={viewMode === 'realtime' && !isPredicting}
            className={`w-full px-4 py-2 font-semibold rounded-lg transition-colors duration-200 flex items-center justify-center ${
              viewMode === 'realtime' && !isPredicting
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-green-600 text-white hover:bg-green-700'
            }`}
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            {t.backToLive}
          </button>
        </div>
      </div>

      {/* Info message */}
      <div className="mt-4 text-sm text-gray-500">
        {isPredicting ? (
          <p>{t.infoMessagePredicting} <strong>{model}</strong> {t.infoMessageFor} <strong>{HORIZONS.find(h => h.value === horizon)?.label}</strong>. {t.infoMessageTrendline}</p>
        ) : isPredictionMode ? (
          <p>{t.infoMessagePredictionMode}</p>
        ) : (
          <p>{t.infoMessageLiveMode}</p>
        )}
      </div>
    </div>
  );
};