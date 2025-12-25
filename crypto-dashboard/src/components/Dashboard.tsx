import React from 'react';
import { ControlPanel } from './ControlPanel';
import { PriceChart } from './PriceChart';
import { MetricsPanel } from './MetricsPanel';
import { LanguageSelector } from '../contexts/LanguageSelector';
import { usePrediction } from '../hooks/usePrediction';
import { useLanguage } from '../contexts/LanguageContext';

export const Dashboard: React.FC = () => {
  const {
    symbol,
    interval,
    model,
    horizon,
    viewMode,
    isLoading,
    isPredicting,
    chartData,
    futurePredictions,
    metrics,
    currentPrice,
    error,
    apiStatus,
    trend,
    setSymbol,
    setInterval,
    setModel,
    setHorizon,
    startPrediction,
    stopPrediction,
    backToRealtime,
  } = usePrediction();

  const { t } = useLanguage();

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-blue-800 text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold">{t.title}</h1>
              <p className="text-blue-100 mt-2">{t.subtitle}</p>
            </div>
            <div className="flex items-center gap-4">
              {/* API Status Indicator */}
              <div className="flex items-center gap-2 bg-white/10 px-3 py-1.5 rounded-lg">
                <div 
                  className={`w-2.5 h-2.5 rounded-full ${
                    apiStatus === 'connected' 
                      ? 'bg-green-400 animate-pulse' 
                      : apiStatus === 'checking'
                      ? 'bg-yellow-400 animate-pulse'
                      : 'bg-red-400'
                  }`}
                />
                <span className="text-sm font-medium">
                  {apiStatus === 'connected' 
                    ? t.apiConnected
                    : apiStatus === 'checking'
                    ? t.checking
                    : t.apiOffline}
                </span>
              </div>
              <LanguageSelector />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Error/Warning Message */}
        {error && (
          <div className={`${
            error.includes('⚠️') 
              ? 'bg-yellow-100 border-yellow-400 text-yellow-700' 
              : 'bg-red-100 border-red-400 text-red-700'
          } border px-4 py-3 rounded-lg mb-6 flex items-center`}>
            <svg
              className="w-5 h-5 mr-2"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              {error.includes('⚠️') ? (
                <path
                  fillRule="evenodd"
                  d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                  clipRule="evenodd"
                />
              ) : (
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                  clipRule="evenodd"
                />
              )}
            </svg>
            <span>{error.replace('⚠️ ', '')}</span>
            {apiStatus === 'disconnected' && (
              <span className="ml-2 text-sm opacity-75">
                ({t.runAPIServer})
              </span>
            )}
          </div>
        )}

        {/* Control Panel */}
        <ControlPanel
          symbol={symbol}
          interval={interval}
          model={model}
          horizon={horizon}
          viewMode={viewMode}
          isLoading={isLoading}
          isPredicting={isPredicting}
          apiStatus={apiStatus}
          trend={trend}
          onSymbolChange={setSymbol}
          onIntervalChange={setInterval}
          onModelChange={setModel}
          onHorizonChange={setHorizon}
          onStartPrediction={startPrediction}
          onStopPrediction={stopPrediction}
          onBackToRealtime={backToRealtime}
        />

        {/* Metrics Panel - only show in prediction mode */}
        {viewMode === 'prediction' && metrics && (
          <div className="mb-6">
            <MetricsPanel metrics={metrics} currentPrice={currentPrice} />
          </div>
        )}

        {/* Current Price Display - for realtime mode */}
        {viewMode === 'realtime' && currentPrice && !isPredicting && (
          <div className="mb-6 bg-white rounded-lg shadow-md p-4">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-gray-500 text-sm">{symbol} {t.currentPriceDisplay}</span>
                <div className="text-2xl font-bold text-gray-800">
                  ${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
              </div>
              <div className="flex items-center text-green-500">
                <span className="w-3 h-3 bg-green-500 rounded-full animate-pulse mr-2"></span>
                {t.live}
              </div>
            </div>
          </div>
        )}

        {/* Prediction Info - when predicting */}
        {isPredicting && currentPrice && (
          <div className="mb-6 bg-gradient-to-r from-purple-500 to-indigo-600 rounded-lg shadow-md p-4 text-white">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-purple-100 text-sm">{symbol} - {t.predictingWith} {model}</span>
                <div className="text-2xl font-bold">
                  ${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
              </div>
              <div className="flex flex-col items-end">
                <div className={`flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                  trend === 'up' ? 'bg-green-400/20 text-green-100' :
                  trend === 'down' ? 'bg-red-400/20 text-red-100' :
                  'bg-gray-400/20 text-gray-100'
                }`}>
                  {trend === 'up' ? '📈 ' + t.trendUp : trend === 'down' ? '📉 ' + t.trendDown : '➡️ ' + t.trendNeutral}
                </div>
                <div className="flex items-center mt-2 text-purple-200 text-sm">
                  <span className="w-2 h-2 bg-purple-300 rounded-full animate-pulse mr-2"></span>
                  {t.updatedEveryMinute}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Price Chart */}
        <PriceChart 
          data={chartData} 
          futurePredictions={futurePredictions}
          isRunning={viewMode === 'prediction'} 
          isPredicting={isPredicting}
          currentPrice={currentPrice}
          horizon={horizon}
          trend={trend}
        />

        {/* Info Section */}
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">
            {t.howItWorks}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm text-blue-800">
            <div>
              <strong>{t.liveMarketTitle}</strong>
              <p className="mt-1">{t.liveMarketDesc}</p>
            </div>
            <div>
              <strong>{t.chooseModelHorizonTitle}</strong>
              <p className="mt-1">{t.chooseModelHorizonDesc}</p>
            </div>
            <div>
              <strong>{t.startPredictionTitle}</strong>
              <p className="mt-1">{t.startPredictionDesc}</p>
            </div>
            <div>
              <strong>{t.autoUpdateTitle}</strong>
              <p className="mt-1">{t.autoUpdateDesc}</p>
            </div>
          </div>
        </div>

        {/* Footer Info */}
        <div className="mt-6 text-center text-gray-500 text-sm">
          <p>{t.dataSource}</p>
          <p className="mt-1">{t.disclaimer}</p>
        </div>
      </main>
    </div>
  );
};