import React from 'react';
import { Metrics } from '../types';
import { formatNumber } from '../services/utils';
import { useLanguage } from '../contexts/LanguageContext';

interface MetricsPanelProps {
  metrics: Metrics | null;
  currentPrice: number | null;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({
  metrics,
  currentPrice,
}) => {
  const { t } = useLanguage();

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-bold text-gray-800 mb-4">
        {t.performanceMetrics}
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Current Price */}
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="text-sm text-gray-600 mb-1">{t.currentPrice}</div>
          <div className="text-2xl font-bold text-blue-600">
            {currentPrice ? `$${formatNumber(currentPrice, 2)}` : '---'}
          </div>
        </div>

        {/* MAE */}
        <div className="bg-purple-50 rounded-lg p-4">
          <div className="text-sm text-gray-600 mb-1">{t.mae}</div>
          <div className="text-2xl font-bold text-purple-600">
            {metrics ? formatNumber(metrics.mae, 2) : '---'}
          </div>
          <div className="text-xs text-gray-500 mt-1">{t.meanAbsoluteError}</div>
        </div>

        {/* RMSE */}
        <div className="bg-orange-50 rounded-lg p-4">
          <div className="text-sm text-gray-600 mb-1">{t.rmse}</div>
          <div className="text-2xl font-bold text-orange-600">
            {metrics ? formatNumber(metrics.rmse, 2) : '---'}
          </div>
          <div className="text-xs text-gray-500 mt-1">{t.rootMeanSquaredError}</div>
        </div>

        {/* Direction Accuracy */}
        <div className="bg-green-50 rounded-lg p-4">
          <div className="text-sm text-gray-600 mb-1">{t.directionAccuracy}</div>
          <div className="text-2xl font-bold text-green-600">
            {metrics ? `${metrics.directionAccuracy.toFixed(1)}%` : '---'}
          </div>
          <div className="text-xs text-gray-500 mt-1">{t.predictionAccuracy}</div>
        </div>
      </div>

      {metrics && (
        <div className="mt-4 text-xs text-gray-500 text-right">
          {t.lastUpdated}: {metrics.lastUpdated.toLocaleTimeString()}
        </div>
      )}
    </div>
  );
};