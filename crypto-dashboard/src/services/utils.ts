import { ChartDataPoint } from '../types';

/**
 * Calculate Mean Absolute Error
 */
export function calculateMAE(
  actual: number[],
  predicted: number[]
): number {
  if (actual.length !== predicted.length || actual.length === 0) {
    return 0;
  }

  const sum = actual.reduce((acc, val, idx) => {
    return acc + Math.abs(val - predicted[idx]);
  }, 0);

  return sum / actual.length;
}

/**
 * Calculate Root Mean Squared Error
 */
export function calculateRMSE(
  actual: number[],
  predicted: number[]
): number {
  if (actual.length !== predicted.length || actual.length === 0) {
    return 0;
  }

  const sumSquared = actual.reduce((acc, val, idx) => {
    return acc + Math.pow(val - predicted[idx], 2);
  }, 0);

  return Math.sqrt(sumSquared / actual.length);
}

/**
 * Calculate direction accuracy 
 */
export function calculateDirectionAccuracy(
  data: ChartDataPoint[]
): number {
  let correct = 0;
  let total = 0;

  for (let i = 1; i < data.length; i++) {
    const prev = data[i - 1];
    const curr = data[i];

    if (
      prev.actual !== undefined &&
      prev.predicted !== undefined &&
      curr.actual !== undefined &&
      curr.predicted !== undefined
    ) {
      const actualDirection = curr.actual - prev.actual;
      const predictedDirection = curr.predicted - prev.predicted;

      if (
        (actualDirection > 0 && predictedDirection > 0) ||
        (actualDirection < 0 && predictedDirection < 0)
      ) {
        correct++;
      }
      total++;
    }
  }

  return total > 0 ? (correct / total) * 100 : 0;
}

/**
 * Format large numbers with K, M, B suffixes
 */
export function formatNumber(num: number, decimals: number = 2): string {
  if (num >= 1e9) {
    return (num / 1e9).toFixed(decimals) + 'B';
  }
  if (num >= 1e6) {
    return (num / 1e6).toFixed(decimals) + 'M';
  }
  if (num >= 1e3) {
    return (num / 1e3).toFixed(decimals) + 'K';
  }
  return num.toFixed(decimals);
}

/**
 * Format timestamp to readable time string
 */
export function formatTime(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Format timestamp to date string
 */
export function formatDate(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}
