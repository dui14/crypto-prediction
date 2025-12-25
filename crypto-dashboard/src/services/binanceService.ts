import axios from 'axios';
import { Candle, TimeInterval } from '../types';

// Có thể config proxy nếu cần
const USE_PROXY = process.env.REACT_APP_USE_PROXY === 'true';
const PROXY_URL = process.env.REACT_APP_PROXY_URL || 'http://localhost:8080';

const BINANCE_API_BASE = USE_PROXY 
  ? `${PROXY_URL}/api/v3` 
  : (process.env.REACT_APP_BINANCE_API || 'https://api.binance.com/api/v3');
const BINANCE_WS_BASE = process.env.REACT_APP_BINANCE_WS || 'wss://stream.binance.com:9443/ws';

console.log('🔧 Binance Service Config:', {
  USE_PROXY,
  BINANCE_API_BASE,
  BINANCE_WS_BASE
});

export class BinanceService {
  private ws: WebSocket | null = null;
  private onPriceUpdate: ((price: number) => void) | null = null;

  /**
   * Fetch historical candlestick data from Binance
   */
  async getKlines(
    symbol: string,
    interval: TimeInterval,
    limit: number = 100
  ): Promise<Candle[]> {
    try {
      const response = await axios.get(`${BINANCE_API_BASE}/klines`, {
        params: {
          symbol: symbol.toUpperCase(),
          interval,
          limit,
        },
      });

      return response.data.map((kline: any[]) => ({
        timestamp: kline[0],
        open: parseFloat(kline[1]),
        high: parseFloat(kline[2]),
        low: parseFloat(kline[3]),
        close: parseFloat(kline[4]),
        volume: parseFloat(kline[5]),
      }));
    } catch (error) {
      console.error('Error fetching klines:', error);
      throw new Error('Failed to fetch market data from Binance');
    }
  }

  /**
   * Get current price for a symbol
   */
  async getCurrentPrice(symbol: string): Promise<number> {
    try {
      const response = await axios.get(`${BINANCE_API_BASE}/ticker/price`, {
        params: {
          symbol: symbol.toUpperCase(),
        },
      });
      return parseFloat(response.data.price);
    } catch (error) {
      console.error('Error fetching current price:', error);
      throw new Error('Failed to fetch current price');
    }
  }

  /**
   * real-time price updates via WebSocket
   */
  subscribeToPrice(
    symbol: string,
    onUpdate: (price: number) => void
  ): void {
    this.onPriceUpdate = onUpdate;
    const wsSymbol = symbol.toLowerCase();
    const wsUrl = `${BINANCE_WS_BASE}/${wsSymbol}@trade`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const price = parseFloat(data.p);
        if (this.onPriceUpdate) {
          this.onPriceUpdate(price);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket connection closed');
    };
  }

  /**
   * Unsubscribe from price updates and close WebSocket
   */
  unsubscribe(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.onPriceUpdate = null;
  }

  /**
   * Check if WebSocket is connected
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

export const binanceService = new BinanceService();
