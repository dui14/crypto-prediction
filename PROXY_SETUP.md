# Setup Proxy cho Binance API

Nếu Binance API bị chặn ở khu vực của bạn, bạn cần dùng proxy.

## Bước 1: Test xem có bị chặn không

Mở file `test-api.html` trong browser:
```
d:\AppDev\code\crypto\crypto-dashboard\test-api.html
```

Click các buttons:
- "Test REST API" - kiểm tra HTTP API
- "Test WebSocket" - kiểm tra WebSocket

Nếu thấy X là bị chặn → cần proxy

## Bước 2: Nếu có proxy sẵn

Giả sử proxy là: `http://your-proxy.com:8080`

### Cách 1: Sửa file `.env`
```bash
REACT_APP_USE_PROXY=true
REACT_APP_PROXY_URL=http://your-proxy.com:8080
```

### Cách 2: Setup proxy trong package.json
Thêm vào `package.json`:
```json
{
  "proxy": "http://your-proxy.com:8080"
}
```

## Bước 3: Nếu proxy yêu cầu forward requests

Proxy server của bạn cần forward:
- `http://your-proxy.com:8080/api/v3/*` → `https://api.binance.com/api/v3/*`
- WebSocket cần hỗ trợ: `wss://stream.binance.com:9443/ws`

## Bước 4: Hoặc dùng CORS Proxy đơn giản

Cài đặt local proxy:
```bash
npm install -g local-cors-proxy
```

Chạy proxy:
```bash
lcp --proxyUrl https://api.binance.com --port 8080
```

Sau đó set trong `.env`:
```bash
REACT_APP_USE_PROXY=true
REACT_APP_PROXY_URL=http://localhost:8080
```

## Bước 5: Restart app

```bash
npm start
```

## Alternative: Sử dụng setupProxy.js

Tạo file `src/setupProxy.js`:
```javascript
const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'https://api.binance.com',
      changeOrigin: true,
      pathRewrite: {
        '^/api': '/api/v3',
      },
    })
  );
};
```

Sau đó update `binanceService.ts`:
```typescript
const BINANCE_API_BASE = '/api';
```

## Kiểm tra logs

Khi app chạy, mở Console (F12) sẽ thấy:
```
🔧 Binance Service Config: {
  USE_PROXY: true/false,
  BINANCE_API_BASE: "...",
  BINANCE_WS_BASE: "..."
}
```

## Nếu vẫn không hoạt động

1. Check firewall
2. Check antivirus
3. Thử VPN
