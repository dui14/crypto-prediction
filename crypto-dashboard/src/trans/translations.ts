export const translations = {
  en: {
    // Header
    title: 'Crypto Price Prediction Dashboard',
    subtitle: 'Real-time cryptocurrency price prediction using machine learning models',
    
    // Control Panel
    configPanel: 'Configuration Panel',
    token: 'Token',
    interval: 'Interval',
    model: 'Model',
    actions: 'Actions',
    startPrediction: 'Start Prediction',
    stopPrediction: 'Stop Prediction',
    minutes: 'minutes',
    
    // Metrics
    performanceMetrics: 'Performance Metrics',
    currentPrice: 'Current Price',
    mae: 'MAE',
    rmse: 'RMSE',
    directionAccuracy: 'Direction Accuracy',
    meanAbsoluteError: 'Mean Absolute Error',
    rootMeanSquaredError: 'Root Mean Squared Error',
    predictionAccuracy: 'Prediction Accuracy',
    lastUpdated: 'Last updated',
    
    // Chart
    liveMarketAnalysis: 'Live Market Analysis',
    comparingPrices: 'Comparing real-time prices vs. predictions',
    actualPrice: 'Actual Price',
    predictedPrice: 'Predicted Price',
    actual: 'Actual',
    predicted: 'Predicted',
    startToSeeChart: 'Click "Start Prediction" button to begin',
    
    // Info Section
    howItWorks: 'How It Works',
    step1Title: '1. Select Configuration',
    step1Desc: 'Choose your token, time interval (5m or 15m), and ML model for predictions.',
    step2Title: '2. Start Prediction',
    step2Desc: 'Click "Start Prediction" to fetch real-time data from Binance and begin analysis.',
    step3Title: '3. Monitor Performance',
    step3Desc: 'Watch the green line (actual price) vs red line (predicted price) and track accuracy metrics.',
    
    // Footer
    dataSource: 'Data sourced from Binance API • Models: LightGBM, XGBoost, LSTM, GRU, CatBoost',
    disclaimer: '⚠️ This is for educational purposes only. Not financial advice.',
    
    // Errors
    errorLoadingData: 'Failed to load market data. Please try again.',
    errorStartingPrediction: 'Failed to start prediction. Please try again.',
    
    // Additional UI text
    apiConnected: 'API Connected',
    checking: 'Checking...',
    apiOffline: 'API Offline',
    runAPIServer: 'Run: python src/api/prediction_api.py',
    live: 'Live',
    predicting: 'Predicting...',
    trendUp: 'Uptrend',
    trendDown: 'Downtrend',
    trendNeutral: 'Neutral',
    liveMode: '🔴 Live Mode',
    predictionMode: '📊 Prediction Mode',
    currentPriceDisplay: 'Current Price',
    predictingWith: 'Predicting with',
    updatedEveryMinute: 'Updated ~every 1 minute',
    liveMarketTitle: 'Live Mode',
    chooseModelHorizonTitle: 'Choose Model & Horizon',
    chooseModelHorizonDesc: 'Select model (LightGBM, XGBoost, CatBoost) and prediction timeframe (4h, 1d, 3d).',
    startPredictionTitle: 'Start Prediction',
    startPredictionDesc: 'Click "Start Prediction" to view price trend predictions.',
    autoUpdateTitle: 'Auto Update',
    autoUpdateDesc: 'Prediction trendline updates automatically every ~1 minute with new data.',
    liveMarketDesc: 'View real-time prices from Binance WebSocket.',
    predictionForLabel: 'Predict for',
    disabledModel: '(Disabled)',
    modelDisabledWarning: '⚠️ This model is currently disabled',
    modelLoading: 'Loading...',
    backToLive: 'Back to Live',
    apiOfflineWarning: 'API offline - Start the prediction server first',
    infoMessagePredicting: 'Running prediction with',
    infoMessageFor: 'for',
    infoMessageTrendline: 'Trendline will update automatically every ~1 minute.',
    infoMessagePredictionMode: '📊 Prediction mode: Showing prediction results vs actual price.',
    infoMessageLiveMode: '🔴 Live mode: Real-time price. Select model, horizon and click "Start Prediction" to view predictions.',
    
    // Price Chart
    currentPriceLabel: 'Current:',
    predictedPriceLabel: 'Predicted:',
    changeLabel: 'Change:',
    predictionSummary: 'Prediction Summary',
    futureAreaName: 'Prediction Zone',
    now: 'Now →',
    
    // Horizon Labels
    horizon5m: '5 Minutes',
    horizon15m: '15 Minutes',
    horizon1h: '1 Hour',
    horizon4h: '4 Hours',
    horizon1d: '1 Day',
    horizon3d: '3 Days',
    
    // Horizon Descriptions
    horizonDesc5m: 'Predict 5 minutes ahead',
    horizonDesc15m: 'Predict 15 minutes ahead',
    horizonDesc1h: 'Predict 1 hour ahead',
    horizonDesc4h: 'Predict 4 hours ahead',
    horizonDesc1d: 'Predict 1 day ahead',
    horizonDesc3d: 'Predict 3 days ahead',
  },
  
  vi: {
    // Header
    title: 'Bảng Điều Khiển Dự Đoán Giá Crypto',
    subtitle: 'Dự đoán giá tiền điện tử theo thời gian thực bằng các mô hình học máy',
    
    // Control Panel
    configPanel: 'Bảng Cấu Hình',
    token: 'Token',
    interval: 'Khoảng Thời Gian',
    model: 'Mô Hình',
    actions: 'Hành Động',
    startPrediction: 'Bắt Đầu Dự Đoán',
    stopPrediction: 'Dừng Dự Đoán',
    minutes: 'phút',
    
    // Metrics
    performanceMetrics: 'Chỉ Số Hiệu Suất',
    currentPrice: 'Giá Hiện Tại',
    mae: 'MAE',
    rmse: 'RMSE',
    directionAccuracy: 'Độ Chính Xác Xu Hướng',
    meanAbsoluteError: 'Sai Số Tuyệt Đối Trung Bình',
    rootMeanSquaredError: 'Căn Bậc Hai Sai Số Bình Phương',
    predictionAccuracy: 'Độ Chính Xác Dự Đoán',
    lastUpdated: 'Cập nhật lần cuối',
    
    // Chart
    liveMarketAnalysis: 'Phân Tích Thị Trường Trực Tiếp',
    comparingPrices: 'So sánh giá thực tế với giá dự đoán',
    actualPrice: 'Giá Thực Tế',
    predictedPrice: 'Giá Dự Đoán',
    actual: 'Thực Tế',
    predicted: 'Dự Đoán',
    startToSeeChart: 'Nhấn nút "Bắt Đầu Dự Đoán" để bắt đầu',
    
    // Info Section
    howItWorks: 'Cách Hoạt Động',
    step1Title: '1. Chọn Cấu Hình',
    step1Desc: 'Chọn token, khoảng thời gian (5 phút hoặc 15 phút) và mô hình ML để dự đoán.',
    step2Title: '2. Bắt Đầu Dự Đoán',
    step2Desc: 'Nhấn "Bắt Đầu Dự Đoán" để lấy dữ liệu thời gian thực từ Binance và bắt đầu phân tích.',
    step3Title: '3. Theo Dõi Hiệu Suất',
    step3Desc: 'Xem đường màu xanh (giá thực tế) so với đường màu đỏ (giá dự đoán) và theo dõi các chỉ số độ chính xác.',
    
    // Footer
    dataSource: 'Dữ liệu từ Binance API • Mô hình: LightGBM, XGBoost, LSTM, GRU, CatBoost',
    disclaimer: '⚠️ Chỉ cho mục đích giáo dục. Không phải lời khuyên tài chính.',
    
    // Errors
    errorLoadingData: 'Không thể tải dữ liệu thị trường. Vui lòng thử lại.',
    errorStartingPrediction: 'Không thể bắt đầu dự đoán. Vui lòng thử lại.',
    
    // Additional UI text
    apiConnected: 'API Kết Nối',
    checking: 'Đang Kiểm Tra...',
    apiOffline: 'API Ngoại Tuyến',
    runAPIServer: 'Chạy: python src/api/prediction_api.py',
    live: 'Trực Tiếp',
    predicting: 'Đang Dự Đoán...',
    trendUp: 'Xu Hướng Tăng',
    trendDown: 'Xu Hướng Giảm',
    trendNeutral: 'Bình Thường',
    liveMode: '🔴 Chế Độ Trực Tiếp',
    predictionMode: '📊 Chế Độ Dự Đoán',
    currentPriceDisplay: 'Giá Hiện Tại',
    predictingWith: 'Đang dự đoán với',
    updatedEveryMinute: 'Cập nhật khoảng ~1 phút',
    liveMarketTitle: 'Chế Độ Trực Tiếp',
    chooseModelHorizonTitle: 'Chọn Mô Hình & Horizon',
    chooseModelHorizonDesc: 'Chọn mô hình (LightGBM, XGBoost, CatBoost) và khung thời gian dự đoán (4h, 1d, 3d).',
    startPredictionTitle: 'Bắt Đầu Dự Đoán',
    startPredictionDesc: 'Nhấn "Bắt Đầu Dự Đoán" để xem trendline dự đoán giá tương lai.',
    autoUpdateTitle: 'Cập Nhật Tự Động',
    autoUpdateDesc: 'Trendline dự đoán tự động cập nhật khoảng ~1 phút với dữ liệu mới.',
    liveMarketDesc: 'Xem giá realtime từ Binance WebSocket.',
    predictionForLabel: 'Dự đoán cho',
    disabledModel: '(Bị Vô Hiệu Hóa)',
    modelDisabledWarning: '⚠️ Mô hình này hiện đang bị vô hiệu hóa',
    modelLoading: 'Đang Tải...',
    backToLive: 'Quay Lại',
    apiOfflineWarning: 'API ngoại tuyến - Hãy khởi động máy chủ dự đoán trước',
    infoMessagePredicting: 'Chạy dự đoán với',
    infoMessageFor: 'cho',
    infoMessageTrendline: 'Trendline sẽ cập nhật tự động mỗi ~1 phút.',
    infoMessagePredictionMode: '📊 Chế độ dự đoán: Đang hiển thị kết quả dự đoán so với giá thực.',
    infoMessageLiveMode: '🔴 Chế độ trực tiếp: Giá realtime. Chọn mô hình, horizon và nhấn "Bắt Đầu Dự Đoán" để xem dự đoán.',
    
    // Price Chart
    currentPriceLabel: 'Giá hiện tại:',
    predictedPriceLabel: 'Giá dự đoán:',
    changeLabel: 'Thay đổi:',
    predictionSummary: 'Tóm tắt dự đoán',
    futureAreaName: 'Vùng dự đoán',
    now: 'Bây giờ →',
    
    // Horizon Labels
    horizon5m: '5 Phút',
    horizon15m: '15 Phút',
    horizon1h: '1 Giờ',
    horizon4h: '4 Giờ',
    horizon1d: '1 Ngày',
    horizon3d: '3 Ngày',
    
    // Horizon Descriptions
    horizonDesc5m: 'Dự đoán 5 phút sau',
    horizonDesc15m: 'Dự đoán 15 phút sau',
    horizonDesc1h: 'Dự đoán 1 giờ sau',
    horizonDesc4h: 'Dự đoán 4 giờ sau',
    horizonDesc1d: 'Dự đoán 1 ngày sau',
    horizonDesc3d: 'Dự đoán 3 ngày sau',
  },
};

export type Language = 'en' | 'vi';
export type TranslationKeys = typeof translations.en;