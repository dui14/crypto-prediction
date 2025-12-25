# Global Copilot Instructions

- Luôn phản hồi bằng tiếng Việt.
- Bạn sẽ là 1 trợ lí AI hướng dẫn tôi tạo và train model dự đoán giá crypto cùng với thiết kế GUI
- Tôi muốn lấy 1 token bất kì thông qua API của Binance (có sẳn) để hiện thị trên chart khi run và trên đó hiển thị đường giá thực màu xanh, đường giá dự đoán màu đỏ, màu x y z theo từng model
- Tôi muốn lấy chỉ 5 phút hoặc 15 phút để vẽ chart (vì lấy nhiều api sẽ làm lag) và lấy dữ liệu trong vòng 1 ngày hoặc 3 ngày tùy chọn
- Thiết kế 1 UI cho phép user chọn các models để dự đoán giá, trong UI có như symbol BTC/USDT, 1 cái select chọn frame cố định là 5 phút hoặc 15 phút, 1 select lịch chọn từ 2020-01-01 to 2025-11-19. Các models như LightGBM, XGBoost, CatBoost, LSTM,GRU