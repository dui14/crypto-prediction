# Train on Google Colab 

# !pip install requests[socks] pandas tqdm -q

import requests
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm

# PROXY Ex
proxy_host = "snvt3.tunproxy.com"
proxy_port = 53531
proxy_user = "dui14"
proxy_pass = "SXXLjNY9"

proxies = {
    'http':  f'socks5://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}',
    'https': f'socks5://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}'
}

def get_all_klines(symbol="BTCUSDT", interval="5m", start_str="1 Jan, 2020", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_str, "%d %b, %Y").timestamp() * 1000)
    
    all_data = []
    pbar = tqdm(total=550000, desc="Đang tải nến 5m", unit="nến")
    
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ts,
            'limit': limit
        }
        
        for _ in range(3):  # thử lại 3 lần nếu lỗi mạng
            try:
                resp = requests.get(url, params=params, proxies=proxies, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    if not data:
                        pbar.close()
                        print("\nĐã lấy hết dữ liệu!")
                        return all_data
                    
                    batch_size = len(data)
                    all_data.extend(data)
                    pbar.update(batch_size)
                    
                    # Cập nhật thời gian cho lần gọi tiếp theo
                    start_ts = data[-1][0] + 1  # mở cửa nến cuối + 1ms
                    
                    if batch_size < limit:
                        pbar.close()
                        print(f"\nHoàn thành! Đã lấy {len(all_data):,} nến")
                        return all_data
                    time.sleep(0.15) 
                    break
                else:
                    print(f"Lỗi {resp.status_code}, thử lại...")
                    time.sleep(1)
            except Exception as e:
                print("Lỗi mạng, thử lại...", e)
                time.sleep(2)
    pbar.close()
    return all_data

# Bắt đầu tải 
print("Bắt đầu tải ~540.000 cây nến 5m BTCUSDT từ 01/01/2020...")
raw_data = get_all_klines()

# CELL 3: Xử lý dữ liệu và xuất CSV
columns = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades', 'taker_base_vol', 
    'taker_quote_vol', 'ignore'
]

df = pd.DataFrame(raw_data, columns=columns)

# Chuyển kiểu dữ liệu
df = df.astype({
    'open': float, 'high': float, 'low': float, 'close': float,
    'volume': float, 'quote_volume': float, 'trades': int
})

# Tạo các cột đúng format 
df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
df['symbol'] = 'BTCUSDT'
df['vwap'] = (df['quote_volume'] / df['volume']).round(8)
df['lastSize'] = 0
df['turnover'] = df['quote_volume']
df['homeNotional'] = df['volume']
df['foreignNotional'] = df['quote_volume']

# Lấy đúng thứ tự cột
final_df = df[[
    'timestamp', 'symbol', 'open', 'high', 'low', 'close',
    'trades', 'volume', 'vwap', 'lastSize', 'turnover',
    'homeNotional', 'foreignNotional'
]].copy()

# Lưu file + tải về ngay
filename = f"BTCUSDT_5m_2020_to_now_{len(final_df)//1000}k_rows.csv"
final_df.to_csv(filename, index=False)

print(f"\nHOÀN TẤT 100%!")
print(f"Tổng cộng: {len(final_df):,} rows")
print(f"Thời gian từ: {final_df['timestamp'].min()} → {final_df['timestamp'].max()}")
print(f"File đã lưu: {filename}")

from google.colab import files
files.download(filename)