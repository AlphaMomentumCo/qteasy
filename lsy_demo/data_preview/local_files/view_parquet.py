import pandas as pd

data_path = "/home/lsy/data/project/Quant/datas/etf_daily.parquet"

df = pd.read_parquet(data_path)

df.info()

# 1. 先查看日期范围
trade_dates = df.index.get_level_values("trade_date")
latest_date = trade_dates.max()
earliest_date = trade_dates.min()
print(f"最早交易日期: {earliest_date}")
print(f"最新交易日期: {latest_date}")

# 2. 选择你想看的某一天（默认看最新日期）
# 也可以改成 target_date = pd.Timestamp("2024-01-02")
target_date = latest_date

# 3. 取出这一天的所有 ETF 代码并统计数量
day_data = df.xs(target_date, level="trade_date")
etf_codes = sorted(day_data.index.unique())

print(f"\n目标日期: {target_date}")
print(f"该日 ETF 数量: {len(etf_codes)}")
print("该日所有 ETF（ts_code）:")
# for code in etf_codes:
#     print(code)
