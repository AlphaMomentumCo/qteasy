import pandas as pd

data_path = "/home/lsy/data/project/Quant/datas/etf_basic_data.parquet"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)

df = pd.read_parquet(data_path)

df.info()

preview = df.head(10)
try:
    print(preview.to_markdown(index=False, tablefmt="grid"))
except ImportError:
    # Fallback when tabulate is not installed.
    print(preview.to_string(index=False))
