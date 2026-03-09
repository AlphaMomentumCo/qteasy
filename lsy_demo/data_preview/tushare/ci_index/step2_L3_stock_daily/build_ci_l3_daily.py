import argparse

import pandas as pd


def to_code_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, tuple) or isinstance(value, set):
        return [str(x).strip() for x in list(value) if str(x).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    s = str(value).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def build_ci_l3_daily(input_parquet: str, output_parquet: str) -> None:
    df = pd.read_parquet(input_parquet)

    required_cols = [
        "open",
        "low",
        "high",
        "close",
        "pre_close",
        "change",
        "pct_change",
        "vol",
        "amount",
        "con_codes",
        "l3_name",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少字段: {missing}")

    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ["trade_date", "ts_code"]:
        raise ValueError("输入 parquet 的 index 必须是 ['trade_date', 'ts_code']")

    df = df[required_cols].copy()
    df["con_codes"] = df["con_codes"].apply(to_code_list)
    df = df.sort_index()

    df.to_parquet(output_parquet)
    print(f"saved: {output_parquet}")
    print(f"rows: {len(df)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert L3 parquet to L1/L2-like format.")
    parser.add_argument(
        "--input-parquet",
        default="/home/lsy/data/project/Quant/AlphaData/data_preview/tushare/ci_index/step1_get_L3_daily/ci_l3_daily.parquet",
    )
    parser.add_argument(
        "--output-parquet",
        default="/home/lsy/data/project/Quant/AlphaData/data_preview/tushare/ci_index/step2_L3_stock_daily/ci_l3_daily.parquet",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_ci_l3_daily(args.input_parquet, args.output_parquet)
