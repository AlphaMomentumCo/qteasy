import argparse

import pandas as pd
import qteasy as qt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="简单查询 qteasy 的 ci_index_daily 数据")
    parser.add_argument("--ts_code", type=str, default="", help="指数代码，多个用逗号分隔，如 CI005201.CI")
    parser.add_argument("--start_date", type=str, default="", help="开始日期 YYYYMMDD")
    parser.add_argument("--end_date", type=str, default="", help="结束日期 YYYYMMDD")
    parser.add_argument("--limit", type=int, default=10, help="展示前N行")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    shares = args.ts_code.strip() or None
    start = args.start_date.strip() or None
    end = args.end_date.strip() or None

    ds = qt.QT_DATA_SOURCE
    print(f"当前数据源: {ds}")

    df = ds.read_table_data(
        "ci_index_daily",
        shares=shares,
        start=start,
        end=end,
        primary_key_in_index=False,
    )

    if df.empty:
        print("ci_index_daily 查询结果为空")
        return

    # 统一排序，便于查看
    if {"trade_date", "ts_code"}.issubset(df.columns):
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    print(f"查询结果行数: {len(df):,}")
    if "trade_date" in df.columns:
        print(f"日期范围: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")
    if "ts_code" in df.columns:
        print(f"指数数量: {df['ts_code'].nunique()}")

    print("\n前几行数据:")
    print(df.head(args.limit).to_string(index=False))


if __name__ == "__main__":
    main()
