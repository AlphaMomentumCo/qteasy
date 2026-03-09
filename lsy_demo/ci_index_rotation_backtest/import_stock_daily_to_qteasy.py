import argparse
import os
from typing import Iterable

import pandas as pd

import qteasy as qt


DEFAULT_STOCK_PARQUET = "/home/lsy/data/project/Quant/datas/stock_daily.parquet"

TARGET_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 stock_daily.parquet 写入 qteasy 的 stock_daily 表")
    parser.add_argument("--source_parquet", type=str, default=DEFAULT_STOCK_PARQUET, help="stock_daily parquet 路径")
    parser.add_argument("--start_date", type=str, default="", help="起始日期 YYYYMMDD")
    parser.add_argument("--end_date", type=str, default="", help="结束日期 YYYYMMDD")
    parser.add_argument("--chunk_size", type=int, default=20, help="按交易日分块写入，每块包含交易日数量")
    parser.add_argument("--merge_type", type=str, default="update", choices=["update", "ignore"], help="主键冲突处理方式")
    parser.add_argument("--dry_run", action="store_true", help="仅预览，不写入")
    return parser.parse_args()


def _read_source(path: str) -> pd.DataFrame:
    # 优先仅读取目标列，降低内存占用
    try:
        df = pd.read_parquet(path, columns=[c for c in TARGET_COLUMNS if c not in {"ts_code", "trade_date"}])
        # 如果主键在 index，reset 后补全
        df = df.reset_index()
    except Exception:
        df = pd.read_parquet(path)
        if not {"ts_code", "trade_date"}.issubset(df.columns):
            df = df.reset_index()

    if not {"ts_code", "trade_date"}.issubset(df.columns):
        raise ValueError("parquet 中未找到 ts_code / trade_date 字段")

    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"源数据缺少必要列: {missing}")

    out = df[TARGET_COLUMNS].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
    out["ts_code"] = out["ts_code"].astype(str).str.upper().str.strip()
    out = out.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    return out


def _filter_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    out = df
    if start_date:
        out = out[out["trade_date"] >= pd.to_datetime(start_date)]
    if end_date:
        out = out[out["trade_date"] <= pd.to_datetime(end_date)]
    return out


def _iter_date_chunks(unique_dates: Iterable[pd.Timestamp], chunk_size: int):
    dates = list(unique_dates)
    for i in range(0, len(dates), chunk_size):
        yield dates[i : i + chunk_size]


def main() -> None:
    args = parse_args()
    src = os.path.abspath(args.source_parquet)
    if not os.path.exists(src):
        raise FileNotFoundError(f"源文件不存在: {src}")

    raw = _read_source(src)
    raw = _filter_date(raw, args.start_date, args.end_date)
    if raw.empty:
        print("筛选后无数据，退出")
        return

    ds = qt.QT_DATA_SOURCE
    dmin = raw["trade_date"].min().strftime("%Y-%m-%d")
    dmax = raw["trade_date"].max().strftime("%Y-%m-%d")
    print(f"数据源: {ds}")
    print(f"待写入行数: {len(raw):,}, 日期范围: {dmin} ~ {dmax}")

    if args.dry_run:
        print("dry_run=True，不写入数据库")
        return

    total_rows = 0
    udates = raw["trade_date"].drop_duplicates().sort_values().tolist()
    for idx, date_chunk in enumerate(_iter_date_chunks(udates, args.chunk_size), start=1):
        part = raw[raw["trade_date"].isin(date_chunk)].copy()
        rows = ds.update_table_data("stock_daily", part, merge_type=args.merge_type)
        total_rows += rows
        print(
            f"chunk {idx:04d}: 写入 {len(part):,} 行, 影响 {rows:,} 行, "
            f"日期 {date_chunk[0].strftime('%Y-%m-%d')} ~ {date_chunk[-1].strftime('%Y-%m-%d')}"
        )

    print(f"完成写入，累计影响行数: {total_rows:,}")


if __name__ == "__main__":
    main()
