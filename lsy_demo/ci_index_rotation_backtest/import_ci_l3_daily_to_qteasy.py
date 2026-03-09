import argparse
import os
from typing import Iterable

import numpy as np
import pandas as pd

import qteasy as qt


DEFAULT_PARQUET = "/home/lsy/data/project/Quant/AlphaData/data_preview/tushare/ci_index/step2_L3_stock_daily/ci_l3_daily.parquet"


TARGET_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "low",
    "high",
    "close",
    "pre_close",
    "change",
    "pct_change",
    "vol",
    "amount",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 ci_l3_daily.parquet 写入 qteasy 的 ci_index_daily 表")
    parser.add_argument("--source_parquet", type=str, default=DEFAULT_PARQUET, help="行业指数 parquet 路径")
    parser.add_argument(
        "--constituents_export",
        type=str,
        default="",
        help="导出行业-成分股映射 parquet 路径，默认导出到当前目录",
    )
    parser.add_argument("--start_date", type=str, default="", help="起始日期 YYYYMMDD")
    parser.add_argument("--end_date", type=str, default="", help="结束日期 YYYYMMDD")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=60,
        help="按交易日分块写入，每块含多少个交易日",
    )
    parser.add_argument(
        "--merge_type",
        type=str,
        default="update",
        choices=["update", "ignore"],
        help="重复主键处理方式",
    )
    parser.add_argument("--dry_run", action="store_true", help="仅预览数据，不实际写入")
    return parser.parse_args()


def _normalize_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    if {"trade_date", "ts_code"}.issubset(df.columns):
        out = df.copy()
    else:
        out = df.reset_index()
        if not {"trade_date", "ts_code"}.issubset(out.columns):
            raise ValueError("parquet 中未找到 trade_date/ts_code 字段，请检查数据结构")

    # 兼容字段名差异
    rename_map = {"pct_chg": "pct_change"}
    out = out.rename(columns=rename_map)

    missing = [c for c in TARGET_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    out = out[TARGET_COLUMNS + [c for c in ["l3_name", "con_codes"] if c in out.columns]].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
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

    raw = pd.read_parquet(src)
    raw = _normalize_raw_frame(raw)
    raw = _filter_date(raw, args.start_date, args.end_date)
    if raw.empty:
        print("筛选后无数据，退出")
        return

    ci_daily = raw[TARGET_COLUMNS].copy()

    ds = qt.QT_DATA_SOURCE
    dmin = raw["trade_date"].min().strftime("%Y-%m-%d")
    dmax = raw["trade_date"].max().strftime("%Y-%m-%d")
    print(f"数据源: {ds}")
    print(f"待写入行数: {len(ci_daily):,}, 日期范围: {dmin} ~ {dmax}")

    if args.dry_run:
        print("dry_run=True，不写入数据库")
    else:
        total = 0
        udates = ci_daily["trade_date"].drop_duplicates().sort_values().tolist()
        for idx, date_chunk in enumerate(_iter_date_chunks(udates, args.chunk_size), start=1):
            part = ci_daily[ci_daily["trade_date"].isin(date_chunk)].copy()
            rows = ds.update_table_data("ci_index_daily", part, merge_type=args.merge_type)
            total += rows
            print(
                f"chunk {idx:03d}: 写入 {len(part):,} 行, 影响 {rows:,} 行, "
                f"日期 {date_chunk[0].strftime('%Y-%m-%d')} ~ {date_chunk[-1].strftime('%Y-%m-%d')}"
            )
        print(f"完成写入，累计影响行数: {total:,}")

    if {"l3_name", "con_codes"}.issubset(raw.columns):
        export_path = args.constituents_export.strip()
        if not export_path:
            export_path = os.path.join(
                os.path.dirname(__file__),
                "output/ci_l3_constituents.parquet",
            )
        export_path = os.path.abspath(export_path)

        con_df = raw[["trade_date", "ts_code", "l3_name", "con_codes"]].copy()
        con_df = con_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        con_df.to_parquet(export_path, index=False)
        print(f"已导出行业-成分股映射: {export_path}, 行数: {len(con_df):,}")


if __name__ == "__main__":
    main()
