import argparse
from collections import defaultdict
from datetime import timedelta

import pandas as pd


def normalize_date_col(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def build_history_daily_sets(
    history_path: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict:
    df = pd.read_csv(history_path, usecols=["ts_code", "in_date", "out_date"])
    df = df.dropna(subset=["ts_code", "in_date"])

    df["in_dt"] = normalize_date_col(df["in_date"])
    df["out_dt"] = normalize_date_col(df["out_date"]).fillna(pd.Timestamp("2099-12-31"))
    df = df.dropna(subset=["in_dt", "out_dt"])

    df["in_dt"] = df["in_dt"].clip(lower=start_date, upper=end_date)
    df["out_dt"] = df["out_dt"].clip(lower=start_date, upper=end_date)
    df = df[df["in_dt"] <= df["out_dt"]].copy()

    add_events = defaultdict(list)
    remove_events = defaultdict(list)

    for row in df.itertuples(index=False):
        add_events[row.in_dt].append(row.ts_code)
        next_day = row.out_dt + timedelta(days=1)
        if next_day <= end_date:
            remove_events[next_day].append(row.ts_code)

    active = set()
    daily_sets = {}
    for dt in pd.date_range(start_date, end_date, freq="D"):
        for code in remove_events.get(dt, []):
            active.discard(code)
        for code in add_events.get(dt, []):
            active.add(code)
        daily_sets[dt] = set(active)
    return daily_sets


def build_parquet_daily_sets(parquet_path: str) -> dict:
    df = pd.read_parquet(parquet_path, columns=["con_codes"])
    if not isinstance(df.index, pd.MultiIndex) or "trade_date" not in df.index.names:
        raise ValueError("Parquet index must be MultiIndex and contain level: trade_date")

    trade_dates = pd.to_datetime(df.index.get_level_values("trade_date"))
    out = defaultdict(set)

    for dt, con_str in zip(trade_dates, df["con_codes"].astype(str).values):
        if not con_str or con_str == "nan":
            continue
        out[dt.normalize()].update(x for x in con_str.split(",") if x)

    return out


def build_daily_stock_list(
    parquet_path: str,
    history_path: str,
    start: str,
    end: str,
    out_parquet: str,
    out_csv: str,
) -> None:
    start_date = pd.to_datetime(start, format="%Y%m%d")
    end_date = pd.to_datetime(end, format="%Y%m%d")

    history_sets = build_history_daily_sets(history_path, start_date, end_date)
    parquet_sets = build_parquet_daily_sets(parquet_path)

    rows = []
    for dt in pd.date_range(start_date, end_date, freq="D"):
        merged = set(history_sets.get(dt, set()))
        merged.update(parquet_sets.get(dt, set()))
        codes = sorted(merged)
        rows.append(
            {
                "trade_date": dt,
                "date": int(dt.strftime("%Y%m%d")),
                "stock_count": len(codes),
                "stock_list": ",".join(codes),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_parquet, index=False)
    out_df.to_csv(out_csv, index=False)

    print(f"Done. rows={len(out_df)}")
    print(f"Saved parquet: {out_parquet}")
    print(f"Saved csv: {out_csv}")
    print(out_df.head(3).to_string(index=False))
    print(out_df.tail(3).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily stock list from 2001 to 20251231.")
    parser.add_argument(
        "--parquet-path",
        default="/home/lsy/data/project/Quant/AlphaData/data_preview/tushare/ci_index/step1_get_L3_daily/ci_l3_daily.parquet",
    )
    parser.add_argument(
        "--history-path",
        default="/home/lsy/data/project/Quant/AlphaData/data_preview/tushare/ci_index/step1_get_L3_daily/citic_members_history_full.csv",
    )
    parser.add_argument("--start", default="20010101")
    parser.add_argument("--end", default="20251231")
    parser.add_argument(
        "--out-parquet",
        default="/home/lsy/data/project/Quant/AlphaData/data_preview/tushare/ci_index/step2_L3_stock_daily/l3_stock_daily_20010101_20251231.parquet",
    )
    parser.add_argument(
        "--out-csv",
        default="/home/lsy/data/project/Quant/AlphaData/data_preview/tushare/ci_index/step2_L3_stock_daily/l3_stock_daily_20010101_20251231.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_daily_stock_list(
        parquet_path=args.parquet_path,
        history_path=args.history_path,
        start=args.start,
        end=args.end,
        out_parquet=args.out_parquet,
        out_csv=args.out_csv,
    )
