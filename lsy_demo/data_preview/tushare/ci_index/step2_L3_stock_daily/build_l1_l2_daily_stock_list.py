import argparse
import ast
from pathlib import Path

import pandas as pd
import numpy as np


def parse_con_codes(value) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, tuple) or isinstance(value, set):
        return [str(x).strip() for x in list(value) if str(x).strip()]
    if isinstance(value, np.ndarray):
        return [str(x).strip() for x in value.tolist() if str(x).strip()]
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []

    s = str(value).strip()
    if not s:
        return []

    # Handle serialized Python list string: "['000001.SZ', '600000.SH']"
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (SyntaxError, ValueError):
            pass

    # Fallback for plain comma string
    return [x.strip() for x in s.split(",") if x.strip()]


def convert_one_level(
    input_path: str,
    name_col: str,
    level_tag: str,
    out_parquet: str,
    out_csv: str,
) -> None:
    df = pd.read_parquet(input_path, columns=["con_codes", name_col])
    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ["trade_date", "ts_code"]:
        raise ValueError(f"{input_path} 的 index 不是 ['trade_date', 'ts_code']")

    out_rows = []
    for (trade_date, ts_code), row in df.iterrows():
        codes = sorted(set(parse_con_codes(row["con_codes"])))
        out_rows.append(
            {
                "trade_date": pd.to_datetime(trade_date),
                "date": int(pd.to_datetime(trade_date).strftime("%Y%m%d")),
                f"{level_tag}_code": ts_code,
                f"{level_tag}_name": row[name_col],
                "stock_count": len(codes),
                "stock_list": ",".join(codes),
            }
        )

    out_df = pd.DataFrame(out_rows).sort_values(["trade_date", f"{level_tag}_code"]).reset_index(drop=True)
    out_df.to_parquet(out_parquet, index=False)
    out_df.to_csv(out_csv, index=False)

    # Daily total unique stock count (union across all categories on the same day)
    daily_union = {}
    for trade_date, codes in zip(out_df["trade_date"], out_df["stock_list"]):
        dt = pd.to_datetime(trade_date)
        if dt not in daily_union:
            daily_union[dt] = set()
        if isinstance(codes, str) and codes:
            daily_union[dt].update(x for x in codes.split(",") if x)

    total_rows = []
    for dt in sorted(daily_union.keys()):
        merged_codes = sorted(daily_union[dt])
        total_rows.append(
            {
                "trade_date": dt,
                "date": int(dt.strftime("%Y%m%d")),
                "stock_count": len(merged_codes),
                "stock_list": ",".join(merged_codes),
            }
        )
    total_df = pd.DataFrame(total_rows)
    total_parquet = str(Path(out_parquet).with_name(f"{level_tag}_stock_daily_total.parquet"))
    total_csv = str(Path(out_csv).with_name(f"{level_tag}_stock_daily_total.csv"))
    total_df.to_parquet(total_parquet, index=False)
    total_df.to_csv(total_csv, index=False)

    print(f"[{level_tag}] rows={len(out_df)}")
    print(f"[{level_tag}] saved parquet: {out_parquet}")
    print(f"[{level_tag}] saved csv: {out_csv}")
    print(f"[{level_tag}] total rows={len(total_df)}")
    print(f"[{level_tag}] saved total parquet: {total_parquet}")
    print(f"[{level_tag}] saved total csv: {total_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build L1/L2 daily stock list by category.")
    parser.add_argument(
        "--l1-input",
        default="/home/lsy/data/project/Quant/datas/ci_l1_daily.parquet",
    )
    parser.add_argument(
        "--l2-input",
        default="/home/lsy/data/project/Quant/datas/ci_l2_daily.parquet",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/lsy/data/project/Quant/AlphaData/data_preview/tushare/ci_index/step2_L3_stock_daily",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    convert_one_level(
        input_path=args.l1_input,
        name_col="l1_name",
        level_tag="l1",
        out_parquet=str(out_dir / "l1_stock_daily_by_category.parquet"),
        out_csv=str(out_dir / "l1_stock_daily_by_category.csv"),
    )
    convert_one_level(
        input_path=args.l2_input,
        name_col="l2_name",
        level_tag="l2",
        out_parquet=str(out_dir / "l2_stock_daily_by_category.parquet"),
        out_csv=str(out_dir / "l2_stock_daily_by_category.csv"),
    )


if __name__ == "__main__":
    main()
