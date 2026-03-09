#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def find_min_in_date(csv_path: Path) -> str:
    min_date = None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "in_date" not in (reader.fieldnames or []):
            raise ValueError("CSV does not contain column: in_date")

        for row in reader:
            raw = (row.get("in_date") or "").strip()
            if not raw:
                continue

            # Keep only 8-digit dates like 20241029.
            date_str = raw.split(".")[0]
            if len(date_str) == 8 and date_str.isdigit():
                if min_date is None or date_str < min_date:
                    min_date = date_str

    if min_date is None:
        raise ValueError("No valid in_date found in CSV")
    return min_date


def main() -> None:
    parser = argparse.ArgumentParser(description="Get earliest in_date from CSV")
    parser.add_argument(
        "--file",
        default="citic_members_history_full.csv",
        help="CSV file path (default: citic_members_history_full.csv)",
    )
    args = parser.parse_args()

    csv_path = Path(args.file)
    min_date = find_min_in_date(csv_path)
    print(min_date)


if __name__ == "__main__":
    main()
