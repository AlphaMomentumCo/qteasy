#!/usr/bin/env bash
set -euo pipefail
cd /home/lsy/data/project/Quant/qteasy/lsy_demo/ci_index_rotation_backtest
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARY_CSV="$BASE_DIR/tradelog_my_run/trade_summary_None_20260308_230237.csv"
LOG_CSV="$BASE_DIR/tradelog_my_run/trade_log_None_20260308_230237.csv"

exec streamlit run "$BASE_DIR/trade_log_analyzer_web.py" -- \
  --summary "$SUMMARY_CSV" \
  --log "$LOG_CSV"
