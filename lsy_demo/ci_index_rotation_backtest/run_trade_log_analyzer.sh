#!/usr/bin/env bash
set -euo pipefail

cd /home/lsy/data/project/Quant/qteasy/lsy_demo/ci_index_rotation_backtest

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$BASE_DIR/trade_log_analyzer_gui.py" "$@"
