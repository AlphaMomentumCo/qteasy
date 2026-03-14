#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EXP_ROOT="${SCRIPT_DIR}/exp"
EXP_TS="$(date +%Y%m%d_%H%M%S)"
EXP_DIR="${EXP_ROOT}/${EXP_TS}"
mkdir -p "${EXP_DIR}"

cp -f "${SCRIPT_DIR}/run_ci_index_rotation.sh" "${EXP_DIR}/"
cp -f "${SCRIPT_DIR}/ci_index_rotation_backtest.py" "${EXP_DIR}/"

echo "Experiment output dir: ${EXP_DIR}"

python ci_index_rotation_backtest.py \
  --ci_parquet /home/lsy/data/project/Quant/AlphaData/data_preview/tushare/ci_index/step2_L3_stock_daily/ci_l3_daily.parquet \
  --start_date 20200101 \
  --end_date 20251231 \
  --m_days 60 \
  --top_k 3 \
  --top_n 3 \
  --switch_th 0.15 \
  --idx_ma_days 20 \
  --stock_ma_days 30 \
  --breadth_ma_days 30 \
  --min_idx_score 0.0 \
  --min_breadth 0.25 \
  --invest_ratio 0.95 \
  --min_amount 50000 \
  --run_freq W \
  --cash 100000 \
  --benchmark_asset 000300.SH \
  --trade_log \
  --exp_dir "${EXP_DIR}" \
  2>&1 | tee "${EXP_DIR}/run.log"
