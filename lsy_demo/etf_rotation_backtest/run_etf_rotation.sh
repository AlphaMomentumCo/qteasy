cd /home/lsy/data/project/Quant/qteasy/lsy_demo

python etf_rotation_backtest.py \
  --asset_pool 518880.SH,513100.SH,159915.SZ,510180.SH,513130.SH,517000.SH,512100.SH,000300.SH,588000.SH,588750.SH \
  --start_date 20200101 \
  --end_date 20251231 \
  --max_sel 1 \
  --benchmark_asset 000300.SH

