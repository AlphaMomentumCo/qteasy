cd /home/lsy/data/project/Quant/qteasy

python lsy_demo/ci_index_rotation_backtest/import_stock_adj_factor_to_qteasy.py \
  --source_parquet /home/lsy/data/project/Quant/datas/stock_daily.parquet \
  --merge_type update \
  --chunk_size 32
