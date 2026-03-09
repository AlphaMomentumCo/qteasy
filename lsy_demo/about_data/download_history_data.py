import qteasy as qt
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from qteasy import utilfuncs as uf


TARGET_TABLE = "stock_indicator"
TARGET_END_DATE = "20260101"
CHANNEL = "tushare"


def _to_ymd_str(value):
    if not value:
        return None
    try:
        return pd.to_datetime(value).strftime("%Y%m%d")
    except Exception:
        return None


def _reload_trade_calendar_cache():
    # qteasy 在 import 时会缓存 QT_TRADE_CALENDAR，下载后需要手动刷新
    qt.QT_TRADE_CALENDAR = qt.QT_DATA_SOURCE.read_table_data("trade_calendar")
    for fn in (
        uf.is_market_trade_day,
        uf.last_known_market_trade_day,
        uf.prev_market_trade_day,
        uf.nearest_market_trade_day,
        uf.next_market_trade_day,
    ):
        fn.cache_clear()


def _get_table_max_date(table_name):
    # 用 DataSource 直接读取，避免 core.get_table_info 默认打印
    info = qt.QT_DATA_SOURCE.get_table_info(table_name, verbose=False, print_info=False)
    pk1 = info.get("primary_key1")
    pk2 = info.get("primary_key2")
    if pk1 == "trade_date":
        return _to_ymd_str(info.get("pk_max1"))
    if pk2 == "trade_date":
        return _to_ymd_str(info.get("pk_max2"))
    # 兜底：尝试能解析为日期的最大主键
    return _to_ymd_str(info.get("pk_max2")) or _to_ymd_str(info.get("pk_max1"))


def refill_stock_indicator_to_20260101():
    before_max = _get_table_max_date(TARGET_TABLE)
    print(f"[Before] {TARGET_TABLE} max trade_date = {before_max}")

    # 1) 先更新交易日历，确保 2026 年交易日可用
    qt.refill_data_source(
        tables="trade_calendar",
        channel=CHANNEL,
        parallel=True,
        merge_type="update",
    )
    _reload_trade_calendar_cache()

    # 2) 增量起点：从现有最大日期的下一天开始
    if before_max:
        start_date = (pd.to_datetime(before_max) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    else:
        start_date = "19990101"

    if start_date > TARGET_END_DATE:
        print(f"No refill needed: start_date={start_date} > end_date={TARGET_END_DATE}")
        return

    # 3) 补充 stock_indicator（映射 tushare daily_basic）
    qt.refill_data_source(
        tables=TARGET_TABLE,
        channel=CHANNEL,
        start_date=start_date,
        end_date=TARGET_END_DATE,
        parallel=True,
        merge_type="update",
        refill_dependent_tables=False,
    )

    after_max = _get_table_max_date(TARGET_TABLE)
    print(f"[After ] {TARGET_TABLE} max trade_date = {after_max}")


if __name__ == "__main__":
    refill_stock_indicator_to_20260101()
