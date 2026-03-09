import argparse
import os
import sys
from datetime import datetime
import re
import numpy as np

# 优先使用当前项目下的qteasy源码
# _LOCAL_QTEASY = os.path.join(os.path.dirname(__file__), "qteasy")
# if os.path.isdir(_LOCAL_QTEASY):
#     sys.path.insert(0, _LOCAL_QTEASY)

import qteasy as qt
qt.configure(trade_log_file_path='tradelog_my_run/')


def _patch_qteasy_fd_trade_price_compat() -> None:
    """兼容部分 qteasy 版本在 SH/SZ ETF 场景下未纳入 FD 价格源的问题。"""
    import qteasy.history as qh
    import qteasy.qt_operator as qo

    original = qh.check_and_prepare_trade_prices

    def _wrapped(op, shares, price_adj, datasource):
        if isinstance(shares, str):
            share_list = [s.strip() for s in shares.split(",") if s.strip()]
        else:
            share_list = list(shares)

        # 若策略本身依赖 FD 数据且资产代码均为场内 SH/SZ，给原函数注入一个 OF 代码触发 FD 数据源查询
        strategy_uses_fd = any(
            any("_FD_" in dtype_id for dtype_id in stg.data_type_ids)
            for stg in op.strategies
        )
        all_exchange_suffix = bool(share_list) and all(
            str(s).upper().endswith((".SH", ".SZ")) for s in share_list
        )

        if strategy_uses_fd and all_exchange_suffix:
            trigger_symbol = f"{str(share_list[0])[:-2]}OF"
            patched_shares = share_list + [trigger_symbol]
            trade_prices = original(op, patched_shares, price_adj, datasource)
            if hasattr(trade_prices, "columns"):
                existing_cols = [c for c in share_list if c in trade_prices.columns]
                if existing_cols:
                    return trade_prices.loc[:, existing_cols]
            return trade_prices

        return original(op, shares, price_adj, datasource)

    qh.check_and_prepare_trade_prices = _wrapped
    qo.check_and_prepare_trade_prices = _wrapped


class EtfRotationStg(qt.GeneralStg):
    """ETF轮动策略（基于聚宽动量打分）：
    - 以 log 价格线性回归斜率估算年化收益
    - 用 R² 作为趋势稳定度
    - 得分 = 年化收益 * R²
    - 周频调仓降低换手，变动阈值控制进一步降低换手
    """

    def __init__(self, pars: tuple):
        m_days, max_sel, switch_th = pars
        window_len = m_days + 5
        super().__init__(
            pars=[
                qt.Parameter((5, 250), name="m_days", par_type="int"),
                qt.Parameter((1, 10), name="max_sel", par_type="int"),
                qt.Parameter((0.0, 1.0), name="switch_th", par_type="float"),
            ],
            par_values=pars,
            name="ETF_ROTATION_SCORE",
            description="Momentum score rotation: annualized return * R^2",
            data_types=qt.StgData(
                "close",
                freq="d",
                asset_type="FD",
                window_length=window_len,
                use_latest_data_cycle=True,
            ),
        )

    @staticmethod
    def _score_one(series: np.ndarray) -> float:
        series = series[np.isfinite(series)]
        if series.size < 5:
            return -np.inf
        y = np.log(series)
        x = np.arange(y.size, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        resid = y - y_hat
        var = np.var(y, ddof=1)
        if not np.isfinite(var) or var == 0:
            r2 = 0.0
        else:
            r2 = 1 - (np.sum(resid ** 2) / ((y.size - 1) * var))
        annualized_returns = np.exp(slope) ** 250 - 1
        return annualized_returns * r2

    def realize(self):
        close = self.get_data("close_FD_d")
        if close.ndim == 1:
            close = close[:, None]

        pars = self.par_values
        if pars is None:
            return np.zeros(close.shape[1], dtype=float)
        m_days, max_sel, switch_th = pars
        if close.shape[0] <= m_days:
            return np.zeros(close.shape[1], dtype=float)

        window = close[-m_days:, :]
        scores = np.array([self._score_one(window[:, i]) for i in range(window.shape[1])])
        ranked = np.argsort(scores)[::-1]
        selected = ranked[:max_sel]

        target = np.zeros_like(scores, dtype=float)
        target[selected] = 1.0 / len(selected)

        # 换手控制：若调整幅度过小则保持当前持仓
        if switch_th > 0:
            own_amounts = self.get_data("proc.own_amounts", lag=0).reshape(-1)
            prices = self.get_data("proc.trade_price", lag=0).reshape(-1)
            # 某些版本在首个信号步会返回空 trade_price，这里回退到当前收盘价
            if prices.size == 0 or prices.size != close.shape[1]:
                prices = close[-1, :].reshape(-1)
            if own_amounts.size == 0:
                own_amounts = np.zeros(close.shape[1], dtype=float)
            elif own_amounts.size != close.shape[1]:
                own_amounts = own_amounts[:close.shape[1]]
                if own_amounts.size < close.shape[1]:
                    own_amounts = np.pad(
                        own_amounts,
                        (0, close.shape[1] - own_amounts.size),
                        mode="constant",
                        constant_values=0.0,
                    )
            current_value = own_amounts * prices
            total_value = np.sum(current_value)
            if total_value > 0:
                current_weight = current_value / total_value
                if np.nansum(np.abs(target - current_weight)) < switch_th:
                    return current_weight

        return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETF轮动策略回测")
    parser.add_argument(
        "--asset_pool",
        type=str,
        default="518880.SH,513100.SH,159915.SZ,510180.SH",
        help="ETF资产池，逗号分隔。默认：华安黄金ETF、纳指ETF、创业板ETF、上证180ETF",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="20200101",
        help="回测起始日期 YYYYMMDD",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="20251231",
        help="回测结束日期 YYYYMMDD",
    )
    parser.add_argument(
        "--m_days",
        type=int,
        default=25,
        help="动量参考天数",
    )
    parser.add_argument(
        "--switch_th",
        type=float,
        default=0.25,
        help="目标仓位变动阈值，小于该值则不换手",
    )
    parser.add_argument(
        "--max_sel",
        type=int,
        default=1,
        help="最多持有的ETF数量",
    )
    parser.add_argument(
        "--run_freq",
        type=str,
        default="W",
        help="策略运行频率，W为周频，D为日频",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=100000.0,
        help="初始资金",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="先下载fund_basic与fund_daily数据",
    )
    parser.add_argument(
        "--visual",
        default=True,
        action="store_true",
        help="显示图表结果",
    )
    parser.add_argument(
        "--trade_log",
        default=True,
        action="store_true",
        help="打印交易日志",
    )
    parser.add_argument(
        "--fig_path",
        type=str,
        default="",
        help="回测图像保存路径，留空则自动生成",
    )
    parser.add_argument(
        "--benchmark_asset",
        type=str,
        default="000300.SH",
        help="",
    )
    return parser.parse_args()


def _prepare_visual_save(fig_path: str) -> None:
    import matplotlib.pyplot as plt

    original_show = plt.show

    def _save_and_show(*args, **kwargs):
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        original_show(*args, **kwargs)

    plt.show = _save_and_show


def _normalize_asset_pool(asset_pool_raw: str) -> list[str]:
    pool = []
    for raw in asset_pool_raw.split(","):
        code = raw.strip().upper()
        if not code:
            continue
        if code.endswith(".SS"):
            code = code[:-3] + ".SH"
        pool.append(code)

    invalid = [c for c in pool if re.fullmatch(r"\d{6}\.(SH|SZ)", c) is None]
    if invalid:
        raise ValueError(
            f"资产代码格式不正确: {invalid}，请使用类似 510300.SH / 159915.SZ 的格式"
        )
    if not pool:
        raise ValueError("asset_pool 不能为空")
    return pool


def main() -> None:
    _patch_qteasy_fd_trade_price_compat()
    args = parse_args()
    asset_pool = _normalize_asset_pool(args.asset_pool)

    if args.visual:
        if args.fig_path:
            fig_path = args.fig_path
        else:
            os.makedirs("output", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_path = os.path.join(
                "output",
                f"etf_rotation_backtest_{args.start_date}_{args.end_date}.png",
            )
        _prepare_visual_save(fig_path)
        print(f"回测图像将保存到: {fig_path}")

    if args.download:
        qt.refill_data_source(
            tables="fund_basic, fund_daily, index_daily",
            start_date=args.start_date,
            end_date=args.end_date,
            symbols=",".join(asset_pool + ["000300.SH"]),
        )

    pars = (args.m_days, args.max_sel, args.switch_th)
    alpha = EtfRotationStg(pars=pars)

    
    op = qt.Operator(
        strategies=[alpha],
        signal_type="PT",
        op_type="step",
        run_freq=args.run_freq,
        run_timing="close",
    )

    qt.run(
        op,
        mode=1,
        asset_pool=asset_pool,
        asset_type="FD",
        benchmark_asset=args.benchmark_asset,
        invest_start=args.start_date,
        invest_end=args.end_date,
        invest_cash_amounts=[args.cash],
        trade_batch_size=0.1,
        sell_batch_size=0.1,
        cost_rate_buy=0.0001,
        cost_rate_sell=0.0001,
        visual=args.visual,
        trade_log=args.trade_log,
    )


if __name__ == "__main__":
    main()
