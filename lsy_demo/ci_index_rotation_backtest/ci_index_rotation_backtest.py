import argparse
import ast
import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import qteasy as qt

DEFAULT_CI_PARQUET = "/home/lsy/data/project/Quant/AlphaData/data_preview/tushare/ci_index/step2_L3_stock_daily/ci_l3_daily.parquet"


def _score_one(series: np.ndarray) -> float:
    """使用波动率调整后的动量得分 (Risk-adjusted momentum)"""
    series = series[np.isfinite(series) & (series > 0)]
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
    
    annualized_returns = np.exp(slope * 250) - 1
    
    # NEW: 计算收益率的年化波动率
    pct_returns = np.diff(series) / series[:-1]
    vol = np.std(pct_returns, ddof=1) * np.sqrt(250)
    if vol == 0 or not np.isfinite(vol):
        return -np.inf
        
    return (annualized_returns * r2) / vol


def _parse_con_codes(v) -> List[str]:
    if isinstance(v, list):
        codes = v
    elif isinstance(v, np.ndarray):
        codes = v.tolist()
    elif isinstance(v, tuple):
        codes = list(v)
    elif isinstance(v, str):
        txt = v.strip()
        if not txt:
            return []
        if txt.startswith("[") and txt.endswith("]"):
            try:
                parsed = ast.literal_eval(txt)
                codes = parsed if isinstance(parsed, list) else [txt]
            except Exception:
                codes = [x.strip().strip("'\"") for x in txt[1:-1].split(",") if x.strip()]
        else:
            codes = [x.strip() for x in txt.split(",") if x.strip()]
    else:
        return []

    out = []
    for c in codes:
        s = str(c).strip().upper()
        if not s:
            continue
        if s.endswith(".SS"):
            s = s[:-3] + ".SH"
        out.append(s)
    return sorted(set(out))


def _normalize_ci_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy() if {"trade_date", "ts_code"}.issubset(df.columns) else df.reset_index()
    if "pct_chg" in out.columns and "pct_change" not in out.columns:
        out = out.rename(columns={"pct_chg": "pct_change"})

    need_cols = ["trade_date", "ts_code", "close", "con_codes"]
    for c in need_cols:
        if c not in out.columns:
            raise ValueError(f"行业数据缺少必要字段: {c}")

    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
    out = out.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    return out


def _build_constituent_history(ci_df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, List[List[str]], List[str]]]:
    out: Dict[str, Tuple[np.ndarray, List[List[str]], List[str]]] = {}
    for ts_code, g in ci_df.groupby("ts_code", sort=False):
        g2 = g.sort_values("trade_date")
        dates = g2["trade_date"].to_numpy(dtype="datetime64[ns]")
        code_lists = [_parse_con_codes(x) for x in g2["con_codes"].tolist()]
        if "l3_name" in g2.columns:
            names = [str(x) if pd.notna(x) else "" for x in g2["l3_name"].tolist()]
        else:
            names = [""] * len(g2)
        out[ts_code] = (dates, code_lists, names)
    return out


def _lookup_constituents(
    hist: Dict[str, Tuple[np.ndarray, List[List[str]], List[str]]],
    idx_code: str,
    date: pd.Timestamp,
) -> Tuple[List[str], str]:
    item = hist.get(idx_code)
    if item is None:
        return [], ""
    dates, code_lists, names = item
    if len(dates) == 0:
        return [], ""
    pos = np.searchsorted(dates, np.datetime64(date), side="right") - 1
    if pos < 0:
        return [], ""
    return code_lists[pos], names[pos]


def _normalize_stock_pool(codes: List[str]) -> List[str]:
    pool = []
    for code in codes:
        s = str(code).strip().upper()
        if not s:
            continue
        if s.endswith(".SS"):
            s = s[:-3] + ".SH"
        pool.append(s)

    valid = [c for c in pool if re.fullmatch(r"\d{6}\.(SH|SZ)", c)]
    return sorted(set(valid))


def _symbols_with_data(
    ds: qt.DataSource,
    table: str,
    symbols: List[str],
    start_date: str,
    end_date: str,
    chunk_size: int = 500,
) -> set[str]:
    """分块查询指定表在区间内有数据的 symbols，避免一次性读入过大。"""
    available: set[str] = set()
    if not symbols:
        return available

    for i in range(0, len(symbols), chunk_size):
        sym_chunk = symbols[i : i + chunk_size]
        try:
            df = ds.read_table_data(
                table,
                shares=sym_chunk,
                start=start_date,
                end=end_date,
            )
        except Exception:
            continue
        if df.empty:
            continue

        if isinstance(df.index, pd.MultiIndex) and "ts_code" in df.index.names:
            available.update(df.index.get_level_values("ts_code").unique().tolist())
        elif "ts_code" in df.columns:
            available.update(df["ts_code"].astype(str).str.upper().str.strip().tolist())

    return available


class CiIndexRotationStg(qt.GeneralStg):
    """行业指数轮动策略：
    1. 用行业指数 close 做动量分数，选最高分行业；
    2. 加入行业趋势过滤 + 市场广度过滤；
    3. 在该行业成分股中，用同样分数选最高分个股并分散持仓；
    4. 输出 PT 目标仓位（允许部分现金仓位）。
    """

    def __init__(
        self,
        pars: tuple,
        ci_close: pd.DataFrame,
        con_hist: Dict[str, Tuple[np.ndarray, List[List[str]], List[str]]],
        use_forward_adj: bool = True,
    ):
        (
            m_days,
            top_k,      # NEW: 持有行业数
            top_n,      # 行业内持有个股数
            switch_th,
            idx_ma_days,
            stock_ma_days,
            breadth_ma_days,
            min_idx_score,
            min_breadth,
            invest_ratio,
            min_amount,
            min_total_mv,
        ) = pars
        window_len = max(m_days, idx_ma_days, stock_ma_days, breadth_ma_days) + 5
        super().__init__(
            pars=[
                qt.Parameter((5, 250), name="m_days", par_type="int"),
                qt.Parameter((1, 20), name="top_k", par_type="int"), # NEW
                qt.Parameter((1, 20), name="top_n", par_type="int"),
                qt.Parameter((0.0, 1.0), name="switch_th", par_type="float"),
                qt.Parameter((5, 250), name="idx_ma_days", par_type="int"),
                qt.Parameter((5, 250), name="stock_ma_days", par_type="int"),
                qt.Parameter((5, 250), name="breadth_ma_days", par_type="int"),
                qt.Parameter((-1.0, 5.0), name="min_idx_score", par_type="float"),
                qt.Parameter((0.0, 1.0), name="min_breadth", par_type="float"),
                qt.Parameter((0.0, 1.0), name="invest_ratio", par_type="float"),
                qt.Parameter((0.0, 1e10), name="min_amount", par_type="float"),
                qt.Parameter((0.0, 1e12), name="min_total_mv", par_type="float"),
            ],
            par_values=pars,
            name="CI_INDEX_ROTATION",
            description="Risk-aware CI rotation: trend filter + breadth filter + diversified stock picks",
            data_types=[
                qt.StgData(
                    "close|f" if use_forward_adj else "close",
                    freq="d",
                    asset_type="E",
                    window_length=window_len,
                    use_latest_data_cycle=True,
                ),
                qt.StgData(
                    "amount",
                    freq="d",
                    asset_type="E",
                    window_length=window_len,
                    use_latest_data_cycle=True,
                ),
                qt.StgData(
                    "total_mv",
                    freq="d",
                    asset_type="E",
                    window_length=window_len,
                    use_latest_data_cycle=True,
                ),
            ],
        )
        self._ci_close = ci_close.sort_index()
        self._con_hist = con_hist
        self._close_dtype_id = "close|f_E_d" if use_forward_adj else "close_E_d"

    def _current_date(self) -> pd.Timestamp:
        op = self._group._operator
        idx = getattr(op, "_current_signal_index", None)
        t_idx = getattr(op, "_process_time_index", None)
        if idx is None or t_idx is None or idx >= len(t_idx):
            return self._ci_close.index[-1]
        return pd.Timestamp(t_idx[idx]).normalize()

    def realize(self):
        close = self.get_data(self._close_dtype_id)
        amount = self.get_data("amount_E_d")
        total_mv = self.get_data("total_mv_E_d")
        
        if close.ndim == 1:
            close = close[:, None]
        if amount.ndim == 1:
            amount = amount[:, None]
        if total_mv.ndim == 1:
            total_mv = total_mv[:, None]

        (
            m_days,
            top_k,
            top_n,
            switch_th,
            idx_ma_days,
            stock_ma_days,
            breadth_ma_days,
            min_idx_score,
            min_breadth,
            invest_ratio,
            min_amount,
            min_total_mv,
        ) = self.par_values
        
        need_len = max(m_days, stock_ma_days, breadth_ma_days, 60)
        if close.shape[0] < need_len:
            return np.zeros(close.shape[1], dtype=float)

        current_dt = self._current_date()
        idx_hist = self._ci_close.loc[self._ci_close.index <= current_dt]
        if len(idx_hist) < max(m_days, 60):
            return np.zeros(close.shape[1], dtype=float)

        # ---------------- 行业初步打分 ----------------
        idx_window = idx_hist.tail(m_days)
        idx_scores = np.array([_score_one(idx_window[c].to_numpy(dtype=float)) for c in idx_window.columns])
        
        if np.all(~np.isfinite(idx_scores)):
            return np.zeros(close.shape[1], dtype=float)

        top_k_industries = int(top_k)
        valid_idx_mask = np.isfinite(idx_scores) & (idx_scores >= min_idx_score)
        if not np.any(valid_idx_mask):
            return np.zeros(close.shape[1], dtype=float)
            
        rank_idx = np.argsort(idx_scores)[::-1]
        best_indices = [idx_window.columns[i] for i in rank_idx if valid_idx_mask[i]][:top_k_industries]
        
        target = np.zeros(close.shape[1], dtype=float)
        if not best_indices:
            return target
            
        share_names = self.share_names

        # ==========================================
        # 颠覆点 1：双引擎广度计算 (Dual-Engine Breadth)
        # ==========================================
        # 引擎 A：全局广度 (Global Breadth)
        close_tail_all = close[-breadth_ma_days:, :]
        ma_all_global = np.nanmean(close_tail_all, axis=0)
        latest_all_global = close[-1, :]
        valid_all_global = np.isfinite(latest_all_global) & np.isfinite(ma_all_global) & (ma_all_global > 0)
        global_breadth = float(np.mean(latest_all_global[valid_all_global] > ma_all_global[valid_all_global])) if np.any(valid_all_global) else 0.0

        # 引擎 B：核心广度 (Core Breadth)
        core_pos = set()
        for current_idx in best_indices:
            candidates, _ = _lookup_constituents(self._con_hist, current_idx, current_dt)
            if candidates:
                core_pos.update([i for i, s in enumerate(share_names) if s in set(candidates)])
        core_pos = list(core_pos)

        if not core_pos:
            return target

        close_core = close[-breadth_ma_days:, core_pos]
        ma_core = np.nanmean(close_core, axis=0)
        latest_core = close[-1, core_pos]
        valid_core = np.isfinite(latest_core) & np.isfinite(ma_core) & (ma_core > 0)
        core_breadth = float(np.mean(latest_core[valid_core] > ma_core[valid_core])) if np.any(valid_core) else 0.0

        # ==========================================
        # 颠覆点 2：牛熊状态机与动态权重配比
        # ==========================================
        # 定义当前大环境状态：全局广度大于 30% 视为暖意环境
        is_warm_market = global_breadth > 0.30

        if is_warm_market:
            # 暖市环境：顺应大盘趋势，全局广度权重占优
            effective_breadth = global_breadth * 0.60 + core_breadth * 0.40
        else:
            # 寒冬环境：大盘全线下跌，必须死死盯住抱团主线，核心广度占据绝对主导
            effective_breadth = global_breadth * 0.20 + core_breadth * 0.80

        # 终极熔断：加权后的广度依然低于 25%，说明是绝望的单边杀跌，一分钱都不投入
        if effective_breadth < 0.25:
            return np.zeros(close.shape[1], dtype=float)

        # 仓位平滑分配：从 0.25 到 0.50 之间动态加满
        smooth_multiplier = min(1.0, (effective_breadth - 0.25) / (0.50 - 0.25))
        dynamic_invest_ratio = float(invest_ratio) * smooth_multiplier
        industry_weight = dynamic_invest_ratio / top_k_industries

        # ---------------- 个股筛选 ----------------
        for current_idx in best_indices:
            idx_series = idx_hist[current_idx].dropna()
            if len(idx_series) < 60: continue
            
            # 【保留 23% 版本的神盾】：行业双均线铁铠，彻底杜绝熊市抄底
            idx_20d_ma = float(idx_series.tail(20).mean())
            idx_60d_ma = float(idx_series.tail(60).mean())
            idx_latest_price = float(idx_series.iloc[-1])
            
            if not (np.isfinite(idx_20d_ma) and np.isfinite(idx_60d_ma)): continue
            if (idx_20d_ma < idx_60d_ma) or (idx_latest_price < idx_20d_ma):
                continue 

            candidates, _ = _lookup_constituents(self._con_hist, current_idx, current_dt)
            if not candidates: continue
            candidate_set = set(candidates)
            pos = [i for i, s in enumerate(share_names) if s in candidate_set]
            if not pos: continue

            latest_mv = total_mv[-1, pos]
            mv_ok = np.isfinite(latest_mv) & (latest_mv >= float(min_total_mv))
            pos = [p for p, ok in zip(pos, mv_ok) if ok]
            if not pos: continue

            amt_ma = np.nanmean(amount[-stock_ma_days:, pos], axis=0)
            liquid_ok = np.isfinite(amt_ma) & (amt_ma >= float(min_amount))
            pos = [p for p, ok in zip(pos, liquid_ok) if ok]
            if not pos: continue

            # 自适应止损：暖市容忍 20% 洗盘，寒冬严苛至 12%
            sl_threshold = 0.80 if is_warm_market else 0.88
            stk_20d_high = np.nanmax(close[-20:, pos], axis=0)
            stk_last_prices = close[-1, pos]
            drawdown_ok = np.isfinite(stk_20d_high) & np.isfinite(stk_last_prices) & (stk_last_prices > stk_20d_high * sl_threshold)
            pos = [p for p, ok in zip(pos, drawdown_ok) if ok]
            if not pos: continue

            # ==========================================
            # 颠覆点 3：状态自适应的 Alpha (Regime-Adaptive Alpha)
            # ==========================================
            stk_last_prices = close[-1, pos]
            stk_20d_ma = np.nanmean(close[-20:, pos], axis=0)
            bias_20 = (stk_last_prices - stk_20d_ma) / (stk_20d_ma + 1e-8)
            
            stk_ret_long = (close[-1, pos] - close[-m_days, pos]) / (close[-m_days, pos] + 1e-8)
            
            prices_m_days = close[-m_days:, pos]
            daily_returns = np.diff(prices_m_days, axis=0) / (prices_m_days[:-1, :] + 1e-8)
            stk_volatility = np.nanstd(daily_returns, axis=0)
            
            vol_5d = np.nanmean(amount[-5:, pos], axis=0)
            vol_15d = np.nanmean(amount[-20:-5, pos], axis=0)
            vol_ratio = vol_5d / (vol_15d + 1e-8)
            
            composite_scores = np.full(len(pos), -np.inf)

            if is_warm_market:
                # 【暖冬模式】：重拳出击，放宽限制，优先绝对动量
                valid_stk_mask = np.isfinite(stk_ret_long) & np.isfinite(stk_volatility) & \
                                 (stk_ret_long > 0) & (bias_20 >= 0.0) & (bias_20 <= 0.15) & (vol_ratio < 2.0)
                
                if np.any(valid_stk_mask):
                    valid_idx = np.where(valid_stk_mask)[0]
                    # 解除大部分波动率惩罚，只做轻微乖离扣分，抓住主升浪猛涨的票
                    composite_scores[valid_idx] = stk_ret_long[valid_idx] - (bias_20[valid_idx] * 1.0)
            else:
                # 【凛冬模式】：极限求稳，极严苛的缩量回踩，强制波动率惩罚
                valid_stk_mask = np.isfinite(stk_ret_long) & np.isfinite(stk_volatility) & \
                                 (stk_ret_long > 0) & (bias_20 >= 0.0) & (bias_20 <= 0.08) & (vol_ratio < 1.2)
                
                if np.any(valid_stk_mask):
                    valid_idx = np.where(valid_stk_mask)[0]
                    # 计算平滑动量，波动越大的票分数越低，并重罚乖离率
                    smooth_momentum = stk_ret_long[valid_idx] / (stk_volatility[valid_idx] + 1e-5)
                    composite_scores[valid_idx] = smooth_momentum - (bias_20[valid_idx] * 3.0) 
            
            if not np.any(valid_stk_mask):
                continue

            rank_stk = np.argsort(composite_scores)[::-1]
            valid_rank_stk = [i for i in rank_stk if valid_stk_mask[i]]
            top_n_actual = min(int(top_n), len(valid_rank_stk))
            
            if top_n_actual > 0:
                chosen = [pos[i] for i in valid_rank_stk[:top_n_actual]]
                stock_weight = industry_weight / float(top_n)
                target[chosen] += stock_weight

        # 换仓摩擦控制
        if switch_th > 0:
            own_amounts = self.get_data("proc.own_amounts", lag=0).reshape(-1)
            prices = self.get_data("proc.trade_price", lag=0).reshape(-1)
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
    parser = argparse.ArgumentParser(description="行业指数轮动（qteasy框架版）")
    parser.add_argument("--ci_parquet", type=str, default=DEFAULT_CI_PARQUET, help="行业指数 parquet")
    parser.add_argument("--start_date", type=str, default="20200101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end_date", type=str, default="20251231", help="结束日期 YYYYMMDD")
    parser.add_argument("--m_days", type=int, default=25, help="动量窗口天数")
    parser.add_argument("--top_k", type=int, default=3, help="持有的行业数量") # NEW
    parser.add_argument("--top_n", type=int, default=3, help="行业内持有个股数量")
    parser.add_argument("--switch_th", type=float, default=0.25, help="仓位变化阈值")
    parser.add_argument("--idx_ma_days", type=int, default=30, help="行业指数趋势均线天数")
    parser.add_argument("--stock_ma_days", type=int, default=20, help="个股趋势均线天数")
    parser.add_argument("--breadth_ma_days", type=int, default=20, help="市场广度均线天数")
    parser.add_argument("--min_idx_score", type=float, default=0.0, help="行业最小动量分数，低于则空仓")
    parser.add_argument("--min_breadth", type=float, default=0.5, help="最小市场广度阈值(0~1)")
    parser.add_argument("--invest_ratio", type=float, default=0.9, help="最大持仓比例(其余为空仓)")
    parser.add_argument("--min_amount", type=float, default=30000.0, help="最近均值最小成交额(千元)")
    parser.add_argument(
        "--min_total_mv",
        type=float,
        default=1_000_000.0,
        help="最小总市值阈值(单位: 万元)，默认100亿",
    )
    parser.add_argument("--run_freq", type=str, default="W", help="策略运行频率，W或D")
    parser.add_argument("--cash", type=float, default=100000.0, help="初始资金")
    parser.add_argument("--download", action="store_true", help="尝试下载 stock_daily/index_daily")
    parser.add_argument("--visual", default=True, action="store_true", help="显示图表")
    parser.add_argument("--trade_log", default=True, action="store_true", help="打印交易日志")
    parser.add_argument("--fig_path", type=str, default="", help="图像保存路径")
    parser.add_argument("--exp_dir", type=str, default="", help="实验输出目录，默认自动创建到脚本目录下 exp/时间戳")
    parser.add_argument("--benchmark_asset", type=str, default="000300.SH", help="基准指数")
    return parser.parse_args()


def _prepare_visual_save(fig_path: str) -> None:
    import matplotlib.pyplot as plt

    original_show = plt.show

    def _save_and_show(*args, **kwargs):
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        original_show(*args, **kwargs)

    plt.show = _save_and_show


def _prepare_experiment_dir(exp_dir: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if exp_dir:
        resolved = os.path.abspath(exp_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved = os.path.join(script_dir, "exp", ts)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _backup_strategy_files(exp_dir: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in ("run_ci_index_rotation.sh", "ci_index_rotation_backtest.py"):
        src = os.path.join(script_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(exp_dir, filename))


def main() -> None:
    args = parse_args()
    exp_dir = _prepare_experiment_dir(args.exp_dir)
    _backup_strategy_files(exp_dir)
    qt.configure(trade_log_file_path=os.path.join(exp_dir, "tradelog"))
    print(f"实验输出目录: {exp_dir}")

    ci_path = os.path.abspath(args.ci_parquet)
    if not os.path.exists(ci_path):
        raise FileNotFoundError(f"行业 parquet 不存在: {ci_path}")

    ci_df = pd.read_parquet(ci_path)
    ci_df = _normalize_ci_df(ci_df)
    start_dt = pd.to_datetime(args.start_date)
    end_dt = pd.to_datetime(args.end_date)
    ci_df = ci_df[(ci_df["trade_date"] >= start_dt) & (ci_df["trade_date"] <= end_dt)].copy()
    if ci_df.empty:
        raise ValueError("行业数据为空，请检查日期范围")

    ci_close = ci_df.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last")
    ci_close = ci_close.sort_index().sort_index(axis=1)

    con_hist = _build_constituent_history(ci_df)
    all_codes = sorted({code for v in ci_df["con_codes"].tolist() for code in _parse_con_codes(v)})
    stock_pool = _normalize_stock_pool(all_codes)
    if not stock_pool:
        raise RuntimeError("从 con_codes 未解析出可交易股票代码")

    ds = qt.QT_DATA_SOURCE
    probe_shares = stock_pool[:200]
    try:
        adj_probe = ds.read_table_data(
            "stock_adj_factor",
            shares=probe_shares,
            start=args.start_date,
            end=args.end_date,
        )
        use_forward_adj = not adj_probe.empty
    except Exception:
        use_forward_adj = False

    print(f"行业指数数量: {ci_close.shape[1]}")
    print(f"候选股票数量: {len(stock_pool)}")
    print(f"价格复权模式: {'forward' if use_forward_adj else 'none'}")

    has_daily = _symbols_with_data(ds, "stock_daily", stock_pool, args.start_date, args.end_date)
    filtered_pool = sorted(set(stock_pool).intersection(has_daily))
    if use_forward_adj:
        has_adj = _symbols_with_data(ds, "stock_adj_factor", filtered_pool, args.start_date, args.end_date)
        filtered_pool = sorted(set(filtered_pool).intersection(has_adj))

    dropped = len(stock_pool) - len(filtered_pool)
    stock_pool = filtered_pool
    print(f"区间内可交易股票数量: {len(stock_pool)} (剔除 {dropped})")
    if not stock_pool:
        raise RuntimeError("过滤后 asset_pool 为空，请检查 stock_daily / stock_adj_factor 数据覆盖区间")

    if args.visual:
        if args.fig_path:
            fig_path = args.fig_path if os.path.isabs(args.fig_path) else os.path.join(exp_dir, args.fig_path)
        else:
            fig_path = os.path.join(
                exp_dir,
                f"ci_index_rotation_backtest_{args.start_date}_{args.end_date}.png",
            )
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        _prepare_visual_save(fig_path)
        print(f"回测图像将保存到: {fig_path}")

    if args.download:
        qt.refill_data_source(
            tables="stock_daily, index_daily",
            start_date=args.start_date,
            end_date=args.end_date,
            symbols=",".join(stock_pool + [args.benchmark_asset]),
        )

    # NEW: 注意 pars 元组里加入了 args.top_k
    pars = (
        args.m_days,
        args.top_k, 
        args.top_n,
        args.switch_th,
        args.idx_ma_days,
        args.stock_ma_days,
        args.breadth_ma_days,
        args.min_idx_score,
        args.min_breadth,
        args.invest_ratio,
        args.min_amount,
        args.min_total_mv,
    )
    alpha = CiIndexRotationStg(
        pars=pars,
        ci_close=ci_close,
        con_hist=con_hist,
        use_forward_adj=use_forward_adj,
    )

    op = qt.Operator(
        strategies=[alpha],
        signal_type="PT",
        op_type="step",
        run_freq=args.run_freq,
        run_timing="close",
    )

    from qteasy.history import check_and_prepare_trade_prices
    op.prepare_running_schedule(start_date=args.start_date, end_date=args.end_date)
    probe_prices = check_and_prepare_trade_prices(
        op=op,
        shares=stock_pool,
        price_adj="f" if use_forward_adj else "none",
        datasource=qt.QT_DATA_SOURCE,
    )
    tradable_cols = [c for c in stock_pool if c in probe_prices.columns]
    if len(tradable_cols) != len(stock_pool):
        print(f"交易价可用列数量: {len(tradable_cols)} (进一步剔除 {len(stock_pool) - len(tradable_cols)})")
        stock_pool = tradable_cols
    if not stock_pool:
        raise RuntimeError("交易价预检查后 asset_pool 为空，请检查价格数据覆盖情况")

    qt.run(
        op,
        mode=1,
        asset_pool=stock_pool,
        asset_type="E",
        benchmark_asset=args.benchmark_asset,
        invest_start=args.start_date,
        invest_end=args.end_date,
        invest_cash_amounts=[args.cash],
        trade_batch_size=100,
        sell_batch_size=100,
        cost_rate_buy=0.0003,
        cost_rate_sell=0.0013,
        backtest_price_adj="f" if use_forward_adj else "none",
        visual=args.visual,
        trade_log=args.trade_log,
    )


if __name__ == "__main__":
    main()
