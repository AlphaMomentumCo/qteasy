#!/usr/bin/env python3
"""Streamlit web dashboard for backtest trade log analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

SUMMARY_ROW_TYPE = "7, summary"
PRICE_ROW_TYPE = "1, price"
OWN_ROW_TYPE = "5, own amounts"
DATE_COL_SUMMARY = "Unnamed: 0"
ROW_TYPE_COL = "Unnamed: 2"
META_COLS = {"Unnamed: 0", "Unnamed: 1", "Unnamed: 2", "add. invest", "own cash", "available cash", "value"}


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_cli_args() -> argparse.Namespace:
    """Parse app-specific args when launched by `streamlit run app.py -- ...`."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--summary", type=str, default="")
    parser.add_argument("--log", type=str, default="")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def _build_drawdown_events(
    eq_df: pd.DataFrame,
    top_n: int = 5,
    min_gap_days: int = 20,
    min_drawdown_abs: float = 0.10,
) -> pd.DataFrame:
    if eq_df.empty:
        return pd.DataFrame(columns=["date", "drawdown", "value"])

    candidates = eq_df[eq_df["drawdown"] <= -abs(min_drawdown_abs)].sort_values("drawdown", ascending=True).copy()
    if candidates.empty:
        return pd.DataFrame(columns=["date", "drawdown", "value"])
    picked = []
    picked_dates: list[pd.Timestamp] = []

    for _, row in candidates.iterrows():
        dt = row["date"]
        if all(abs((dt - p).days) >= min_gap_days for p in picked_dates):
            picked.append(row)
            picked_dates.append(dt)
        if len(picked) >= top_n:
            break

    if not picked:
        return pd.DataFrame(columns=["date", "drawdown", "value"])
    return pd.DataFrame(picked).sort_values("drawdown", ascending=True).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_data(summary_path: str, log_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_file = Path(summary_path)
    log_file = Path(log_path)
    if not summary_file.exists():
        raise FileNotFoundError(f"trade_summary 文件不存在: {summary_file}")
    if not log_file.exists():
        raise FileNotFoundError(f"trade_log 文件不存在: {log_file}")

    ts = pd.read_csv(summary_file)
    tl = pd.read_csv(log_file, low_memory=False)

    ts["date"] = pd.to_datetime(ts[DATE_COL_SUMMARY], errors="coerce")
    ts = ts[ts["date"].notna()].copy()
    for col in [
        "0, trade signal",
        "1, price",
        "2, traded amounts",
        "3, cash changed",
        "4, trade cost",
        "5, own amounts",
        "6, available amounts",
        "7, summary",
    ]:
        if col in ts.columns:
            ts[col] = _safe_to_numeric(ts[col])
    ts["code"] = ts.get("code", "").astype(str)
    ts["side"] = ts["2, traded amounts"].map(lambda x: "BUY" if x > 0 else "SELL")
    ts = ts.sort_values("date")

    tl["date"] = pd.to_datetime(tl[DATE_COL_SUMMARY], errors="coerce")
    tl = tl[tl["date"].notna()].copy()

    eq = tl[tl[ROW_TYPE_COL] == SUMMARY_ROW_TYPE].copy()
    for col in ["value", "own cash", "available cash"]:
        if col in eq.columns:
            eq[col] = _safe_to_numeric(eq[col])
    eq = eq[["date", "value", "own cash", "available cash"]].sort_values("date")
    eq["value"] = eq["value"].ffill()
    eq["peak"] = eq["value"].cummax()
    eq["drawdown"] = eq["value"] / eq["peak"] - 1.0

    stock_cols = [c for c in tl.columns if c not in META_COLS and c != "date"]

    own = tl[tl[ROW_TYPE_COL] == OWN_ROW_TYPE][["date", *stock_cols]].copy().sort_values("date")
    price = tl[tl[ROW_TYPE_COL] == PRICE_ROW_TYPE][["date", *stock_cols]].copy().sort_values("date")

    for col in stock_cols:
        own[col] = _safe_to_numeric(own[col]).fillna(0.0)
        price[col] = _safe_to_numeric(price[col]).fillna(0.0)

    return ts, eq, own, price


def _latest_not_after(df: pd.DataFrame, dt: pd.Timestamp) -> pd.Series | None:
    sub = df[df["date"] <= dt]
    if sub.empty:
        return None
    return sub.iloc[-1]


def _last_entry_date_for_code(own_df: pd.DataFrame, code: str, trough_dt: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    series = own_df.loc[own_df["date"] <= trough_dt, ["date", code]].copy()
    if series.empty:
        return pd.NaT
    series[code] = _safe_to_numeric(series[code]).fillna(0.0)
    series["prev"] = series[code].shift(1).fillna(0.0)
    entries = series[(series["prev"] <= 0) & (series[code] > 0)]
    if not entries.empty:
        return entries["date"].iloc[-1]
    if float(series[code].iloc[-1]) > 0:
        first_positive = series[series[code] > 0]
        if not first_positive.empty:
            return first_positive["date"].iloc[0]
    return pd.NaT


def _build_holdings_at_trough(
    own_df: pd.DataFrame,
    price_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    trough_dt: pd.Timestamp,
) -> pd.DataFrame:
    own_row = _latest_not_after(own_df, trough_dt)
    price_row = _latest_not_after(price_df, trough_dt)
    if own_row is None or price_row is None:
        return pd.DataFrame()

    stock_cols = [c for c in own_df.columns if c != "date"]
    records = []

    for code in stock_cols:
        amount = float(own_row.get(code, 0.0) or 0.0)
        if amount <= 0:
            continue
        px = float(price_row.get(code, 0.0) or 0.0)
        mv = amount * px

        entry_date = _last_entry_date_for_code(own_df, code, trough_dt)
        days_held = (trough_dt.date() - entry_date.date()).days if pd.notna(entry_date) else None

        entry_price = None
        if pd.notna(entry_date):
            price_entry_row = _latest_not_after(price_df, entry_date)
            if price_entry_row is not None:
                entry_price = float(price_entry_row.get(code, 0.0) or 0.0)

        last_buy = trades_df[
            (trades_df["code"] == code) & (trades_df["side"] == "BUY") & (trades_df["date"] <= trough_dt)
        ]
        last_buy_dt = last_buy["date"].iloc[-1] if not last_buy.empty else pd.NaT

        records.append(
            {
                "code": code,
                "amount": amount,
                "price_at_trough": px,
                "market_value": mv,
                "entry_date": entry_date,
                "days_held": days_held,
                "entry_price": entry_price,
                "last_buy_trade_date": last_buy_dt,
            }
        )

    holdings = pd.DataFrame(records)
    if holdings.empty:
        return holdings

    total_mv = holdings["market_value"].sum()
    holdings["weight"] = holdings["market_value"] / total_mv if total_mv else 0.0
    return holdings.sort_values("market_value", ascending=False).reset_index(drop=True)


def main() -> None:
    cli_args = _parse_cli_args()
    base = Path(__file__).resolve().parent / "tradelog_my_run"
    default_summary = Path(cli_args.summary) if cli_args.summary else base / "trade_summary_None_20260308_221445.csv"
    default_log = Path(cli_args.log) if cli_args.log else base / "trade_log_None_20260308_221445.csv"

    st.set_page_config(page_title="回测交易日志分析", layout="wide")
    st.title("回测交易日志分析")
    st.caption("聚焦大回撤：定位谷底附近持仓、建仓时间与交易行为")

    with st.sidebar:
        st.header("数据文件")
        summary_path = st.text_input("trade_summary", value=str(default_summary))
        log_path = st.text_input("trade_log", value=str(default_log))

    try:
        trades_df, eq_df, own_df, price_df = load_data(summary_path, log_path)
    except Exception as exc:
        st.error(str(exc))
        return

    min_date = trades_df["date"].min().date()
    max_date = trades_df["date"].max().date()

    with st.sidebar:
        st.header("交易筛选")
        date_range = st.date_input("日期区间", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        code_options = ["ALL"] + sorted(trades_df["code"].dropna().unique().tolist())
        code = st.selectbox("股票代码", options=code_options)
        side = st.selectbox("方向", options=["ALL", "BUY", "SELL"])

        st.header("回撤分析")
        dd_top_n = st.slider("候选大回撤数量", min_value=3, max_value=12, value=6)
        dd_threshold = st.slider("最小回撤阈值(%)", min_value=5, max_value=40, value=10, step=1)
        dd_gap = st.slider("回撤事件最小间隔(天)", min_value=5, max_value=90, value=20)
        around_days = st.slider("谷底前后观察窗口(天)", min_value=10, max_value=120, value=40)
        holdings_top_n = st.slider("持仓展示数量", min_value=5, max_value=50, value=20)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = min_date

    trades = trades_df[(trades_df["date"].dt.date >= start_date) & (trades_df["date"].dt.date <= end_date)].copy()
    eq = eq_df[(eq_df["date"].dt.date >= start_date) & (eq_df["date"].dt.date <= end_date)].copy()

    if code != "ALL":
        trades = trades[trades["code"] == code]
    if side in {"BUY", "SELL"}:
        trades = trades[trades["side"] == side]

    trade_count = int(len(trades))
    buy_count = int((trades["side"] == "BUY").sum()) if not trades.empty else 0
    sell_count = int((trades["side"] == "SELL").sum()) if not trades.empty else 0
    turnover = float(trades["3, cash changed"].abs().sum()) if not trades.empty else 0.0
    fee = float(trades["4, trade cost"].sum()) if not trades.empty else 0.0
    cashflow = float(trades["3, cash changed"].sum()) if not trades.empty else 0.0
    final_value = float(eq["value"].iloc[-1]) if not eq.empty else float("nan")
    max_dd = float(eq["drawdown"].min()) if not eq.empty else float("nan")

    st.markdown("### 总览指标")
    cols = st.columns(8)
    cols[0].metric("交易笔数", f"{trade_count:,}")
    cols[1].metric("买入", f"{buy_count:,}")
    cols[2].metric("卖出", f"{sell_count:,}")
    cols[3].metric("成交额(绝对值)", f"{turnover:,.2f}")
    cols[4].metric("手续费", f"{fee:,.2f}")
    cols[5].metric("净现金流", f"{cashflow:,.2f}")
    cols[6].metric("期末净值", "-" if pd.isna(final_value) else f"{final_value:,.2f}")
    cols[7].metric("最大回撤", "-" if pd.isna(max_dd) else f"{max_dd:.2%}")

    drawdown_events = _build_drawdown_events(
        eq,
        top_n=dd_top_n,
        min_gap_days=dd_gap,
        min_drawdown_abs=dd_threshold / 100.0,
    )

    selected_trough = pd.NaT
    selected_dd = None
    if not drawdown_events.empty:
        st.markdown("### 回撤事件选择")
        options = {
            f"{r['date'].strftime('%Y-%m-%d')} | 回撤 {r['drawdown']:.2%} | 净值 {r['value']:.2f}": r
            for _, r in drawdown_events.iterrows()
        }
        selected_label = st.selectbox("选择回撤事件(谷底)", list(options.keys()))
        selected_row = options[selected_label]
        selected_trough = selected_row["date"]
        selected_dd = float(selected_row["drawdown"])
        with st.expander("候选回撤事件列表", expanded=False):
            dd_view = drawdown_events.copy()
            dd_view["date"] = dd_view["date"].dt.strftime("%Y-%m-%d")
            dd_view["drawdown"] = dd_view["drawdown"].map(lambda x: f"{x:.2%}")
            dd_view["value"] = dd_view["value"].map(lambda x: f"{x:,.2f}")
            st.dataframe(
                dd_view.rename(columns={"date": "谷底日期", "drawdown": "回撤", "value": "净值"}),
                use_container_width=True,
            )
    else:
        st.warning(f"当前区间内没有回撤超过 {dd_threshold}% 的事件。可下调阈值或扩大日期范围。")

    fig_eq = make_subplots(specs=[[{"secondary_y": True}]])
    if not eq.empty:
        fig_eq.add_trace(go.Scatter(x=eq["date"], y=eq["value"], mode="lines", name="净值", line=dict(width=2)), secondary_y=False)
        fig_eq.add_trace(go.Scatter(x=eq["date"], y=eq["own cash"], mode="lines", name="own cash"), secondary_y=False)
        fig_eq.add_trace(go.Scatter(x=eq["date"], y=eq["available cash"], mode="lines", name="available cash"), secondary_y=False)
        fig_eq.add_trace(
            go.Scatter(x=eq["date"], y=eq["drawdown"], mode="lines", fill="tozeroy", name="回撤", opacity=0.25),
            secondary_y=True,
        )

        if pd.notna(selected_trough):
            win_start = selected_trough - pd.Timedelta(days=around_days)
            win_end = selected_trough + pd.Timedelta(days=around_days)
            fig_eq.add_vrect(x0=win_start, x1=win_end, fillcolor="orange", opacity=0.1, line_width=0)
            fig_eq.add_vline(x=selected_trough, line_color="red", line_dash="dash")

    fig_eq.update_layout(
        height=430,
        title="净值/现金与回撤（含选中回撤窗口）",
        legend_orientation="h",
        hovermode="x unified",
        template="plotly_white",
    )
    fig_eq.update_yaxes(title_text="Value", secondary_y=False)
    fig_eq.update_yaxes(title_text="Drawdown", secondary_y=True)
    st.markdown("### 净值与回撤")
    st.plotly_chart(fig_eq, use_container_width=True)

    if pd.notna(selected_trough):
        st.subheader("回撤谷底持仓与建仓时间")
        info_cols = st.columns(4)
        info_cols[0].metric("谷底日期", selected_trough.strftime("%Y-%m-%d"))
        info_cols[1].metric("谷底回撤", f"{selected_dd:.2%}" if selected_dd is not None else "-")

        holdings = _build_holdings_at_trough(own_df, price_df, trades_df, selected_trough)
        if holdings.empty:
            st.warning("该回撤谷底附近未解析到持仓数据")
        else:
            total_mv = float(holdings["market_value"].sum())
            weighted_age = (holdings["weight"] * holdings["days_held"].fillna(0)).sum()
            info_cols[2].metric("谷底持仓市值(估算)", f"{total_mv:,.2f}")
            info_cols[3].metric("持仓加权持有天数", f"{weighted_age:,.1f}")

            show_holdings = holdings.head(holdings_top_n).copy()

            fig_hold = go.Figure()
            fig_hold.add_trace(
                go.Bar(
                    x=show_holdings["code"],
                    y=show_holdings["weight"] * 100,
                    customdata=show_holdings[["entry_date", "days_held", "market_value"]],
                    hovertemplate=(
                        "代码: %{x}<br>权重: %{y:.2f}%<br>建仓时间: %{customdata[0]}"
                        "<br>持有天数: %{customdata[1]}<br>市值: %{customdata[2]:,.2f}<extra></extra>"
                    ),
                    name="仓位权重",
                )
            )
            fig_hold.update_layout(height=340, title=f"谷底持仓权重 Top {holdings_top_n}")
            st.plotly_chart(fig_hold, use_container_width=True)

            scatter_df = show_holdings.dropna(subset=["entry_date"]).copy()
            if not scatter_df.empty:
                fig_entry = go.Figure()
                fig_entry.add_trace(
                    go.Scatter(
                        x=scatter_df["entry_date"],
                        y=scatter_df["market_value"],
                        mode="markers+text",
                        text=scatter_df["code"],
                        textposition="top center",
                        marker=dict(size=10),
                        name="建仓点",
                    )
                )
                fig_entry.add_vline(x=selected_trough, line_color="red", line_dash="dash")
                fig_entry.update_layout(height=320, title="当前持仓的建仓时间分布（越靠右越晚买）")
                fig_entry.update_yaxes(title_text="谷底时市值")
                st.plotly_chart(fig_entry, use_container_width=True)

            table_cols = [
                "code",
                "amount",
                "price_at_trough",
                "market_value",
                "weight",
                "entry_date",
                "entry_price",
                "days_held",
                "last_buy_trade_date",
            ]
            show_df = show_holdings[table_cols].copy()
            show_df["weight"] = show_df["weight"].map(lambda x: f"{x:.2%}")
            show_df["amount"] = show_df["amount"].map(lambda x: f"{x:,.0f}")
            show_df["price_at_trough"] = show_df["price_at_trough"].map(lambda x: f"{x:,.3f}")
            show_df["market_value"] = show_df["market_value"].map(lambda x: f"{x:,.2f}")
            show_df["entry_price"] = show_df["entry_price"].map(lambda x: "-" if pd.isna(x) else f"{x:,.3f}")
            show_df["entry_date"] = show_df["entry_date"].map(lambda x: "-" if pd.isna(x) else x.strftime("%Y-%m-%d"))
            show_df["last_buy_trade_date"] = show_df["last_buy_trade_date"].map(
                lambda x: "-" if pd.isna(x) else x.strftime("%Y-%m-%d %H:%M:%S")
            )
            st.dataframe(
                show_df.rename(
                    columns={
                        "code": "代码",
                        "amount": "持仓数量",
                        "price_at_trough": "谷底价格",
                        "market_value": "谷底市值",
                        "weight": "仓位权重",
                        "entry_date": "建仓日期",
                        "entry_price": "建仓价",
                        "days_held": "持有天数",
                        "last_buy_trade_date": "最近买入交易时间",
                    }
                ),
                use_container_width=True,
                height=360,
            )

            if not show_holdings.empty:
                selected_code = st.selectbox("查看某个持仓在回撤窗口内的交易", show_holdings["code"].tolist())
                win_start = selected_trough - pd.Timedelta(days=around_days)
                win_end = selected_trough + pd.Timedelta(days=around_days)
                code_trades = trades_df[
                    (trades_df["code"] == selected_code)
                    & (trades_df["date"] >= win_start)
                    & (trades_df["date"] <= win_end)
                ].copy()
                st.caption(f"{selected_code} 在窗口 {win_start.date()} ~ {win_end.date()} 的交易记录")
                if code_trades.empty:
                    st.info("该窗口内没有该股票交易")
                else:
                    st.dataframe(
                        code_trades.sort_values("date")[
                            [
                                "date",
                                "side",
                                "1, price",
                                "2, traded amounts",
                                "3, cash changed",
                                "4, trade cost",
                                "5, own amounts",
                            ]
                        ],
                        use_container_width=True,
                        height=240,
                    )

    daily = pd.DataFrame()
    if not trades.empty:
        daily = trades.groupby(trades["date"].dt.date).agg(
            trade_count=("code", "count"), turnover=("3, cash changed", lambda s: s.abs().sum())
        )

    fig_daily = make_subplots(specs=[[{"secondary_y": True}]])
    if not daily.empty:
        x = pd.to_datetime(daily.index)
        fig_daily.add_trace(go.Bar(x=x, y=daily["turnover"], name="日成交额(绝对值)", opacity=0.65), secondary_y=False)
        fig_daily.add_trace(
            go.Scatter(x=x, y=daily["trade_count"], mode="lines+markers", name="日交易笔数"),
            secondary_y=True,
        )
    fig_daily.update_layout(height=320, title="交易活跃度", legend_orientation="h", template="plotly_white")
    fig_daily.update_yaxes(title_text="Turnover", secondary_y=False)
    fig_daily.update_yaxes(title_text="Count", secondary_y=True)
    st.markdown("### 交易活跃度")
    st.plotly_chart(fig_daily, use_container_width=True)

    st.subheader("交易明细")
    if trades.empty:
        st.info("当前筛选条件下没有交易记录")
    else:
        show_cols = [
            "date",
            "code",
            "side",
            "1, price",
            "2, traded amounts",
            "3, cash changed",
            "4, trade cost",
            "0, trade signal",
            "5, own amounts",
        ]
        detail_df = trades.sort_values("date", ascending=False)[show_cols].copy()
        detail_df["date"] = detail_df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        detail_df = detail_df.rename(
            columns={
                "date": "日期",
                "code": "代码",
                "side": "方向",
                "1, price": "价格",
                "2, traded amounts": "成交数量",
                "3, cash changed": "现金变化",
                "4, trade cost": "手续费",
                "0, trade signal": "信号",
                "5, own amounts": "持仓数量",
            }
        )
        st.dataframe(detail_df, use_container_width=True, height=400)


if __name__ == "__main__":
    main()
