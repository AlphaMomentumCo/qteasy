#!/usr/bin/env python3
"""Tkinter GUI for backtest trade log analysis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


SUMMARY_ROW_TYPE = "7, summary"
DATE_COL_SUMMARY = "Unnamed: 0"
ROW_TYPE_COL = "Unnamed: 2"


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


class TradeLogAnalyzerApp:
    def __init__(self, root: tk.Tk, summary_path: Path, log_path: Path) -> None:
        self.root = root
        self.root.title("Backtest Trade Log Analyzer")
        self.root.geometry("1420x920")

        self.summary_path_var = tk.StringVar(value=str(summary_path))
        self.log_path_var = tk.StringVar(value=str(log_path))

        self.code_var = tk.StringVar(value="ALL")
        self.side_var = tk.StringVar(value="ALL")
        self.start_date_var = tk.StringVar(value="")
        self.end_date_var = tk.StringVar(value="")

        self.trade_summary_df = pd.DataFrame()
        self.equity_df = pd.DataFrame()
        self.ax_equity_twin = None

        self.metric_vars: dict[str, tk.StringVar] = {}

        self._build_layout()
        self._load_data(initial=True)

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="trade_summary:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(top, textvariable=self.summary_path_var, width=95).grid(
            row=0, column=1, sticky="we", padx=4, pady=3
        )
        ttk.Button(top, text="Browse", command=self._choose_summary_file).grid(
            row=0, column=2, padx=4, pady=3
        )

        ttk.Label(top, text="trade_log:").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(top, textvariable=self.log_path_var, width=95).grid(
            row=1, column=1, sticky="we", padx=4, pady=3
        )
        ttk.Button(top, text="Browse", command=self._choose_log_file).grid(row=1, column=2, padx=4, pady=3)

        ttk.Button(top, text="Reload", command=self._load_data).grid(row=0, column=3, rowspan=2, padx=8)

        top.columnconfigure(1, weight=1)

        filter_frame = ttk.LabelFrame(self.root, text="Filters", padding=8)
        filter_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        ttk.Label(filter_frame, text="Symbol").grid(row=0, column=0, padx=4)
        self.code_combo = ttk.Combobox(filter_frame, textvariable=self.code_var, state="readonly", width=16)
        self.code_combo.grid(row=0, column=1, padx=4)

        ttk.Label(filter_frame, text="Side").grid(row=0, column=2, padx=4)
        side_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.side_var,
            state="readonly",
            values=["ALL", "BUY", "SELL"],
            width=10,
        )
        side_combo.grid(row=0, column=3, padx=4)

        ttk.Label(filter_frame, text="Start (YYYY-MM-DD)").grid(row=0, column=4, padx=4)
        ttk.Entry(filter_frame, textvariable=self.start_date_var, width=14).grid(row=0, column=5, padx=4)

        ttk.Label(filter_frame, text="End (YYYY-MM-DD)").grid(row=0, column=6, padx=4)
        ttk.Entry(filter_frame, textvariable=self.end_date_var, width=14).grid(row=0, column=7, padx=4)

        ttk.Button(filter_frame, text="Apply", command=self._apply_filters).grid(row=0, column=8, padx=8)

        metrics = ttk.Frame(self.root, padding=(8, 4))
        metrics.pack(side=tk.TOP, fill=tk.X)
        metric_labels = [
            "Trades",
            "Buys",
            "Sells",
            "Turnover (Abs)",
            "Fees",
            "Net Cashflow",
            "Final Equity",
            "Max Drawdown",
        ]
        for i, label in enumerate(metric_labels):
            card = ttk.Frame(metrics, relief=tk.RIDGE, borderwidth=1, padding=6)
            card.grid(row=0, column=i, sticky="nsew", padx=4)
            ttk.Label(card, text=label).pack(anchor="w")
            var = tk.StringVar(value="-")
            self.metric_vars[label] = var
            ttk.Label(card, textvariable=var, font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
            metrics.columnconfigure(i, weight=1)

        chart_frame = ttk.Frame(self.root, padding=8)
        chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(14, 8), dpi=100)
        self.ax_equity = self.figure.add_subplot(211)
        self.ax_activity = self.figure.add_subplot(212)
        self.figure.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.LabelFrame(self.root, text="Trade Details", padding=8)
        table_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        columns = ["date", "code", "side", "price", "amount", "cash_changed", "cost", "signal", "position"]
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)

        col_titles = {
            "date": "Datetime",
            "code": "Code",
            "side": "Side",
            "price": "Price",
            "amount": "Amount",
            "cash_changed": "Cash Change",
            "cost": "Cost",
            "signal": "Signal",
            "position": "Position",
        }
        widths = {
            "date": 155,
            "code": 100,
            "side": 70,
            "price": 85,
            "amount": 90,
            "cash_changed": 110,
            "cost": 90,
            "signal": 80,
            "position": 90,
        }

        for col in columns:
            self.tree.heading(col, text=col_titles[col])
            self.tree.column(col, width=widths[col], anchor="center")

        yscroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _choose_summary_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*")])
        if path:
            self.summary_path_var.set(path)

    def _choose_log_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*")])
        if path:
            self.log_path_var.set(path)

    def _load_data(self, initial: bool = False) -> None:
        try:
            summary_path = Path(self.summary_path_var.get()).expanduser()
            log_path = Path(self.log_path_var.get()).expanduser()

            if not summary_path.exists():
                raise FileNotFoundError(f"trade_summary not found: {summary_path}")
            if not log_path.exists():
                raise FileNotFoundError(f"trade_log not found: {log_path}")

            trade_summary = pd.read_csv(summary_path)
            usecols = [DATE_COL_SUMMARY, ROW_TYPE_COL, "add. invest", "own cash", "available cash", "value"]
            trade_log = pd.read_csv(log_path, usecols=usecols)

            trade_summary = self._prepare_trade_summary(trade_summary)
            equity_df = self._prepare_equity(trade_log)

            if trade_summary.empty:
                raise ValueError("trade_summary is empty after parsing")
            if equity_df.empty:
                raise ValueError("No '7, summary' rows found in trade_log")

            self.trade_summary_df = trade_summary
            self.equity_df = equity_df

            code_values = ["ALL"] + sorted(self.trade_summary_df["code"].dropna().astype(str).unique().tolist())
            self.code_combo["values"] = code_values
            if self.code_var.get() not in code_values:
                self.code_var.set("ALL")

            min_date = self.trade_summary_df["date"].min().date().isoformat()
            max_date = self.trade_summary_df["date"].max().date().isoformat()

            if initial or not self.start_date_var.get():
                self.start_date_var.set(min_date)
            if initial or not self.end_date_var.get():
                self.end_date_var.set(max_date)

            self._apply_filters()
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Load Error", str(exc))

    def _prepare_trade_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df[DATE_COL_SUMMARY], errors="coerce")
        df = df[df["date"].notna()]

        numeric_cols = [
            "0, trade signal",
            "1, price",
            "2, traded amounts",
            "3, cash changed",
            "4, trade cost",
            "5, own amounts",
            "6, available amounts",
            "7, summary",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = _safe_to_numeric(df[col])

        df["code"] = df.get("code", "").astype(str)
        df["side"] = df["2, traded amounts"].map(lambda x: "BUY" if x > 0 else "SELL")
        return df.sort_values("date")

    def _prepare_equity(self, df: pd.DataFrame) -> pd.DataFrame:
        if ROW_TYPE_COL not in df.columns:
            raise ValueError("trade_log missing column: Unnamed: 2")

        eq = df[df[ROW_TYPE_COL] == SUMMARY_ROW_TYPE].copy()
        eq["date"] = pd.to_datetime(eq[DATE_COL_SUMMARY], errors="coerce")
        eq = eq[eq["date"].notna()].sort_values("date")

        for col in ["value", "own cash", "available cash", "add. invest"]:
            if col in eq.columns:
                eq[col] = _safe_to_numeric(eq[col])

        if "value" not in eq.columns:
            raise ValueError("trade_log missing value column")

        eq = eq[["date", "value", "own cash", "available cash"]].copy()
        eq["value"] = eq["value"].ffill()
        eq["peak"] = eq["value"].cummax()
        eq["drawdown"] = eq["value"] / eq["peak"] - 1.0
        return eq

    def _apply_filters(self) -> None:
        if self.trade_summary_df.empty or self.equity_df.empty:
            return

        try:
            start = pd.to_datetime(self.start_date_var.get())
            end = pd.to_datetime(self.end_date_var.get())
            if pd.isna(start) or pd.isna(end):
                raise ValueError("Invalid date format")
            if start > end:
                start, end = end, start

            mask_date = (self.trade_summary_df["date"] >= start) & (self.trade_summary_df["date"] <= end)
            trades = self.trade_summary_df.loc[mask_date].copy()

            code = self.code_var.get().strip()
            if code and code != "ALL":
                trades = trades[trades["code"] == code]

            side = self.side_var.get().strip().upper()
            if side in {"BUY", "SELL"}:
                trades = trades[trades["side"] == side]

            eq = self.equity_df[(self.equity_df["date"] >= start) & (self.equity_df["date"] <= end)].copy()

            self._update_metrics(trades, eq)
            self._update_charts(trades, eq)
            self._update_table(trades)
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Filter Error", str(exc))

    def _update_metrics(self, trades: pd.DataFrame, eq: pd.DataFrame) -> None:
        trade_count = int(len(trades))
        buy_count = int((trades["side"] == "BUY").sum()) if not trades.empty else 0
        sell_count = int((trades["side"] == "SELL").sum()) if not trades.empty else 0

        turnover = float(trades["3, cash changed"].abs().sum()) if not trades.empty else 0.0
        fee = float(trades["4, trade cost"].sum()) if not trades.empty else 0.0
        cashflow = float(trades["3, cash changed"].sum()) if not trades.empty else 0.0

        if not eq.empty:
            final_value = float(eq["value"].iloc[-1])
            max_dd = float(eq["drawdown"].min())
        else:
            final_value = math.nan
            max_dd = math.nan

        self.metric_vars["Trades"].set(f"{trade_count:,}")
        self.metric_vars["Buys"].set(f"{buy_count:,}")
        self.metric_vars["Sells"].set(f"{sell_count:,}")
        self.metric_vars["Turnover (Abs)"].set(f"{turnover:,.2f}")
        self.metric_vars["Fees"].set(f"{fee:,.2f}")
        self.metric_vars["Net Cashflow"].set(f"{cashflow:,.2f}")
        self.metric_vars["Final Equity"].set("-" if math.isnan(final_value) else f"{final_value:,.2f}")
        self.metric_vars["Max Drawdown"].set("-" if math.isnan(max_dd) else f"{max_dd:.2%}")

    def _update_charts(self, trades: pd.DataFrame, eq: pd.DataFrame) -> None:
        if self.ax_equity_twin is not None:
            self.ax_equity_twin.remove()
            self.ax_equity_twin = None

        self.ax_equity.clear()
        self.ax_activity.clear()

        if not eq.empty:
            self.ax_equity.plot(eq["date"], eq["value"], label="Equity", linewidth=1.8)
            if "own cash" in eq.columns:
                self.ax_equity.plot(eq["date"], eq["own cash"], label="Own Cash", linewidth=1.0, alpha=0.85)
            if "available cash" in eq.columns:
                self.ax_equity.plot(
                    eq["date"],
                    eq["available cash"],
                    label="Available Cash",
                    linewidth=1.0,
                    alpha=0.85,
                )
            self.ax_equity_twin = self.ax_equity.twinx()
            self.ax_equity_twin.fill_between(eq["date"], eq["drawdown"], 0, color="tab:red", alpha=0.18)
            self.ax_equity_twin.set_ylabel("Drawdown")
            self.ax_equity_twin.set_ylim(min(eq["drawdown"].min() * 1.2, -0.01), 0.02)
            self.ax_equity.set_title("Equity / Cash and Drawdown")
            self.ax_equity.set_ylabel("Value")
            self.ax_equity.grid(alpha=0.25)
            self.ax_equity.legend(loc="upper left")
        else:
            self.ax_equity.text(0.5, 0.5, "No equity data in selected range", ha="center", va="center")

        if not trades.empty:
            daily = trades.groupby(trades["date"].dt.date).agg(
                trade_count=("code", "count"), turnover=("3, cash changed", lambda s: s.abs().sum())
            )
            x = pd.to_datetime(daily.index)
            self.ax_activity.bar(x, daily["turnover"], width=2.0, alpha=0.6, label="Daily Turnover (Abs)")
            ax2 = self.ax_activity.twinx()
            ax2.plot(x, daily["trade_count"], color="tab:orange", linewidth=1.5, label="Daily Trade Count")
            ax2.set_ylabel("Count")

            self.ax_activity.set_title("Trading Activity")
            self.ax_activity.set_ylabel("Turnover")
            self.ax_activity.grid(alpha=0.25)
            self.ax_activity.legend(loc="upper left")
            ax2.legend(loc="upper right")
        else:
            self.ax_activity.text(0.5, 0.5, "No trades under current filters", ha="center", va="center")

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def _update_table(self, trades: pd.DataFrame) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        if trades.empty:
            return

        view = trades.sort_values("date", ascending=False).head(300)
        for _, row in view.iterrows():
            vals = (
                row["date"].strftime("%Y-%m-%d %H:%M:%S"),
                row.get("code", ""),
                row.get("side", ""),
                f"{row.get('1, price', float('nan')):.3f}",
                f"{row.get('2, traded amounts', float('nan')):.0f}",
                f"{row.get('3, cash changed', float('nan')):.2f}",
                f"{row.get('4, trade cost', float('nan')):.2f}",
                f"{row.get('0, trade signal', float('nan')):.3f}",
                f"{row.get('5, own amounts', float('nan')):.0f}",
            )
            self.tree.insert("", tk.END, values=vals)


def parse_args() -> argparse.Namespace:
    default_base = Path(__file__).resolve().parent / "tradelog_my_run"
    parser = argparse.ArgumentParser(description="Backtest trade log visual analyzer")
    parser.add_argument(
        "--summary",
        type=Path,
        default=default_base / "trade_summary_None_20260308_221445.csv",
        help="Path to trade_summary CSV",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=default_base / "trade_log_None_20260308_221445.csv",
        help="Path to trade_log CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = TradeLogAnalyzerApp(root, args.summary, args.log)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
