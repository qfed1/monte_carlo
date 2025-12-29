from __future__ import annotations

import base64
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

POLICY_SUFFIX = "_FAIRVALUE_SIGMA_POLICY_"
DEFAULT_TAIL_DAYS = 10
DEFAULT_BOOTSTRAP_SIMS = 2000


@dataclass
class ContractData:
    contract: str
    split: str
    df: pd.DataFrame


@dataclass
class SummaryStats:
    contract: str
    split: str
    bars: int
    days: int
    total_pnl: float
    avg_daily_pnl: float
    median_daily_pnl: float
    win_rate: float
    pnl_std_daily: float
    sharpe_daily: float
    pnl_per_bar: float
    pnl_std_bar: float


@dataclass
class MonteCarloResult:
    totals: List[float]
    max_drawdowns: List[float]
    pass_rate: float
    rules: Dict[str, float]


def _parse_contract_split(filename: str) -> Optional[Tuple[str, str]]:
    if POLICY_SUFFIX not in filename:
        return None
    base = os.path.basename(filename)
    if not base.endswith(".csv"):
        return None
    left, right = base.split(POLICY_SUFFIX, 1)
    split = right.replace(".csv", "")
    return left, split


def load_policy_csvs(data_dir: str) -> List[ContractData]:
    results: List[ContractData] = []
    for entry in os.listdir(data_dir):
        if POLICY_SUFFIX not in entry:
            continue
        parsed = _parse_contract_split(entry)
        if not parsed:
            continue
        contract, split = parsed
        path = os.path.join(data_dir, entry)
        df = pd.read_csv(path)
        results.append(ContractData(contract=contract, split=split, df=df))
    return results


def ensure_day_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Day" in df.columns:
        return df
    if "BarTime" in df.columns:
        df = df.copy()
        df["Day"] = pd.to_datetime(df["BarTime"]).dt.date.astype(str)
        return df
    raise ValueError("CSV missing 'Day' or 'BarTime' columns.")


def compute_daily_pnl(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_day_column(df)
    if "PnL_ticks" not in df.columns:
        raise ValueError("CSV missing 'PnL_ticks' column.")
    daily = df.groupby("Day", sort=True)["PnL_ticks"].sum().reset_index()
    daily.rename(columns={"PnL_ticks": "DailyPnL"}, inplace=True)
    return daily


def compute_summary(df: pd.DataFrame, contract: str, split: str) -> SummaryStats:
    df = ensure_day_column(df)
    daily = compute_daily_pnl(df)
    bars = len(df)
    days = len(daily)
    total = float(daily["DailyPnL"].sum())
    avg_daily = float(daily["DailyPnL"].mean()) if days else 0.0
    median_daily = float(daily["DailyPnL"].median()) if days else 0.0
    win_rate = float((daily["DailyPnL"] > 0).mean()) if days else 0.0
    pnl_std_daily = float(daily["DailyPnL"].std(ddof=0)) if days else 0.0
    sharpe_daily = float(avg_daily / pnl_std_daily) if pnl_std_daily else 0.0
    pnl_per_bar = float(df["PnL_ticks"].mean()) if bars else 0.0
    pnl_std_bar = float(df["PnL_ticks"].std(ddof=0)) if bars else 0.0
    return SummaryStats(
        contract=contract,
        split=split,
        bars=bars,
        days=days,
        total_pnl=total,
        avg_daily_pnl=avg_daily,
        median_daily_pnl=median_daily,
        win_rate=win_rate,
        pnl_std_daily=pnl_std_daily,
        sharpe_daily=sharpe_daily,
        pnl_per_bar=pnl_per_bar,
        pnl_std_bar=pnl_std_bar,
    )


def state_pnl_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "StateID" not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby("StateID")["PnL_ticks"].agg(["count", "sum", "mean", "std"]).reset_index()
    grouped.rename(columns={"sum": "total_pnl", "mean": "mean_pnl", "std": "std_pnl"}, inplace=True)
    grouped["std_pnl"] = grouped["std_pnl"].fillna(0.0)
    return grouped


def tail_risk_state_attribution(df: pd.DataFrame, tail_days: int = DEFAULT_TAIL_DAYS) -> pd.DataFrame:
    if "StateID" not in df.columns:
        return pd.DataFrame()
    df = ensure_day_column(df)
    daily = compute_daily_pnl(df)
    worst_days = daily.nsmallest(tail_days, "DailyPnL")["Day"].tolist()
    tail = df[df["Day"].isin(worst_days)]
    if tail.empty:
        return pd.DataFrame()
    state_sum = tail.groupby("StateID")["PnL_ticks"].sum().reset_index()
    total = float(state_sum["PnL_ticks"].sum())
    if total == 0:
        state_sum["contrib"] = 0.0
    else:
        state_sum["contrib"] = state_sum["PnL_ticks"] / total
    state_sum.rename(columns={"PnL_ticks": "tail_pnl"}, inplace=True)
    return state_sum.sort_values("tail_pnl")


def _bootstrap_sample(daily: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(daily), len(daily))
    return daily[idx]


def max_drawdown(series: np.ndarray) -> float:
    cumulative = np.cumsum(series)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    return float(drawdown.min(initial=0.0))


def monte_carlo_bootstrap(
    daily_pnl: Iterable[float],
    sims: int = DEFAULT_BOOTSTRAP_SIMS,
    rules: Optional[Dict[str, float]] = None,
) -> MonteCarloResult:
    pnl = np.array(list(daily_pnl), dtype=np.float64)
    if len(pnl) == 0:
        return MonteCarloResult(totals=[], max_drawdowns=[], pass_rate=0.0, rules=rules or {})

    rng = np.random.default_rng(7)
    totals = []
    mdds = []
    for _ in range(sims):
        sample = _bootstrap_sample(pnl, rng)
        totals.append(float(sample.sum()))
        mdds.append(max_drawdown(sample))

    rules = rules or {"min_total_pnl": 0.0, "max_drawdown": -2000.0}
    passes = 0
    for total, mdd in zip(totals, mdds):
        if total >= rules["min_total_pnl"] and mdd >= rules["max_drawdown"]:
            passes += 1
    pass_rate = passes / sims if sims else 0.0
    return MonteCarloResult(totals=totals, max_drawdowns=mdds, pass_rate=pass_rate, rules=rules)


def markov_monte_carlo(
    df: pd.DataFrame,
    transition: np.ndarray,
    sims: int = DEFAULT_BOOTSTRAP_SIMS,
) -> MonteCarloResult:
    if "StateID" not in df.columns:
        return MonteCarloResult(totals=[], max_drawdowns=[], pass_rate=0.0, rules={})
    df = ensure_day_column(df)
    pnl_by_state = df.groupby("StateID")["PnL_ticks"].apply(list).to_dict()
    daily_counts = df.groupby("Day").size().values
    if len(daily_counts) == 0:
        return MonteCarloResult(totals=[], max_drawdowns=[], pass_rate=0.0, rules={})

    rng = np.random.default_rng(7)
    states = sorted(pnl_by_state.keys())
    state_index = {s: i for i, s in enumerate(states)}
    start_dist = df["StateID"].value_counts(normalize=True).reindex(states).fillna(0.0).values

    totals = []
    mdds = []
    for _ in range(sims):
        daily_pnl = []
        current = rng.choice(states, p=start_dist)
        for bars in daily_counts:
            day_total = 0.0
            for _ in range(bars):
                pnl_choices = pnl_by_state.get(current, [0.0])
                day_total += float(rng.choice(pnl_choices))
                row = transition[state_index[current]]
                if row.sum() <= 0:
                    current = rng.choice(states, p=start_dist)
                else:
                    current = rng.choice(states, p=row)
            daily_pnl.append(day_total)
        totals.append(float(sum(daily_pnl)))
        mdds.append(max_drawdown(np.array(daily_pnl)))

    rules = {"min_total_pnl": 0.0, "max_drawdown": -2000.0}
    passes = sum(1 for total, mdd in zip(totals, mdds) if total >= 0 and mdd >= rules["max_drawdown"])
    pass_rate = passes / sims if sims else 0.0
    return MonteCarloResult(totals=totals, max_drawdowns=mdds, pass_rate=pass_rate, rules=rules)


def format_llm_summary(summaries: List[SummaryStats]) -> str:
    lines = ["Policy Performance Summary (copy-friendly)", ""]
    for summary in summaries:
        lines.append(f"{summary.contract} {summary.split}")
        lines.append(
            f"  Bars={summary.bars} Days={summary.days} TotalPnL={summary.total_pnl:.2f} "
            f"AvgDaily={summary.avg_daily_pnl:.2f} MedianDaily={summary.median_daily_pnl:.2f} "
            f"WinRate={summary.win_rate:.2%} SharpeDaily={summary.sharpe_daily:.2f}"
        )
    return "\n".join(lines)


def summary_to_csv(summaries: List[SummaryStats]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "contract": s.contract,
            "split": s.split,
            "bars": s.bars,
            "days": s.days,
            "total_pnl": s.total_pnl,
            "avg_daily_pnl": s.avg_daily_pnl,
            "median_daily_pnl": s.median_daily_pnl,
            "win_rate": s.win_rate,
            "pnl_std_daily": s.pnl_std_daily,
            "sharpe_daily": s.sharpe_daily,
            "pnl_per_bar": s.pnl_per_bar,
            "pnl_std_bar": s.pnl_std_bar,
        }
        for s in summaries
    ])


def save_outputs(
    output_dir: str,
    summaries: List[SummaryStats],
    tail_attrib: Dict[Tuple[str, str], pd.DataFrame],
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "contract_summary.csv")
    summary_to_csv(summaries).to_csv(summary_path, index=False)

    tail_paths = {}
    for key, df in tail_attrib.items():
        contract, split = key
        if df.empty:
            continue
        path = os.path.join(output_dir, f"{contract}_{split}_tail_state_attrib.csv")
        df.to_csv(path, index=False)
        tail_paths[f"{contract}-{split}"] = path

    return {"summary": summary_path, **tail_paths}
