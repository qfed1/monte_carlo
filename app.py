#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Monte Carlo Policy Diagnostics", layout="wide")


@dataclass
class ContractData:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    markov_p: np.ndarray | None


DEFAULT_PATTERN = "*_FAIRVALUE_SIGMA_POLICY_{split}.csv"


def _run_script(script_path: Path, workdir: Path, cmd_args: str) -> Tuple[int, str]:
    cmd = f"python {script_path} {cmd_args}".strip()
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(workdir),
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return result.returncode, output.strip()


def _load_policy_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "BarTime" in df.columns:
        df["BarTime"] = pd.to_datetime(df["BarTime"])
    if "Day" not in df.columns and "BarTime" in df.columns:
        df["Day"] = df["BarTime"].dt.date
    return df


def _discover_contracts(root: Path) -> Dict[str, ContractData]:
    contracts: Dict[str, ContractData] = {}
    for split in ("TRAIN", "TEST"):
        pattern = DEFAULT_PATTERN.format(split=split)
        for path in root.glob(pattern):
            name = path.name.split("_FAIRVALUE_SIGMA_POLICY_")[0]
            entry = contracts.get(name)
            if entry is None:
                entry = ContractData(name=name, train=pd.DataFrame(), test=pd.DataFrame(), markov_p=None)
            if split == "TRAIN":
                entry.train = _load_policy_csv(path)
            else:
                entry.test = _load_policy_csv(path)
            contracts[name] = entry

    for name, entry in contracts.items():
        markov_path = root / f"{name}_TEST_MARKOV_P.npy"
        if markov_path.exists():
            entry.markov_p = np.load(markov_path)
    return contracts


def _daily_pnl(df: pd.DataFrame) -> pd.DataFrame:
    if "Day" not in df.columns:
        return pd.DataFrame(columns=["Day", "PnL_ticks"])
    pnl = df.groupby("Day")["PnL_ticks"].sum().reset_index()
    pnl.rename(columns={"PnL_ticks": "DailyPnL_ticks"}, inplace=True)
    return pnl


def _bar_stats(df: pd.DataFrame) -> Dict[str, float]:
    pnl = df.get("PnL_ticks")
    if pnl is None or pnl.empty:
        return {"mean": 0.0, "std": 0.0, "win_rate": 0.0}
    return {
        "mean": float(pnl.mean()),
        "std": float(pnl.std(ddof=1) if len(pnl) > 1 else 0.0),
        "win_rate": float((pnl > 0).mean()),
    }


def _summarize_contract(name: str, df: pd.DataFrame) -> Dict[str, float]:
    daily = _daily_pnl(df)
    daily_pnl = daily["DailyPnL_ticks"] if not daily.empty else pd.Series(dtype=float)
    bars = len(df)
    days = len(daily_pnl)
    bar_stats = _bar_stats(df)
    return {
        "Contract": name,
        "Bars": bars,
        "Days": days,
        "DailyPnL_mean": float(daily_pnl.mean()) if days else 0.0,
        "DailyPnL_std": float(daily_pnl.std(ddof=1) if days > 1 else 0.0),
        "DailyPnL_min": float(daily_pnl.min()) if days else 0.0,
        "DailyPnL_p05": float(np.percentile(daily_pnl, 5)) if days else 0.0,
        "DailyPnL_p95": float(np.percentile(daily_pnl, 95)) if days else 0.0,
        "BarPnL_mean": bar_stats["mean"],
        "BarPnL_std": bar_stats["std"],
        "BarWinRate": bar_stats["win_rate"],
    }


def _state_pnl_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "StateID" not in df.columns:
        return pd.DataFrame()
    return df.groupby("StateID")["PnL_ticks"].agg(
        mean="mean",
        std="std",
        count="count",
        median="median",
    ).reset_index()


def _bootstrap_daily_pnl(daily: pd.Series, n_paths: int, n_days: int, stop_ticks: float, target_ticks: float) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    if daily.empty:
        return pd.DataFrame()
    values = daily.values
    paths = []
    for _ in range(n_paths):
        draws = rng.choice(values, size=n_days, replace=True)
        draws = np.clip(draws, stop_ticks, target_ticks)
        paths.append(
            {
                "TotalPnL": float(draws.sum()),
                "MeanDaily": float(draws.mean()),
                "StdDaily": float(draws.std(ddof=1) if len(draws) > 1 else 0.0),
                "WorstDay": float(draws.min()),
                "BestDay": float(draws.max()),
            }
        )
    return pd.DataFrame(paths)


def _markov_mc(markov_p: np.ndarray, df: pd.DataFrame, n_paths: int, n_days: int) -> pd.DataFrame:
    if "StateID" not in df.columns:
        return pd.DataFrame()
    if markov_p is None or markov_p.size == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(7)
    state_pnl = df.groupby("StateID")["PnL_ticks"].apply(lambda s: s.values).to_dict()
    if not state_pnl:
        return pd.DataFrame()

    bars_per_day = int(round(df.groupby("Day").size().mean())) if "Day" in df.columns else 0
    if bars_per_day <= 0:
        return pd.DataFrame()

    paths = []
    for _ in range(n_paths):
        total = 0.0
        for _day in range(n_days):
            state = int(rng.integers(0, markov_p.shape[0]))
            day_pnl = 0.0
            for _ in range(bars_per_day):
                pnl_samples = state_pnl.get(state)
                if pnl_samples is None or len(pnl_samples) == 0:
                    pnl = 0.0
                else:
                    pnl = float(rng.choice(pnl_samples))
                day_pnl += pnl
                probs = markov_p[state]
                if probs.sum() == 0:
                    state = int(rng.integers(0, markov_p.shape[0]))
                else:
                    state = int(rng.choice(np.arange(markov_p.shape[0]), p=probs))
            total += day_pnl
        paths.append({"TotalPnL": total})
    return pd.DataFrame(paths)


def _tail_risk_attribution(df: pd.DataFrame, worst_days: Iterable) -> pd.DataFrame:
    if "StateID" not in df.columns:
        return pd.DataFrame()
    worst_df = df[df["Day"].isin(set(worst_days))]
    if worst_df.empty:
        return pd.DataFrame()
    return (
        worst_df.groupby("StateID")["PnL_ticks"]
        .sum()
        .sort_values()
        .reset_index()
        .rename(columns={"PnL_ticks": "TotalPnL_ticks"})
    )


def _markdown_summary(name: str, summary: Dict[str, float]) -> str:
    return (
        f"### {name}\n"
        f"- Bars: {summary['Bars']:,}\n"
        f"- Days: {summary['Days']}\n"
        f"- Daily PnL mean/std: {summary['DailyPnL_mean']:.2f} / {summary['DailyPnL_std']:.2f}\n"
        f"- Daily PnL p05/p95: {summary['DailyPnL_p05']:.2f} / {summary['DailyPnL_p95']:.2f}\n"
        f"- Bar PnL mean/std: {summary['BarPnL_mean']:.4f} / {summary['BarPnL_std']:.4f}\n"
        f"- Bar win rate: {summary['BarWinRate']:.2%}\n"
    )


st.title("Monte Carlo Policy Diagnostics")
st.caption("Run the policy pipeline, then explore contract performance, state distributions, and Monte Carlo risk.")

with st.sidebar:
    st.header("Data & Script")
    root_dir = Path(st.text_input("Results folder", value=str(Path.cwd())))
    script_path = Path(st.text_input("Script path", value=str(root_dir / "fairvalue_sigma_process.py")))
    cmd_args = st.text_input("Script args", value="")
    run_script = st.button("Run policy script")
    st.divider()
    st.header("Monte Carlo Settings")
    n_paths = st.number_input("Paths", min_value=100, max_value=20000, value=2000, step=100)
    n_days = st.number_input("Days per path", min_value=10, max_value=500, value=120, step=5)
    stop_ticks = st.number_input("Prop stop (ticks)", value=-600.0)
    target_ticks = st.number_input("Prop target (ticks)", value=600.0)

if run_script:
    if not script_path.exists():
        st.error(f"Script not found: {script_path}")
    else:
        code, output = _run_script(script_path, root_dir, cmd_args)
        st.subheader("Script Output")
        st.code(output or "(no output)")
        if code != 0:
            st.error(f"Script exited with code {code}")


contracts = _discover_contracts(root_dir)

if not contracts:
    st.warning("No policy CSVs found yet. Run the script or point to a directory with *_FAIRVALUE_SIGMA_POLICY_{TRAIN,TEST}.csv files.")
    st.stop()

contract_names = sorted(contracts.keys())
selected = st.selectbox("Contract", contract_names)
data = contracts[selected]

st.subheader(f"{selected} — Summary")
summary_train = _summarize_contract(f"{selected} TRAIN", data.train)
summary_test = _summarize_contract(f"{selected} TEST", data.test)
summary_df = pd.DataFrame([summary_train, summary_test])
st.dataframe(summary_df, use_container_width=True)

summary_markdown = _markdown_summary(selected + " TRAIN", summary_train) + "\n" + _markdown_summary(
    selected + " TEST", summary_test
)
st.text_area("Copy-ready summary", value=summary_markdown, height=200)

st.subheader("Daily PnL")
for label, df_split in [("TRAIN", data.train), ("TEST", data.test)]:
    daily = _daily_pnl(df_split)
    if daily.empty:
        st.info(f"No daily PnL for {label}.")
        continue
    fig = px.line(daily, x="Day", y="DailyPnL_ticks", title=f"{selected} {label} Daily PnL")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Per-state PnL distribution")
state_dist = _state_pnl_distribution(data.test)
if state_dist.empty:
    st.info("No StateID column found in TEST data.")
else:
    fig = px.bar(state_dist, x="StateID", y="mean", error_y="std", title="Per-State Mean PnL (TEST)")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Monte Carlo — Bootstrap Daily PnL")
daily_test = _daily_pnl(data.test)
bootstrap = _bootstrap_daily_pnl(
    daily_test["DailyPnL_ticks"] if not daily_test.empty else pd.Series(dtype=float),
    int(n_paths),
    int(n_days),
    float(stop_ticks),
    float(target_ticks),
)
if bootstrap.empty:
    st.info("No Monte Carlo results yet.")
else:
    fig = px.histogram(bootstrap, x="TotalPnL", nbins=40, title="Total PnL Distribution (Bootstrap)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(bootstrap.describe(percentiles=[0.05, 0.5, 0.95]), use_container_width=True)

st.subheader("Monte Carlo — Markov-driven")
markov_mc = _markov_mc(data.markov_p, data.test, int(n_paths), int(n_days))
if markov_mc.empty:
    st.info("No Markov MC results. Ensure StateID exists and *_TEST_MARKOV_P.npy is present.")
else:
    fig = px.histogram(markov_mc, x="TotalPnL", nbins=40, title="Total PnL Distribution (Markov)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(markov_mc.describe(percentiles=[0.05, 0.5, 0.95]), use_container_width=True)

st.subheader("Tail-risk state attribution")
if "Day" in data.test.columns:
    daily_test = _daily_pnl(data.test)
    worst_days = daily_test.nsmallest(5, "DailyPnL_ticks")["Day"].tolist() if not daily_test.empty else []
    tail_states = _tail_risk_attribution(data.test, worst_days)
    if tail_states.empty:
        st.info("No tail-risk attribution available.")
    else:
        fig = px.bar(tail_states, x="StateID", y="TotalPnL_ticks", title="Worst-Day State Attribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tail_states, use_container_width=True)

st.subheader("Exports")
summary_csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button("Download summary CSV", summary_csv, file_name=f"{selected}_summary.csv", mime="text/csv")

if not state_dist.empty:
    state_csv = state_dist.to_csv(index=False).encode("utf-8")
    st.download_button("Download per-state PnL CSV", state_csv, file_name=f"{selected}_state_pnl.csv", mime="text/csv")

if not bootstrap.empty:
    bootstrap_csv = bootstrap.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download bootstrap Monte Carlo CSV", bootstrap_csv, file_name=f"{selected}_bootstrap_mc.csv", mime="text/csv"
    )

if not markov_mc.empty:
    markov_csv = markov_mc.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Markov Monte Carlo CSV", markov_csv, file_name=f"{selected}_markov_mc.csv", mime="text/csv"
    )
