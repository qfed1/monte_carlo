import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "fairvalue_sigma_process.py")


@dataclass
class PolicyData:
    contract: str
    split: str
    frame: pd.DataFrame


def load_policy_files(root_dir: str) -> List[PolicyData]:
    policies: List[PolicyData] = []
    for name in os.listdir(root_dir):
        if not name.endswith(".csv"):
            continue
        if "FAIRVALUE_SIGMA_POLICY" not in name:
            continue
        parts = name.split("_FAIRVALUE_SIGMA_POLICY_")
        if len(parts) != 2:
            continue
        contract = parts[0]
        split_part = parts[1].replace(".csv", "")
        split = split_part.upper()
        df = pd.read_csv(os.path.join(root_dir, name), parse_dates=["BarTime"], infer_datetime_format=True)
        df["Contract"] = contract
        df["Split"] = split
        policies.append(PolicyData(contract=contract, split=split, frame=df))
    return policies


def daily_pnl(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("Day", as_index=False)["PnL_ticks"].sum()
    daily = daily.rename(columns={"PnL_ticks": "DailyPnL"})
    return daily


def bar_stats(df: pd.DataFrame) -> pd.Series:
    pnl = df["PnL_ticks"]
    wins = (pnl > 0).sum()
    losses = (pnl < 0).sum()
    return pd.Series(
        {
            "bars": len(df),
            "total_pnl": pnl.sum(),
            "mean_pnl": pnl.mean(),
            "std_pnl": pnl.std(ddof=0),
            "win_rate": wins / max(len(df), 1),
            "loss_rate": losses / max(len(df), 1),
        }
    )


def daily_stats(daily: pd.DataFrame) -> pd.Series:
    pnl = daily["DailyPnL"]
    return pd.Series(
        {
            "days": len(daily),
            "daily_mean": pnl.mean(),
            "daily_std": pnl.std(ddof=0),
            "daily_min": pnl.min(),
            "daily_max": pnl.max(),
            "daily_sharpe": pnl.mean() / pnl.std(ddof=0) if pnl.std(ddof=0) > 0 else 0.0,
        }
    )


def build_state_pnl(df: pd.DataFrame) -> pd.DataFrame:
    if "StateID" not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby("StateID")["PnL_ticks"].agg(["count", "mean", "std", "sum"]).reset_index()
    grouped = grouped.sort_values("sum")
    return grouped


def bootstrap_daily_pnl(daily: pd.DataFrame, sims: int, horizon: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = daily["DailyPnL"].values
    if len(values) == 0:
        return pd.DataFrame()
    samples = rng.choice(values, size=(sims, horizon), replace=True)
    equity = samples.cumsum(axis=1)
    total = equity[:, -1]
    min_dd = np.min(equity - np.maximum.accumulate(equity, axis=1), axis=1)
    return pd.DataFrame({"TotalPnL": total, "MaxDrawdown": min_dd})


def evaluate_prop_rules(results: pd.DataFrame, profit_target: float, max_drawdown: float) -> Dict[str, float]:
    if results.empty:
        return {"pass_rate": 0.0}
    passes = (results["TotalPnL"] >= profit_target) & (results["MaxDrawdown"] >= -abs(max_drawdown))
    return {"pass_rate": float(passes.mean())}


def load_markov_matrix(root_dir: str, contract: str) -> Optional[np.ndarray]:
    path = os.path.join(root_dir, f"{contract}_TEST_MARKOV_P.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def markov_monte_carlo(
    P: np.ndarray,
    state_pnl: pd.Series,
    sims: int,
    horizon: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_states = P.shape[0]
    if n_states == 0:
        return pd.DataFrame()

    pnl_values = state_pnl.reindex(range(n_states)).fillna(0.0).values
    states = np.zeros((sims, horizon), dtype=int)
    states[:, 0] = rng.integers(0, n_states, size=sims)
    for t in range(1, horizon):
        prev = states[:, t - 1]
        probs = P[prev]
        states[:, t] = [rng.choice(n_states, p=probs[i]) for i in range(sims)]

    pnl = pnl_values[states]
    equity = pnl.cumsum(axis=1)
    total = equity[:, -1]
    min_dd = np.min(equity - np.maximum.accumulate(equity, axis=1), axis=1)
    return pd.DataFrame({"TotalPnL": total, "MaxDrawdown": min_dd})


def tail_risk_attribution(df: pd.DataFrame, worst_days: int) -> pd.DataFrame:
    if df.empty or "StateID" not in df.columns:
        return pd.DataFrame()
    daily = df.groupby("Day", as_index=False)["PnL_ticks"].sum()
    worst = daily.nsmallest(worst_days, "PnL_ticks")
    worst_days_set = set(worst["Day"])
    worst_df = df[df["Day"].isin(worst_days_set)]
    attrib = worst_df.groupby("StateID")["PnL_ticks"].sum().reset_index()
    attrib = attrib.sort_values("PnL_ticks")
    return attrib


def render_summary_markdown(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No summaries available yet."
    lines = ["# Policy Summary", ""]
    for _, row in summary.iterrows():
        lines.append(f"## {row['Contract']} ({row['Split']})")
        lines.append(f"- Bars: {row['bars']:,}")
        lines.append(f"- Total PnL (ticks): {row['total_pnl']:.2f}")
        lines.append(f"- Mean Bar PnL: {row['mean_pnl']:.4f}")
        lines.append(f"- Win Rate: {row['win_rate']:.2%}")
        lines.append(f"- Days: {row['days']:,}")
        lines.append(f"- Mean Daily PnL: {row['daily_mean']:.2f}")
        lines.append(f"- Daily Sharpe: {row['daily_sharpe']:.2f}")
        lines.append("")
    return "\n".join(lines)


def run_script(root_dir: str, contracts: List[Tuple[str, str]], episodes: int, max_workers: int) -> Tuple[int, str]:
    config = {
        "root_dir": root_dir,
        "max_workers": max_workers,
        "episodes": episodes,
        "contracts": [{"name": name, "path": path} for name, path in contracts],
    }
    cfg_path = os.path.join(root_dir, "fairvalue_config.json")
    os.makedirs(root_dir, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    result = subprocess.run(
        [sys.executable, SCRIPT_PATH, "--config", cfg_path],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__),
    )
    output = result.stdout + "\n" + result.stderr
    return result.returncode, output


def main() -> None:
    st.set_page_config(page_title="Fair-Value Sigma Dashboard", layout="wide")
    st.title("Fair-Value Sigma Policy Lab")

    with st.sidebar:
        st.header("Run Configuration")
        root_dir = st.text_input("Root directory", value=os.environ.get("FAIRVALUE_ROOT_DIR", "/Users/evanb/Desktop/orderflowv5"))
        contracts_text = st.text_area(
            "Contracts (name,path per line)",
            value="NQMAR25,/Users/evanb/Desktop/orderflowv5/NQMAR25.csv\n"
            "NQJUN25,/Users/evanb/Desktop/orderflowv5/NQJUN25.csv\n"
            "NQSEP25,/Users/evanb/Desktop/orderflowv5/NQSEP25.csv\n"
            "NQDEC25,/Users/evanb/Desktop/orderflowv5/NQDEC25.csv",
            height=160,
        )
        episodes = st.number_input("RL episodes", min_value=1, max_value=50, value=6)
        max_workers = st.number_input("Max workers", min_value=1, max_value=16, value=4)
        run_button = st.button("Run pipeline")

        st.header("Monte Carlo")
        sims = st.number_input("Simulations", min_value=100, max_value=10000, value=2000, step=100)
        horizon = st.number_input("Horizon (days)", min_value=5, max_value=200, value=30, step=5)
        profit_target = st.number_input("Profit target (ticks)", min_value=0.0, value=2000.0)
        max_drawdown = st.number_input("Max drawdown (ticks)", min_value=0.0, value=1000.0)
        tail_days = st.number_input("Worst days for attribution", min_value=5, max_value=50, value=10)
        seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=7)

    if run_button:
        contracts: List[Tuple[str, str]] = []
        for line in contracts_text.splitlines():
            if not line.strip():
                continue
            if "," not in line:
                st.warning(f"Skipping line without comma: {line}")
                continue
            name, path = [part.strip() for part in line.split(",", 1)]
            contracts.append((name, path))

        code, output = run_script(root_dir, contracts, int(episodes), int(max_workers))
        st.subheader("Pipeline Output")
        st.code(output)
        st.success("Pipeline completed." if code == 0 else "Pipeline finished with errors.")

    st.header("Policy Performance")
    policies = load_policy_files(root_dir) if os.path.isdir(root_dir) else []

    if not policies:
        st.info("No policy CSVs found yet. Run the pipeline to populate outputs.")
        return

    rows = []
    for policy in policies:
        daily = daily_pnl(policy.frame)
        stats = bar_stats(policy.frame)
        d_stats = daily_stats(daily)
        row = {"Contract": policy.contract, "Split": policy.split}
        row.update(stats.to_dict())
        row.update(d_stats.to_dict())
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Copy/paste summary for LLM edits")
    markdown = render_summary_markdown(summary_df)
    st.text_area("Markdown summary", value=markdown, height=240)

    for policy in policies:
        st.subheader(f"{policy.contract} ({policy.split})")
        daily = daily_pnl(policy.frame)
        fig = px.line(daily, x="Day", y="DailyPnL", title="Daily PnL")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(policy.frame, x="PnL_ticks", nbins=80, title="Bar PnL distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Per-State PnL Distribution**")
        state_stats = build_state_pnl(policy.frame)
        if state_stats.empty:
            st.caption("StateID not available in this dataset.")
        else:
            st.dataframe(state_stats, use_container_width=True)
            fig = px.bar(state_stats, x="StateID", y="sum", title="Total PnL by State")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Bootstrap Monte Carlo (Daily)**")
        mc_results = bootstrap_daily_pnl(daily, int(sims), int(horizon), int(seed))
        if not mc_results.empty:
            evals = evaluate_prop_rules(mc_results, float(profit_target), float(max_drawdown))
            st.write({"pass_rate": f"{evals['pass_rate']:.2%}"})
            fig = px.histogram(mc_results, x="TotalPnL", nbins=60, title="Bootstrap total PnL")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Markov Monte Carlo (State-driven)**")
        P = load_markov_matrix(root_dir, policy.contract)
        if P is None or state_stats.empty:
            st.caption("Markov matrix or state stats not available.")
        else:
            state_pnl = state_stats.set_index("StateID")["mean"]
            mc_markov = markov_monte_carlo(P, state_pnl, int(sims), int(horizon), int(seed))
            evals = evaluate_prop_rules(mc_markov, float(profit_target), float(max_drawdown))
            st.write({"pass_rate": f"{evals['pass_rate']:.2%}"})
            fig = px.histogram(mc_markov, x="TotalPnL", nbins=60, title="Markov total PnL")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Tail-risk state attribution**")
        tail = tail_risk_attribution(policy.frame, int(tail_days))
        if tail.empty:
            st.caption("Tail-risk attribution unavailable.")
        else:
            st.dataframe(tail, use_container_width=True)
            fig = px.bar(tail, x="StateID", y="PnL_ticks", title="State contribution on worst days")
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
