#!/usr/bin/env python3
"""
FAIR-VALUE SIGMA REGIMES + ORB BIAS + (OPTIONAL) GLOBAL TABULAR RL
=================================================================

PROCESS PARALLEL (FAST + SAFE):
- Phase 1 PREP runs per-contract in ProcessPool workers, writes cache to disk.
- Phase 2 GLOBAL RL trains in the main process (deterministic).
- Phase 3 SIM runs per-contract in ProcessPool workers using cached prepared data + Q.

DETERMINISM:
- GLOBAL RL training uses stable sort by BarTime only:
    data.sort_values("BarTime", kind="mergesort")
  so BarTime ties preserve concat order (contract order).
- RL uses local RNG (np.random.default_rng(SEED)).

If you want the "legacy" behavior that likely produced your baseline run:
- RL_EPISODE_KEY = "DAY"  (default)
If you want the "correct" separation across overlapping calendar days:
- RL_EPISODE_KEY = "CONTRACT_DAY"

Deps: numpy, pandas
Run:
  conda activate py312
  python fairvalue_sigma_process.py
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIG
# ──────────────────────────────────────────────────────────────────────────────

ROOT_DIR = os.environ.get("FAIRVALUE_ROOT_DIR", "/Users/evanb/Desktop/orderflowv5")

CONTRACTS = [
    ("NQMAR25", os.path.join(ROOT_DIR, "NQMAR25.csv")),
    ("NQJUN25", os.path.join(ROOT_DIR, "NQJUN25.csv")),
    ("NQSEP25", os.path.join(ROOT_DIR, "NQSEP25.csv")),
    ("NQDEC25", os.path.join(ROOT_DIR, "NQDEC25.csv")),
]

# Use processes for real parallelism (pandas/groupby is CPU heavy)
MAX_WORKERS = 4

# Deterministic seed for RL
SEED = 7

# Bar build
BAR_SECONDS = 60
CHUNK_ROWS = 2_000_000

# Market
TICK_SIZE = 0.25

# Session control
TRADE_SESSION = "RTH"   # "RTH", "ORB", "ALL"
RTH_START = "07:30"     # Mountain time (Denver)
RTH_END   = "14:00"
ORB_START = "07:30"
ORB_END   = "08:30"

# Non-oracle execution
EXEC_LAG_BARS = 1  # decision at bar t applies to bar t+1

# Daily risk controls (ticks)
DAILY_STOP_TICKS = -600
DAILY_TARGET_TICKS = 600

# Regime calibration target (Gaussian 1-sigma ~ 68.27%)
TARGET_INSIDE_FRAC = 0.6827

# Thresholds (base, then scaled by ENTER_SCALE)
BAL_Z_ENTER_BASE = 1.00
IMB_Z_ENTER_BASE = 1.25
CAND_Z_ENTER_FRAC = 0.90

# State machine behavior
WARMUP_BARS_PER_DAY = 10
EXIT_INSIDE_BARS = 3
EXIT_BREAK_Z_MAX = 1.00
CAND_MAX_BARS = 6

# BreakDir definition (trend proxy)
BREAK_LAG = 5
BREAK_EPS_SIG = 0.10

# Trading principle
TRADE_IMB_ONLY = True
BAL_SCALP_ENABLED = True
BAL_SCALP_Z_ENTER = 0.55
BAL_SCALP_Z_MAX = 1.05
BAL_SCALP_SIZE = 0.25
IMB_SIZE = 1.0

# ORB bias gate
ORB_BIAS_GATE = True
ORB_BREAK_BUFFER_TICKS = 2

# Costs
COST_TICKS_PER_UNIT_CHANGE = 0.0

# RL
USE_RL = True
GLOBAL_RL = True
EPISODES = 6
EPS_START = 0.25
EPS_END = 0.05
ALPHA = 0.15
GAMMA = 0.20

# IMPORTANT: choose episode grouping key
# "DAY" preserves legacy behavior (likely your baseline).
# "CONTRACT_DAY" prevents mixing when calendar days overlap across contracts.
RL_EPISODE_KEY = "CONTRACT_DAY"  # or "CONTRACT_DAY"

# Calibration search
SCALE_GRID = np.round(np.arange(0.80, 1.51, 0.025), 4)
MIN_IMB_DAYS_FRAC = 0.90
IMB_DAYS_PENALTY_W = 1.0

# Train/test split by days (chronological)
TRAIN_FRAC = 0.70

# Cache + logs
CACHE_DIR = os.path.join(ROOT_DIR, "_prepared_cache")
LOG_DIR = os.path.join(ROOT_DIR, "_worker_logs")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG OVERRIDES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfigOverrides:
    root_dir: str | None = None
    contracts: List[Tuple[str, str]] | None = None
    max_workers: int | None = None
    episodes: int | None = None
    seed: int | None = None
    use_rl: bool | None = None
    global_rl: bool | None = None
    orb_bias_gate: bool | None = None
    trade_session: str | None = None


def apply_overrides(overrides: ConfigOverrides) -> None:
    global ROOT_DIR, CONTRACTS, MAX_WORKERS, EPISODES, SEED, USE_RL, GLOBAL_RL
    global ORB_BIAS_GATE, TRADE_SESSION, CACHE_DIR, LOG_DIR

    if overrides.root_dir:
        ROOT_DIR = overrides.root_dir
    if overrides.contracts:
        CONTRACTS = overrides.contracts
    if overrides.max_workers is not None:
        MAX_WORKERS = overrides.max_workers
    if overrides.episodes is not None:
        EPISODES = overrides.episodes
    if overrides.seed is not None:
        SEED = overrides.seed
    if overrides.use_rl is not None:
        USE_RL = overrides.use_rl
    if overrides.global_rl is not None:
        GLOBAL_RL = overrides.global_rl
    if overrides.orb_bias_gate is not None:
        ORB_BIAS_GATE = overrides.orb_bias_gate
    if overrides.trade_session is not None:
        TRADE_SESSION = overrides.trade_session

    CACHE_DIR = os.path.join(ROOT_DIR, "_prepared_cache")
    LOG_DIR = os.path.join(ROOT_DIR, "_worker_logs")
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def parse_args() -> ConfigOverrides:
    parser = argparse.ArgumentParser(description="Run fair-value sigma pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--root-dir", type=str, default=None, help="Root directory for data and outputs.")
    parser.add_argument("--max-workers", type=int, default=None, help="Number of worker processes.")
    parser.add_argument("--episodes", type=int, default=None, help="RL episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--use-rl", type=int, default=None, help="1 to use RL, 0 to disable.")
    parser.add_argument("--global-rl", type=int, default=None, help="1 for global RL, 0 for per-contract only.")
    parser.add_argument("--orb-bias-gate", type=int, default=None, help="1 to enable ORB bias gate.")
    parser.add_argument("--trade-session", type=str, default=None, help="Session filter: RTH, ORB, ALL.")

    args = parser.parse_args()

    overrides = ConfigOverrides(
        root_dir=args.root_dir,
        max_workers=args.max_workers,
        episodes=args.episodes,
        seed=args.seed,
        use_rl=(None if args.use_rl is None else bool(args.use_rl)),
        global_rl=(None if args.global_rl is None else bool(args.global_rl)),
        orb_bias_gate=(None if args.orb_bias_gate is None else bool(args.orb_bias_gate)),
        trade_session=args.trade_session,
    )

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        overrides.root_dir = cfg.get("root_dir", overrides.root_dir)
        overrides.max_workers = cfg.get("max_workers", overrides.max_workers)
        overrides.episodes = cfg.get("episodes", overrides.episodes)
        overrides.seed = cfg.get("seed", overrides.seed)
        if "use_rl" in cfg:
            overrides.use_rl = bool(cfg["use_rl"])
        if "global_rl" in cfg:
            overrides.global_rl = bool(cfg["global_rl"])
        if "orb_bias_gate" in cfg:
            overrides.orb_bias_gate = bool(cfg["orb_bias_gate"])
        if "trade_session" in cfg:
            overrides.trade_session = cfg["trade_session"]
        contracts_cfg = cfg.get("contracts")
        if contracts_cfg:
            overrides.contracts = [(c["name"], c["path"]) for c in contracts_cfg]

    return overrides

# ──────────────────────────────────────────────────────────────────────────────
# TIME HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _hhmm_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

RTH_START_MIN = _hhmm_to_minutes(RTH_START)
RTH_END_MIN   = _hhmm_to_minutes(RTH_END)
ORB_START_MIN = _hhmm_to_minutes(ORB_START)
ORB_END_MIN   = _hhmm_to_minutes(ORB_END)

def _tod_minutes(ts: pd.Timestamp) -> int:
    return ts.hour * 60 + ts.minute

def in_session(ts: pd.Timestamp) -> bool:
    if TRADE_SESSION == "ALL":
        return True
    mins = _tod_minutes(ts)
    if TRADE_SESSION == "RTH":
        return (RTH_START_MIN <= mins) and (mins <= RTH_END_MIN)
    if TRADE_SESSION == "ORB":
        return (ORB_START_MIN <= mins) and (mins <= ORB_END_MIN)
    raise ValueError(f"Unknown TRADE_SESSION={TRADE_SESSION}")

def in_orb_window(ts: pd.Timestamp) -> bool:
    mins = _tod_minutes(ts)
    return (ORB_START_MIN <= mins) and (mins <= ORB_END_MIN)

def after_orb(ts: pd.Timestamp) -> bool:
    return _tod_minutes(ts) > ORB_END_MIN

def sign3(x: float, eps: float) -> int:
    if x > eps:
        return 2  # up
    if x < -eps:
        return 0  # down
    return 1      # flat

# ──────────────────────────────────────────────────────────────────────────────
# STREAM TICKS -> BARS (chunk-safe)
# ──────────────────────────────────────────────────────────────────────────────

def build_bars_from_ticks(csv_path: str, bar_sec: int, chunk_rows: int, log) -> pd.DataFrame:
    """
    Streams tick CSV and builds OHLCV bars.
    Expected columns: Time, Price, Volume (TickStreamer24x7).
    """
    hdr = pd.read_csv(csv_path, nrows=1)
    cols = list(hdr.columns)
    if not {"Time", "Price"}.issubset(set(cols)):
        raise ValueError(f"[{os.path.basename(csv_path)}] Expected columns include Time, Price. Found={cols}")
    vol_col = "Volume" if "Volume" in cols else None
    if vol_col is None:
        raise ValueError(f"[{os.path.basename(csv_path)}] Missing Volume column.")

    usecols = ["Time", "Price", vol_col]
    pending = None
    bars_out: List[pd.DataFrame] = []

    reader = pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=chunk_rows,
        low_memory=True,
    )

    for chunk_idx, chunk in enumerate(reader, start=1):
        chunk["Time"] = pd.to_datetime(chunk["Time"])
        chunk["BarTime"] = chunk["Time"].dt.floor(f"{bar_sec}s")

        if pending is not None and len(pending) > 0:
            chunk = pd.concat([pending, chunk], ignore_index=True)
            pending = None

        last_bt = chunk["BarTime"].iloc[-1]
        mask_last = (chunk["BarTime"] == last_bt)
        pending = chunk.loc[mask_last, ["BarTime", "Price", vol_col]].copy()
        chunk2 = chunk.loc[~mask_last, ["BarTime", "Price", vol_col]]

        if len(chunk2) == 0:
            continue

        gb = chunk2.groupby("BarTime", sort=True)
        o = gb["Price"].first()
        h = gb["Price"].max()
        l = gb["Price"].min()
        c = gb["Price"].last()
        v = gb[vol_col].sum()

        bars = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}).reset_index()
        bars_out.append(bars)

        if (chunk_idx % 5) == 0:
            log(f"  ..chunk {chunk_idx} processed, bars so far ~ {sum(len(b) for b in bars_out):,}")

    if pending is not None and len(pending) > 0:
        gb = pending.groupby("BarTime", sort=True)
        o = gb["Price"].first()
        h = gb["Price"].max()
        l = gb["Price"].min()
        c = gb["Price"].last()
        v = gb[vol_col].sum()
        bars = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}).reset_index()
        bars_out.append(bars)

    df = pd.concat(bars_out, ignore_index=True)
    df = df.sort_values("BarTime").reset_index(drop=True)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# NON-ORACLE PER-DAY KALMAN FILTER (1D)
# ──────────────────────────────────────────────────────────────────────────────

def kalman_filter_1d(x: np.ndarray, q: float = 1e-4, r: float = 2e-2) -> np.ndarray:
    if len(x) == 0:
        return x
    xf = np.empty_like(x, dtype=np.float64)
    est = float(x[0])
    p = 1.0
    xf[0] = est
    for i in range(1, len(x)):
        p = p + q
        k = p / (p + r)
        est = est + k * (float(x[i]) - est)
        p = (1.0 - k) * p
        xf[i] = est
    return xf

# ──────────────────────────────────────────────────────────────────────────────
# FEATURES: VWAP/SIGMA + ZSCORES
# ──────────────────────────────────────────────────────────────────────────────

def add_vwap_sigma(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Day"] = df["BarTime"].dt.date

    v = df["Volume"].astype(np.float64).values
    p = df["Close"].astype(np.float64).values
    pv = p * v
    pv2 = (p * p) * v

    day = df["Day"].values
    v_cum = np.zeros(len(df), dtype=np.float64)
    pv_cum = np.zeros(len(df), dtype=np.float64)
    pv2_cum = np.zeros(len(df), dtype=np.float64)

    last_day = None
    sv = spv = spv2 = 0.0
    for i in range(len(df)):
        if last_day is None or day[i] != last_day:
            sv = spv = spv2 = 0.0
            last_day = day[i]
        sv += v[i]
        spv += pv[i]
        spv2 += pv2[i]
        v_cum[i] = sv
        pv_cum[i] = spv
        pv2_cum[i] = spv2

    vwap = pv_cum / np.maximum(v_cum, 1e-12)
    e2 = pv2_cum / np.maximum(v_cum, 1e-12)
    var = np.maximum(e2 - vwap * vwap, 1e-12)
    sigma = np.sqrt(var)

    df["VWAP_raw"] = vwap
    df["Sigma_raw"] = sigma
    return df


def apply_per_day_kalman(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Close_f"] = df["Close"].astype(np.float64)
    df["VWAP_f"] = df["VWAP_raw"].astype(np.float64)

    out_close = []
    out_vwap = []
    for _, g in df.groupby("Day", sort=True):
        c = g["Close_f"].values.astype(np.float64)
        w = g["VWAP_f"].values.astype(np.float64)
        out_close.append(kalman_filter_1d(c))
        out_vwap.append(kalman_filter_1d(w))

    df["Close_f"] = np.concatenate(out_close)
    df["VWAP_f"] = np.concatenate(out_vwap)
    df["Sigma_f"] = df["Sigma_raw"].astype(np.float64)
    return df


def add_zscores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sig = np.maximum(df["Sigma_f"].values.astype(np.float64), 1e-12)

    zc = (df["Close_f"].values.astype(np.float64) - df["VWAP_f"].values.astype(np.float64)) / sig
    zh = (df["High"].values.astype(np.float64) - df["VWAP_f"].values.astype(np.float64)) / sig
    zl = (df["Low"].values.astype(np.float64) - df["VWAP_f"].values.astype(np.float64)) / sig
    zbreak = np.where(np.abs(zh) >= np.abs(zl), zh, zl)

    df["ZClose"] = zc
    df["ZHigh"] = zh
    df["ZLow"] = zl
    df["ZBreak"] = zbreak
    return df

# ──────────────────────────────────────────────────────────────────────────────
# ORB LEVELS (day-local)
# ──────────────────────────────────────────────────────────────────────────────

def add_orb_levels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    buf = ORB_BREAK_BUFFER_TICKS * TICK_SIZE

    orb_high = np.full(len(df), np.nan, dtype=np.float64)
    orb_low = np.full(len(df), np.nan, dtype=np.float64)
    orb_bias = np.zeros(len(df), dtype=np.int8)

    idx0 = 0
    for _, g in df.groupby("Day", sort=True):
        n = len(g)
        idxs = np.arange(idx0, idx0 + n)

        in_orb = g["BarTime"].apply(in_orb_window).values
        if np.any(in_orb):
            oh = float(np.max(g.loc[in_orb, "High"].values))
            ol = float(np.min(g.loc[in_orb, "Low"].values))
        else:
            oh = np.nan
            ol = np.nan

        orb_high[idxs] = oh
        orb_low[idxs] = ol

        bias = 0
        for j in range(n):
            ts = g["BarTime"].iloc[j]
            if not after_orb(ts):
                orb_bias[idx0 + j] = 0
                continue
            if math.isnan(oh) or math.isnan(ol):
                orb_bias[idx0 + j] = 0
                continue

            close = float(g["Close_f"].iloc[j])
            if bias == 0:
                if close > oh + buf:
                    bias = +1
                elif close < ol - buf:
                    bias = -1
            orb_bias[idx0 + j] = bias

        idx0 += n

    df["ORB_H"] = orb_high
    df["ORB_L"] = orb_low
    df["ORB_Bias"] = orb_bias
    return df

# ──────────────────────────────────────────────────────────────────────────────
# CALIBRATION (TRAIN ONLY)
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_enter_scale(train_df: pd.DataFrame) -> Tuple[float, float, float, float]:
    zc = train_df["ZClose"].values.astype(np.float64)
    zb = train_df["ZBreak"].values.astype(np.float64)
    days = train_df["Day"].values

    day_ids, day_starts = np.unique(days, return_index=True)
    day_starts = list(day_starts) + [len(train_df)]

    best = None
    for scale in SCALE_GRID:
        bal_th = scale * BAL_Z_ENTER_BASE
        inside = np.mean(np.abs(zc) <= bal_th)

        imb_th = scale * IMB_Z_ENTER_BASE
        imb_day_count = 0
        for k in range(len(day_ids)):
            a, b = day_starts[k], day_starts[k + 1]
            if np.any(np.abs(zb[a:b]) >= imb_th):
                imb_day_count += 1
        imb_days_frac = imb_day_count / max(len(day_ids), 1)

        penalty = 0.0
        if imb_days_frac < MIN_IMB_DAYS_FRAC:
            penalty = IMB_DAYS_PENALTY_W * (MIN_IMB_DAYS_FRAC - imb_days_frac) ** 2

        obj = (inside - TARGET_INSIDE_FRAC) ** 2 + penalty

        if best is None or obj < best[3]:
            best = (float(scale), float(inside), float(imb_days_frac), float(obj))

    assert best is not None
    return best

# ──────────────────────────────────────────────────────────────────────────────
# REGIME STATE MACHINE + 27-STATE ENCODING
# ──────────────────────────────────────────────────────────────────────────────

def state_id(breakdir: int, zband: int, phase: int) -> int:
    return breakdir * 9 + zband * 3 + phase


def decode_state(sid: int) -> Tuple[int, int, int]:
    bd = sid // 9
    rem = sid % 9
    zb = rem // 3
    ph = rem % 3
    return bd, zb, ph


def print_state_table() -> None:
    print("State ID decoding (BreakDir, ZBand, Phase):")
    for sid in range(27):
        bd, zb, ph = decode_state(sid)
        bd_s = "down" if bd == 0 else "flat" if bd == 1 else "up"
        zb_s = "< -bal" if zb == 0 else "inside" if zb == 1 else "> +bal"
        ph_s = "BAL" if ph == 0 else "CAND" if ph == 1 else "IMB"
        print(f"State {sid:2d}: ({bd},{zb},{ph})  {ph_s:<4} | break={bd_s:<4} | z={zb_s}")


def add_regimes(df: pd.DataFrame, enter_scale: float) -> pd.DataFrame:
    df = df.copy()
    bal_th = enter_scale * BAL_Z_ENTER_BASE
    imb_th = enter_scale * IMB_Z_ENTER_BASE
    cand_th = enter_scale * (CAND_Z_ENTER_FRAC * IMB_Z_ENTER_BASE)
    exit_break_max = enter_scale * EXIT_BREAK_Z_MAX

    zc = df["ZClose"].values.astype(np.float64)
    zb = df["ZBreak"].values.astype(np.float64)
    sig = np.maximum(df["Sigma_f"].values.astype(np.float64), 1e-12)

    close = df["Close_f"].values.astype(np.float64)
    trend = np.zeros(len(df), dtype=np.float64)
    trend[BREAK_LAG:] = close[BREAK_LAG:] - close[:-BREAK_LAG]

    trend_sig = np.zeros(len(df), dtype=np.float64)
    trend_sig[BREAK_LAG:] = trend[BREAK_LAG:] / sig[BREAK_LAG:]

    bd = np.array([sign3(x, BREAK_EPS_SIG) for x in trend_sig], dtype=np.int8)

    zbnd = np.ones(len(df), dtype=np.int8)
    zbnd[zc < -bal_th] = 0
    zbnd[zc >  bal_th] = 2

    phase = np.zeros(len(df), dtype=np.int8)
    regime = np.array(["BAL"] * len(df), dtype=object)

    idx0 = 0
    for _, g in df.groupby("Day", sort=True):
        n = len(g)
        zc_d = zc[idx0:idx0+n]
        zb_d = zb[idx0:idx0+n]

        ph = 0
        cand_left = 0
        imb_dir = 0
        inside_streak = 0

        for j in range(n):
            if j < WARMUP_BARS_PER_DAY:
                phase[idx0+j] = 0
                regime[idx0+j] = "BAL"
                ph = 0
                cand_left = 0
                imb_dir = 0
                inside_streak = 0
                continue

            inside = (abs(zc_d[j]) <= bal_th)
            zbreak_abs = abs(zb_d[j])

            if ph == 0:
                if (zbreak_abs >= cand_th) and (abs(zc_d[j]) >= 0.5 * bal_th):
                    ph = 1
                    cand_left = CAND_MAX_BARS
                    phase[idx0+j] = 1
                    regime[idx0+j] = "CAND_UP" if zb_d[j] > 0 else "CAND_DOWN"
                else:
                    phase[idx0+j] = 0
                    regime[idx0+j] = "BAL"

            elif ph == 1:
                cand_left -= 1
                if zbreak_abs >= imb_th:
                    ph = 2
                    imb_dir = +1 if zb_d[j] > 0 else -1
                    inside_streak = 0
                    phase[idx0+j] = 2
                    regime[idx0+j] = "IMB_UP" if imb_dir > 0 else "IMB_DOWN"
                elif inside or cand_left <= 0:
                    ph = 0
                    phase[idx0+j] = 0
                    regime[idx0+j] = "BAL"
                else:
                    phase[idx0+j] = 1
                    regime[idx0+j] = "CAND_UP" if zb_d[j] > 0 else "CAND_DOWN"

            else:
                if zbreak_abs >= 0.75 * imb_th:
                    imb_dir = +1 if zb_d[j] > 0 else -1

                if inside and (zbreak_abs <= exit_break_max):
                    inside_streak += 1
                else:
                    inside_streak = 0

                if inside_streak >= EXIT_INSIDE_BARS:
                    ph = 0
                    phase[idx0+j] = 0
                    regime[idx0+j] = "BAL"
                    imb_dir = 0
                    inside_streak = 0
                else:
                    phase[idx0+j] = 2
                    regime[idx0+j] = "IMB_UP" if imb_dir > 0 else "IMB_DOWN"

        idx0 += n

    df["BreakDir"] = bd
    df["ZBand"] = zbnd
    df["Phase"] = phase
    df["StateID"] = np.array([state_id(int(bd[i]), int(zbnd[i]), int(phase[i])) for i in range(len(df))], dtype=np.int16)
    df["Regime"] = regime
    df["RegimeClass"] = np.where(df["Regime"] == "BAL", "BALANCE",
                                 np.where(df["Regime"].astype(str).str.startswith("IMB"), "IMBALANCE", "CANDIDATE"))
    return df

# ──────────────────────────────────────────────────────────────────────────────
# TRADING + RL
# ──────────────────────────────────────────────────────────────────────────────

ACTIONS = np.array([-1, 0, +1], dtype=np.int8)  # maps action index -> position sign


def allowed_actions_for_bar(row: pd.Series, day_locked: bool) -> np.ndarray:
    if day_locked:
        return np.array([1], dtype=np.int8)

    ts = row["BarTime"]
    if not in_session(ts):
        return np.array([1], dtype=np.int8)

    if TRADE_IMB_ONLY and int(row["Phase"]) != 2:
        return np.array([1], dtype=np.int8)

    if ORB_BIAS_GATE:
        if not after_orb(ts):
            return np.array([1], dtype=np.int8)
        bias = int(row["ORB_Bias"])
        if bias == 0:
            return np.array([1], dtype=np.int8)
        if bias > 0:
            return np.array([1, 2], dtype=np.int8)
        return np.array([0, 1], dtype=np.int8)

    return np.array([0, 1, 2], dtype=np.int8)


def deterministic_scalp_desired_pos(row: pd.Series) -> float:
    if not BAL_SCALP_ENABLED:
        return 0.0
    if int(row["Phase"]) != 0:
        return 0.0
    ts = row["BarTime"]
    if not in_session(ts):
        return 0.0
    z = float(row["ZClose"])
    az = abs(z)
    if (az >= BAL_SCALP_Z_ENTER) and (az <= BAL_SCALP_Z_MAX):
        return (-1.0 if z > 0 else +1.0) * float(BAL_SCALP_SIZE)
    return 0.0


def simulate_with_policy(df: pd.DataFrame, Q: np.ndarray | None, greedy: bool) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    desired = np.zeros(n, dtype=np.float64)
    mode = np.array(["FLAT"] * n, dtype=object)

    # Decide desired positions
    idx0 = 0
    for _, g in df.groupby("Day", sort=True):
        m = len(g)
        locked = False
        dp = 0.0
        for j in range(m):
            i = idx0 + j
            if locked:
                desired[i] = 0.0
                mode[i] = "FLAT"
                continue

            row = df.iloc[i]
            scalp_pos = deterministic_scalp_desired_pos(row)

            imb_pos = 0.0
            if int(row["Phase"]) == 2 and in_session(row["BarTime"]):
                if Q is not None and USE_RL:
                    allowed = allowed_actions_for_bar(row, locked)
                    s = int(row["StateID"])
                    qrow = Q[s]
                    ai = int(allowed[np.argmax(qrow[allowed])]) if greedy else int(allowed[0])
                    a = int(ACTIONS[ai])
                    imb_pos = float(a) * float(IMB_SIZE)
                else:
                    reg = str(row["Regime"])
                    if reg == "IMB_UP":
                        imb_pos = +float(IMB_SIZE)
                    elif reg == "IMB_DOWN":
                        imb_pos = -float(IMB_SIZE)

                if ORB_BIAS_GATE:
                    ts = row["BarTime"]
                    if not after_orb(ts):
                        imb_pos = 0.0
                    else:
                        bias = int(row["ORB_Bias"])
                        if bias == 0:
                            imb_pos = 0.0
                        elif bias > 0 and imb_pos < 0:
                            imb_pos = 0.0
                        elif bias < 0 and imb_pos > 0:
                            imb_pos = 0.0

            if abs(imb_pos) > 1e-12:
                desired[i] = imb_pos
                mode[i] = "IMB"
            elif abs(scalp_pos) > 1e-12:
                desired[i] = scalp_pos
                mode[i] = "SCALP"
            else:
                desired[i] = 0.0
                mode[i] = "FLAT"
        idx0 += m

    # Apply execution lag
    exec_pos = np.zeros(n, dtype=np.float64)
    if EXEC_LAG_BARS <= 0:
        exec_pos[:] = desired
    else:
        exec_pos[EXEC_LAG_BARS:] = desired[:-EXEC_LAG_BARS]
        exec_pos[:EXEC_LAG_BARS] = 0.0

    close = df["Close_f"].values.astype(np.float64)
    dpx = np.zeros(n, dtype=np.float64)
    dpx[1:] = (close[1:] - close[:-1]) / TICK_SIZE

    pnl = np.zeros(n, dtype=np.float64)
    pnl[1:] = exec_pos[1:] * dpx[1:]

    if COST_TICKS_PER_UNIT_CHANGE > 0:
        chg = np.zeros(n, dtype=np.float64)
        chg[1:] = np.abs(exec_pos[1:] - exec_pos[:-1])
        pnl[1:] -= COST_TICKS_PER_UNIT_CHANGE * chg[1:]

    # Daily lockout (stop/target) based on realized PnL
    day_locked = np.zeros(n, dtype=np.int8)
    day_pnl = np.zeros(n, dtype=np.float64)

    idx0 = 0
    for _, g in df.groupby("Day", sort=True):
        m = len(g)
        dp = 0.0
        locked = False
        for j in range(m):
            i = idx0 + j
            day_pnl[i] = dp
            day_locked[i] = 1 if locked else 0
            if locked:
                continue
            dp += pnl[i]
            if dp <= DAILY_STOP_TICKS or dp >= DAILY_TARGET_TICKS:
                locked = True
        idx0 += m

    df["DesiredPos"] = desired
    df["ExecPos"] = exec_pos
    df["TradeMode"] = mode
    df["PnL_ticks"] = pnl
    df["DayPnL_ticks"] = day_pnl
    df["DayLocked"] = day_locked
    return df


def train_q_global(train_frames_in_contract_order: List[pd.DataFrame],
                   enter_scales_in_contract_order: List[float],
                   contract_names_in_order: List[str]) -> np.ndarray:
    """
    Deterministic GLOBAL Q-learning:
    - local RNG (default_rng)
    - stable sort by BarTime (mergesort)
    - optional episode grouping: DAY (legacy) vs CONTRACT_DAY
    """
    Q = np.zeros((27, 3), dtype=np.float64)
    rng = np.random.default_rng(SEED)

    big = []
    for df, sc, name in zip(train_frames_in_contract_order, enter_scales_in_contract_order, contract_names_in_order):
        tmp = df.copy()
        tmp["ENTER_SCALE"] = float(sc)
        tmp["Contract"] = name
        big.append(tmp)

    data = pd.concat(big, ignore_index=True)
    data = data.sort_values("BarTime", kind="mergesort").reset_index(drop=True)

    eps_vals = np.linspace(EPS_START, EPS_END, num=max(EPISODES, 2))

    # Build episode key array
    if RL_EPISODE_KEY.upper() == "DAY":
        key = data["Day"].astype(str).values
    elif RL_EPISODE_KEY.upper() == "CONTRACT_DAY":
        key = (data["Contract"].astype(str) + "::" + data["Day"].astype(str)).values
    else:
        raise ValueError(f"RL_EPISODE_KEY must be DAY or CONTRACT_DAY, got {RL_EPISODE_KEY}")

    starts = np.r_[0, 1 + np.where(key[1:] != key[:-1])[0], len(data)]
    print(f"Training RL on {len(starts)-1} episode(s), {len(data):,} bars (TRAIN, global). key={RL_EPISODE_KEY}")

    for ep in range(EPISODES):
        eps = float(eps_vals[min(ep, len(eps_vals)-1)])
        total_r = 0.0

        for k in range(len(starts) - 1):
            a0, b0 = int(starts[k]), int(starts[k + 1])
            dp = 0.0
            locked = False

            for t in range(a0, b0 - 1):
                row = data.iloc[t]
                allowed = allowed_actions_for_bar(row, locked)
                s = int(row["StateID"])

                if rng.random() < eps:
                    ai = int(rng.choice(allowed))
                else:
                    qrow = Q[s]
                    ai = int(allowed[np.argmax(qrow[allowed])])
                a = int(ACTIONS[ai])

                scalp_pos = deterministic_scalp_desired_pos(row)

                desired_pos = 0.0
                if int(row["Phase"]) == 2:
                    desired_pos = float(a) * float(IMB_SIZE)
                    if ORB_BIAS_GATE:
                        ts = row["BarTime"]
                        if not after_orb(ts):
                            desired_pos = 0.0
                        else:
                            bias = int(row["ORB_Bias"])
                            if bias == 0:
                                desired_pos = 0.0
                            elif bias > 0 and desired_pos < 0:
                                desired_pos = 0.0
                            elif bias < 0 and desired_pos > 0:
                                desired_pos = 0.0
                elif abs(scalp_pos) > 1e-12:
                    desired_pos = float(scalp_pos)

                close_t = float(data["Close_f"].iloc[t])
                close_n = float(data["Close_f"].iloc[t + 1])
                ret_ticks = (close_n - close_t) / TICK_SIZE

                r = float(desired_pos) * float(ret_ticks)
                if COST_TICKS_PER_UNIT_CHANGE > 0:
                    r -= COST_TICKS_PER_UNIT_CHANGE * abs(desired_pos)

                total_r += r
                dp += r
                if dp <= DAILY_STOP_TICKS or dp >= DAILY_TARGET_TICKS:
                    locked = True

                s2 = int(data["StateID"].iloc[t + 1])
                td_target = r + GAMMA * np.max(Q[s2])
                Q[s, ai] += ALPHA * (td_target - Q[s, ai])

        print(f"Episode {ep+1}/{EPISODES} shaped reward (TRAIN): {total_r:.2f}")

    return Q


def save_markov(df: pd.DataFrame, out_path: str) -> None:
    s = df["StateID"].values.astype(np.int16)
    P = np.zeros((27, 27), dtype=np.float64)
    for i in range(len(s) - 1):
        P[int(s[i]), int(s[i + 1])] += 1.0
    row_sums = P.sum(axis=1, keepdims=True)
    P = np.divide(P, np.maximum(row_sums, 1e-12))
    np.save(out_path, P)


def summarize_split(df: pd.DataFrame, tag: str) -> str:
    pnl = float(df["PnL_ticks"].sum())
    bars = len(df)
    days = df["Day"].nunique()
    out = []
    out.append(f"\n=== {tag} ===")
    out.append(f"Bars: {bars:,}  Days: {days}")
    out.append(f"Total PnL: {pnl:.2f}")

    rc = df["RegimeClass"].value_counts(normalize=True)
    out.append("RegimeClass fractions (bars):")
    out.append(rc.to_string())

    rr = df["Regime"].value_counts(normalize=True)
    out.append("Regime fractions (bars):")
    out.append(rr.to_string())

    exec_only = df[df["ExecPos"].abs() > 1e-12]
    if len(exec_only) > 0:
        tm = exec_only["TradeMode"].value_counts(normalize=True)
        out.append("TradeMode fractions (executed bars only):")
        out.append(tm.to_string())

    return "\n".join(out)

# ──────────────────────────────────────────────────────────────────────────────
# WORKERS
# ──────────────────────────────────────────────────────────────────────────────

def _worker_log_factory(contract: str) -> Tuple[io.StringIO, Any, Any]:
    buf = io.StringIO()
    log_path = os.path.join(LOG_DIR, f"{contract}.log")
    f = open(log_path, "w", buffering=1)
    def log(msg: str) -> None:
        buf.write(msg + "\n")
        f.write(msg + "\n")
    return buf, f, log


def prep_worker(contract_name: str, csv_path: str) -> Dict[str, Any]:
    buf, f, log = _worker_log_factory(contract_name)
    try:
        log(f"\n================ PREP {contract_name} ================")
        log(f"[worker] pid={os.getpid()}")
        log(f"Loading CSV: {csv_path}")
        bars = build_bars_from_ticks(csv_path, BAR_SECONDS, CHUNK_ROWS, log)
        log(f"Built {len(bars):,} bars. Range {bars['BarTime'].iloc[0]} -> {bars['BarTime'].iloc[-1]}")

        bars = add_vwap_sigma(bars)
        bars = apply_per_day_kalman(bars)
        bars = add_zscores(bars)
        bars = add_orb_levels(bars)

        bars["InSession"] = bars["BarTime"].apply(in_session).astype(np.int8)

        days = np.array(sorted(bars["Day"].unique()))
        n_days = len(days)
        n_train = max(1, int(round(n_days * TRAIN_FRAC)))
        train_days = set(days[:n_train])
        test_days = set(days[n_train:])

        log(f"Total days: {n_days} | Train: {len(train_days)} | Test: {len(test_days)}")

        train_df = bars[bars["Day"].isin(train_days)].copy()
        cal_df = train_df[train_df["InSession"] == 1].copy()

        best_scale, inside, imb_days_frac, obj = calibrate_enter_scale(cal_df)
        log(f"[CALIBRATE] ENTER_SCALE={best_scale:.4f} inside={inside*100:.2f}% "
            f"(target {TARGET_INSIDE_FRAC*100:.2f}%) IMB-days={imb_days_frac*100:.1f}% obj={obj:.6f}")

        all_df = add_regimes(bars, best_scale)

        train_all = all_df[all_df["Day"].isin(train_days)].copy()
        test_all = all_df[all_df["Day"].isin(test_days)].copy()

        train_pkl = os.path.join(CACHE_DIR, f"{contract_name}_TRAIN_prepared.pkl")
        test_pkl  = os.path.join(CACHE_DIR, f"{contract_name}_TEST_prepared.pkl")
        train_all.to_pickle(train_pkl, protocol=4)
        test_all.to_pickle(test_pkl, protocol=4)

        meta = {
            "name": contract_name,
            "csv": csv_path,
            "enter_scale": float(best_scale),
            "train_pkl": train_pkl,
            "test_pkl": test_pkl,
            "days_total": int(n_days),
            "days_train": int(len(train_days)),
            "days_test": int(len(test_days)),
            "bar_first": str(all_df["BarTime"].iloc[0]),
            "bar_last": str(all_df["BarTime"].iloc[-1]),
            "logfile": os.path.join(LOG_DIR, f"{contract_name}.log"),
        }
        meta_path = os.path.join(CACHE_DIR, f"{contract_name}_META.json")
        with open(meta_path, "w") as jf:
            json.dump(meta, jf, indent=2)

        log(f"Cached TRAIN: {train_pkl}")
        log(f"Cached TEST : {test_pkl}")
        log(f"[LOGFILE] {meta['logfile']}")
        return {"ok": True, "meta": meta, "stdout": buf.getvalue()}
    except Exception:
        tb = traceback.format_exc()
        log("EXCEPTION:\n" + tb)
        return {"ok": False, "error": tb, "stdout": buf.getvalue()}
    finally:
        f.close()


def sim_worker(contract_name: str, enter_scale: float, train_pkl: str, test_pkl: str,
               Q: np.ndarray | None) -> Dict[str, Any]:
    buf, f, log = _worker_log_factory(contract_name)
    try:
        log(f"\n================ SIM {contract_name} ================")
        log(f"[worker] pid={os.getpid()}")
        train_df = pd.read_pickle(train_pkl)
        test_df  = pd.read_pickle(test_pkl)

        train_sim = simulate_with_policy(train_df, Q, greedy=True)
        test_sim  = simulate_with_policy(test_df, Q, greedy=True)

        train_sim_sess = train_sim[train_sim["InSession"] == 1].copy()
        test_sim_sess  = test_sim[test_sim["InSession"] == 1].copy()

        out_train = os.path.join(ROOT_DIR, f"{contract_name}_FAIRVALUE_SIGMA_POLICY_TRAIN.csv")
        out_test  = os.path.join(ROOT_DIR, f"{contract_name}_FAIRVALUE_SIGMA_POLICY_TEST.csv")
        mk        = os.path.join(ROOT_DIR, f"{contract_name}_TEST_MARKOV_P.npy")

        train_sim_sess.to_csv(out_train, index=False)
        test_sim_sess.to_csv(out_test, index=False)
        save_markov(test_sim_sess, mk)

        summ_train = summarize_split(train_sim_sess, f"{contract_name} TRAIN")
        summ_test  = summarize_split(test_sim_sess, f"{contract_name} TEST")

        log(summ_train)
        log(f"Saved TRAIN policy CSV: {out_train}")
        log(summ_test)
        log(f"Saved TEST policy CSV: {out_test}")
        log(f"Saved Markov matrix: {mk}")

        tr_pnl = float(train_sim_sess["PnL_ticks"].sum())
        te_pnl = float(test_sim_sess["PnL_ticks"].sum())
        log(f"{contract_name:8s}  TrainPnL={tr_pnl:10.2f}  TestPnL={te_pnl:10.2f}")
        log(f"[LOGFILE] {os.path.join(LOG_DIR, f'{contract_name}.log')}")

        return {"ok": True, "stdout": buf.getvalue(), "train_pnl": tr_pnl, "test_pnl": te_pnl}
    except Exception:
        tb = traceback.format_exc()
        log("EXCEPTION:\n" + tb)
        return {"ok": False, "error": tb, "stdout": buf.getvalue()}
    finally:
        f.close()

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    overrides = parse_args()
    apply_overrides(overrides)
    np.random.seed(SEED)  # legacy; RL uses local RNG anyway

    print_state_table()
    print("\n================ SETTINGS ================")
    print(f"MAX_WORKERS={MAX_WORKERS}  BAR_SECONDS={BAR_SECONDS}  CHUNK_ROWS={CHUNK_ROWS:,}")
    print(f"SESSION={TRADE_SESSION}  ORB={ORB_START}-{ORB_END}  ORB_BIAS_GATE={ORB_BIAS_GATE}")
    print(f"USE_RL={USE_RL}  GLOBAL_RL={GLOBAL_RL}  EPISODES={EPISODES}  SEED={SEED}")
    print(f"RL_EPISODE_KEY={RL_EPISODE_KEY}")
    print(f"CACHE_DIR={CACHE_DIR}")
    print(f"LOG_DIR={LOG_DIR}")

    # Spawn context (safe on macOS)
    ctx = mp.get_context("spawn")

    # ── Phase 1: PREP in parallel
    print("\n================ PHASE 1: PREP (parallel) ================")
    prep_results: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as ex:
        futs = {ex.submit(prep_worker, name, path): name for name, path in CONTRACTS}
        for fut in as_completed(futs):
            name = futs[fut]
            res = fut.result()
            prep_results[name] = res
            if not res.get("ok", False):
                errors[name] = res.get("error", "unknown error")

    print("\n================ PREP OUTPUTS (ORDERED) ================")
    metas_in_order: List[Dict[str, Any]] = []
    for name, _ in CONTRACTS:
        print("\n" + "=" * 80)
        print(f"PREP OUTPUT — {name}")
        print("=" * 80)
        res = prep_results.get(name)
        if res is None:
            print("[MISSING RESULT]")
            continue
        print(res.get("stdout", ""))
        if res.get("ok"):
            metas_in_order.append(res["meta"])
        else:
            print(f"[ERROR] {name}: see logfile {os.path.join(LOG_DIR, f'{name}.log')}")

    if errors:
        print("\n================ PREP ERRORS SUMMARY ================")
        for k, v in errors.items():
            print(f"{k}:\n{v}")
        print("\nStop: fix PREP errors first.")
        return

    Q = None
    if USE_RL and GLOBAL_RL:
        # ── Phase 2: GLOBAL RL in main process (deterministic)
        print("\n================ PHASE 2: GLOBAL RL TRAIN ================")
        train_frames: List[pd.DataFrame] = []
        scales: List[float] = []
        names: List[str] = []
        for meta in metas_in_order:
            name = meta["name"]
            df_train = pd.read_pickle(meta["train_pkl"])
            df_train = df_train[df_train["InSession"] == 1].copy()
            train_frames.append(df_train)
            scales.append(float(meta["enter_scale"]))
            names.append(name)

        Q = train_q_global(train_frames, scales, names)

    # ── Phase 3: SIM in parallel
    print("\n================ PHASE 3: SIM (parallel) ================")
    sim_results: Dict[str, Dict[str, Any]] = {}
    sim_errors: Dict[str, str] = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as ex:
        futs = {}
        for meta in metas_in_order:
            futs[ex.submit(
                sim_worker,
                meta["name"],
                float(meta["enter_scale"]),
                meta["train_pkl"],
                meta["test_pkl"],
                Q
            )] = meta["name"]

        for fut in as_completed(futs):
            name = futs[fut]
            res = fut.result()
            sim_results[name] = res
            if not res.get("ok", False):
                sim_errors[name] = res.get("error", "unknown error")

    print("\n================ SIM OUTPUTS (ORDERED) ================")
    for name, _ in CONTRACTS:
        print("\n" + "=" * 80)
        print(f"SIM OUTPUT — {name}")
        print("=" * 80)
        res = sim_results.get(name)
        if res is None:
            print("[MISSING RESULT]")
            continue
        print(res.get("stdout", ""))
        if not res.get("ok", False):
            print(f"[ERROR] {name}: see logfile {os.path.join(LOG_DIR, f'{name}.log')}")

    if sim_errors:
        print("\n================ SIM ERRORS SUMMARY ================")
        for k, v in sim_errors.items():
            print(f"{k}:\n{v}")

    print("\nDone.")

if __name__ == "__main__":
    main()
