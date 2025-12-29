from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

import analysis

app = Flask(__name__)

DEFAULT_DATA_DIR = os.environ.get("MONTE_CARLO_DATA_DIR", os.getcwd())
DEFAULT_SCRIPT = os.environ.get("FAIRVALUE_SCRIPT", "fairvalue_sigma_process.py")


def _coerce_rules(payload: Dict[str, str]) -> Dict[str, float]:
    return {
        "min_total_pnl": float(payload.get("min_total_pnl", 0) or 0),
        "max_drawdown": float(payload.get("max_drawdown", -2000) or -2000),
    }


@app.route("/")
def index() -> str:
    return render_template(
        "index.html",
        default_data_dir=DEFAULT_DATA_DIR,
        default_script=DEFAULT_SCRIPT,
    )


@app.route("/api/run", methods=["POST"])
def run_script():
    payload = request.get_json() or {}
    script_path = payload.get("script_path", DEFAULT_SCRIPT)
    root_dir = payload.get("root_dir", DEFAULT_DATA_DIR)

    if not Path(script_path).exists():
        return jsonify({"ok": False, "error": f"Script not found: {script_path}"}), 400

    env = os.environ.copy()
    env["ROOT_DIR_OVERRIDE"] = root_dir
    proc = subprocess.run(["python", script_path], capture_output=True, text=True, env=env)
    return jsonify(
        {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    )


@app.route("/api/analyze", methods=["POST"])
def analyze():
    payload = request.get_json() or {}
    data_dir = payload.get("data_dir", DEFAULT_DATA_DIR)
    output_dir = payload.get("output_dir", os.path.join(data_dir, "analysis_outputs"))
    rules = _coerce_rules(payload)
    sims = int(payload.get("sims", analysis.DEFAULT_BOOTSTRAP_SIMS))
    tail_days = int(payload.get("tail_days", analysis.DEFAULT_TAIL_DAYS))

    records = analysis.load_policy_csvs(data_dir)
    if not records:
        return jsonify({"ok": False, "error": "No policy CSVs found."}), 404

    summaries = []
    plots = {}
    tail_attrib = {}

    for record in records:
        df = record.df
        summary = analysis.compute_summary(df, record.contract, record.split)
        summaries.append(summary)

        daily = analysis.compute_daily_pnl(df)
        state_dist = analysis.state_pnl_distribution(df)
        tail = analysis.tail_risk_state_attribution(df, tail_days=tail_days)
        tail_attrib[(record.contract, record.split)] = tail

        mc = analysis.monte_carlo_bootstrap(daily["DailyPnL"].values, sims=sims, rules=rules)

        markov = None
        markov_path = os.path.join(data_dir, f"{record.contract}_TEST_MARKOV_P.npy")
        if os.path.exists(markov_path):
            transition = np.load(markov_path)
            markov = analysis.markov_monte_carlo(df, transition, sims=sims)

        plots[f"{record.contract}-{record.split}"] = {
            "daily": daily.to_dict(orient="list"),
            "state": state_dist.to_dict(orient="list"),
            "tail": tail.to_dict(orient="list"),
            "mc": {
                "totals": mc.totals,
                "max_drawdowns": mc.max_drawdowns,
                "pass_rate": mc.pass_rate,
            },
            "markov": None if markov is None else {
                "totals": markov.totals,
                "max_drawdowns": markov.max_drawdowns,
                "pass_rate": markov.pass_rate,
            },
        }

    llm_summary = analysis.format_llm_summary(summaries)
    output_paths = analysis.save_outputs(output_dir, summaries, tail_attrib)

    return jsonify(
        {
            "ok": True,
            "summaries": [summary.__dict__ for summary in summaries],
            "plots": plots,
            "llm_summary": llm_summary,
            "output_paths": output_paths,
        }
    )


@app.route("/api/download", methods=["GET"])
def download():
    path = request.args.get("path")
    if not path:
        return jsonify({"ok": False, "error": "Missing path"}), 400
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": "File not found"}), 404
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
