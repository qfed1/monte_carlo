const runButton = document.getElementById("run-script");
const runOutput = document.getElementById("run-output");
const analyzeButton = document.getElementById("run-analysis");
const results = document.getElementById("results");
const llmSummary = document.getElementById("llm-summary");
const copySummary = document.getElementById("copy-summary");
const downloadLinks = document.getElementById("download-links");

function addDownloadLink(label, path) {
  const link = document.createElement("a");
  link.textContent = label;
  link.href = `/api/download?path=${encodeURIComponent(path)}`;
  link.className = "download-link";
  downloadLinks.appendChild(link);
}

runButton.addEventListener("click", async () => {
  runOutput.textContent = "Running...";
  const payload = {
    script_path: document.getElementById("script-path").value,
    root_dir: document.getElementById("root-dir").value,
  };
  const res = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!data.ok) {
    runOutput.textContent = data.error || "Script failed.";
    return;
  }
  runOutput.textContent = data.stdout || data.stderr || "Done.";
});

copySummary.addEventListener("click", () => {
  llmSummary.select();
  document.execCommand("copy");
});

function renderContractSection(contractKey, payload) {
  const section = document.createElement("section");
  section.className = "panel";
  section.innerHTML = `<h2>${contractKey}</h2>`;

  const daily = payload.daily;
  const dailyTrace = {
    x: daily.Day,
    y: daily.DailyPnL,
    type: "bar",
    name: "Daily PnL",
  };
  const cumPnL = daily.DailyPnL.reduce((acc, val) => {
    acc.push((acc.at(-1) || 0) + val);
    return acc;
  }, []);
  const cumTrace = {
    x: daily.Day,
    y: cumPnL,
    type: "scatter",
    name: "Cumulative PnL",
  };

  const dailyDiv = document.createElement("div");
  dailyDiv.className = "chart";
  section.appendChild(dailyDiv);
  Plotly.newPlot(dailyDiv, [dailyTrace, cumTrace], {
    title: "Daily PnL",
    yaxis: { title: "Ticks" },
    xaxis: { title: "Day" },
  });

  if (payload.state.StateID) {
    const stateDiv = document.createElement("div");
    stateDiv.className = "chart";
    section.appendChild(stateDiv);
    Plotly.newPlot(
      stateDiv,
      [
        {
          x: payload.state.StateID,
          y: payload.state.mean_pnl,
          type: "bar",
          name: "Mean PnL",
        },
      ],
      {
        title: "Per-State Mean PnL",
        xaxis: { title: "StateID" },
        yaxis: { title: "Mean PnL (ticks)" },
      }
    );
  }

  const mcDiv = document.createElement("div");
  mcDiv.className = "chart";
  section.appendChild(mcDiv);
  Plotly.newPlot(
    mcDiv,
    [{ x: payload.mc.totals, type: "histogram", name: "Total PnL" }],
    { title: `Bootstrap MC (pass rate ${(payload.mc.pass_rate * 100).toFixed(1)}%)` }
  );

  if (payload.markov) {
    const markovDiv = document.createElement("div");
    markovDiv.className = "chart";
    section.appendChild(markovDiv);
    Plotly.newPlot(
      markovDiv,
      [{ x: payload.markov.totals, type: "histogram", name: "Total PnL" }],
      { title: `Markov MC (pass rate ${(payload.markov.pass_rate * 100).toFixed(1)}%)` }
    );
  }

  if (payload.tail.StateID) {
    const tailDiv = document.createElement("div");
    tailDiv.className = "chart";
    section.appendChild(tailDiv);
    Plotly.newPlot(
      tailDiv,
      [
        {
          x: payload.tail.StateID,
          y: payload.tail.tail_pnl,
          type: "bar",
          name: "Tail PnL",
        },
      ],
      {
        title: "Tail-Risk State Attribution",
        xaxis: { title: "StateID" },
        yaxis: { title: "Tail PnL (ticks)" },
      }
    );
  }

  results.appendChild(section);
}

analyzeButton.addEventListener("click", async () => {
  results.innerHTML = "";
  downloadLinks.innerHTML = "";
  const payload = {
    data_dir: document.getElementById("data-dir").value,
    output_dir: document.getElementById("output-dir").value,
    sims: document.getElementById("sims").value,
    tail_days: document.getElementById("tail-days").value,
    min_total_pnl: document.getElementById("min-total").value,
    max_drawdown: document.getElementById("max-dd").value,
  };
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!data.ok) {
    results.innerHTML = `<div class="error">${data.error}</div>`;
    return;
  }
  llmSummary.value = data.llm_summary || "";

  if (data.output_paths?.summary) {
    addDownloadLink("Summary CSV", data.output_paths.summary);
  }
  Object.entries(data.output_paths || {}).forEach(([key, path]) => {
    if (key === "summary") return;
    addDownloadLink(`Tail Attribution (${key})`, path);
  });

  Object.entries(data.plots).forEach(([key, payload]) => {
    renderContractSection(key, payload);
  });
});
