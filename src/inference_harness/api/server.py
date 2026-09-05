from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from .ws_manager import WebSocketManager
from ..types import RunResult, AggregatedMetrics


_manager = WebSocketManager()


def build_app() -> FastAPI:
    app = FastAPI(title="Inference Harness Dashboard", version="0.1.0")

    @app.get("/")
    async def root() -> HTMLResponse:
        return HTMLResponse(_DASHBOARD_HTML)

    @app.websocket("/ws/stream")
    async def ws_endpoint(ws: WebSocket) -> None:
        await _manager.connect(ws)
        try:
            while True:
                await ws.receive_text()   # keep connection alive (client pings)
        except WebSocketDisconnect:
            _manager.disconnect(ws)

    @app.post("/api/result")
    async def post_result(result: dict) -> dict:
        """Backends push RunResult JSON here; dashboard receives it via WebSocket."""
        await _manager.broadcast(json.dumps({"type": "run_result", "data": result}))
        return {"ok": True}

    @app.post("/api/aggregated")
    async def post_aggregated(agg: dict) -> dict:
        await _manager.broadcast(json.dumps({"type": "aggregated", "data": agg}))
        return {"ok": True}

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "connections": len(_manager.active)}

    return app


# Minimal self-contained dashboard — no build step required
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Inference Harness Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }
  h1 { color: #38bdf8; font-size: 1.4rem; margin-bottom: 4px; }
  .subtitle { color: #94a3b8; font-size: 0.85rem; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .card { background: #1e293b; border-radius: 10px; padding: 20px; }
  .card h2 { font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 12px; }
  canvas { max-height: 220px; }
  #log { background: #0f172a; border-radius: 6px; padding: 12px; font-size: 0.78rem; font-family: monospace; max-height: 180px; overflow-y: auto; color: #7dd3fc; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
  .decode { background: #dc2626; color: #fff; }
  .prefill { background: #d97706; color: #fff; }
  .balanced { background: #16a34a; color: #fff; }
  #status { position: fixed; top: 12px; right: 20px; font-size: 0.8rem; color: #64748b; }
  .connected { color: #4ade80; }
</style>
</head>
<body>
<div id="status">⬤ connecting…</div>
<h1>Inference Harness Dashboard</h1>
<p class="subtitle">Live benchmark results via WebSocket</p>
<div class="grid">
  <div class="card">
    <h2>TTFT (ms) — live</h2>
    <canvas id="ttftChart"></canvas>
  </div>
  <div class="card">
    <h2>Throughput (tok/s)</h2>
    <canvas id="tpsChart"></canvas>
  </div>
  <div class="card">
    <h2>Prefill vs Decode Latency</h2>
    <canvas id="phaseChart"></canvas>
  </div>
  <div class="card">
    <h2>GPU Utilization %</h2>
    <canvas id="gpuChart"></canvas>
  </div>
</div>
<div class="card" style="margin-top:20px">
  <h2>Event log</h2>
  <div id="log">Waiting for data…</div>
</div>
<script>
const colors = ['#38bdf8','#4ade80','#f472b6','#fb923c','#a78bfa','#fbbf24'];
const mkChart = (id, label, color) => new Chart(document.getElementById(id), {
  type: 'line',
  data: { labels: [], datasets: [{ label, data: [], borderColor: color, tension: 0.3, pointRadius: 3, fill: false }] },
  options: { animation: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { color:'#94a3b8' }, grid: { color:'#1e293b' } }, y: { ticks: { color:'#94a3b8' }, grid: { color:'#334155' } } } }
});
const ttftChart = mkChart('ttftChart', 'TTFT ms', colors[0]);
const tpsChart  = mkChart('tpsChart',  'tok/s',   colors[1]);
const phaseChart = new Chart(document.getElementById('phaseChart'), {
  type: 'bar',
  data: { labels: [], datasets: [
    { label: 'Prefill ms', data: [], backgroundColor: colors[3] },
    { label: 'Decode ms',  data: [], backgroundColor: colors[2] },
  ]},
  options: { animation: false, scales: { x: { stacked: true, ticks:{color:'#94a3b8'}, grid:{color:'#1e293b'} }, y: { stacked: true, ticks:{color:'#94a3b8'}, grid:{color:'#334155'} } } }
});
const gpuChart  = mkChart('gpuChart',  'GPU Util%', colors[4]);

let runN = 0;

function push(chart, label, val) {
  chart.data.labels.push(label);
  chart.data.datasets[0].data.push(val);
  if (chart.data.labels.length > 40) { chart.data.labels.shift(); chart.data.datasets[0].data.shift(); }
  chart.update();
}

function log(msg) {
  const el = document.getElementById('log');
  el.innerHTML = `[${new Date().toLocaleTimeString()}] ${msg}<br>` + el.innerHTML;
}

const ws = new WebSocket(`ws://${location.host}/ws/stream`);
ws.onopen = () => { document.getElementById('status').textContent = '⬤ connected'; document.getElementById('status').className = 'connected'; };
ws.onclose = () => { document.getElementById('status').textContent = '⬤ disconnected'; document.getElementById('status').className = ''; };
ws.onmessage = (evt) => {
  const msg = JSON.parse(evt.data);
  if (msg.type === 'run_result') {
    const r = msg.data; runN++;
    const lbl = `#${runN} ${r.scenario||''}`;
    push(ttftChart, lbl, r.ttft_ms);
    push(tpsChart,  lbl, r.throughput_tps);
    if (r.gpu_utilization_pct) push(gpuChart, lbl, r.gpu_utilization_pct);
    if (r.prefill_ms || r.decode_ms) {
      phaseChart.data.labels.push(lbl);
      phaseChart.data.datasets[0].data.push(r.prefill_ms || r.ttft_ms);
      phaseChart.data.datasets[1].data.push(r.decode_ms || (r.tpot_ms * (r.output_tokens || 1)));
      if (phaseChart.data.labels.length > 20) { phaseChart.data.labels.shift(); phaseChart.data.datasets.forEach(d=>d.data.shift()); }
      phaseChart.update();
    }
    const bn = r.bottleneck || 'unknown';
    log(`${r.backend}/${r.scenario} — TTFT ${r.ttft_ms?.toFixed(0)}ms  TPS ${r.throughput_tps?.toFixed(1)}  <span class="badge ${bn}">${bn.toUpperCase()}</span>`);
  } else if (msg.type === 'aggregated') {
    const a = msg.data;
    log(`AGG ${a.scenario}: TTFT p50=${a.ttft_p50_ms?.toFixed(0)}ms  bottleneck=${a.dominant_bottleneck}`);
  }
};
</script>
</body>
</html>"""
