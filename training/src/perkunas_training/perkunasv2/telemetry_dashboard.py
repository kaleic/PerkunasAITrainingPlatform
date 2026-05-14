from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


TIMING_FIELDS = (
    "param_load_seconds",
    "module_build_seconds",
    "h2d_seconds",
    "forward_kernel_seconds",
    "backward_kernel_seconds",
    "activation_cpu_copy_seconds",
    "gradient_cpu_copy_seconds",
    "trace_stage_seconds",
    "trace_restore_seconds",
    "optimizer_load_seconds",
    "optimizer_math_seconds",
    "param_save_stage_seconds",
    "optimizer_save_stage_seconds",
)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")


def _rolling_std(values: list[float | None], window: int = 25) -> list[float | None]:
    out: list[float | None] = []
    recent: list[float] = []
    for value in values:
        if value is not None:
            recent.append(value)
        if len(recent) > window:
            recent.pop(0)
        out.append(statistics.pstdev(recent) if len(recent) >= 2 else None)
    return out


def _ema(values: list[float | None], alpha: float = 0.08) -> list[float | None]:
    out: list[float | None] = []
    current: float | None = None
    for value in values:
        if value is None:
            out.append(current)
            continue
        current = value if current is None else (alpha * value) + ((1.0 - alpha) * current)
        out.append(current)
    return out


def _extract_train(entry: dict[str, Any], order: int) -> dict[str, Any]:
    guard = entry.get("guarded_step_replay") or {}
    grad_norm = entry.get("grad_norm") or {}
    memory = entry.get("memory") or {}
    residency = entry.get("residency") or {}
    timing = entry.get("timing_breakdown") or {}
    attempt_log = guard.get("attempt_log") or []
    rejected_attempts = sum(1 for item in attempt_log if not item.get("accepted"))

    row: dict[str, Any] = {
        "i": order,
        "step": _int(entry.get("step")),
        "train_loss": _float(entry.get("train_loss", entry.get("loss"))),
        "lr": _float(entry.get("lr")),
        "base_lr": _float(entry.get("base_lr")),
        "tokens_per_sec": _float(entry.get("tokens_per_sec")),
        "step_seconds": _float(entry.get("step_seconds")),
        "forward_trace_seconds": _float(entry.get("forward_trace_seconds")),
        "data_load_seconds": _float(entry.get("data_load_seconds")),
        "forward_compute_seconds": _float(entry.get("forward_compute_seconds")),
        "shard_update_seconds": _float(entry.get("shard_update_seconds")),
        "backward_update_seconds": _float(entry.get("backward_update_seconds")),
        "commit_seconds": _float(entry.get("commit_seconds")),
        "grad_mean": _float(grad_norm.get("mean")),
        "grad_max": _float(grad_norm.get("max")),
        "global_grad_norm": _float(entry.get("global_grad_norm")),
        "global_grad_clip_scale": _float(entry.get("global_grad_clip_scale")),
        "guard_active": bool(guard.get("active", False)),
        "guard_accepted": guard.get("accepted"),
        "guard_attempts": _int(guard.get("attempts")),
        "guard_max_attempts": _int(guard.get("max_attempts")),
        "guard_loss_before": _float(guard.get("loss_before")),
        "guard_loss_after": _float(guard.get("accepted_loss_after")),
        "guard_lr_scale": _float(guard.get("accepted_lr_scale")),
        "guard_grad_norm_scale": _float(guard.get("accepted_grad_norm_scale")),
        "guard_rejected_attempts": rejected_attempts,
        "updated_shards": _int(entry.get("updated_shards")),
        "optimizer_shards_touched": _int(entry.get("optimizer_shards_touched")),
        "max_active_param_shards": _int(entry.get("max_active_param_shards_observed")),
        "max_active_optimizer_shards": _int(entry.get("max_active_optimizer_shards_observed")),
        "memory_allocated_mb": _float(memory.get("allocated_mb")),
        "memory_reserved_mb": _float(memory.get("reserved_mb")),
        "memory_peak_allocated_mb": _float(memory.get("peak_allocated_mb")),
        "cached_param_shard_count": len(residency.get("cached_param_shards") or []),
        "cached_optimizer_shard_count": len(residency.get("cached_optimizer_shards") or []),
        "storage_shard_count": _int(residency.get("storage_shard_count")),
        "max_resident_shards": _int(residency.get("max_resident_shards")),
    }
    for field in TIMING_FIELDS:
        row[field] = _float(entry.get(field, timing.get(field)))

    tokens_per_sec = row["tokens_per_sec"]
    step_seconds = row["step_seconds"]
    row["estimated_tokens"] = (
        tokens_per_sec * step_seconds
        if tokens_per_sec is not None and step_seconds is not None
        else None
    )
    loss_after = row["guard_loss_after"]
    loss_before = row["guard_loss_before"]
    row["guard_loss_delta"] = (
        loss_after - loss_before
        if loss_after is not None and loss_before is not None
        else None
    )
    return row


def _extract_validation(entry: dict[str, Any], order: int) -> dict[str, Any]:
    return {
        "i": order,
        "step": _int(entry.get("step")),
        "val_loss": _float(entry.get("val_loss")),
        "val_perplexity": _float(entry.get("val_perplexity")),
        "validation_batches": _float(entry.get("validation_batches")),
    }


def parse_train_log(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    parsed = 0
    skipped = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for order, line in enumerate(handle):
            raw = line.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                skipped += 1
                continue
            parsed += 1
            if "train_loss" in entry or "loss" in entry:
                train_rows.append(_extract_train(entry, order))
            elif "val_loss" in entry:
                val_rows.append(_extract_validation(entry, order))

    losses = [row.get("train_loss") for row in train_rows]
    smooth = _ema(losses)
    rolling = _rolling_std(losses)
    cumulative_tokens = 0.0
    previous_loss: float | None = None
    seen_steps: set[int] = set()
    duplicate_steps = 0
    for row, ema_value, std_value in zip(train_rows, smooth, rolling, strict=False):
        row["train_loss_ema"] = ema_value
        row["train_loss_rolling_std"] = std_value
        loss = row.get("train_loss")
        row["train_loss_delta"] = (
            loss - previous_loss if loss is not None and previous_loss is not None else None
        )
        if loss is not None:
            previous_loss = loss
        tokens = row.get("estimated_tokens")
        if tokens is not None:
            cumulative_tokens += tokens
        row["cumulative_estimated_tokens"] = cumulative_tokens
        step = row.get("step")
        if step is not None:
            duplicate_steps += 1 if step in seen_steps else 0
            seen_steps.add(step)

    summary = build_summary(path, train_rows, val_rows, parsed, skipped, duplicate_steps)
    return train_rows, val_rows, summary


def build_summary(
    path: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    parsed: int,
    skipped: int,
    duplicate_steps: int,
) -> dict[str, Any]:
    train_losses = [row["train_loss"] for row in train_rows if row.get("train_loss") is not None]
    val_losses = [row["val_loss"] for row in val_rows if row.get("val_loss") is not None]
    token_estimates = [
        row["estimated_tokens"] for row in train_rows if row.get("estimated_tokens") is not None
    ]
    first_step = train_rows[0]["step"] if train_rows else None
    latest_step = train_rows[-1]["step"] if train_rows else None
    best_val = min(val_rows, key=lambda row: row["val_loss"] or float("inf")) if val_rows else None
    latest_val = val_rows[-1] if val_rows else None
    best_train = min(train_rows, key=lambda row: row["train_loss"] or float("inf")) if train_rows else None

    return {
        "input_path": str(path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parsed_lines": parsed,
        "skipped_lines": skipped,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "first_step": first_step,
        "latest_step": latest_step,
        "duplicate_train_steps": duplicate_steps,
        "latest_train_loss": train_losses[-1] if train_losses else None,
        "best_train_loss": best_train.get("train_loss") if best_train else None,
        "best_train_step": best_train.get("step") if best_train else None,
        "latest_val_step": latest_val.get("step") if latest_val else None,
        "latest_val_loss": latest_val.get("val_loss") if latest_val else None,
        "latest_val_perplexity": latest_val.get("val_perplexity") if latest_val else None,
        "best_val_step": best_val.get("step") if best_val else None,
        "best_val_loss": best_val.get("val_loss") if best_val else None,
        "best_val_perplexity": best_val.get("val_perplexity") if best_val else None,
        "median_tokens_per_update": statistics.median(token_estimates) if token_estimates else None,
        "estimated_tokens_seen": sum(token_estimates) if token_estimates else None,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0d1117;
      --panel: #151b23;
      --panel-2: #101720;
      --line: #263241;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #2f81f7;
      --good: #3fb950;
      --warn: #d29922;
      --bad: #f85149;
      --cyan: #39c5cf;
      --purple: #a371f7;
      --orange: #db6d28;
      --radius: 8px;
      --shadow: 0 18px 44px rgba(0, 0, 0, 0.24);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      min-width: 1040px;
    }
    header {
      padding: 24px 28px 18px;
      border-bottom: 1px solid var(--line);
      background: #111821;
      position: sticky;
      top: 0;
      z-index: 20;
      box-shadow: var(--shadow);
    }
    h1 {
      font-size: 24px;
      margin: 0 0 8px;
      font-weight: 720;
      letter-spacing: 0;
    }
    .subline {
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
      word-break: break-all;
    }
    main { padding: 22px 28px 34px; }
    .grid {
      display: grid;
      gap: 14px;
    }
    .kpis {
      grid-template-columns: repeat(8, minmax(0, 1fr));
      margin-bottom: 18px;
    }
    .kpi, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
    }
    .kpi {
      padding: 13px 14px;
      min-height: 82px;
    }
    .kpi .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }
    .kpi .value {
      font-size: 20px;
      font-weight: 760;
      line-height: 1.15;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .kpi .hint {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: 1fr 170px 136px 112px 112px 130px 130px;
      gap: 12px;
      align-items: end;
      margin-bottom: 18px;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 7px;
    }
    input, select, button {
      width: 100%;
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      font-size: 13px;
    }
    input[type="range"] {
      padding-left: 0;
      padding-right: 0;
      accent-color: var(--accent);
    }
    button {
      cursor: pointer;
      font-weight: 650;
    }
    button:hover { border-color: var(--accent); }
    .chart-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .panel {
      min-height: 332px;
      overflow: hidden;
    }
    .panel-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      padding: 14px 16px 8px;
      border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .panel-title {
      font-weight: 720;
      font-size: 14px;
    }
    .panel-note {
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }
    canvas {
      display: block;
      width: 100%;
      height: 270px;
    }
    .wide { grid-column: 1 / -1; }
    .timing-bars {
      padding: 14px 16px 16px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 230px 1fr 80px;
      gap: 12px;
      align-items: center;
      margin: 9px 0;
      color: var(--muted);
      font-size: 12px;
    }
    .bar-track {
      height: 12px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: var(--accent);
      border-radius: 999px;
    }
    .tables {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 14px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      border-bottom: 1px solid rgba(255,255,255,0.06);
      padding: 8px 9px;
      text-align: right;
      white-space: nowrap;
    }
    th:first-child, td:first-child { text-align: left; }
    th {
      color: var(--muted);
      font-weight: 650;
      position: sticky;
      top: 0;
      background: var(--panel);
      z-index: 1;
    }
    .table-wrap {
      max-height: 320px;
      overflow: auto;
      padding: 0 8px 10px;
    }
    .tag {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 54px;
      padding: 3px 7px;
      border-radius: 999px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 11px;
      font-weight: 650;
    }
    .tag.good { color: var(--good); border-color: rgba(63,185,80,0.45); }
    .tag.warn { color: var(--warn); border-color: rgba(210,153,34,0.45); }
    .tag.bad { color: var(--bad); border-color: rgba(248,81,73,0.45); }
    .empty {
      color: var(--muted);
      padding: 16px;
      font-size: 13px;
    }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <div class="subline">
      <span id="filePath"></span>
      <span id="generatedAt"></span>
      <span id="lineCounts"></span>
      <span id="rangeLabel"></span>
    </div>
  </header>
  <main>
    <section class="grid kpis" id="kpis"></section>
    <section class="toolbar">
      <div>
        <label for="windowStart">Gradient playback window</label>
        <input id="windowStart" type="range" value="0" min="0" max="0" step="1" />
      </div>
      <div>
        <label for="windowSize">Window size</label>
        <select id="windowSize">
          <option value="100">100 train rows</option>
          <option value="250" selected>250 train rows</option>
          <option value="500">500 train rows</option>
          <option value="1000">1000 train rows</option>
          <option value="all">All rows</option>
        </select>
      </div>
      <div>
        <label for="xMode">X axis</label>
        <select id="xMode">
          <option value="step">Step</option>
          <option value="order">Log order</option>
        </select>
      </div>
      <div>
        <label for="spikeThreshold">Spike floor</label>
        <input id="spikeThreshold" type="number" value="4" step="0.05" />
      </div>
      <div>
        <label>&nbsp;</label>
        <button id="playButton">Play</button>
      </div>
      <div>
        <label>&nbsp;</label>
        <button id="fitButton">Fit all</button>
      </div>
      <div>
        <label>&nbsp;</label>
        <button id="csvButton">Export window CSV</button>
      </div>
    </section>

    <section class="grid chart-grid">
      <article class="panel wide">
        <div class="panel-header">
          <div class="panel-title">Learning Curve</div>
          <div class="panel-note">train loss, EMA, validation checkpoints</div>
        </div>
        <canvas id="lossChart"></canvas>
      </article>
      <article class="panel">
        <div class="panel-header">
          <div class="panel-title">Gradient Magnitude</div>
          <div class="panel-note">global, mean, max</div>
        </div>
        <canvas id="gradChart"></canvas>
      </article>
      <article class="panel">
        <div class="panel-header">
          <div class="panel-title">Clip And Replay</div>
          <div class="panel-note">clip scale, attempts, accepted scales</div>
        </div>
        <canvas id="guardChart"></canvas>
      </article>
      <article class="panel">
        <div class="panel-header">
          <div class="panel-title">Noise</div>
          <div class="panel-note">loss delta, rolling std, guarded delta</div>
        </div>
        <canvas id="noiseChart"></canvas>
      </article>
      <article class="panel">
        <div class="panel-header">
          <div class="panel-title">Throughput</div>
          <div class="panel-note">tokens/sec and update time</div>
        </div>
        <canvas id="throughputChart"></canvas>
      </article>
      <article class="panel">
        <div class="panel-header">
          <div class="panel-title">Timing Lines</div>
          <div class="panel-note">forward, update, backward, build</div>
        </div>
        <canvas id="timingChart"></canvas>
      </article>
      <article class="panel">
        <div class="panel-header">
          <div class="panel-title">Memory</div>
          <div class="panel-note">allocated, reserved, peak allocated MB</div>
        </div>
        <canvas id="memoryChart"></canvas>
      </article>
      <article class="panel wide">
        <div class="panel-header">
          <div class="panel-title">Average Timing Composition</div>
          <div class="panel-note">current playback window</div>
        </div>
        <div class="timing-bars" id="timingBars"></div>
      </article>
    </section>

    <section class="grid tables">
      <article class="panel">
        <div class="panel-header">
          <div class="panel-title">Validation Checkpoints</div>
          <div class="panel-note">all rows in current window</div>
        </div>
        <div class="table-wrap" id="valTable"></div>
      </article>
      <article class="panel">
        <div class="panel-header">
          <div class="panel-title">Spike Rows</div>
          <div class="panel-note">loss above floor, sorted by loss</div>
        </div>
        <div class="table-wrap" id="spikeTable"></div>
      </article>
      <article class="panel">
        <div class="panel-header">
          <div class="panel-title">Replay Events</div>
          <div class="panel-note">attempts, rejected candidates, scaled accepts</div>
        </div>
        <div class="table-wrap" id="replayTable"></div>
      </article>
      <article class="panel">
        <div class="panel-header">
          <div class="panel-title">Slowest Updates</div>
          <div class="panel-note">timing outliers in current window</div>
        </div>
        <div class="table-wrap" id="slowTable"></div>
      </article>
    </section>
  </main>

  <script>
    const TRAIN = __TRAIN_DATA__;
    const VAL = __VAL_DATA__;
    const META = __META_DATA__;
    const TIMING_FIELDS = __TIMING_FIELDS__;

    const colors = {
      train: "#58a6ff",
      ema: "#3fb950",
      val: "#f0883e",
      grad: "#d2a8ff",
      gradMean: "#39c5cf",
      gradMax: "#f85149",
      clip: "#7ee787",
      attempts: "#d29922",
      lrScale: "#a371f7",
      gradScale: "#db6d28",
      noise: "#ff7b72",
      std: "#79c0ff",
      throughput: "#56d364",
      seconds: "#f2cc60",
      memory: "#39c5cf"
    };

    const state = { start: 0, playing: false };
    const $ = (id) => document.getElementById(id);

    function fmt(value, digits = 4) {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return "-";
      const n = Number(value);
      if (Math.abs(n) >= 1000000) return n.toExponential(3);
      if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 1 });
      if (Math.abs(n) > 0 && Math.abs(n) < 0.001) return n.toExponential(3);
      return n.toLocaleString(undefined, { maximumFractionDigits: digits });
    }

    function windowSize() {
      const raw = $("windowSize").value;
      return raw === "all" ? TRAIN.length : Number(raw);
    }

    function currentWindow() {
      const size = windowSize();
      const maxStart = Math.max(0, TRAIN.length - size);
      state.start = Math.min(state.start, maxStart);
      const end = Math.min(TRAIN.length, state.start + size);
      const rows = TRAIN.slice(state.start, end);
      const firstOrder = rows.length ? rows[0].i : 0;
      const lastOrder = rows.length ? rows[rows.length - 1].i : 0;
      const vals = VAL.filter((row) => row.i >= firstOrder && row.i <= lastOrder);
      return { rows, vals, firstOrder, lastOrder };
    }

    function point(row, field) {
      const xField = $("xMode").value === "order" ? "i" : "step";
      const x = row[xField];
      const y = row[field];
      if (x === null || x === undefined || y === null || y === undefined) return null;
      return { x: Number(x), y: Number(y), step: row.step, i: row.i };
    }

    function drawChart(canvasId, series, options = {}) {
      const canvas = $(canvasId);
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      const pad = { left: 58, right: 22, top: 20, bottom: 42 };
      const w = rect.width - pad.left - pad.right;
      const h = rect.height - pad.top - pad.bottom;
      const all = series.flatMap((s) => s.points);
      if (!all.length || w <= 0 || h <= 0) {
        ctx.fillStyle = "#8b949e";
        ctx.font = "13px sans-serif";
        ctx.fillText("No data in this window", pad.left, pad.top + 20);
        return;
      }
      let xMin = Math.min(...all.map((p) => p.x));
      let xMax = Math.max(...all.map((p) => p.x));
      let yMin = options.yMin ?? Math.min(...all.map((p) => p.y));
      let yMax = options.yMax ?? Math.max(...all.map((p) => p.y));
      if (xMin === xMax) { xMin -= 1; xMax += 1; }
      if (yMin === yMax) { yMin -= 1; yMax += 1; }
      const yPad = (yMax - yMin) * 0.08;
      yMin -= yPad;
      yMax += yPad;
      if (options.zeroFloor) yMin = Math.min(0, yMin);

      const sx = (x) => pad.left + ((x - xMin) / (xMax - xMin)) * w;
      const sy = (y) => pad.top + h - ((y - yMin) / (yMax - yMin)) * h;

      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 1;
      ctx.fillStyle = "#8b949e";
      ctx.font = "11px sans-serif";
      for (let i = 0; i <= 4; i++) {
        const y = pad.top + (h * i) / 4;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(pad.left + w, y);
        ctx.stroke();
        const label = yMax - ((yMax - yMin) * i) / 4;
        ctx.fillText(fmt(label, 3), 8, y + 4);
      }
      for (let i = 0; i <= 4; i++) {
        const x = pad.left + (w * i) / 4;
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, pad.top + h);
        ctx.stroke();
        const label = xMin + ((xMax - xMin) * i) / 4;
        ctx.fillText(fmt(label, 0), x - 16, pad.top + h + 24);
      }

      for (const s of series) {
        if (!s.points.length) continue;
        ctx.strokeStyle = s.color;
        ctx.fillStyle = s.color;
        ctx.lineWidth = s.width || 2;
        ctx.setLineDash(s.dash || []);
        ctx.beginPath();
        s.points.forEach((p, idx) => {
          const x = sx(p.x);
          const y = sy(p.y);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
        if (s.points.length <= 120 || s.pointsOnly) {
          for (const p of s.points) {
            ctx.beginPath();
            ctx.arc(sx(p.x), sy(p.y), s.radius || 2.5, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      let lx = pad.left;
      let ly = 12;
      ctx.font = "12px sans-serif";
      for (const s of series) {
        ctx.fillStyle = s.color;
        ctx.fillRect(lx, ly - 8, 10, 3);
        ctx.fillStyle = "#c9d1d9";
        ctx.fillText(s.name, lx + 14, ly - 4);
        lx += Math.min(170, 24 + s.name.length * 7);
      }
    }

    function avg(rows, field) {
      const vals = rows.map((r) => r[field]).filter((v) => v !== null && v !== undefined && Number.isFinite(Number(v))).map(Number);
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    }

    function minBy(rows, field) {
      const vals = rows.filter((r) => r[field] !== null && r[field] !== undefined);
      if (!vals.length) return null;
      return vals.reduce((a, b) => Number(a[field]) <= Number(b[field]) ? a : b);
    }

    function maxBy(rows, field) {
      const vals = rows.filter((r) => r[field] !== null && r[field] !== undefined);
      if (!vals.length) return null;
      return vals.reduce((a, b) => Number(a[field]) >= Number(b[field]) ? a : b);
    }

    function renderKpis(win) {
      const rows = win.rows;
      const latest = rows[rows.length - 1] || {};
      const bestWin = minBy(rows, "train_loss") || {};
      const bestVal = META.best_val_loss;
      const latestVal = META.latest_val_loss;
      const replayRows = rows.filter((r) => (r.guard_attempts || 0) > 1 || (r.guard_rejected_attempts || 0) > 0);
      const kpis = [
        ["Latest train", fmt(latest.train_loss, 4), `step ${latest.step ?? "-"}`],
        ["Best train window", fmt(bestWin.train_loss, 4), `step ${bestWin.step ?? "-"}`],
        ["Latest val", fmt(latestVal, 4), `step ${META.latest_val_step ?? "-"} / ppl ${fmt(META.latest_val_perplexity, 2)}`],
        ["Best val", fmt(bestVal, 4), `step ${META.best_val_step ?? "-"} / ppl ${fmt(META.best_val_perplexity, 2)}`],
        ["Avg grad norm", fmt(avg(rows, "global_grad_norm"), 3), `clip ${fmt(avg(rows, "global_grad_clip_scale"), 5)}`],
        ["Avg tokens/sec", fmt(avg(rows, "tokens_per_sec"), 1), `step sec ${fmt(avg(rows, "step_seconds"), 2)}`],
        ["Replay pressure", fmt(replayRows.length, 0), `${fmt((replayRows.length / Math.max(1, rows.length)) * 100, 1)}% window`],
        ["Median tokens/update", fmt(META.median_tokens_per_update, 0), `${fmt(META.estimated_tokens_seen, 0)} est total`],
      ];
      $("kpis").innerHTML = kpis.map(([label, value, hint]) => `
        <div class="kpi">
          <div class="label">${label}</div>
          <div class="value">${value}</div>
          <div class="hint">${hint}</div>
        </div>
      `).join("");
    }

    function renderTimingBars(rows) {
      const values = TIMING_FIELDS.map((field) => [field, avg(rows, field) || 0]).filter(([, value]) => value > 0);
      const total = values.reduce((sum, [, value]) => sum + value, 0) || 1;
      values.sort((a, b) => b[1] - a[1]);
      $("timingBars").innerHTML = values.map(([field, value], index) => {
        const pct = (value / total) * 100;
        const palette = ["#58a6ff", "#3fb950", "#d29922", "#a371f7", "#39c5cf", "#f0883e", "#ff7b72"];
        return `
          <div class="bar-row">
            <div>${field.replaceAll("_", " ")}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${pct.toFixed(2)}%;background:${palette[index % palette.length]}"></div></div>
            <div>${fmt(value, 3)}s</div>
          </div>
        `;
      }).join("") || `<div class="empty">No timing fields in this window.</div>`;
    }

    function table(headers, rows, mapper) {
      if (!rows.length) return `<div class="empty">No rows in this window.</div>`;
      return `
        <table>
          <thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead>
          <tbody>${rows.map((row) => `<tr>${mapper(row).map((v) => `<td>${v}</td>`).join("")}</tr>`).join("")}</tbody>
        </table>
      `;
    }

    function renderTables(win) {
      const rows = win.rows;
      const spikeFloor = Number($("spikeThreshold").value || 4);
      const vals = win.vals.slice().sort((a, b) => a.step - b.step);
      $("valTable").innerHTML = table(["step", "val loss", "ppl", "batches"], vals, (r) => [
        r.step, fmt(r.val_loss, 4), fmt(r.val_perplexity, 2), fmt(r.validation_batches, 0)
      ]);

      const spikes = rows.filter((r) => Number(r.train_loss) >= spikeFloor)
        .sort((a, b) => b.train_loss - a.train_loss)
        .slice(0, 40);
      $("spikeTable").innerHTML = table(["step", "loss", "delta", "gnorm", "clip", "after"], spikes, (r) => [
        r.step, fmt(r.train_loss, 4), fmt(r.train_loss_delta, 4), fmt(r.global_grad_norm, 3),
        fmt(r.global_grad_clip_scale, 5), fmt(r.guard_loss_after, 4)
      ]);

      const replay = rows.filter((r) =>
        (r.guard_attempts || 0) > 1 ||
        (r.guard_rejected_attempts || 0) > 0 ||
        (r.guard_lr_scale !== null && r.guard_lr_scale !== 1) ||
        (r.guard_grad_norm_scale !== null && r.guard_grad_norm_scale !== 1)
      ).slice(-80).reverse();
      $("replayTable").innerHTML = table(["step", "attempts", "rejected", "lr scale", "grad scale", "before", "after"], replay, (r) => [
        r.step, r.guard_attempts ?? "-", r.guard_rejected_attempts ?? 0, fmt(r.guard_lr_scale, 3),
        fmt(r.guard_grad_norm_scale, 3), fmt(r.guard_loss_before, 4), fmt(r.guard_loss_after, 4)
      ]);

      const slow = rows.filter((r) => r.step_seconds !== null && r.step_seconds !== undefined)
        .sort((a, b) => b.step_seconds - a.step_seconds)
        .slice(0, 40);
      $("slowTable").innerHTML = table(["step", "sec", "tok/s", "forward", "update", "build", "h2d"], slow, (r) => [
        r.step, fmt(r.step_seconds, 2), fmt(r.tokens_per_sec, 1), fmt(r.forward_trace_seconds, 2),
        fmt(r.shard_update_seconds, 2), fmt(r.module_build_seconds, 2), fmt(r.h2d_seconds, 2)
      ]);
    }

    function renderCharts(win) {
      const rows = win.rows;
      const vals = win.vals;
      drawChart("lossChart", [
        { name: "train loss", color: colors.train, points: rows.map((r) => point(r, "train_loss")).filter(Boolean), width: 1.5 },
        { name: "train EMA", color: colors.ema, points: rows.map((r) => point(r, "train_loss_ema")).filter(Boolean), width: 2.4 },
        { name: "val loss", color: colors.val, points: vals.map((r) => point(r, "val_loss")).filter(Boolean), width: 0, pointsOnly: true, radius: 4 },
      ]);
      drawChart("gradChart", [
        { name: "global grad norm", color: colors.grad, points: rows.map((r) => point(r, "global_grad_norm")).filter(Boolean) },
        { name: "grad mean", color: colors.gradMean, points: rows.map((r) => point(r, "grad_mean")).filter(Boolean) },
        { name: "grad max", color: colors.gradMax, points: rows.map((r) => point(r, "grad_max")).filter(Boolean) },
      ], { zeroFloor: true });
      drawChart("guardChart", [
        { name: "clip scale", color: colors.clip, points: rows.map((r) => point(r, "global_grad_clip_scale")).filter(Boolean) },
        { name: "attempts", color: colors.attempts, points: rows.map((r) => point(r, "guard_attempts")).filter(Boolean) },
        { name: "lr scale", color: colors.lrScale, points: rows.map((r) => point(r, "guard_lr_scale")).filter(Boolean), dash: [5, 4] },
        { name: "grad scale", color: colors.gradScale, points: rows.map((r) => point(r, "guard_grad_norm_scale")).filter(Boolean), dash: [5, 4] },
      ], { zeroFloor: true });
      drawChart("noiseChart", [
        { name: "loss delta", color: colors.noise, points: rows.map((r) => point(r, "train_loss_delta")).filter(Boolean) },
        { name: "rolling std", color: colors.std, points: rows.map((r) => point(r, "train_loss_rolling_std")).filter(Boolean) },
        { name: "guard delta", color: colors.warn, points: rows.map((r) => point(r, "guard_loss_delta")).filter(Boolean), dash: [5, 4] },
      ]);
      drawChart("throughputChart", [
        { name: "tokens/sec", color: colors.throughput, points: rows.map((r) => point(r, "tokens_per_sec")).filter(Boolean) },
        { name: "step seconds", color: colors.seconds, points: rows.map((r) => point(r, "step_seconds")).filter(Boolean) },
      ], { zeroFloor: true });
      drawChart("timingChart", [
        { name: "forward trace", color: "#58a6ff", points: rows.map((r) => point(r, "forward_trace_seconds")).filter(Boolean) },
        { name: "update", color: "#d29922", points: rows.map((r) => point(r, "shard_update_seconds")).filter(Boolean) },
        { name: "backward", color: "#a371f7", points: rows.map((r) => point(r, "backward_update_seconds")).filter(Boolean) },
        { name: "module build", color: "#3fb950", points: rows.map((r) => point(r, "module_build_seconds")).filter(Boolean) },
        { name: "h2d", color: "#f0883e", points: rows.map((r) => point(r, "h2d_seconds")).filter(Boolean) },
      ], { zeroFloor: true });
      drawChart("memoryChart", [
        { name: "allocated MB", color: colors.memory, points: rows.map((r) => point(r, "memory_allocated_mb")).filter(Boolean) },
        { name: "reserved MB", color: "#f2cc60", points: rows.map((r) => point(r, "memory_reserved_mb")).filter(Boolean) },
        { name: "peak allocated MB", color: "#ff7b72", points: rows.map((r) => point(r, "memory_peak_allocated_mb")).filter(Boolean) },
      ], { zeroFloor: true });
    }

    function updateRangeControls() {
      const size = windowSize();
      const maxStart = Math.max(0, TRAIN.length - size);
      $("windowStart").max = String(maxStart);
      $("windowStart").value = String(Math.min(state.start, maxStart));
    }

    function render() {
      updateRangeControls();
      const win = currentWindow();
      renderKpis(win);
      renderCharts(win);
      renderTimingBars(win.rows);
      renderTables(win);
      const first = win.rows[0] || {};
      const last = win.rows[win.rows.length - 1] || {};
      $("rangeLabel").textContent = `window rows ${state.start + 1}-${state.start + win.rows.length} / steps ${first.step ?? "-"}-${last.step ?? "-"}`;
    }

    function exportCsv() {
      const rows = currentWindow().rows;
      const cols = [
        "i","step","train_loss","train_loss_ema","train_loss_delta","lr","base_lr",
        "tokens_per_sec","step_seconds","global_grad_norm","global_grad_clip_scale",
        "grad_mean","grad_max","guard_attempts","guard_lr_scale","guard_grad_norm_scale",
        "guard_loss_before","guard_loss_after","forward_trace_seconds","shard_update_seconds",
        "backward_update_seconds","module_build_seconds","h2d_seconds","memory_reserved_mb"
      ];
      const csv = [cols.join(",")].concat(rows.map((r) =>
        cols.map((c) => r[c] === null || r[c] === undefined ? "" : JSON.stringify(r[c])).join(",")
      )).join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "telemetry_window.csv";
      a.click();
      URL.revokeObjectURL(url);
    }

    let timer = null;
    function setPlaying(value) {
      state.playing = value;
      $("playButton").textContent = value ? "Pause" : "Play";
      if (timer) clearInterval(timer);
      if (value) {
        timer = setInterval(() => {
          const size = windowSize();
          const step = Math.max(1, Math.floor(size / 12));
          const maxStart = Math.max(0, TRAIN.length - size);
          state.start = state.start >= maxStart ? 0 : Math.min(maxStart, state.start + step);
          render();
        }, 900);
      }
    }

    function init() {
      $("filePath").textContent = META.input_path;
      $("generatedAt").textContent = `generated ${META.generated_at}`;
      $("lineCounts").textContent = `${META.train_rows} train rows, ${META.validation_rows} validation rows, ${META.skipped_lines} skipped`;
      $("windowStart").addEventListener("input", (event) => {
        state.start = Number(event.target.value);
        render();
      });
      $("windowSize").addEventListener("change", () => render());
      $("xMode").addEventListener("change", () => render());
      $("spikeThreshold").addEventListener("change", () => render());
      $("playButton").addEventListener("click", () => setPlaying(!state.playing));
      $("fitButton").addEventListener("click", () => {
        $("windowSize").value = "all";
        state.start = 0;
        render();
      });
      $("csvButton").addEventListener("click", exportCsv);
      window.addEventListener("resize", () => render());
      render();
    }

    init();
  </script>
</body>
</html>
"""


def build_html(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    title: str,
) -> str:
    return (
        HTML_TEMPLATE.replace("__TITLE__", title)
        .replace("__TRAIN_DATA__", _safe_json(train_rows))
        .replace("__VAL_DATA__", _safe_json(val_rows))
        .replace("__META_DATA__", _safe_json(summary))
        .replace("__TIMING_FIELDS__", _safe_json(TIMING_FIELDS))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML telemetry dashboard from a Perkunas train_log.jsonl."
    )
    parser.add_argument("-input", "--input", dest="input_path", required=True, help="Input train_log.jsonl")
    parser.add_argument(
        "-output",
        "--output",
        dest="output_path",
        help="Output HTML path. Defaults to ./<input-stem>_telemetry.html",
    )
    parser.add_argument("--title", default="Perkunas Training Telemetry")
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input log not found: {input_path}")
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else Path.cwd() / f"{input_path.stem}_telemetry.html"
    )

    train_rows, val_rows, summary = parse_train_log(input_path)
    html = build_html(train_rows, val_rows, summary, args.title)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path}")
    print(
        f"Parsed {summary['train_rows']} train rows and {summary['validation_rows']} validation rows "
        f"from {summary['parsed_lines']} JSON lines."
    )
    if summary["best_val_loss"] is not None:
        print(
            "Best validation: "
            f"step={summary['best_val_step']} "
            f"loss={summary['best_val_loss']:.6f} "
            f"ppl={summary['best_val_perplexity']:.3f}"
        )


if __name__ == "__main__":
    main()
