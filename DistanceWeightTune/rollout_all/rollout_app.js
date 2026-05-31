const DATA = window.ROLLOUT_DATA || { rows: [], summaries: [], latDomain: [30, 55] };
const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  setup: "",
  pathId: "",
  viewMode: "path",
  activeDot: null,
};

const els = {
  setupSelect: document.getElementById("setupSelect"),
  pathSelect: document.getElementById("pathSelect"),
  showObserved: document.getElementById("showObserved"),
  showTruth: document.getElementById("showTruth"),
  showDirect: document.getElementById("showDirect"),
  showLstm: document.getElementById("showLstm"),
  showTransformer: document.getElementById("showTransformer"),
  showPredictedPoints: document.getElementById("showPredictedPoints"),
  showPersistence: document.getElementById("showPersistence"),
  showVelocity: document.getElementById("showVelocity"),
  plot: document.getElementById("rolloutPlot"),
  tooltip: document.getElementById("tooltip"),
  selectedPathName: document.getElementById("selectedPathName"),
  mapTitle: document.getElementById("mapTitle"),
  observedDays: document.getElementById("observedDays"),
  rolloutDays: document.getElementById("rolloutDays"),
  directMean: document.getElementById("directMean"),
  lstmMean: document.getElementById("lstmMean"),
  transformerMean: document.getElementById("transformerMean"),
  directFinal: document.getElementById("directFinal"),
  lstmFinal: document.getElementById("lstmFinal"),
  transformerFinal: document.getElementById("transformerFinal"),
  sourceBird: document.getElementById("sourceBird"),
  dateRange: document.getElementById("dateRange"),
  observedDistance: document.getElementById("observedDistance"),
  modelNames: document.getElementById("modelNames"),
  fitPathButton: document.getElementById("fitPathButton"),
  fitAllButton: document.getElementById("fitAllButton"),
};

const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const precise = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });

const summariesByKey = new Map(DATA.summaries.map((row) => [`${row.setup}::${row.path_id}`, row]));

function rowsForSelection() {
  return DATA.rows.filter((row) => row.setup === state.setup && row.path_id === state.pathId);
}

function rowsForDomain(pathRows) {
  if (state.viewMode === "all") {
    return DATA.rows.filter((row) => row.setup === state.setup);
  }
  return pathRows;
}

function extent(rows, keys) {
  let min = Infinity;
  let max = -Infinity;
  rows.forEach((row) => {
    keys.forEach((key) => {
      const value = Number(row[key]);
      if (Number.isFinite(value)) {
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
    });
  });
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [0, 1];
  }
  if (min === max) {
    return [min - 0.5, max + 0.5];
  }
  return [min, max];
}

function paddedDomain([min, max], ratio = 0.08) {
  const span = max - min || 1;
  return [min - span * ratio, max + span * ratio];
}

function buildTicks([min, max], count = 6) {
  const ticks = [];
  const step = (max - min) / Math.max(count - 1, 1);
  for (let i = 0; i < count; i += 1) {
    ticks.push(min + step * i);
  }
  return ticks;
}

function formatKm(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${fmt.format(n)} km` : "--";
}

function formatDegree(value, axis) {
  const direction = axis === "lat"
    ? value >= 0 ? "N" : "S"
    : value >= 0 ? "E" : "W";
  return `${Math.abs(value).toFixed(2)} deg ${direction}`;
}

function createSvgElement(name, attrs = {}) {
  const el = document.createElementNS(SVG_NS, name);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
  return el;
}

function pathData(rows, latKey, lonKey, x, y) {
  return rows
    .map((row, index) => `${index === 0 ? "M" : "L"} ${x(Number(row[lonKey])).toFixed(2)} ${y(Number(row[latKey])).toFixed(2)}`)
    .join(" ");
}

function drawLine(svg, rows, latKey, lonKey, className, x, y) {
  const valid = rows.filter((row) => Number.isFinite(Number(row[latKey])) && Number.isFinite(Number(row[lonKey])));
  if (valid.length < 2) return;
  svg.appendChild(createSvgElement("path", {
    class: `series-line ${className}`,
    d: pathData(valid, latKey, lonKey, x, y),
  }));
}

function flyThresholdFor(row, modelKey) {
  const summary = summariesByKey.get(`${row.setup}::${row.path_id}`) || {};
  const threshold = Number(summary[`${modelKey}_fly_threshold`]);
  return Number.isFinite(threshold) ? threshold : 0.5;
}

function isFlyPrediction(row, modelKey) {
  if (row.phase !== "rollout") return false;
  const probability = Number(row[`${modelKey}_fly_probability`]);
  return Number.isFinite(probability) && probability >= flyThresholdFor(row, modelKey);
}

function drawFlyGatedLine(svg, rows, latKey, lonKey, className, modelKey, x, y) {
  const segments = [];
  for (let i = 1; i < rows.length; i += 1) {
    const previous = rows[i - 1];
    const row = rows[i];
    if (!isFlyPrediction(row, modelKey)) continue;
    const prevLat = Number(previous[latKey]);
    const prevLon = Number(previous[lonKey]);
    const lat = Number(row[latKey]);
    const lon = Number(row[lonKey]);
    if (![prevLat, prevLon, lat, lon].every(Number.isFinite)) continue;
    segments.push(`M ${x(prevLon).toFixed(2)} ${y(prevLat).toFixed(2)} L ${x(lon).toFixed(2)} ${y(lat).toFixed(2)}`);
  }
  if (!segments.length) return;
  svg.appendChild(createSvgElement("path", {
    class: `series-line ${className}`,
    d: segments.join(" "),
  }));
}

function drawPredictedDots(svg, rows, latKey, lonKey, className, label, errorKey, x, y, modelKey = null) {
  const group = createSvgElement("g");
  rows
    .filter((row) => row.phase === "rollout")
    .forEach((row) => {
      const lat = Number(row[latKey]);
      const lon = Number(row[lonKey]);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
      const noFly = modelKey ? !isFlyPrediction(row, modelKey) : false;
      const dot = createSvgElement("circle", {
        class: `path-dot predicted-dot ${className}${noFly ? " nofly-dot" : ""}`,
        cx: x(lon),
        cy: y(lat),
        r: noFly ? 3.2 : 4.2,
        tabindex: "0",
      });
      dot.addEventListener("mouseenter", (event) => showPredictionTooltip(event, row, label, lat, lon, errorKey, modelKey));
      dot.addEventListener("mousemove", positionTooltip);
      dot.addEventListener("mouseleave", hideTooltip);
      dot.addEventListener("focus", (event) => showPredictionTooltip(event, row, label, lat, lon, errorKey, modelKey));
      dot.addEventListener("blur", hideTooltip);
      group.appendChild(dot);
    });
  svg.appendChild(group);
}

function addEndpoint(svg, cx, cy, label, className) {
  svg.appendChild(createSvgElement("circle", { class: className, cx, cy, r: 6.8 }));
  const text = createSvgElement("text", { class: "marker-label", x: cx + 10, y: cy - 10 });
  text.textContent = label;
  svg.appendChild(text);
}

function drawPlot(rows) {
  const svg = els.plot;
  svg.replaceChildren();
  const rect = svg.getBoundingClientRect();
  const width = Math.max(rect.width, 720);
  const height = Math.max(rect.height, 420);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  if (!rows.length) return;

  const margin = { top: 28, right: 34, bottom: 48, left: 70 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const domainRows = rowsForDomain(rows);
  const lonDomain = paddedDomain(extent(domainRows, [
    "true_lon", "direct_lon", "lstm_lon", "transformer_lon", "persistence_lon", "const_velocity_lon",
  ]));
  const latDomain = [Number(DATA.latDomain[0]), Number(DATA.latDomain[1])];
  const x = (lon) => margin.left + ((lon - lonDomain[0]) / (lonDomain[1] - lonDomain[0])) * plotWidth;
  const y = (lat) => margin.top + (1 - ((lat - latDomain[0]) / (latDomain[1] - latDomain[0]))) * plotHeight;

  const grid = createSvgElement("g");
  buildTicks(lonDomain, 7).forEach((tick) => {
    const tx = x(tick);
    grid.appendChild(createSvgElement("line", {
      class: "grid-line", x1: tx, y1: margin.top, x2: tx, y2: margin.top + plotHeight,
    }));
    const label = createSvgElement("text", {
      class: "tick-label", x: tx, y: height - 22, "text-anchor": "middle",
    });
    label.textContent = formatDegree(tick, "lon");
    grid.appendChild(label);
  });
  [55, 50, 45, 40, 35, 30].forEach((tick) => {
    const ty = y(tick);
    grid.appendChild(createSvgElement("line", {
      class: "grid-line", x1: margin.left, y1: ty, x2: margin.left + plotWidth, y2: ty,
    }));
    const label = createSvgElement("text", { class: "tick-label", x: 14, y: ty + 4 });
    label.textContent = formatDegree(tick, "lat");
    grid.appendChild(label);
  });
  grid.appendChild(createSvgElement("line", {
    class: "axis-line", x1: margin.left, y1: margin.top + plotHeight, x2: margin.left + plotWidth, y2: margin.top + plotHeight,
  }));
  grid.appendChild(createSvgElement("line", {
    class: "axis-line", x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotHeight,
  }));
  svg.appendChild(grid);

  if (els.showTruth.checked) drawLine(svg, rows, "true_lat", "true_lon", "truth-line", x, y);
  if (els.showObserved.checked) drawLine(svg, rows.filter((row) => row.phase === "observed"), "true_lat", "true_lon", "observed-line", x, y);
  if (els.showDirect.checked) drawLine(svg, rows, "direct_lat", "direct_lon", "direct-line", x, y);
  if (els.showLstm.checked) drawFlyGatedLine(svg, rows, "lstm_lat", "lstm_lon", "lstm-line", "lstm", x, y);
  if (els.showTransformer.checked) drawFlyGatedLine(svg, rows, "transformer_lat", "transformer_lon", "transformer-line", "transformer", x, y);
  if (els.showPersistence.checked) drawLine(svg, rows, "persistence_lat", "persistence_lon", "persistence-line", x, y);
  if (els.showVelocity.checked) drawLine(svg, rows, "const_velocity_lat", "const_velocity_lon", "velocity-line", x, y);
  if (els.showPredictedPoints.checked) {
    if (els.showDirect.checked) drawPredictedDots(svg, rows, "direct_lat", "direct_lon", "direct-dot", "Direct", "direct_error_km", x, y);
    if (els.showLstm.checked) drawPredictedDots(svg, rows, "lstm_lat", "lstm_lon", "lstm-dot", "LSTM", "lstm_error_km", x, y, "lstm");
    if (els.showTransformer.checked) drawPredictedDots(svg, rows, "transformer_lat", "transformer_lon", "transformer-dot", "Transformer", "transformer_error_km", x, y, "transformer");
  }

  const dots = createSvgElement("g");
  rows.forEach((row, index) => {
    const dot = createSvgElement("circle", {
      class: `path-dot ${row.phase === "observed" ? "context-dot" : "rollout-dot"}`,
      cx: x(Number(row.true_lon)),
      cy: y(Number(row.true_lat)),
      r: row.phase === "observed" ? 4.4 : 5.4,
      tabindex: "0",
      "data-index": index,
    });
    dot.addEventListener("mouseenter", (event) => showTooltip(event, row, index, rows.length));
    dot.addEventListener("mousemove", positionTooltip);
    dot.addEventListener("mouseleave", hideTooltip);
    dot.addEventListener("focus", (event) => showTooltip(event, row, index, rows.length));
    dot.addEventListener("blur", hideTooltip);
    dots.appendChild(dot);
  });
  svg.appendChild(dots);

  const first = rows[0];
  const lastObserved = [...rows].reverse().find((row) => row.phase === "observed") || first;
  const last = rows[rows.length - 1];
  addEndpoint(svg, x(Number(first.true_lon)), y(Number(first.true_lat)), "Start", "start-marker");
  addEndpoint(svg, x(Number(lastObserved.true_lon)), y(Number(lastObserved.true_lat)), "Observed", "start-marker");
  addEndpoint(svg, x(Number(last.true_lon)), y(Number(last.true_lat)), "End", "end-marker");
}

function showTooltip(event, row, index, total) {
  if (state.activeDot) state.activeDot.classList.remove("is-active");
  state.activeDot = event.currentTarget;
  state.activeDot.classList.add("is-active");
  els.tooltip.innerHTML = `
    <strong>${row.date} (${index + 1} / ${total})</strong>
    ${row.phase} day ${row.step_index}<br>
    True ${Number(row.true_lat).toFixed(5)}, ${Number(row.true_lon).toFixed(5)}<br>
    Step ${formatKm(row.target_step_km)}<br>
    Direct error ${formatKm(row.direct_error_km)}<br>
    LSTM error ${formatKm(row.lstm_error_km)}<br>
    Transformer error ${formatKm(row.transformer_error_km)}
  `;
  els.tooltip.hidden = false;
  positionTooltip(event);
}

function showPredictionTooltip(event, row, label, lat, lon, errorKey, modelKey = null) {
  if (state.activeDot) state.activeDot.classList.remove("is-active");
  state.activeDot = event.currentTarget;
  state.activeDot.classList.add("is-active");
  const probability = modelKey ? Number(row[`${modelKey}_fly_probability`]) : NaN;
  const threshold = modelKey ? flyThresholdFor(row, modelKey) : NaN;
  const decision = modelKey && Number.isFinite(probability)
    ? `<br>Fly ${probability >= threshold ? "yes" : "no"} (${precise.format(probability)} / ${precise.format(threshold)})`
    : "";
  els.tooltip.innerHTML = `
    <strong>${label} prediction: ${row.date}</strong>
    Rollout day ${row.rollout_step}<br>
    Pred ${lat.toFixed(5)}, ${lon.toFixed(5)}<br>
    True ${Number(row.true_lat).toFixed(5)}, ${Number(row.true_lon).toFixed(5)}<br>
    Error ${formatKm(row[errorKey])}${decision}
  `;
  els.tooltip.hidden = false;
  positionTooltip(event);
}

function positionTooltip(event) {
  const wrap = els.plot.parentElement.getBoundingClientRect();
  const tooltip = els.tooltip;
  const tipRect = tooltip.getBoundingClientRect();
  const sourceX = event.clientX ?? wrap.left + Number(event.currentTarget.getAttribute("cx"));
  const sourceY = event.clientY ?? wrap.top + Number(event.currentTarget.getAttribute("cy"));
  let left = sourceX - wrap.left + 14;
  let top = sourceY - wrap.top + 14;
  if (left + tipRect.width > wrap.width - 10) left = sourceX - wrap.left - tipRect.width - 14;
  if (top + tipRect.height > wrap.height - 10) top = sourceY - wrap.top - tipRect.height - 14;
  tooltip.style.left = `${Math.max(10, left)}px`;
  tooltip.style.top = `${Math.max(10, top)}px`;
}

function hideTooltip() {
  if (state.activeDot) {
    state.activeDot.classList.remove("is-active");
    state.activeDot = null;
  }
  els.tooltip.hidden = true;
}

function updateSummary(rows) {
  const summary = summariesByKey.get(`${state.setup}::${state.pathId}`) || {};
  const first = rows[0] || {};
  const last = rows[rows.length - 1] || {};
  els.selectedPathName.textContent = state.pathId || "Select a path";
  els.mapTitle.textContent = `${state.setup}: ${state.pathId}`;
  els.observedDays.textContent = `${summary.observed_days ?? "--"} days`;
  els.rolloutDays.textContent = `${summary.rollout_steps ?? "--"} days`;
  els.directMean.textContent = formatKm(summary.direct_mean_rollout_error_km);
  els.lstmMean.textContent = formatKm(summary.lstm_mean_rollout_error_km);
  els.transformerMean.textContent = formatKm(summary.transformer_mean_rollout_error_km);
  els.directFinal.textContent = formatKm(summary.direct_final_rollout_error_km);
  els.lstmFinal.textContent = formatKm(summary.lstm_final_rollout_error_km);
  els.transformerFinal.textContent = formatKm(summary.transformer_final_rollout_error_km);
  els.sourceBird.textContent = summary.source_bird_id || first.source_bird_id || "--";
  els.dateRange.textContent = first.date && last.date ? `${first.date} to ${last.date}` : "--";
  els.observedDistance.textContent = formatKm(summary.observed_displacement_km);
  els.modelNames.textContent = `Direct: ${summary.direct_model || first.direct_model || "--"}; LSTM: ${summary.lstm_model || first.lstm_model || "--"}; Transformer: ${summary.transformer_model || first.transformer_model || "--"}`;
}

function render() {
  const rows = rowsForSelection();
  hideTooltip();
  updateSummary(rows);
  drawPlot(rows);
}

function populateSetupSelect() {
  const setups = [...new Set(DATA.summaries.map((row) => row.setup))].sort();
  els.setupSelect.replaceChildren();
  setups.forEach((setup) => {
    const option = document.createElement("option");
    option.value = setup;
    option.textContent = setup;
    els.setupSelect.appendChild(option);
  });
  state.setup = setups[0] || "";
  els.setupSelect.value = state.setup;
}

function populatePathSelect() {
  const summaries = DATA.summaries
    .filter((row) => row.setup === state.setup)
    .sort((a, b) => Number(b.rollout_steps) - Number(a.rollout_steps) || a.path_id.localeCompare(b.path_id));
  els.pathSelect.replaceChildren();
  summaries.forEach((row) => {
    const option = document.createElement("option");
    option.value = row.path_id;
    option.textContent = `${row.source_bird_id} / ${row.path_year} / ${row.rollout_steps} rollout days`;
    els.pathSelect.appendChild(option);
  });
  state.pathId = summaries[0]?.path_id || "";
  els.pathSelect.value = state.pathId;
}

els.setupSelect.addEventListener("change", () => {
  state.setup = els.setupSelect.value;
  state.viewMode = "path";
  populatePathSelect();
  render();
});

els.pathSelect.addEventListener("change", () => {
  state.pathId = els.pathSelect.value;
  state.viewMode = "path";
  render();
});

[els.showObserved, els.showTruth, els.showDirect, els.showLstm, els.showTransformer, els.showPredictedPoints, els.showPersistence, els.showVelocity]
  .forEach((input) => input.addEventListener("input", render));

els.fitPathButton.addEventListener("click", () => {
  state.viewMode = "path";
  render();
});

els.fitAllButton.addEventListener("click", () => {
  state.viewMode = "all";
  render();
});

window.addEventListener("resize", render);

populateSetupSelect();
populatePathSelect();
render();
