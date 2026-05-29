const CSV_PATH = "../data/filtered/dataset2_daily_movement_lat30_50_birds100_all_month_context.csv";
const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  rows: [],
  birds: new Map(),
  selectedBird: "",
  viewMode: "bird",
  activeDot: null,
};

const els = {
  birdSelect: document.getElementById("birdSelect"),
  thresholdInput: document.getElementById("thresholdInput"),
  plot: document.getElementById("migrationPlot"),
  tooltip: document.getElementById("tooltip"),
  mapTitle: document.getElementById("mapTitle"),
  selectedBirdName: document.getElementById("selectedBirdName"),
  startDate: document.getElementById("startDate"),
  endDate: document.getElementById("endDate"),
  distanceTravelled: document.getElementById("distanceTravelled"),
  flydays: document.getElementById("flydays"),
  daysTracked: document.getElementById("daysTracked"),
  recordCount: document.getElementById("recordCount"),
  latRange: document.getElementById("latRange"),
  lonRange: document.getElementById("lonRange"),
  longestMove: document.getElementById("longestMove"),
  fitBirdButton: document.getElementById("fitBirdButton"),
  fitAllButton: document.getElementById("fitAllButton"),
};

const numberFormat = new Intl.NumberFormat("en-US");
const compactFormat = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 1,
});

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        field += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      row.push(field);
      field = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") {
        i += 1;
      }
      row.push(field);
      if (row.some((value) => value.length > 0)) {
        rows.push(row);
      }
      row = [];
      field = "";
    } else {
      field += char;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  const headers = rows.shift();
  return rows.map((values) => {
    const obj = {};
    headers.forEach((header, index) => {
      obj[header] = values[index] ?? "";
    });
    return obj;
  });
}

function toRecord(row) {
  return {
    bird: row.individual_local_identifier,
    date: row.date.slice(0, 10),
    lat: Number(row.lat_median),
    lon: Number(row.lon_median),
    nPoints: Number(row.n_points),
    timestamp: row.timestamp_min_utc,
    stepKm: Number(row.step_length_km) || 0,
    heading: Number(row.heading_deg),
    speed: Number(row.speed_km_per_day) || 0,
    stopoverDays: Number(row.stopover_duration_days) || 0,
  };
}

function groupBirds(rows) {
  const birds = new Map();
  rows.forEach((row) => {
    if (!birds.has(row.bird)) {
      birds.set(row.bird, []);
    }
    birds.get(row.bird).push(row);
  });

  birds.forEach((birdRows) => {
    birdRows.sort((a, b) => a.date.localeCompare(b.date));
  });

  return birds;
}

function extent(rows, key) {
  let min = Infinity;
  let max = -Infinity;
  rows.forEach((row) => {
    min = Math.min(min, row[key]);
    max = Math.max(max, row[key]);
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

function summarize(rows, thresholdKm) {
  const lat = extent(rows, "lat");
  const lon = extent(rows, "lon");
  const distance = rows.reduce((sum, row) => sum + row.stepKm, 0);
  const flydays = rows.filter((row) => row.stepKm >= thresholdKm).length;
  const longest = rows.reduce((max, row) => Math.max(max, row.stepKm), 0);
  const start = rows[0]?.date ?? "--";
  const end = rows[rows.length - 1]?.date ?? "--";
  const daysTracked = start !== "--" && end !== "--"
    ? Math.round((new Date(`${end}T00:00:00Z`) - new Date(`${start}T00:00:00Z`)) / 86400000) + 1
    : 0;

  return { start, end, distance, flydays, longest, daysTracked, lat, lon };
}

function formatKm(value) {
  return `${compactFormat.format(value)} km`;
}

function formatDegree(value, axis) {
  const direction = axis === "lat"
    ? value >= 0 ? "N" : "S"
    : value >= 0 ? "E" : "W";
  return `${Math.abs(value).toFixed(2)}°${direction}`;
}

function colorForIndex(index, total) {
  const t = total <= 1 ? 0 : index / (total - 1);
  const hue = 220 - 215 * t;
  const lightness = 45 + Math.sin(t * Math.PI) * 5;
  return `hsl(${hue}, 78%, ${lightness}%)`;
}

function createSvgElement(name, attrs = {}) {
  const el = document.createElementNS(SVG_NS, name);
  Object.entries(attrs).forEach(([key, value]) => {
    el.setAttribute(key, String(value));
  });
  return el;
}

function buildTicks([min, max], count = 6) {
  const ticks = [];
  const span = max - min;
  if (span <= 0) {
    return [min];
  }
  const step = span / (count - 1);
  for (let i = 0; i < count; i += 1) {
    ticks.push(min + step * i);
  }
  return ticks;
}

function drawPlot(rows, domainRows) {
  const svg = els.plot;
  svg.replaceChildren();

  const rect = svg.getBoundingClientRect();
  const width = Math.max(rect.width, 720);
  const height = Math.max(rect.height, 420);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  if (!rows.length) {
    return;
  }

  const margin = { top: 28, right: 34, bottom: 48, left: 70 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const lonDomain = paddedDomain(extent(domainRows, "lon"));
  const latDomain = paddedDomain(extent(domainRows, "lat"));

  const x = (lon) => margin.left + ((lon - lonDomain[0]) / (lonDomain[1] - lonDomain[0])) * plotWidth;
  const y = (lat) => margin.top + (1 - ((lat - latDomain[0]) / (latDomain[1] - latDomain[0]))) * plotHeight;

  const grid = createSvgElement("g");
  buildTicks(lonDomain).forEach((tick) => {
    const tx = x(tick);
    grid.appendChild(createSvgElement("line", {
      class: "grid-line",
      x1: tx,
      y1: margin.top,
      x2: tx,
      y2: margin.top + plotHeight,
    }));
    const label = createSvgElement("text", {
      class: "tick-label",
      x: tx,
      y: height - 22,
      "text-anchor": "middle",
    });
    label.textContent = formatDegree(tick, "lon");
    grid.appendChild(label);
  });

  buildTicks(latDomain).forEach((tick) => {
    const ty = y(tick);
    grid.appendChild(createSvgElement("line", {
      class: "grid-line",
      x1: margin.left,
      y1: ty,
      x2: margin.left + plotWidth,
      y2: ty,
    }));
    const label = createSvgElement("text", {
      class: "tick-label",
      x: 14,
      y: ty + 4,
    });
    label.textContent = formatDegree(tick, "lat");
    grid.appendChild(label);
  });

  grid.appendChild(createSvgElement("line", {
    class: "axis-line",
    x1: margin.left,
    y1: margin.top + plotHeight,
    x2: margin.left + plotWidth,
    y2: margin.top + plotHeight,
  }));
  grid.appendChild(createSvgElement("line", {
    class: "axis-line",
    x1: margin.left,
    y1: margin.top,
    x2: margin.left,
    y2: margin.top + plotHeight,
  }));
  svg.appendChild(grid);

  const pathData = rows.map((row, index) => `${index === 0 ? "M" : "L"} ${x(row.lon).toFixed(2)} ${y(row.lat).toFixed(2)}`).join(" ");
  svg.appendChild(createSvgElement("path", { class: "path-line", d: pathData }));

  const dots = createSvgElement("g");
  rows.forEach((row, index) => {
    const radius = row.stepKm >= Number(els.thresholdInput.value) ? 5.5 : 4.2;
    const dot = createSvgElement("circle", {
      class: "path-dot",
      cx: x(row.lon),
      cy: y(row.lat),
      r: radius,
      fill: colorForIndex(index, rows.length),
      tabindex: "0",
      "data-index": index,
    });
    dot.addEventListener("mouseenter", (event) => showTooltip(event, row, index, rows.length));
    dot.addEventListener("mousemove", (event) => positionTooltip(event));
    dot.addEventListener("mouseleave", hideTooltip);
    dot.addEventListener("focus", (event) => showTooltip(event, row, index, rows.length));
    dot.addEventListener("blur", hideTooltip);
    dots.appendChild(dot);
  });
  svg.appendChild(dots);

  const first = rows[0];
  const last = rows[rows.length - 1];
  addEndpoint(svg, x(first.lon), y(first.lat), "Start", "start-marker");
  addEndpoint(svg, x(last.lon), y(last.lat), "End", "end-marker");
}

function addEndpoint(svg, cx, cy, label, className) {
  svg.appendChild(createSvgElement("circle", {
    class: className,
    cx,
    cy,
    r: 6.8,
  }));
  const text = createSvgElement("text", {
    class: "marker-label",
    x: cx + 10,
    y: cy - 10,
  });
  text.textContent = label;
  svg.appendChild(text);
}

function showTooltip(event, row, index, total) {
  const dot = event.currentTarget;
  if (state.activeDot) {
    state.activeDot.classList.remove("is-active");
  }
  state.activeDot = dot;
  dot.classList.add("is-active");
  els.tooltip.innerHTML = `
    <strong>${row.date} (${index + 1} / ${total})</strong>
    Lat ${row.lat.toFixed(5)}, Lon ${row.lon.toFixed(5)}<br>
    Step ${formatKm(row.stepKm)} · Speed ${formatKm(row.speed)}/day<br>
    GPS points ${numberFormat.format(row.nPoints)} · Stopover ${numberFormat.format(row.stopoverDays)} days
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

  if (left + tipRect.width > wrap.width - 10) {
    left = sourceX - wrap.left - tipRect.width - 14;
  }
  if (top + tipRect.height > wrap.height - 10) {
    top = sourceY - wrap.top - tipRect.height - 14;
  }

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
  const threshold = Number(els.thresholdInput.value) || 30;
  const summary = summarize(rows, threshold);
  els.selectedBirdName.textContent = state.selectedBird;
  els.mapTitle.textContent = state.selectedBird;
  els.startDate.textContent = summary.start;
  els.endDate.textContent = summary.end;
  els.distanceTravelled.textContent = formatKm(summary.distance);
  els.flydays.textContent = `${numberFormat.format(summary.flydays)} days`;
  els.daysTracked.textContent = `${numberFormat.format(summary.daysTracked)} days`;
  els.recordCount.textContent = numberFormat.format(rows.length);
  els.latRange.textContent = `${formatDegree(summary.lat[0], "lat")} to ${formatDegree(summary.lat[1], "lat")}`;
  els.lonRange.textContent = `${formatDegree(summary.lon[0], "lon")} to ${formatDegree(summary.lon[1], "lon")}`;
  els.longestMove.textContent = formatKm(summary.longest);
}

function render() {
  const rows = state.birds.get(state.selectedBird) ?? [];
  const domainRows = state.viewMode === "all" ? state.rows : rows;
  hideTooltip();
  updateSummary(rows);
  drawPlot(rows, domainRows);
}

function populateBirdSelect() {
  const options = [...state.birds.entries()]
    .sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));

  els.birdSelect.replaceChildren();
  options.forEach(([bird, rows]) => {
    const option = document.createElement("option");
    option.value = bird;
    option.textContent = `${bird} (${numberFormat.format(rows.length)} records)`;
    els.birdSelect.appendChild(option);
  });

  state.selectedBird = options[0]?.[0] ?? "";
  els.birdSelect.value = state.selectedBird;
  els.birdSelect.disabled = false;
}

async function loadData() {
  const response = await fetch(CSV_PATH);
  if (!response.ok) {
    throw new Error(`Could not load CSV at ${CSV_PATH}`);
  }
  const text = await response.text();
  state.rows = parseCsv(text)
    .map(toRecord)
    .filter((row) => row.bird && Number.isFinite(row.lat) && Number.isFinite(row.lon));
  state.birds = groupBirds(state.rows);
  populateBirdSelect();
  render();
}

els.birdSelect.addEventListener("change", () => {
  state.selectedBird = els.birdSelect.value;
  state.viewMode = "bird";
  render();
});

els.thresholdInput.addEventListener("input", render);

els.fitBirdButton.addEventListener("click", () => {
  state.viewMode = "bird";
  render();
});

els.fitAllButton.addEventListener("click", () => {
  state.viewMode = "all";
  render();
});

window.addEventListener("resize", () => {
  render();
});

loadData().catch((error) => {
  els.mapTitle.textContent = "Could not load dataset";
  els.selectedBirdName.textContent = error.message;
  console.error(error);
});
