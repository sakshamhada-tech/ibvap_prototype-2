// IBVAP Dashboard client
// Connects to /ws for live alerts + stats, updates the DOM directly
// (no framework needed for a page this size).

const MAX_ALERTS_SHOWN = 40;

// Maps backend alert_type -> visual severity + human-readable label.
// Severity drives color: critical=red, caution=amber, info=green.
const ALERT_META = {
  VIRTUAL_FENCE_INTRUSION: { level: "critical", label: "Fence intrusion" },
  SUSPICIOUS_LOITERING:    { level: "caution",  label: "Loitering" },
  NIGHT_MOVEMENT:          { level: "caution",  label: "Night movement" },
  ANPR_READ:               { level: "info",     label: "Plate read" },
};

const el = {
  video: document.getElementById("video"),
  connDot: document.getElementById("conn-dot"),
  connLabel: document.getElementById("conn-label"),
  clock: document.getElementById("clock"),
  modeBadge: document.getElementById("mode-badge"),
  frameCounter: document.getElementById("frame-counter"),
  alertLog: document.getElementById("alert-log"),
  alertCount: document.getElementById("alert-count"),
  statPeople: document.getElementById("stat-people"),
  statVehicles: document.getElementById("stat-vehicles"),
  statAlerts: document.getElementById("stat-alerts"),
  statUptime: document.getElementById("stat-uptime"),
  statFps: document.getElementById("stat-fps"),
};

function tickClock() {
  const now = new Date();
  el.clock.textContent = now.toLocaleTimeString("en-GB", { hour12: false });
}
setInterval(tickClock, 1000);
tickClock();

function formatUptime(seconds) {
  const s = Math.floor(seconds % 60);
  const m = Math.floor((seconds / 60) % 60);
  const h = Math.floor(seconds / 3600);
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)}`;
}

function renderAlert(alert) {
  // clear the "no alerts yet" placeholder on first real alert
  const placeholder = el.alertLog.querySelector(".alert-empty");
  if (placeholder) placeholder.remove();

  const meta = ALERT_META[alert.alert_type] || { level: "info", label: alert.alert_type };

  const item = document.createElement("li");
  item.className = "alert-item";
  item.dataset.level = meta.level;

  const top = document.createElement("div");
  top.className = "alert-item__top";

  const type = document.createElement("span");
  type.className = "alert-item__type";
  type.textContent = meta.label + (alert.track_id !== null ? ` · #${alert.track_id}` : "");

  const time = document.createElement("span");
  time.className = "mono muted";
  time.textContent = alert.timestamp.split(" ")[1] || alert.timestamp;

  top.appendChild(type);
  top.appendChild(time);

  const detail = document.createElement("div");
  detail.className = "alert-item__detail";
  detail.textContent = alert.details;

  item.appendChild(top);
  item.appendChild(detail);

  el.alertLog.prepend(item);

  // cap the visible list so the DOM doesn't grow unbounded over a long demo
  const items = el.alertLog.querySelectorAll(".alert-item");
  if (items.length > MAX_ALERTS_SHOWN) {
    items[items.length - 1].remove();
  }
}

function applyStats(stats) {
  if (stats.people !== undefined) el.statPeople.textContent = stats.people;
  if (stats.vehicles !== undefined) el.statVehicles.textContent = stats.vehicles;
  if (stats.alerts_total !== undefined) {
    el.statAlerts.textContent = stats.alerts_total;
    el.alertCount.textContent = `${stats.alerts_total} total`;
  }
  if (stats.uptime_seconds !== undefined) {
    el.statUptime.textContent = formatUptime(stats.uptime_seconds);
  }
  if (stats.fps !== undefined) {
    el.statFps.textContent = stats.fps.toFixed(1);
  }
  if (stats.frame_number !== undefined) {
    el.frameCounter.textContent = `frame ${stats.frame_number}`;
  }
  if (stats.night_mode !== undefined) {
    el.modeBadge.textContent = stats.night_mode ? "night" : "day";
    el.modeBadge.dataset.mode = stats.night_mode ? "night" : "day";
  }
}

function connect() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${window.location.host}/ws`);

  ws.onopen = () => {
    el.connDot.dataset.state = "live";
    el.connLabel.textContent = "live";
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "alert") {
      renderAlert(msg.data);
    } else if (msg.type === "stats") {
      applyStats(msg.data);
    }
  };

  ws.onclose = () => {
    el.connDot.dataset.state = "down";
    el.connLabel.textContent = "reconnecting";
    setTimeout(connect, 1500);
  };

  ws.onerror = () => ws.close();
}

connect();
