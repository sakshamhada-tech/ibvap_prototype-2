const MAX_ALERTS_SHOWN = 40;
const ALERT_META = {
  VIRTUAL_FENCE_INTRUSION: { level: "critical", label: "Fence intrusion" },
  SUSPICIOUS_LOITERING: { level: "caution", label: "Loitering" },
  NIGHT_MOVEMENT: { level: "caution", label: "Night movement" },
  ANPR_READ: { level: "info", label: "Plate read" },
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
  el.clock.textContent = new Date().toLocaleTimeString("en-GB", { hour12: false });
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
  const placeholder = el.alertLog.querySelector(".alert-empty");
  if (placeholder) placeholder.remove();
  const meta = ALERT_META[alert.alert_type] || {
    level: "info",
    label: alert.alert_type || "Unknown event",
  };
  const item = document.createElement("li");
  item.className = "alert-item";
  item.dataset.level = meta.level;

  const top = document.createElement("div");
  top.className = "alert-item__top";
  const type = document.createElement("span");
  type.className = "alert-item__type";
  type.textContent = `${meta.label}${alert.track_id != null ? ` · #${alert.track_id}` : ""}`;
  const eventTime = document.createElement("span");
  eventTime.className = "mono muted";
  const timestamp = typeof alert.timestamp === "string" ? alert.timestamp : "";
  eventTime.textContent = timestamp.includes("T")
    ? timestamp.split("T")[1].replace(/\+.*/, "")
    : timestamp.split(" ")[1] || timestamp;
  top.append(type, eventTime);

  const detail = document.createElement("div");
  detail.className = "alert-item__detail";
  detail.textContent = alert.details || "";
  item.append(top, detail);
  el.alertLog.prepend(item);

  const items = el.alertLog.querySelectorAll(".alert-item");
  if (items.length > MAX_ALERTS_SHOWN) items[items.length - 1].remove();
}

function applyStats(stats) {
  if (stats.people !== undefined) el.statPeople.textContent = stats.people;
  if (stats.vehicles !== undefined) el.statVehicles.textContent = stats.vehicles;
  if (stats.alerts_total !== undefined) {
    el.statAlerts.textContent = stats.alerts_total;
    el.alertCount.textContent = `${stats.alerts_total} this session`;
  }
  if (stats.uptime_seconds !== undefined) {
    el.statUptime.textContent = formatUptime(stats.uptime_seconds);
  }
  if (stats.fps !== undefined) el.statFps.textContent = Number(stats.fps).toFixed(1);
  if (stats.frame_number !== undefined) {
    el.frameCounter.textContent = `frame ${stats.frame_number}`;
  }
  if (stats.night_mode !== undefined) {
    el.modeBadge.textContent = stats.night_mode ? "night" : "day";
    el.modeBadge.dataset.mode = stats.night_mode ? "night" : "day";
  }
  if (stats.service_status) {
    const running = stats.service_status === "running";
    el.connDot.dataset.state = running ? "live" : "down";
    el.connLabel.textContent = running ? "live" : stats.service_status;
    el.video.parentElement.dataset.state = running ? "live" : "down";
  }
}

let reconnectTimer;
function connect() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);
  socket.onopen = () => {
    el.connDot.dataset.state = "live";
    el.connLabel.textContent = "live";
  };
  socket.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "alert") renderAlert(message.data);
      if (message.type === "stats") applyStats(message.data);
    } catch (error) {
      console.error("Invalid dashboard message", error);
    }
  };
  socket.onclose = (event) => {
    el.connDot.dataset.state = "down";
    if (event.code === 4403) {
      window.location.assign("/login");
      return;
    }
    el.connLabel.textContent = event.code === 1013 ? "capacity reached" : "reconnecting";
    reconnectTimer = window.setTimeout(connect, 1500);
  };
  socket.onerror = () => socket.close();
}

window.addEventListener("beforeunload", () => window.clearTimeout(reconnectTimer));
el.video.addEventListener("error", () => {
  el.connDot.dataset.state = "down";
  el.connLabel.textContent = "feed unavailable";
  el.video.parentElement.dataset.state = "down";
});
connect();
