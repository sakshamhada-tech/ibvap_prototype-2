const MAX_ALERTS_SHOWN = 40;
const ALERT_META = {
  VIRTUAL_FENCE_INTRUSION: { level: "critical", label: "Fence intrusion" },
  FIREARM_DETECTED: { level: "critical", label: "Firearm review" },
  CONTEXTUAL_RISK: { level: "critical", label: "Contextual risk" },
  GROUP_APPROACH: { level: "caution", label: "Group approach" },
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
  enhancedFaces: document.getElementById("enhanced-faces"),
  faceStatus: document.getElementById("face-status"),
  statPeople: document.getElementById("stat-people"),
  statVehicles: document.getElementById("stat-vehicles"),
  statAlerts: document.getElementById("stat-alerts"),
  statUptime: document.getElementById("stat-uptime"),
  statFps: document.getElementById("stat-fps"),
  statRecording: document.getElementById("stat-recording"),
  statAlarm: document.getElementById("stat-alarm"),
  statAnpr: document.getElementById("stat-anpr"),
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
  const timestamp =
    typeof alert.timestamp_utc === "string"
      ? alert.timestamp_utc
      : typeof alert.timestamp === "string"
        ? alert.timestamp
        : "";
  const instant = new Date(timestamp);
  if (timestamp && !Number.isNaN(instant.getTime())) {
    eventTime.textContent = instant.toLocaleTimeString("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
    eventTime.dateTime = instant.toISOString();
    eventTime.title = `System time: ${instant.toLocaleString()} · UTC: ${instant.toISOString()}`;
  } else {
    eventTime.textContent = timestamp;
  }
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
  if (stats.recording_status) el.statRecording.textContent = stats.recording_status;
  if (stats.alarm_status) el.statAlarm.textContent = stats.alarm_status;
  if (stats.anpr) {
    const attempts = Number(stats.anpr.attempts || 0);
    const valid = Number(stats.anpr.valid_observations || 0);
    const result = stats.anpr.last_result || stats.anpr.status || "unknown";
    el.statAnpr.textContent =
      stats.anpr.status === "disabled" ? "disabled" : `${result} · ${valid}/${attempts}`;
    el.statAnpr.title = [
      `candidates=${stats.anpr.detector_candidates || 0}`,
      `small=${stats.anpr.rejected_small || 0}`,
      `blurry=${stats.anpr.rejected_blurry || 0}`,
      `no-text=${stats.anpr.ocr_no_text || 0}`,
      `low-confidence=${stats.anpr.ocr_low_confidence || 0}`,
      `region-rejected=${stats.anpr.region_rejected || 0}`,
      `stable=${stats.anpr.stable_reads || 0}`,
    ].join(" · ");
  }
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

function renderFaces(snapshot) {
  el.enhancedFaces.replaceChildren();
  el.faceStatus.textContent = snapshot.status || "unknown";
  const faces = Array.isArray(snapshot.faces) ? snapshot.faces : [];
  if (faces.length === 0) {
    const empty = document.createElement("p");
    empty.className = "face-empty";
    if (snapshot.status === "disabled") {
      empty.textContent = "Face extraction is disabled. Enable SCRFD to use this panel.";
    } else if (snapshot.status === "loading") {
      empty.textContent = "Loading SCRFD face detector…";
    } else if (snapshot.status === "unavailable" || snapshot.status === "error") {
      empty.textContent = snapshot.message || "Face enhancement is unavailable.";
    } else {
      empty.textContent = "No tracked faces are currently available.";
    }
    el.enhancedFaces.append(empty);
    return;
  }

  const rejectionLabels = {
    source_face_too_small: "GFPGAN REJECTED · source face too small",
    source_face_too_blurry: "GFPGAN REJECTED · source face too blurred",
    alignment_failed: "ALIGNMENT REJECTED · landmarks invalid",
    alignment_encoding_failed: "ALIGNMENT REJECTED · encoding failed",
    restoration_failed: "SOURCE ONLY · GFPGAN failed",
    restoration_encoding_failed: "SOURCE ONLY · GFPGAN encoding failed",
  };

  const makeVariant = (url, label, alt) => {
    const variant = document.createElement("figure");
    variant.className = "face-variant";
    const image = document.createElement("img");
    image.src = url;
    image.alt = alt;
    const caption = document.createElement("figcaption");
    caption.textContent = label;
    variant.append(image, caption);
    return variant;
  };

  for (const face of faces) {
    const card = document.createElement("article");
    card.className = "face-card";
    const comparison = document.createElement("div");
    comparison.className = "face-comparison";
    comparison.append(
      makeVariant(
        face.source_image_url,
        "DETECTED SOURCE · NO AI",
        `Original detected face pixels for track ${face.track_id}`,
      ),
    );
    if (face.aligned_image_url) {
      comparison.append(
        makeVariant(
          face.aligned_image_url,
          "LANDMARK ALIGNED · NO AI",
          `Landmark-aligned source face for track ${face.track_id}`,
        ),
      );
    }

    if (face.review_image_url) {
      const blendWeight = Number(face.blend_weight);
      const blendLabel =
        face.blend_weight !== null && Number.isFinite(blendWeight)
          ? `EXPERIMENTAL GFPGAN · ${Math.round(blendWeight * 100)}%`
          : "EXPERIMENTAL GFPGAN";
      comparison.append(
        makeVariant(
          face.review_image_url,
          blendLabel,
          `Conservative AI face blend for track ${face.track_id}`,
        ),
      );
    } else if (face.restoration_status === "rejected") {
      const rejected = document.createElement("div");
      rejected.className = "face-rejected";
      rejected.textContent = rejectionLabels[face.quality_reason] || "SOURCE ONLY · GFPGAN rejected";
      comparison.append(rejected);
    }

    const details = document.createElement("p");
    details.className = "face-details mono";
    const confidence = Number(face.confidence);
    const sourceSize = Number(face.source_face_size_px);
    const confidenceText =
      face.confidence !== null && Number.isFinite(confidence)
        ? ` · detection ${confidence.toFixed(2)}`
        : "";
    const sizeText =
      face.source_face_size_px !== null && Number.isFinite(sourceSize)
        ? ` · source ${Math.round(sourceSize)}px`
        : "";
    details.textContent = `TRACK #${face.track_id}${confidenceText}${sizeText}`;
    card.append(comparison, details);
    el.enhancedFaces.append(card);
  }
}

let facePollTimer;
async function pollFaces() {
  try {
    const response = await fetch("/api/faces", { cache: "no-store" });
    if (response.status === 401) {
      window.location.assign("/login");
      return;
    }
    if (!response.ok) throw new Error(`face endpoint returned ${response.status}`);
    renderFaces(await response.json());
  } catch (error) {
    el.faceStatus.textContent = "unavailable";
    console.error("Could not update enhanced faces", error);
  } finally {
    facePollTimer = window.setTimeout(pollFaces, 1500);
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

window.addEventListener("beforeunload", () => {
  window.clearTimeout(reconnectTimer);
  window.clearTimeout(facePollTimer);
});
el.video.addEventListener("error", () => {
  el.connDot.dataset.state = "down";
  el.connLabel.textContent = "feed unavailable";
  el.video.parentElement.dataset.state = "down";
});
connect();
pollFaces();
