/**
 * EmotiSense AI – script.js
 * Full frontend logic: upload, webcam, history, dashboard, theme, toasts
 */

"use strict";

// ── Constants ────────────────────────────────────────────────────────────────

const EMOTION_EMOJIS = {
  Angry:    "😠",
  Disgust:  "🤢",
  Fear:     "😨",
  Happy:    "😊",
  Neutral:  "😐",
  Sad:      "😢",
  Surprise: "😲",
};

const EMOTION_COLORS = {
  Angry:    "#ef4444",
  Disgust:  "#84cc16",
  Fear:     "#f59e0b",
  Happy:    "#10b981",
  Neutral:  "#94a3b8",
  Sad:      "#3b82f6",
  Surprise: "#a855f7",
};

const HISTORY_KEY = "emotisense_history";
const MAX_HISTORY = 50;

// ── State ────────────────────────────────────────────────────────────────────

let selectedFile = null;
let webcamStream = null;
let autoPredictTimer = null;
let sessionCounts = {};

// ── DOM helpers ──────────────────────────────────────────────────────────────

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function show(el)  { el && (el.style.display = ""); }
function hide(el)  { el && (el.style.display = "none"); }
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls)  e.className = cls;
  if (html) e.innerHTML = html;
  return e;
}

// ── Toast system ─────────────────────────────────────────────────────────────

function toast(message, type = "info", duration = 3500) {
  const icons = { success: "✅", error: "❌", info: "💡" };
  const t = el("div", `toast toast-${type}`, `<span>${icons[type]}</span><span>${message}</span>`);
  const container = $("#toastContainer");
  container.append(t);

  setTimeout(() => {
    t.classList.add("fade-out");
    t.addEventListener("animationend", () => t.remove());
  }, duration);
}

// ── Smooth scroll helper ──────────────────────────────────────────────────────

function scrollToSection(selector) {
  const target = $(selector);
  if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
}
window.scrollToSection = scrollToSection; // expose for inline onclick

// ── Navbar scroll effect ─────────────────────────────────────────────────────

window.addEventListener("scroll", () => {
  const nav = $("#navbar");
  nav.classList.toggle("scrolled", window.scrollY > 40);
}, { passive: true });

// ── Theme toggle ─────────────────────────────────────────────────────────────

const themeToggle = $("#themeToggle");
const htmlEl      = document.documentElement;

function applyTheme(theme) {
  htmlEl.dataset.theme = theme;
  localStorage.setItem("emotisense_theme", theme);
  themeToggle.querySelector(".theme-icon").textContent = theme === "dark" ? "🌙" : "☀️";
}

themeToggle.addEventListener("click", () => {
  applyTheme(htmlEl.dataset.theme === "dark" ? "light" : "dark");
});

// init
applyTheme(localStorage.getItem("emotisense_theme") || "dark");

// ── Reveal on scroll ──────────────────────────────────────────────────────────

const revealIO = new IntersectionObserver((entries) => {
  entries.forEach((e, i) => {
    if (e.isIntersecting) {
      setTimeout(() => e.target.classList.add("visible"), i * 80);
      revealIO.unobserve(e.target);
    }
  });
}, { threshold: 0.12 });

$$(".reveal").forEach(el => revealIO.observe(el));

// ── Score chart builder ───────────────────────────────────────────────────────

function buildScoreChart(scores, containerId, highlight) {
  const container = $("#" + containerId);
  if (!container) return;
  container.innerHTML = "";

  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  sorted.forEach(([emotion, pct]) => {
    const isTop = emotion === highlight;
    const row = el("div", "score-row");
    row.innerHTML = `
      <span class="score-label">${EMOTION_EMOJIS[emotion]} ${emotion}</span>
      <div class="score-bar-wrap">
        <div class="score-bar${isTop ? " top" : ""}" style="width:0%; --color:${EMOTION_COLORS[emotion]}"></div>
      </div>
      <span class="score-pct">${pct.toFixed(1)}%</span>
    `;
    container.append(row);
    // Animate bar
    requestAnimationFrame(() => {
      const bar = row.querySelector(".score-bar");
      setTimeout(() => bar.style.width = pct + "%", 50);
    });
  });
}

// ── Upload ────────────────────────────────────────────────────────────────────

const dropZone    = $("#dropZone");
const fileInput   = $("#fileInput");
const dropContent = $("#dropContent");
const dropPreview = $("#dropPreview");
const previewImg  = $("#previewImg");
const removeImg   = $("#removeImg");
const analyzeBtn  = $("#analyzeBtn");
const resultIdle    = $("#resultIdle");
const resultLoading = $("#resultLoading");
const resultContent = $("#resultContent");

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    toast("Please select a valid image file.", "error");
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    toast("File exceeds 16 MB limit.", "error");
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  hide(dropContent);
  show(dropPreview);
  analyzeBtn.disabled = false;
  // Reset result
  hide(resultLoading);
  hide(resultContent);
  show(resultIdle);
}

function clearFile() {
  selectedFile = null;
  previewImg.src = "";
  hide(dropPreview);
  show(dropContent);
  analyzeBtn.disabled = true;
  hide(resultContent);
  show(resultIdle);
}

dropZone.addEventListener("click", (e) => {
  if (!e.target.closest(".remove-img") && !e.target.closest("button[onclick]")) {
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
  fileInput.value = "";
});
removeImg.addEventListener("click", (e) => { e.stopPropagation(); clearFile(); });

// Drag & drop
["dragenter", "dragover"].forEach(ev => {
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
});
["dragleave", "drop"].forEach(ev => {
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove("drag-over"); });
});
dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

// Analyze
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  hide(resultIdle);
  hide(resultContent);
  show(resultLoading);
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const res  = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.error || "Prediction failed");
    showUploadResult(data);
    addToHistory({ ...data, source: "upload" });
    updateDashboard(data.emotion);
    toast(`Detected: ${data.emotion} (${data.confidence.toFixed(1)}%)`, "success");
  } catch (err) {
    toast(err.message, "error");
    show(resultIdle);
    hide(resultLoading);
  } finally {
    analyzeBtn.disabled = false;
  }
});

function showUploadResult(data) {
  hide(resultLoading);
  hide(resultIdle);
  show(resultContent);

  $("#uploadEmoji").textContent   = EMOTION_EMOJIS[data.emotion] || "🎭";
  $("#uploadEmotion").textContent = data.emotion;
  $("#uploadConf").textContent    = `Confidence: ${data.confidence.toFixed(2)}%`;

  const bar = $("#uploadBar");
  bar.style.removeProperty("--pct");
  bar.classList.remove("animated");
  requestAnimationFrame(() => {
    bar.style.setProperty("--pct", data.confidence.toFixed(1) + "%");
    setTimeout(() => bar.classList.add("animated"), 30);
  });

  buildScoreChart(data.all_scores, "uploadChart", data.emotion);
}

// Download result
$("#downloadBtn").addEventListener("click", () => {
  const emotion  = $("#uploadEmotion").textContent;
  const conf     = $("#uploadConf").textContent;
  const timestamp = new Date().toLocaleString();

  const canvas = document.createElement("canvas");
  canvas.width  = 400;
  canvas.height = 220;
  const ctx = canvas.getContext("2d");

  // Background
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, 400, 220);

  // Border
  ctx.strokeStyle = "#6366f1";
  ctx.lineWidth = 2;
  ctx.roundRect(8, 8, 384, 204, 16);
  ctx.stroke();

  // Emoji
  ctx.font = "56px serif";
  ctx.textAlign = "center";
  ctx.fillText(EMOTION_EMOJIS[emotion] || "🎭", 80, 110);

  // Text
  ctx.fillStyle = "#f1f5f9";
  ctx.font = "bold 28px Syne, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(emotion, 130, 90);

  ctx.fillStyle = "#94a3b8";
  ctx.font = "16px DM Sans, sans-serif";
  ctx.fillText(conf, 130, 118);
  ctx.fillText("EmotiSense AI", 130, 148);
  ctx.fillStyle = "#64748b";
  ctx.font = "12px DM Sans, sans-serif";
  ctx.fillText(timestamp, 130, 170);

  const a = document.createElement("a");
  a.href = canvas.toDataURL("image/png");
  a.download = `emotion_${emotion}_${Date.now()}.png`;
  a.click();
  toast("Result downloaded!", "success");
});

// Try Another
$("#tryAnotherBtn").addEventListener("click", () => { clearFile(); scrollToSection("#upload-section"); });

// ── Webcam ────────────────────────────────────────────────────────────────────

const startCamBtn     = $("#startCam");
const stopCamBtn      = $("#stopCam");
const captureFrameBtn = $("#captureFrame");
const webcamVideo     = $("#webcamVideo");
const webcamCanvas    = $("#webcamCanvas");
const webcamIdle      = $("#webcamIdle");
const webcamOverlay   = $("#webcamOverlay");
const autoPredictToggle = $("#autoPredictToggle");

function startAutoPredict() {
  stopAutoPredict();
  if (autoPredictToggle.checked) {
    autoPredictTimer = setInterval(() => {
      if (webcamStream) predictFromWebcam();
    }, 2000);
  }
}
function stopAutoPredict() {
  clearInterval(autoPredictTimer);
  autoPredictTimer = null;
}

startCamBtn.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    webcamStream = stream;
    webcamVideo.srcObject = stream;
    webcamVideo.style.display = "block";
    hide(webcamIdle);
    show(webcamOverlay);
    startCamBtn.disabled   = true;
    stopCamBtn.disabled    = false;
    captureFrameBtn.disabled = false;
    startAutoPredict();
    toast("Camera started!", "success");
  } catch (err) {
    console.error("Camera error:", err);
    let msg = "Camera access denied or not available.";
    if (err.name === "NotAllowedError") {
      msg = "⚠️ Camera permission denied. Check your browser/system settings.";
    } else if (err.name === "NotFoundError") {
      msg = "❌ No camera device found.";
    } else if (err.name === "NotReadableError") {
      msg = "❌ Camera is already in use by another application.";
    }
    toast(msg, "error");
  }
});

stopCamBtn.addEventListener("click", () => {
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  stopAutoPredict();
  webcamVideo.srcObject = null;
  webcamVideo.style.display = "none";
  show(webcamIdle);
  hide(webcamOverlay);
  startCamBtn.disabled     = false;
  stopCamBtn.disabled      = true;
  captureFrameBtn.disabled = true;
  toast("Camera stopped.", "info");
});

captureFrameBtn.addEventListener("click", () => predictFromWebcam());

autoPredictToggle.addEventListener("change", () => {
  if (autoPredictToggle.checked) {
    startAutoPredict();
    toast("Auto-predict enabled.", "info");
  } else {
    stopAutoPredict();
    toast("Auto-predict disabled.", "info");
  }
});

async function predictFromWebcam() {
  if (!webcamVideo.videoWidth) return;
  webcamCanvas.width  = webcamVideo.videoWidth;
  webcamCanvas.height = webcamVideo.videoHeight;
  const ctx = webcamCanvas.getContext("2d");
  if (!ctx) {
    console.error("Failed to get canvas context");
    return;
  }
  
  try {
    ctx.drawImage(webcamVideo, 0, 0);
    const b64 = webcamCanvas.toDataURL("image/jpeg", 0.7).split(",")[1];
    
    if (!b64) {
      console.error("Failed to convert canvas to base64");
      return;
    }

    const res  = await fetch("/webcam", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: b64 }),
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      console.warn("Prediction failed:", data.error);
      return;
    }
    updateLiveResult(data);
    updateCamBadge(data);
    addToHistory({ ...data, source: "webcam" });
    updateDashboard(data.emotion);
  } catch (err) {
    console.error("Webcam prediction error:", err);
    /* silent fail during auto-predict */
  }
}

function updateCamBadge(data) {
  $("#camEmoji").textContent   = EMOTION_EMOJIS[data.emotion] || "🎭";
  $("#camEmotion").textContent = data.emotion;
  $("#camConf").textContent    = data.confidence.toFixed(1) + "%";
}

function updateLiveResult(data) {
  const emoji      = $("#liveEmoji");
  const emotionEl  = $("#liveEmotionName");
  const confEl     = $("#liveConf");
  const liveBars   = $("#liveBars");

  emoji.textContent     = EMOTION_EMOJIS[data.emotion] || "🎭";
  emotionEl.textContent = data.emotion;
  confEl.textContent    = `${data.confidence.toFixed(2)}% confidence`;

  // Rebuild live bars
  liveBars.innerHTML = "";
  Object.entries(data.all_scores)
    .sort((a, b) => b[1] - a[1])
    .forEach(([em, pct]) => {
      const row = el("div", "live-bar-row");
      row.innerHTML = `
        <span class="live-bar-label">${EMOTION_EMOJIS[em]} ${em}</span>
        <div class="live-bar-track">
          <div class="live-bar-fill" style="width:${pct}%; background:${EMOTION_COLORS[em] || "#6366f1"}"></div>
        </div>
        <span class="live-bar-pct">${pct.toFixed(0)}%</span>
      `;
      liveBars.append(row);
    });
}

// ── History ───────────────────────────────────────────────────────────────────

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch { return []; }
}
function saveHistory(hist) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(hist.slice(0, MAX_HISTORY)));
}

function addToHistory(data) {
  const hist = loadHistory();
  hist.unshift({
    emotion:    data.emotion,
    confidence: data.confidence,
    emoji:      EMOTION_EMOJIS[data.emotion] || "🎭",
    source:     data.source || "upload",
    timestamp:  new Date().toISOString(),
  });
  saveHistory(hist);
  renderHistory();
  updateTotalCount();
}

function renderHistory() {
  const hist  = loadHistory();
  const list  = $("#historyList");
  const empty = $("#historyEmpty");
  const count = $("#historyCount");

  if (!list || !count) return; // guard against missing DOM elements

  count.textContent = `${hist.length} record${hist.length !== 1 ? "s" : ""}`;

  if (!hist.length) {
    list.innerHTML = "";
    if (empty) list.append(empty);
    return;
  }
  if (empty && empty.parentNode === list) list.removeChild(empty);

  // Efficient re-render: only add new items
  list.innerHTML = "";
  hist.forEach((item, i) => {
    const timeAgo = formatTime(item.timestamp);
    const row = el("div", "history-item");
    row.style.animationDelay = `${i * 30}ms`;
    row.innerHTML = `
      <span class="hist-emoji">${item.emoji}</span>
      <div class="hist-info">
        <div class="hist-emotion">${item.emotion}</div>
        <div class="hist-conf">${item.confidence.toFixed(2)}% · ${item.source}</div>
      </div>
      <div class="hist-bar"><div class="hist-bar-fill" style="width:${item.confidence}%"></div></div>
      <span class="hist-time">${timeAgo}</span>
    `;
    list.append(row);
  });
}

function formatTime(iso) {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60)   return "just now";
  if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
  if (diff < 86400)return `${Math.floor(diff/3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}

$("#clearHistory").addEventListener("click", () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
  sessionCounts = {};
  updateDistribution();
  updateTotalCount();
  toast("History cleared.", "info");
});

// ── Dashboard ────────────────────────────────────────────────────────────────

function updateTotalCount() {
  const hist = loadHistory();
  $("#totalPreds").textContent = hist.length;
}

function updateDashboard(emotion) {
  sessionCounts[emotion] = (sessionCounts[emotion] || 0) + 1;
  updateDistribution();
  updateTotalCount();
}

function updateDistribution() {
  const chart = $("#distChart");
  const total = Object.values(sessionCounts).reduce((a, b) => a + b, 0);

  if (!total) {
    chart.innerHTML = `<p class="dist-empty">No predictions yet — upload an image or use your webcam.</p>`;
    return;
  }

  chart.innerHTML = "";
  Object.keys(EMOTION_EMOJIS).forEach(emotion => {
    const count = sessionCounts[emotion] || 0;
    const pct   = total ? (count / total) * 100 : 0;
    const col = el("div", "dist-col");
    col.innerHTML = `
      <div class="dist-count">${count}</div>
      <div class="dist-bar-wrap">
        <div class="dist-bar-fill" style="height:${pct}%; background:${EMOTION_COLORS[emotion]}"></div>
      </div>
      <div class="dist-label">${EMOTION_EMOJIS[emotion]}<br/>${emotion}</div>
    `;
    chart.append(col);
  });
}

// ── Init ──────────────────────────────────────────────────────────────────────

function init() {
  renderHistory();
  updateTotalCount();
  updateDistribution();

  // Re-observe any late-added reveal elements
  $$(".reveal").forEach(el => revealIO.observe(el));

  // Health check
  fetch("/health")
    .then(r => r.json())
    .then(d => {
      if (!d.model_ready) {
        toast("⚠️ Model running in DEMO mode (weights not found).", "info", 6000);
      }
    })
    .catch(() => {});
}

document.addEventListener("DOMContentLoaded", init);
