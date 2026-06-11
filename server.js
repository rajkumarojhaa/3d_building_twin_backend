"use strict";
// server.js — CAMEX v8: Single Source of Truth
// All room dims in METRES throughout. Python receives FEET (converted once at spawn).

const express = require("express");
const http = require("http");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const { WebSocketServer } = require("ws");
const { v4: uuidv4 } = require("uuid");

const multer = require("multer");
const UPLOADS_DIR = path.join(__dirname, "uploads");
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => {
    cb(
      null,
      `${Date.now()}_${file.originalname.replace(/[^a-zA-Z0-9._-]/g, "_")}`,
    );
  },
});
const upload = multer({
  storage,
  limits: { fileSize: 4 * 1024 * 1024 * 1024 },
});

const PORT = process.env.PORT || 3000;
const HLS_ROOT = path.join(__dirname, "hls");
const TRACKER_SCRIPT =
  process.env.TRACKER_SCRIPT || path.join(__dirname, "person_tracker_bbox2.py");
const PYTHON_EXE =
  process.env.PYTHON_EXE ||
  (process.platform === "win32" ? "python" : "python3");

if (!fs.existsSync(HLS_ROOT)) fs.mkdirSync(HLS_ROOT, { recursive: true });

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

app.use(cors({ origin: "*" }));
app.use(express.json({ limit: "10mb" }));

// ─── CONVERSION CONSTANTS ─────────────────────────────────────────────────────
const FT_TO_M = 0.3048;
const M_TO_FT = 3.28084;

// ─── DEFAULT ROOM (metres) — matches ROOM constant in App.jsx ─────────────────
const DEFAULT_ROOM = { W: 12.0, D: 12.0, H: 12.0 };

let roomConfig = {
  roomW: DEFAULT_ROOM.W,
  roomD: DEFAULT_ROOM.D,
  roomH: DEFAULT_ROOM.H,
  camMountHeight: 2.4,
  camTiltDeg: 30,
  camFovH: 90,
  camFovV: 60,
  camOffsetX: 0.0,
};

// pendingTrackerConfig.roomSizeM = [widthMetres, depthMetres]
let pendingTrackerConfig = null;

const sessions = new Map();
const trackers = new Map();
let pythonProc = null;
let pythonCamId = null;
let pythonStatus = "stopped";

// ─── BROADCAST ────────────────────────────────────────────────────────────────
function broadcast(data) {
  const payload = JSON.stringify(data);
  for (const c of wss.clients) if (c.readyState === 1) c.send(payload);
}

// ─── ft → world metres (centred at room centre) ───────────────────────────────
// Python outputs x_ft, z_ft with origin at room corner (0,0).
// Three.js room is centred at (0,0). So shift by half dimensions.
function ftToWorld(x_ft, z_ft, roomW_m, roomD_m) {
  const x_m = x_ft * FT_TO_M;
  const z_m = z_ft * FT_TO_M;
  return {
    x: +(x_m - roomW_m / 2).toFixed(3),
    z: +(z_m - roomD_m / 2).toFixed(3),
  };
}

// ─── STABLE TRACKING (threshold in METRES) ────────────────────────────────────
const MATCH_RADIUS_M = 0.6; // 60 cm

function updateTracks(cameraId, detections) {
  if (!trackers.has(cameraId)) trackers.set(cameraId, new Map());
  const tracks = trackers.get(cameraId);
  const now = Date.now();
  tracks.forEach((t) => {
    t.updated = false;
  });

  for (const det of detections) {
    let best = null,
      bestDist = Infinity;
    tracks.forEach((t) => {
      const d = Math.hypot(t.x - det.x, t.z - det.z);
      if (d < MATCH_RADIUS_M && d < bestDist) {
        best = t;
        bestDist = d;
      }
    });
    if (best) {
      best.x = best.x * 0.6 + det.x * 0.4;
      best.z = best.z * 0.6 + det.z * 0.4;
      best.pyId = det.pyId;
      best.updated = true;
      best.lastSeen = now;
      best.hits++;
      best.missed = 0;
    } else {
      const id = uuidv4();
      tracks.set(id, {
        id,
        pyId: det.pyId,
        x: det.x,
        z: det.z,
        updated: true,
        hits: 1,
        missed: 0,
        lastSeen: now,
        confidence: 0.9,
      });
    }
  }
  tracks.forEach((t, id) => {
    if (!t.updated) t.missed++;
    if (t.missed > 8 || now - t.lastSeen > 3000) tracks.delete(id);
  });
  const stable = [];
  tracks.forEach((t) => {
    if (t.hits >= 2)
      stable.push({
        id: t.id,
        pyId: t.pyId,
        x: +t.x.toFixed(3),
        z: +t.z.toFixed(3),
        confidence: t.confidence,
      });
  });
  return stable;
}

// ─── START PYTHON TRACKER ─────────────────────────────────────────────────────
function startPythonTracker(opts) {
  stopPythonTracker();
  const {
    videoPath,
    floorPolygon,
    roomSizeM,
    cameraId = "CAM-001",
    yoloModel,
  } = opts;

  // Convert metres → feet for Python (single conversion point)
  const roomSizeFt = [
    +(roomSizeM[0] * M_TO_FT).toFixed(4),
    +(roomSizeM[1] * M_TO_FT).toFixed(4),
  ];

  console.log("\n[Tracker] ══════════════════════════════════");
  console.log(`[Tracker] cam=${cameraId}  video=${videoPath}`);
  console.log(
    `[Tracker] roomSizeM=${JSON.stringify(roomSizeM)}  →  roomSizeFt=${JSON.stringify(roomSizeFt)}`,
  );
  console.log(`[Tracker] polygon pts=${floorPolygon?.length ?? 0}`);

  if (!fs.existsSync(TRACKER_SCRIPT)) {
    console.error(`[Tracker] ✗ Script not found: ${TRACKER_SCRIPT}`);
    broadcast({
      type: "tracker_error",
      cameraId,
      error: "tracker script not found",
    });
    pythonStatus = "error";
    return null;
  }

  const scriptDir = path.dirname(TRACKER_SCRIPT);
  const yoloArg = yoloModel || path.join(scriptDir, "yolo26n.pt");
  const wrapperPath = path.join(scriptDir, "_tracker_run.py");

  const wrapperCode = `import sys, os
sys.path.insert(0, ${JSON.stringify(scriptDir)})
from person_tracker_bbox2 import track_persons
track_persons(
    video_path    = ${JSON.stringify(videoPath)},
    floor_polygon = ${JSON.stringify(floorPolygon)},
    room_size_ft  = ${JSON.stringify(roomSizeFt)},
    camera_id     = ${JSON.stringify(cameraId)},
    save_video    = False,
    yolo_model    = ${JSON.stringify(yoloArg)},
)
`;

  try {
    fs.writeFileSync(wrapperPath, wrapperCode, "utf8");
  } catch (err) {
    console.error(`[Tracker] ✗ Write wrapper failed: ${err.message}`);
    pythonStatus = "error";
    return null;
  }

  const proc = spawn(PYTHON_EXE, [wrapperPath], {
    stdio: ["ignore", "pipe", "pipe"],
    cwd: scriptDir,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  pythonProc = proc;
  pythonCamId = cameraId;
  pythonStatus = "running";
  broadcast({ type: "tracker_started", cameraId });

  let lineBuffer = "",
    frameCount = 0,
    personsSeen = 0;

  proc.stdout.on("data", (chunk) => {
    lineBuffer += chunk.toString();
    const lines = lineBuffer.split("\n");
    lineBuffer = lines.pop();

    for (const line of lines) {
      const trimmed = line.trim();
      if (frameCount < 3) console.log(`[Tracker raw] ${trimmed}`);
      if (!trimmed.startsWith("{")) continue;

      let msg;
      try {
        msg = JSON.parse(trimmed);
      } catch {
        console.warn(`[Tracker] JSON parse error: ${trimmed.slice(0, 60)}`);
        continue;
      }

      if (msg.type !== "tracking") {
        broadcast(msg);
        continue;
      }

      frameCount++;
      if (frameCount % 100 === 0)
        console.log(`[Tracker] frame=${frameCount}  people=${msg.count}`);

      // Convert feet → world metres using the ACTUAL room size in metres
      const detections = (msg.people || []).map((p) => {
        const world = ftToWorld(p.x, p.z, roomSizeM[0], roomSizeM[1]);
        return { ...world, pyId: p.id };
      });

      const stable = updateTracks(cameraId, detections);
      if (stable.length !== personsSeen) {
        personsSeen = stable.length;
        console.log(`[Tracker] 👤 count → ${stable.length}`);
      }

      broadcast({
        type: "persons",
        cameraId,
        persons: stable,
        count: stable.length,
        ts: Date.now(),
      });
    }
  });

  proc.stderr.on("data", (d) => {
    const t = d.toString().trim();
    if (t) console.log(`[Tracker stderr] ${t}`);
  });

  proc.on("error", (err) => {
    console.error(`[Tracker] ✗ Spawn failed: ${err.message}`);
    pythonStatus = "error";
    pythonProc = null;
    broadcast({ type: "tracker_error", cameraId, error: err.message });
  });

  proc.on("exit", (code, signal) => {
    console.log(
      `[Tracker] Exit code=${code} signal=${signal}  frames=${frameCount}`,
    );
    pythonStatus = code === 0 ? "stopped" : "error";
    broadcast({ type: "tracker_stopped", cameraId, code });

    // Only auto-restart for RTSP streams, NOT local video files
    const isLocalFile = pendingTrackerConfig?.videoPath &&
      /\.(mp4|avi|mkv|mov|webm)$/i.test(pendingTrackerConfig.videoPath);

    if (code !== 0 && pendingTrackerConfig && !isLocalFile) {
      console.log("[Tracker] Auto-restarting RTSP stream in 4s...");
      setTimeout(() => {
        if (pendingTrackerConfig)
          startPythonTracker({ ...pendingTrackerConfig });
      }, 4000);
    } else if (code !== 0 && isLocalFile) {
      console.log("[Tracker] Local video file tracker exited with error — not auto-restarting.");
      broadcast({ type: "tracker_error", cameraId, error: "Tracker crashed on video file — check stderr above" });
    }
  });

  return proc;
}

// ─── STOP PYTHON TRACKER ──────────────────────────────────────────────────────
function stopPythonTracker() {
  if (!pythonProc) return;
  console.log(`[Tracker] Stopping pid=${pythonProc.pid}`);
  try {
    process.platform === "win32"
      ? spawn("taskkill", ["/pid", String(pythonProc.pid), "/f", "/t"])
      : pythonProc.kill("SIGTERM");
  } catch (_) {}
  pythonProc = null;
  pythonStatus = "stopped";
  if (pythonCamId) {
    trackers.delete(pythonCamId);
    broadcast({
      type: "persons",
      cameraId: pythonCamId,
      persons: [],
      count: 0,
      ts: Date.now(),
    });
  }
  pythonCamId = null;
}

// ─── API: /api/room ───────────────────────────────────────────────────────────
app.post("/api/room", (req, res) => {
  const {
    roomW,
    roomD,
    roomH,
    camMountHeight,
    camTiltDeg,
    camFovH,
    camFovV,
    camOffsetX,
  } = req.body || {};
  if (typeof roomW === "number" && roomW > 0) roomConfig.roomW = roomW;
  if (typeof roomD === "number" && roomD > 0) roomConfig.roomD = roomD;
  if (typeof roomH === "number" && roomH > 0) roomConfig.roomH = roomH;
  if (typeof camMountHeight === "number")
    roomConfig.camMountHeight = camMountHeight;
  if (typeof camTiltDeg === "number") roomConfig.camTiltDeg = camTiltDeg;
  if (typeof camFovH === "number" && camFovH > 0) roomConfig.camFovH = camFovH;
  if (typeof camFovV === "number" && camFovV > 0) roomConfig.camFovV = camFovV;
  if (typeof camOffsetX === "number") roomConfig.camOffsetX = camOffsetX;

  console.log(
    `[Room] Updated: ${roomConfig.roomW}×${roomConfig.roomD}×${roomConfig.roomH} m`,
  );
  broadcast({ type: "room_config", roomConfig });

  // Keep tracker config room size in sync
  // Keep tracker config room size in sync + restart if running
  if (pendingTrackerConfig) {
    const roomChanged =
      pendingTrackerConfig.roomSizeM[0] !== roomConfig.roomW ||
      pendingTrackerConfig.roomSizeM[1] !== roomConfig.roomD;

    pendingTrackerConfig.roomSizeM = [roomConfig.roomW, roomConfig.roomD];
    console.log(
      `[Room] pendingTrackerConfig.roomSizeM → ${JSON.stringify(pendingTrackerConfig.roomSizeM)}`,
    );

    if (roomChanged && pythonStatus === "running") {
      console.log("[Room] Room dims changed while tracker running — restarting tracker with new dims...");
      // Stop first, wait for OS to release file handle, then restart
      stopPythonTracker();
      setTimeout(() => {
        if (pendingTrackerConfig) {
          console.log("[Room] Restarting tracker after room dim change...");
          startPythonTracker({ ...pendingTrackerConfig });
        }
      }, 2500);
    }
  }
  res.json({ success: true, roomConfig });
});

app.get("/api/room", (req, res) => res.json(roomConfig));

app.post("/api/upload-video", upload.single("video"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file received" });
  const serverPath = req.file.path;
  console.log(`[Upload] Saved video: ${serverPath}`);
  res.json({ success: true, serverPath, originalName: req.file.originalname });
});

// ─── API: /api/start-tracker ──────────────────────────────────────────────────
// Frontend ALWAYS sends roomSizeM (metres). Legacy roomSizeFt still accepted.
app.post("/api/start-tracker", (req, res) => {
  const {
    videoPath,
    floorPolygon,
    roomSizeM,
    roomSizeFt,
    cameraId,
    yoloModel,
  } = req.body || {};

  console.log(
    `[API] start-tracker  videoPath=${videoPath}  polygon=${floorPolygon?.length ?? "?"}`,
  );

  if (!videoPath) return res.status(400).json({ error: "videoPath required" });
  if (!floorPolygon || !Array.isArray(floorPolygon) || floorPolygon.length < 4)
    return res
      .status(400)
      .json({ error: "floorPolygon must be ≥4 [x,y] pairs" });

  // Resolve room size in METRES
  let rsm;
  if (Array.isArray(roomSizeM) && roomSizeM.length >= 2) {
    rsm = [+roomSizeM[0], +roomSizeM[1]];
  } else if (Array.isArray(roomSizeFt) && roomSizeFt.length >= 2) {
    rsm = [
      +(roomSizeFt[0] * FT_TO_M).toFixed(4),
      +(roomSizeFt[1] * FT_TO_M).toFixed(4),
    ];
    console.log(`[API] legacy roomSizeFt → metres: ${JSON.stringify(rsm)}`);
  } else {
    rsm = [roomConfig.roomW, roomConfig.roomD];
    console.log(
      `[API] roomSize fallback from roomConfig: ${JSON.stringify(rsm)}`,
    );
  }

  pendingTrackerConfig = {
    videoPath,
    floorPolygon,
    roomSizeM: rsm,
    cameraId: cameraId || "CAM-001",
    yoloModel,
  };
  console.log(
    `[API] pendingTrackerConfig.roomSizeM=${JSON.stringify(rsm)} m  (→ ${rsm.map((v) => +(v * M_TO_FT).toFixed(2))} ft for Python)`,
  );

  try {
    startPythonTracker({
      videoPath,
      floorPolygon,
      roomSizeM: rsm,
      cameraId: cameraId || "CAM-001",
      yoloModel,
    });
    res.json({
      success: true,
      status: "starting",
      cameraId: cameraId || "CAM-001",
      roomSizeM: rsm,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── API: /api/stop-tracker ───────────────────────────────────────────────────
app.post("/api/stop-tracker", (req, res) => {
  pendingTrackerConfig = null;
  stopPythonTracker();
  res.json({ success: true });
});

app.get("/api/tracker-status", (req, res) => {
  res.json({
    status: pythonStatus,
    cameraId: pythonCamId,
    pid: pythonProc?.pid || null,
    hasPendingConfig: !!pendingTrackerConfig,
  });
});

// ─── API: /api/connect ────────────────────────────────────────────────────────
app.post("/api/connect", (req, res) => {
  const {
    rtspUrl,
    cameraId = "CAM-001",
    floorPolygon,
    roomSizeM,
    roomSizeFt,
    yoloModel,
  } = req.body || {};

  console.log(`\n[API] connect  cam=${cameraId}  url=${rtspUrl}`);
  if (!rtspUrl) return res.status(400).json({ error: "rtspUrl required" });

  stopHlsSession(cameraId);

  if (floorPolygon) {
    let rsm;
    if (Array.isArray(roomSizeM) && roomSizeM.length >= 2) {
      rsm = [+roomSizeM[0], +roomSizeM[1]];
    } else if (Array.isArray(roomSizeFt) && roomSizeFt.length >= 2) {
      rsm = [
        +(roomSizeFt[0] * FT_TO_M).toFixed(4),
        +(roomSizeFt[1] * FT_TO_M).toFixed(4),
      ];
    } else {
      rsm = [roomConfig.roomW, roomConfig.roomD];
    }
    pendingTrackerConfig = {
      videoPath: rtspUrl,
      floorPolygon,
      roomSizeM: rsm,
      cameraId,
      yoloModel,
    };
    console.log(
      `[API] connect: set pendingTrackerConfig roomSizeM=${JSON.stringify(rsm)}`,
    );
  }

  const hlsDir = path.join(HLS_ROOT, cameraId);
  const playlist = path.join(hlsDir, "stream.m3u8");
  const segPat = path.join(hlsDir, "seg_%05d.ts");
  ensureCleanDir(hlsDir);

  const ffArgs = ["-hide_banner", "-loglevel", "warning"];
  if (/\.(mp4|webm|mkv|avi|mov)$/i.test(rtspUrl))
    ffArgs.push("-stream_loop", "-1");
  if (rtspUrl.startsWith("rtsp://"))
    ffArgs.push(
      "-rtsp_transport",
      "tcp",
      "-fflags",
      "+genpts+discardcorrupt",
      "-flags",
      "low_delay",
    );
  ffArgs.push(
    "-i",
    rtspUrl,
    "-an",
    "-c:v",
    "libx264",
    "-preset",
    "ultrafast",
    "-tune",
    "zerolatency",
    "-f",
    "hls",
    "-hls_time",
    "1",
    "-hls_list_size",
    "6",
    "-hls_flags",
    "delete_segments+append_list+omit_endlist",
    "-hls_segment_filename",
    segPat,
    playlist,
  );

  const ff = spawn(
    process.platform === "win32" ? "ffmpeg.exe" : "ffmpeg",
    ffArgs,
  );
  ff.stderr.on("data", (d) => console.log("[FFmpeg]", d.toString().trim()));
  sessions.set(cameraId, { ffmpegProc: ff, rtspUrl });

  const start = Date.now();
  const poll = setInterval(() => {
    if (fs.existsSync(playlist)) {
      clearInterval(poll);
      broadcast({
        type: "camera_connected",
        cameraId,
        hlsUrl: `/hls/${cameraId}/stream.m3u8`,
      });

      if (pendingTrackerConfig) {
        console.log("[API] Auto-starting tracker...");
        startPythonTracker({ videoPath: rtspUrl, ...pendingTrackerConfig });
        pendingTrackerConfig.videoPath = rtspUrl;
      } else {
        console.warn("[API] ⚠ No pendingTrackerConfig — tracker not started");
      }

      return res.json({
        success: true,
        hlsUrl: `/hls/${cameraId}/stream.m3u8`,
        trackerAutoStarted: pythonStatus === "running",
      });
    }
    if (Date.now() - start > 15000) {
      clearInterval(poll);
      stopHlsSession(cameraId);
      return res.status(500).json({ error: "FFmpeg timeout" });
    }
  }, 300);
});

app.delete("/api/disconnect", (req, res) => {
  const { cameraId } = req.body || {};
  pendingTrackerConfig = null;
  if (cameraId) stopHlsSession(cameraId);
  stopPythonTracker();
  res.json({ success: true });
});

app.get("/api/sessions", (req, res) => {
  const list = [];
  sessions.forEach((s, id) => list.push({ id, rtspUrl: s.rtspUrl }));
  res.json(list);
});

// ─── API: /api/debug ──────────────────────────────────────────────────────────
app.get("/api/debug", (req, res) => {
  res.json({
    pythonStatus,
    pythonCamId,
    pythonPid: pythonProc?.pid || null,
    pendingTrackerConfig: pendingTrackerConfig
      ? {
          videoPath: pendingTrackerConfig.videoPath,
          polygonPts: pendingTrackerConfig.floorPolygon?.length,
          roomSizeM: pendingTrackerConfig.roomSizeM,
          roomSizeFtComputed: pendingTrackerConfig.roomSizeM?.map(
            (v) => +(v * M_TO_FT).toFixed(3),
          ),
          cameraId: pendingTrackerConfig.cameraId,
          yoloModel: pendingTrackerConfig.yoloModel,
        }
      : null,
    activeSessions: [...sessions.keys()],
    trackerScript: TRACKER_SCRIPT,
    trackerScriptExists: fs.existsSync(TRACKER_SCRIPT),
    pythonExe: PYTHON_EXE,
    roomConfig,
    defaults: DEFAULT_ROOM,
    note: "All room dims in METRES. Python receives FEET (converted at spawn). Three.js centred at 0,0.",
  });
});

// HLS static
app.use(
  "/hls",
  (req, res, next) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
    if (req.path.endsWith(".m3u8"))
      res.setHeader("Content-Type", "application/vnd.apple.mpegurl");
    if (req.path.endsWith(".ts")) res.setHeader("Content-Type", "video/mp2t");
    next();
  },
  express.static(HLS_ROOT),
);

const distDir = path.join(__dirname, "dist");
if (fs.existsSync(distDir)) {
  app.use(express.static(distDir));
  app.get("*", (req, res) => res.sendFile(path.join(distDir, "index.html")));
}

// ─── WebSocket ────────────────────────────────────────────────────────────────
wss.on("connection", (ws, req) => {
  ws.isAlive = true;
  console.log(`[WS] Client connected ${req.socket.remoteAddress}`);
  ws.on("pong", () => {
    ws.isAlive = true;
  });
  ws.send(
    JSON.stringify({
      type: "server_ready",
      roomConfig,
      trackerStatus: pythonStatus,
      hasPendingConfig: !!pendingTrackerConfig,
    }),
  );
});
setInterval(() => {
  for (const ws of wss.clients) {
    if (!ws.isAlive) {
      ws.terminate();
      continue;
    }
    ws.isAlive = false;
    try {
      ws.ping();
    } catch (_) {}
  }
}, 20000);

// ─── Utils ────────────────────────────────────────────────────────────────────
function stopHlsSession(id) {
  const s = sessions.get(id);
  if (!s) return;
  safeKill(s.ffmpegProc);
  sessions.delete(id);
  broadcast({ type: "camera_disconnected", cameraId: id, ts: Date.now() });
}
function safeKill(proc) {
  if (!proc) return;
  try {
    process.platform === "win32"
      ? spawn("taskkill", ["/pid", String(proc.pid), "/f", "/t"])
      : proc.kill("SIGKILL");
  } catch (_) {}
}
function ensureCleanDir(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
    return;
  }
  for (const f of fs.readdirSync(dir)) {
    try {
      fs.rmSync(path.join(dir, f), { recursive: true, force: true });
    } catch (_) {}
  }
}

// ─── Start ────────────────────────────────────────────────────────────────────
server.listen(PORT, () => {
  console.log("\n╔══════════════════════════════════════╗");
  console.log("║   CAMEX v8 — Single Source of Truth  ║");
  console.log("╚══════════════════════════════════════╝");
  console.log(`HTTP  http://localhost:${PORT}`);
  console.log(`WS    ws://localhost:${PORT}`);
  console.log(`Debug http://localhost:${PORT}/api/debug\n`);
  console.log(
    `Default room: ${DEFAULT_ROOM.W}×${DEFAULT_ROOM.D}×${DEFAULT_ROOM.H} m`,
  );
  console.log(`Tracker: ${TRACKER_SCRIPT}`);
  console.log(
    `Script exists: ${fs.existsSync(TRACKER_SCRIPT) ? "✓" : "✗ MISSING"}`,
  );
  console.log(`Python: ${PYTHON_EXE}\n`);
  console.log("Unit contract:");
  console.log("  Frontend → server:  METRES (roomSizeM)");
  console.log("  Server → Python:    FEET   (converted at spawn)");
  console.log("  Python → server:    FEET   (in tracking JSON)");
  console.log("  Server → Three.js:  METRES (ftToWorld, centred at 0,0)\n");
});
