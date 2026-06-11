"""
Person Floor Tracker  ── CAMEX v7
Usage: track_persons(video_path, floor_polygon, room_size_ft, camera_id)

Improvements over v1:
  • RTSP reconnect loop (up to MAX_RECONNECT attempts)
  • Consecutive-failure guard before declaring stream dead
  • Lower CONF_THRESH (0.20) for better detection of distant/partial persons
  • Polygon coverage diagnostic on startup
  • All print() calls use flush=True so Node.js reads them immediately
  • Graceful exit codes: 0 = clean end, 1 = hard error
"""

import cv2
import numpy as np
import json
import time
import sys
from ultralytics import YOLO


# ── Constants ─────────────────────────────────────────────────────────────────
YOLO_MODEL          = "yolo26n.pt"
FLOOR_CANVAS        = 600
CONF_THRESH         = 0.20    # lowered from 0.35 — catches distant/partial persons
MAX_RECONNECT       = 10      # max RTSP reconnect attempts after stream drop
RECONNECT_DELAY_SEC = 3       # seconds to wait between reconnect attempts
MAX_CONSEC_FAIL     = 30      # consecutive failed cap.read() before reconnect


# ── Helpers ───────────────────────────────────────────────────────────────────
def _emit(obj: dict):
    """Print a JSON payload to stdout immediately (flush=True critical for Node pipe)."""
    print(json.dumps(obj), flush=True)


def _log(msg: str):
    """Print a plain tracker log line (also flushed)."""
    print(msg, flush=True)


# ── Main entry point ──────────────────────────────────────────────────────────
def track_persons(
    video_path:    str,
    floor_polygon: list[list[float]],
    room_size_ft:  tuple[float, float],
    camera_id:     str  = "CAM-001",
    save_video:    bool = False,
    output_path:   str  = "tracked_output.mp4",
    yolo_model:    str  = YOLO_MODEL,
):
    room_w_ft, room_d_ft = room_size_ft

    # ── Homography setup ──────────────────────────────────────────────────────
    polygon = np.array(floor_polygon, dtype=np.float32)

    if len(polygon) < 4:
        _emit({"type": "error", "cameraId": camera_id,
               "msg": f"floor_polygon must have ≥4 points, got {len(polygon)}"})
        sys.exit(1)

    # Use first, second, third and last points as homography corners
    # This maps the trapezoid floor region → square canvas
    src_pts = np.array([
        polygon[0],
        polygon[1],
        polygon[2],
        polygon[-1],
    ], dtype=np.float32)

    dst_pts = np.array([
        [0,            0           ],
        [FLOOR_CANVAS, 0           ],
        [FLOOR_CANVAS, FLOOR_CANVAS],
        [0,            FLOOR_CANVAS],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)

    if H is None:
        _emit({"type": "error", "cameraId": camera_id,
               "msg": "Homography computation failed — check floor_polygon points"})
        sys.exit(1)

    def to_real_coords(px: float, py: float) -> tuple[float, float]:
        pt     = np.array([[[px, py]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)[0][0]
        x_ft   = round(float(np.clip(mapped[0] / FLOOR_CANVAS, 0, 1) * room_w_ft), 2)
        z_ft   = round(float(np.clip(mapped[1] / FLOOR_CANVAS, 0, 1) * room_d_ft), 2)
        return x_ft, z_ft

    # ── Load YOLO model ───────────────────────────────────────────────────────
    try:
        model = YOLO(yolo_model)
    except Exception as e:
        _emit({"type": "error", "cameraId": camera_id,
               "msg": f"Failed to load YOLO model '{yolo_model}': {e}"})
        sys.exit(1)

    # ── Open video / RTSP stream ──────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        _emit({"type": "error", "cameraId": camera_id,
               "msg": f"Cannot open video source: {video_path}"})
        sys.exit(1)

    # ── Polygon coverage diagnostic ───────────────────────────────────────────
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    poly_area   = float(cv2.contourArea(polygon))
    frame_area  = float(frame_w * frame_h) if frame_w and frame_h else 1.0
    coverage    = round(poly_area / frame_area, 3) if frame_area > 0 else 0.0

    _emit({
        "type":            "startup_info",
        "cameraId":        camera_id,
        "frameSize":       [frame_w, frame_h],
        "polygonPts":      len(polygon),
        "polygonCoverage": coverage,
        "roomFt":          [room_w_ft, room_d_ft],
        "confThresh":      CONF_THRESH,
        "yoloModel":       yolo_model,
    })

    if coverage < 0.25:
        _emit({
            "type":     "warning",
            "cameraId": camera_id,
            "msg":      (
                f"Polygon covers only {coverage*100:.1f}% of frame. "
                "Persons near edges may be missed. "
                "Consider expanding your floor polygon in Setup."
            ),
        })

    _log(f"[tracker] Started — camera={camera_id}  "
         f"room={room_w_ft}×{room_d_ft} ft  "
         f"frame={frame_w}×{frame_h}  "
         f"coverage={coverage*100:.1f}%  "
         f"conf={CONF_THRESH}  "
         f"save_video={save_video}")

    # ── Video writer setup (optional) ─────────────────────────────────────────
    writer = None
    if save_video:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_w, frame_h),
        )

    # ── Main loop with RTSP reconnect ─────────────────────────────────────────
    frame_count        = 0
    reconnect_attempt  = 0
    consecutive_fails  = 0

    while reconnect_attempt <= MAX_RECONNECT:

        # ── Reconnect attempt (skip on first pass) ────────────────────────────
        if reconnect_attempt > 0:
            _emit({
                "type":     "reconnecting",
                "cameraId": camera_id,
                "attempt":  reconnect_attempt,
                "maxRetry": MAX_RECONNECT,
            })
            time.sleep(RECONNECT_DELAY_SEC)
            cap.release()
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                _emit({"type": "reconnect_failed", "cameraId": camera_id,
                       "attempt": reconnect_attempt})
                reconnect_attempt += 1
                continue

            consecutive_fails = 0
            _emit({"type": "reconnected", "cameraId": camera_id,
                   "attempt": reconnect_attempt})

        # ── Inner read loop ───────────────────────────────────────────────────
        stream_dead = False

        while True:
            ret, frame = cap.read()

            # ── Handle failed read ────────────────────────────────────────────
            if not ret:
                consecutive_fails += 1
                if consecutive_fails >= MAX_CONSEC_FAIL:
                    _emit({
                        "type":       "stream_lost",
                        "cameraId":   camera_id,
                        "framesRead": frame_count,
                        "msg":        f"{MAX_CONSEC_FAIL} consecutive read failures — triggering reconnect",
                    })
                    stream_dead = True
                    break
                time.sleep(0.05)
                continue

            consecutive_fails = 0
            frame_count += 1

           # ── YOLO tracking ─────────────────────────────────────────────────
            try:
                results = model.track(
                    frame,
                    classes=[0],
                    persist=True,
                    conf=CONF_THRESH,
                    verbose=False,
                )
            except Exception as e:
                _emit({"type": "tracking_error", "cameraId": camera_id, "msg": str(e)})
                # Reset tracker state and try next frame without persist
                try:
                    results = model.predict(
                        frame,
                        classes=[0],
                        conf=CONF_THRESH,
                        verbose=False,
                    )
                    _emit({"type": "tracking_reset", "cameraId": camera_id,
                           "msg": "Tracker state reset after error"})
                except Exception as e2:
                    _emit({"type": "tracking_error", "cameraId": camera_id,
                           "msg": f"Reset also failed: {e2}"})
                continue

            people = []

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    # Skip untracked detections (no ID assigned yet)
                    if box.id is None:
                        continue
                    pid = int(box.id[0])

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    foot_x = float((x1 + x2) / 2.0)
                    foot_y = float(y2)

                    # Only include persons whose foot point is inside the floor polygon
                    inside = cv2.pointPolygonTest(polygon, (foot_x, foot_y), False)
                    if inside < 0:
                        continue

                    x_ft, z_ft = to_real_coords(foot_x, foot_y)
                    conf_val   = round(float(box.conf[0].cpu().numpy()), 3)
                    people.append({"id": pid, "x": x_ft, "z": z_ft, "conf": conf_val})

                    # Draw annotations if saving video
                    if save_video:
                        cv2.rectangle(
                            frame,
                            (int(x1), int(y1)), (int(x2), int(y2)),
                            (0, 200, 255), 2,
                        )
                        cv2.putText(
                            frame,
                            f"P{pid} ({x_ft},{z_ft}) {conf_val:.2f}",
                            (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 2,
                        )
                        cv2.circle(frame, (int(foot_x), int(foot_y)), 5, (0, 255, 0), -1)

            # ── Emit tracking payload ─────────────────────────────────────────
            _emit({
                "type":     "tracking",
                "cameraId": camera_id,
                "ts":       int(time.time() * 1000),
                "count":    len(people),
                "people":   sorted(people, key=lambda p: p["id"]),
            })

            # ── Log every 150 frames so Node knows data is flowing ────────────
            if frame_count % 150 == 0:
                _log(f"[tracker] ✓ frame={frame_count}  "
                     f"people={len(people)}  "
                     f"cam={camera_id}")

            # ── Write annotated frame ─────────────────────────────────────────
            if save_video and writer:
                cv2.polylines(
                    frame,
                    [polygon.astype(np.int32)],
                    isClosed=True,
                    color=(0, 255, 100),
                    thickness=2,
                )
                writer.write(frame)

        # ── After inner loop: decide whether to reconnect or stop cleanly ─────
        # ── After inner loop: decide whether to reconnect or stop cleanly ─────
        if not stream_dead:
            _log("[tracker] Video source ended cleanly.")
            # For video files: loop back to start instead of exiting
            is_local_file = not (video_path.startswith("rtsp://") or
                                  video_path.startswith("http://") or
                                  video_path.startswith("https://"))
            if is_local_file:
                _log("[tracker] Looping video file back to start...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                consecutive_fails = 0
                continue  # restart inner loop from frame 0
            else:
                break

        reconnect_attempt += 1

    # ── Final exceeded reconnect limit message ────────────────────────────────
    if reconnect_attempt > MAX_RECONNECT:
        _emit({
            "type":     "error",
            "cameraId": camera_id,
            "msg":      f"Stream could not be recovered after {MAX_RECONNECT} reconnect attempts",
        })

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        _log(f"[tracker] Video saved to: {output_path}")

    _log(f"[tracker] Done — total frames processed: {frame_count}")


# ── Example / standalone usage ────────────────────────────────────────────────
if __name__ == "__main__":
    FLOOR_POLYGON = [
        [340.33,  387.54],
        [1138.69, 231.80],
        [1699.34, 710.49],
        [1243.10, 1079.0],
        [394.43,  1079.0],
    ]

    track_persons(
        video_path    = "depth_person_tracking.mp4",
        floor_polygon = FLOOR_POLYGON,
        room_size_ft  = (12, 12),
        camera_id     = "CAM-001",
        save_video    = False,
        output_path   = "tracked_output.mp4",
    )