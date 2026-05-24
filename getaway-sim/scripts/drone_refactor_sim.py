"""
Simulation port of drone_refactor.py.

Differences vs. the on-board script:
  - MAVLink over UDP 14550 instead of /dev/ttyACM0
  - pyzed.sl replaced with zed_sim (ROS 2 → Gazebo bridge)
  - Ultralytics .pt model instead of TensorRT .engine
  - Skip params that don't apply to SITL (DISARM_DELAY, FENCE_ACTION 4 = Brake)
Everything else (control loop, BODY_OFFSET_NED commands, shutdown handler,
HTTP stream) stays identical so the same behavior gets validated.
"""
import math
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from pymavlink import mavutil
from ultralytics import YOLO

import zed_sim as sl   # drop-in replacement for pyzed.sl

# ── Config ────────────────────────────────────────────────────
MODEL_PATH    = os.environ.get("YOLO_MODEL", "/sim/models/best.pt")
MAVLINK_URL   = os.environ.get("MAVLINK_URL", "udp:127.0.0.1:14551")
DIST_OBJETIVO = 12.0  # m, target chase distance (safe trailing gap)
TAKEOFF_ALT   = 6.0   # m (higher to see far + safety margin)

# ── Velocity-based PID gains ──
# Forward: P on (depth - DIST_OBJETIVO) → m/s forward. Match ambulance speed.
KP_VX = 0.4           # m/s per m error
V_FWD_MAX = 8.0       # m/s forward cap (~29 km/h — keep up with 18 km/h amb + close gap)

# Lateral: P on (cx - CX) / CX → m/s right (BODY frame)
KP_VY = 1.5           # m/s per normalized pixel error
V_LAT_MAX = 2.0       # m/s lateral cap

# Yaw rate: P on horizontal pixel angle → rad/s
KP_YAW_RATE = 0.8     # rad/s per rad error
YAW_RATE_MAX = 0.5    # rad/s cap (~28°/s)

# Altitude hold: P on (TAKEOFF_ALT - rel_alt) → m/s down
KP_VZ = 0.4
V_VERT_MAX = 1.0

DEAD_ZONE_PX = 60     # ignore offsets within ±N px of center
MIN_ALT_M    = 3.0
CMD_RATE_HZ  = 10.0   # velocity setpoints — need higher rate

shutdown_flag  = threading.Event()
emergency_kill = threading.Event()

CLASS_NAMES = {0: 'ambulance', 1: 'fire_truck', 2: 'police_car', 3: 'other_vehicle'}
USING_PLACEHOLDER = False
COCO_VEHICLE_IDS = {2, 5, 7}  # car, bus, truck — for yolov8n fallback
ALLOWED_CLASSES = {0, 1, 2}  # ambulance, fire_truck, police_car (skip other_vehicle)

# ── Model ─────────────────────────────────────────────────────
print(f"Cargando modelo YOLO: {MODEL_PATH}")
if MODEL_PATH.endswith("yolov8n.pt") or "yolov8" in MODEL_PATH.lower():
    USING_PLACEHOLDER = True
    print("ℹ️  Usando modelo COCO placeholder — filtrando car/bus/truck")
elif not os.path.exists(MODEL_PATH):
    print(f"⚠️  No existe {MODEL_PATH} — usando yolov8n.pt como placeholder")
    MODEL_PATH = "yolov8n.pt"
    USING_PLACEHOLDER = True
model = YOLO(MODEL_PATH, task="detect")
print(f"📋 Modelo clases ({len(model.names)}): {model.names}")

# ── Cámara (simulada) ─────────────────────────────────────────
cam = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps        = 30
init_params.depth_mode        = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_units  = sl.UNIT.METER

if cam.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ Cámara simulada no entregó frames en 10s — ¿está corriendo Gazebo + ros_gz_bridge?")
    sys.exit(1)

calib = cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
FX_ZED, FY_ZED = calib.fx, calib.fy
CX_ZED, CY_ZED = calib.cx, calib.cy
print(f"✅ Intrínsecos sim: fx={FX_ZED:.1f} fy={FY_ZED:.1f} cx={CX_ZED:.1f} cy={CY_ZED:.1f}")

runtime = sl.RuntimeParameters()
image   = sl.Mat()
depth   = sl.Mat()

latest_frame = None
frame_lock   = threading.Lock()
heading_actual = 0.0
heading_lock   = threading.Lock()
modo_actual = None
modo_lock   = threading.Lock()
rel_alt_m   = 0.0
alt_lock    = threading.Lock()
roll_actual  = 0.0
pitch_actual = 0.0
attitude_lock = threading.Lock()

# ── MAVLink ───────────────────────────────────────────────────
print(f"Conectando MAVLink: {MAVLINK_URL}")
master = mavutil.mavlink_connection(MAVLINK_URL, source_system=255)
master.wait_heartbeat()
print(f"✅ HEARTBEAT recibido (sys={master.target_system}, comp={master.target_component})")

# Stream rates — replicar lo del script real
for msg_id, period_us in [
    (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 200000),
    (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 100000),
    (mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 200000),
]:
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0, msg_id, period_us, 0, 0, 0, 0, 0
    )
    time.sleep(0.2)

# Params — solo los que aplican en SITL
params = {
    b'FS_THR_ENABLE':  0,
    b'FS_GCS_ENABLE':  0,
    b'ARMING_CHECK':   0,
}
for p, v in params.items():
    master.mav.param_set_send(
        master.target_system, master.target_component,
        p, float(v), mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    time.sleep(0.2)

# GPS lock (SITL lo da rápido)
print("Esperando GPS lock (SITL)...")
for _ in range(20):
    msg = master.recv_match(type='GPS_RAW_INT', blocking=True, timeout=2)
    if msg and msg.fix_type >= 3:
        print(f"✅ GPS — fix={msg.fix_type}, sats={msg.satellites_visible}")
        break

# GUIDED
modos = master.mode_mapping()
modos_inv = {v: k for k, v in modos.items()}
master.mav.set_mode_send(
    master.target_system,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    modos['GUIDED']
)
time.sleep(1.5)
print("✅ Modo GUIDED")


def drenar_statustext(prefijo="  FC:", duracion=1.0):
    t_end = time.time() + duracion
    while time.time() < t_end:
        m = master.recv_match(type='STATUSTEXT', blocking=False)
        if m:
            print(f"{prefijo} {m.text}")
        else:
            time.sleep(0.05)


def confirmar_armado(timeout=3.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=0.5)
        if hb and (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            return True
    return False


# Espera EKF posición lista (hasta 30s)
def esperar_ekf_pos(timeout=30.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        m = master.recv_match(type='EKF_STATUS_REPORT', blocking=True, timeout=1.0)
        if m and (m.flags & 0x10) and (m.flags & 0x20):  # horiz_pos_abs + horiz_pos_rel
            return True
    return False

print("Esperando EKF position estimate...")
if esperar_ekf_pos(30.0):
    print("✅ EKF posición lista")
else:
    print("⚠ EKF no confirmó posición — intentando arm igual")

# ARM con reintentos (hasta 6 intentos × 3s)
armed = False
for intento in range(6):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
    print(f"  intento {intento+1}: ACK arm = {ack.result if ack else 'timeout'}")
    drenar_statustext(duracion=0.5)
    if confirmar_armado(2.0):
        armed = True
        break
    time.sleep(3.0)

if not armed:
    print("❌ Arm falló tras 6 intentos — abortando")
    sys.exit(1)
print("✅ Armado")

# Takeoff
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    0, 0, 0, 0, 0, 0, 0, TAKEOFF_ALT
)
ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
print(f"  ACK takeoff: {ack.result if ack else 'timeout'}")

# Wait until reach ~90% of target altitude (max 20s)
t_to = time.time()
alt_ok = False
while time.time() - t_to < 20.0:
    m = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1.0)
    if m:
        rel_alt_m = m.relative_alt / 1000.0
        if rel_alt_m >= TAKEOFF_ALT * 0.9:
            print(f"✅ Altitud alcanzada: {rel_alt_m:.2f}m")
            alt_ok = True
            break
if not alt_ok:
    print("⚠ Altitud no confirmada — continuando igual")
time.sleep(2)


# ── Hilo: heartbeat + lectura de mensajes ─────────────────────
def heartbeat_y_watchdog():
    global heading_actual, modo_actual, rel_alt_m, roll_actual, pitch_actual
    while not shutdown_flag.is_set():
        master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0
        )
        while True:
            msg = master.recv_match(blocking=False)
            if msg is None:
                break
            t = msg.get_type()
            if t == 'HEARTBEAT':
                nuevo = modos_inv.get(msg.custom_mode, f"unknown({msg.custom_mode})")
                with modo_lock:
                    if nuevo != modo_actual:
                        print(f"📻 Modo: {modo_actual} → {nuevo}")
                        modo_actual = nuevo
            elif t == 'ATTITUDE':
                with heading_lock:
                    heading_actual = msg.yaw
                with attitude_lock:
                    roll_actual = msg.roll
                    pitch_actual = msg.pitch
            elif t == 'GLOBAL_POSITION_INT':
                with alt_lock:
                    rel_alt_m = msg.relative_alt / 1000.0
        time.sleep(0.1)


threading.Thread(target=heartbeat_y_watchdog, daemon=True).start()


# ── Velocity setpoint (BODY frame) ────────────────────────────
def send_velocity_body(vx, vy, vz, yaw_rate=0.0):
    """vx=forward, vy=right, vz=down (m/s), yaw_rate (rad/s) — BODY_OFFSET_NED."""
    # type_mask: ignore pos (bits 0-2), use vel (3-5 clear), ignore accel (6-8),
    # ignore force (9), ignore yaw absolute (10), use yaw_rate (11 clear)
    type_mask = 0b0000011111000111  # = 0x07C7
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        type_mask,
        0, 0, 0,           # pos ignored
        vx, vy, vz,        # velocity
        0, 0, 0,           # acc ignored
        0,                 # yaw absolute ignored
        yaw_rate           # yaw rate rad/s
    )


def hover():
    send_velocity_body(0.0, 0.0, 0.0, 0.0)


def _clamp(x, lim):
    return max(-lim, min(lim, x))


def calcular_velocidad(cx, cy, depth_val, cur_alt):
    """Pixel error + depth → BODY velocity setpoint (vx fwd, vy right, vz down, yaw_rate)."""
    px_err = cx - CX_ZED
    if abs(px_err) < DEAD_ZONE_PX:
        px_err = 0.0

    # Forward: close gap toward DIST_OBJETIVO
    vx = (depth_val - DIST_OBJETIVO) * KP_VX
    vx = _clamp(vx, V_FWD_MAX)

    # Lateral: normalized px error × gain
    x_norm = px_err / CX_ZED  # ~[-1, +1]
    vy = x_norm * KP_VY
    vy = _clamp(vy, V_LAT_MAX)

    # Yaw rate: P on angular error (small angle ≈ px_err / FX)
    yaw_err = math.atan2(px_err / FX_ZED, 1.0)
    yaw_rate = yaw_err * KP_YAW_RATE
    yaw_rate = _clamp(yaw_rate, YAW_RATE_MAX)

    # Altitude hold (vz positive = down). Climb if below TAKEOFF_ALT.
    alt_err = TAKEOFF_ALT - cur_alt
    vz = -alt_err * KP_VZ  # negative because down=+
    vz = _clamp(vz, V_VERT_MAX)

    return vx, vy, vz, yaw_rate


# ── HTTP MJPEG stream ─────────────────────────────────────────
INDEX_HTML = b"""<!DOCTYPE html><html><head><title>getaway-sim</title>
<style>body{margin:0;background:#111;color:#eee;font-family:monospace}
img{display:block;margin:auto;max-width:100%}h3{text-align:center;padding:8px}</style></head>
<body><h3>getaway-sim - FPV stream</h3>
<img src="/stream"></body></html>"""


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *a, **k): pass

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Content-Length', str(len(INDEX_HTML)))
            self.end_headers()
            try: self.wfile.write(INDEX_HTML)
            except Exception: pass
            return

        if self.path != '/stream':
            self.send_response(404); self.end_headers(); return

        self.send_response(200)
        self.send_header('Age', '0')
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        last_send = 0.0
        FPS_LIMIT = 15.0
        try:
            while not shutdown_flag.is_set():
                now = time.time()
                if now - last_send < 1.0 / FPS_LIMIT:
                    time.sleep(0.01); continue
                with frame_lock:
                    f = None if latest_frame is None else latest_frame.copy()
                if f is None:
                    time.sleep(0.05); continue
                ok, jpg = cv2.imencode('.jpg', f, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ok:
                    time.sleep(0.05); continue
                data = jpg.tobytes()
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ')
                self.wfile.write(str(len(data)).encode())
                self.wfile.write(b'\r\n\r\n')
                self.wfile.write(data)
                self.wfile.write(b'\r\n')
                self.wfile.flush()
                last_send = now
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


server = ThreadingHTTPServer(('0.0.0.0', 8080), StreamHandler)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True).start()
print("📺 Stream: http://localhost:8080")


# ── Shutdown ──────────────────────────────────────────────────
def shutdown_seguro(signum=None, frame=None):
    if shutdown_flag.is_set():
        if not emergency_kill.is_set():
            emergency_kill.set()
            print("\n⚠️ FORCE-DISARM")
        return
    shutdown_flag.set()
    print("\n🛑 LAND...")
    for _ in range(3):
        try: hover()
        except Exception: pass
        time.sleep(0.05)
    try:
        master.mav.set_mode_send(
            master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            master.mode_mapping()['LAND']
        )
    except Exception as e:
        print(f"  ⚠️ LAND falló: {e}")
    t0 = time.time()
    while time.time() - t0 < 8.0 and not emergency_kill.is_set():
        msg = master.recv_match(type='HEARTBEAT', blocking=False)
        if msg and not (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print("✅ Desarmado tras touchdown")
            break
        time.sleep(0.2)
    else:
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 21196, 0, 0, 0, 0, 0
        )
        print("✅ Force-disarm enviado")
    try: cam.close()
    except Exception: pass
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown_seguro)
signal.signal(signal.SIGTERM, shutdown_seguro)

# ── Loop principal ────────────────────────────────────────────
ultimo_comando = time.time()
print("Sistema activo — buscando vehículos...\n")

try:
    while not shutdown_flag.is_set():
        t_ini = time.time()
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.01); continue

        cam.retrieve_image(image, sl.VIEW.LEFT)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
        frame = image.get_data()[:, :, :3]

        results = model(frame, verbose=False)
        annotated = frame.copy()
        allowed = COCO_VEHICLE_IDS if USING_PLACEHOLDER else ALLOWED_CLASSES

        r_boxes = results[0].boxes
        vehiculo = False
        chosen = None
        names_dict = model.names

        def _name(cid):
            try: return names_dict.get(cid, f"class_{cid}")
            except AttributeError: return names_dict[cid] if cid in names_dict else f"class_{cid}"

        if r_boxes is not None and len(r_boxes) > 0:
            cls_arr = r_boxes.cls.cpu().numpy().astype(int)
            conf_arr = r_boxes.conf.cpu().numpy()
            xyxy_arr = r_boxes.xyxy.cpu().numpy()
            keep = np.isin(cls_arr, list(allowed))
            for j in np.where(keep)[0]:
                x1, y1, x2, y2 = map(int, xyxy_arr[j])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(annotated, f"{_name(int(cls_arr[j]))} {conf_arr[j]:.2f}",
                            (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            if keep.any():
                masked = np.where(keep, conf_arr, -1.0)
                i = int(np.argmax(masked))
                chosen = (int(cls_arr[i]), float(conf_arr[i]), xyxy_arr[i])

        with alt_lock:
            cur_alt = rel_alt_m
        with modo_lock:
            en_guided = (modo_actual == 'GUIDED')

        if chosen is not None:
            clase_id, conf, xyxy = chosen
            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            clase = _name(clase_id)

            err, dv = depth.get_value(cx, cy)
            depth_ok = np.isfinite(dv)

            cv2.circle(annotated, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(annotated,
                        f"{clase} {conf:.2f} {dv:.1f}m" if depth_ok else f"{clase} {conf:.2f}",
                        (cx, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if conf > 0.40 and depth_ok:
                vehiculo = True
                vx, vy, vz, yr = calcular_velocidad(cx, cy, dv, cur_alt)
                # Safety: don't descend if already too low
                if cur_alt < MIN_ALT_M and vz > 0:
                    vz = -0.5
                if en_guided and (time.time() - ultimo_comando) >= (1.0 / CMD_RATE_HZ):
                    send_velocity_body(vx, vy, vz, yaw_rate=yr)
                    ultimo_comando = time.time()
                    print(f"🎯 {clase} c={conf:.2f} d={dv:.1f}m "
                          f"vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f} "
                          f"yawR={math.degrees(yr):+.0f}°/s alt={cur_alt:.1f}m "
                          f"lat={(time.time()-t_ini)*1000:.0f}ms")
                elif not en_guided:
                    print(f"👁 {clase} c={conf:.2f} d={dv:.1f}m (MANUAL — no command)")

        # If no target, stream hover (zero velocity) — keeps ArduCopter happy
        if not vehiculo and en_guided:
            if (time.time() - ultimo_comando) >= (1.0 / CMD_RATE_HZ):
                # Hold altitude only
                alt_err = TAKEOFF_ALT - cur_alt
                vz_hold = _clamp(-alt_err * KP_VZ, V_VERT_MAX)
                send_velocity_body(0.0, 0.0, vz_hold, yaw_rate=0.0)
                ultimo_comando = time.time()

        # ── HUD overlay: modo + auto/manual badge ────────────
        with modo_lock:
            modo_str = modo_actual or "?"
        is_auto = (modo_str == 'GUIDED')
        badge_txt = "AUTO" if is_auto else "MANUAL"
        badge_col = (0, 220, 0) if is_auto else (0, 165, 255)
        with alt_lock:
            cur_alt = rel_alt_m
        cv2.rectangle(annotated, (10, 10), (220, 90), (0, 0, 0), -1)
        cv2.putText(annotated, badge_txt, (18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, badge_col, 2)
        cv2.putText(annotated, f"mode: {modo_str}", (18, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        cv2.putText(annotated, f"alt:  {cur_alt:.1f}m", (18, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        # ── Digital stabilization: counter-rotate by drone roll/pitch ──
        with attitude_lock:
            r_deg = math.degrees(roll_actual)
            p_rad = pitch_actual
        h, w = annotated.shape[:2]
        # Counter-rotate by roll
        M = cv2.getRotationMatrix2D((w // 2, h // 2), r_deg, 1.0)
        # Vertical translation by pitch (focal length × pitch radians)
        ty = -FY_ZED * p_rad
        M[1, 2] += ty
        stabilized = cv2.warpAffine(annotated, M, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)

        with frame_lock:
            latest_frame = stabilized

except Exception as e:
    import traceback
    print(f"❌ Error loop: {e!r}")
    traceback.print_exc()
finally:
    shutdown_seguro()
