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
from norfair import Detection, Tracker as NFTracker
from pymavlink import mavutil
from ultralytics import YOLO

import zed_sim as sl   # drop-in replacement for pyzed.sl

# ── Config ────────────────────────────────────────────────────
MODEL_PATH    = os.environ.get("YOLO_MODEL", "/sim/models/best.pt")
MAVLINK_URL   = os.environ.get("MAVLINK_URL", "udp:127.0.0.1:14551")
DIST_OBJETIVO = 8.0   # m, target chase distance (más cerca para no perder lock)
TAKEOFF_ALT   = 10.0  # m (higher to see far + safety margin)

# ── Velocity-based controller, P-only con suavizado en inputs ──
# Lecciones de iteraciones anteriores:
#   - Feedforward sobre d(depth)/dt = mal: range rate es velocidad RELATIVA;
#     cuando el drone acelera, ddepth cae → FF acelera más → loop positivo.
#   - Feedforward sobre dcx/dt = mal: el propio yaw del drone hace que el bbox
#     se mueva en imagen, FF interpreta como movimiento del target → oscilación.
#   - Suavizar derivadas no arregla nada; la señal es la equivocada.
# Solución: P-only con EMA en INPUTS (cx, depth) — suaviza el jitter del YOLO
# sin meter el ruido del propio control. Prioriza centrar (yaw) sobre cerrar
# distancia: si target off-center, escala vx hacia 0 hasta que drone gire.

KP_VX = 0.6           # P sobre (depth - DIST_OBJETIVO) — bajo para no pasarse de largo
V_FWD_MAX = 10.0      # m/s — cap forward (>ambulancia 7 m/s pero sin overshoot brutal)
A_BRAKE = 1.5         # m/s² — decel asumida para curva de frenado
V_MATCH = 6.0         # m/s — base de la curva (~velocidad ambulancia): cerca del
                      # target el drone converge a seguirla, no a frenar en seco

KP_VY = 1.0           # P sobre (px_err / CX)
V_LAT_MAX = 2.0       # m/s — cap lateral

KP_YAW_RATE = 1.2     # P sobre ángulo pixel — gana prioridad para centrar
YAW_RATE_MAX = 1.0    # rad/s — cap alto para curvas cerradas

KP_VZ = 0.4
V_VERT_MAX = 1.0

# Smoothing en inputs + slew limit en comandos
ALPHA_EMA   = 0.5     # EMA sobre cx y depth. ↑ = más smoothing, más lag
SLEW_VX_UP   = 2.0    # m/s² — aceleración suave (evita lanzarse al detectar)
SLEW_VX_DOWN = 6.0    # m/s² — frenado rápido (clave anti-overshoot)
SLEW_VY     = 2.0
SLEW_YAW    = 2.0     # rad/s²
LOCK_TIMEOUT_S = 3.0  # tolerancia alta — trasera de ambulancia da detecciones intermitentes

DEAD_ZONE_PX    = 50  # ignora errores ±N px del centro
DEAD_ZONE_DEPTH = 2.0 # ignora errores ±N m de DIST_OBJETIVO
CONF_LOCK    = 0.25   # filtro mínimo para crear/mantener Detection en norfair
CONF_ACQUIRE = 0.40   # filtro mínimo para crear nuevo track norfair
MIN_ALT_M    = 3.0
CMD_RATE_HZ  = 15.0   # rate de setpoints

# ── Norfair tracker (Kalman + ID persistence a través de oclusiones) ──
# A diferencia del lock simple, norfair extrapola posición de bbox via filtro
# Kalman durante gaps. Sobrevive oclusiones (edificios/árboles/puentes) que
# tumban YOLO por varios frames. La identidad del track persiste, así el
# drone vuelve a engancharse al MISMO objeto cuando reaparece cerca de la
# trayectoria predicha — no a otro vehículo cercano.
NF_DIST_THRESHOLD   = 120     # px — match radius entre detección y track
NF_HIT_COUNTER_MAX  = 45      # frames sin detección antes de matar track (~3s a 15Hz)
NF_INIT_DELAY       = 2       # frames consecutivos para confirmar nuevo track
NF_PAST_DETS        = 4       # historial usado por Kalman de norfair

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


def _last_conf(obj) -> float:
    """Norfair TrackedObject → confidence de la última detección viva."""
    if obj.last_detection is not None and obj.last_detection.data:
        return float(obj.last_detection.data.get("conf", 0.0))
    return 0.0


def _last_cls(obj):
    if obj.last_detection is not None and obj.last_detection.data:
        return obj.last_detection.data.get("cls")
    return None


# ── Tracker de estado entre frames ──
class TargetTracker:
    """P-only controller con suavizado en inputs y slew limit en comandos.

    Diseño:
      - EMA en (cx, depth) — suaviza jitter del bounding box / depth lookup.
      - NO derivadas / feedforward (causaban oscilaciones).
      - Dead-zone en px_err y depth_err para no perseguir errores chicos.
      - center_factor: cuando target está off-center, escala vx hacia 0 →
        el drone gira primero (yaw), avanza después. Evita pasar de largo
        en chases circulares.
      - Hysteresis de confidence (CONF_ACQUIRE > CONF_LOCK).
      - Lock timeout grande (LOCK_TIMEOUT_S) tolera detecciones intermitentes.
    """

    def __init__(self):
        self.cx_ema = None
        self.depth_ema = None
        self.prev_vx_cmd = 0.0
        self.prev_vy_cmd = 0.0
        self.prev_yaw_cmd = 0.0
        self.locked = False
        self.last_seen_t = 0.0

    def update_lock(self, conf: float, t_now: float) -> bool:
        if self.locked:
            if conf >= CONF_LOCK:
                self.last_seen_t = t_now
                return True
            return False
        if conf >= CONF_ACQUIRE:
            self.locked = True
            self.last_seen_t = t_now
            self.cx_ema = None       # reset EMA al adquirir lock fresco
            self.depth_ema = None
            return True
        return False

    def check_timeout(self, t_now: float) -> None:
        if self.locked and (t_now - self.last_seen_t) > LOCK_TIMEOUT_S:
            self.locked = False
            self.cx_ema = None
            self.depth_ema = None
            print(f"🔓 Lock perdido (>{LOCK_TIMEOUT_S}s sin detección estable)")

    def calcular(self, cx, cy, depth_val, cur_alt, t_now):
        """Pixel + depth + alt → (vx, vy, vz, yaw_rate) en BODY NED."""
        # EMA en inputs (smooth bbox + depth jitter)
        if self.cx_ema is None:
            self.cx_ema = float(cx)
            self.depth_ema = float(depth_val)
        else:
            self.cx_ema    = ALPHA_EMA * self.cx_ema    + (1 - ALPHA_EMA) * cx
            self.depth_ema = ALPHA_EMA * self.depth_ema + (1 - ALPHA_EMA) * depth_val

        cx_s    = self.cx_ema
        depth_s = self.depth_ema

        # Pixel error con dead-zone
        px_err = cx_s - CX_ZED
        if abs(px_err) < DEAD_ZONE_PX:
            px_err_eff = 0.0
        else:
            px_err_eff = px_err

        # Depth error con dead-zone
        depth_err = depth_s - DIST_OBJETIVO
        if abs(depth_err) < DEAD_ZONE_DEPTH:
            depth_err = 0.0

        # ── Yaw (prioridad: centrar) ──
        yaw_err  = math.atan2(px_err_eff / FX_ZED, 1.0)
        yaw_rate_target = _clamp(yaw_err * KP_YAW_RATE, YAW_RATE_MAX)

        # ── Lateral (ayuda a centrar) ──
        vy_target = _clamp((px_err_eff / CX_ZED) * KP_VY, V_LAT_MAX)

        # ── Forward (depende de qué tan centrado está el target) ──
        # Si centrado → 1.0. Al borde → 0.7 (piso para asegurar avance
        # cuando target en curva mantiene px_err no-cero permanente).
        center_factor = max(0.7, 1.0 - abs(px_err) / CX_ZED)
        vx_target = _clamp(depth_err * KP_VX * center_factor, V_FWD_MAX)

        # ── Curva de frenado: exceso sobre V_MATCH limitado por √(2·a·err) ──
        # Lejos: sprint hasta V_FWD_MAX. Cerca: converge a V_MATCH (velocidad
        # de la ambulancia), no a cero → cierra el gap sin pasarse de largo.
        # Target estático lo cubre el término P (KP_VX·err → 0 en el target).
        if depth_err > 0:
            vx_target = min(vx_target,
                            V_MATCH + math.sqrt(2.0 * A_BRAKE * depth_err))

        # ── Altitud ──
        alt_err = TAKEOFF_ALT - cur_alt
        vz_target = _clamp(-alt_err * KP_VZ, V_VERT_MAX)

        # ── Slew rate limit (asimétrico en vx: acelera lento, frena rápido) ──
        dt_cmd = 1.0 / CMD_RATE_HZ
        dvx = vx_target - self.prev_vx_cmd
        slew_vx = SLEW_VX_UP if dvx * self.prev_vx_cmd >= 0 and abs(vx_target) > abs(self.prev_vx_cmd) else SLEW_VX_DOWN
        vx = self.prev_vx_cmd + _clamp(dvx, slew_vx * dt_cmd)
        vy = self.prev_vy_cmd + _clamp(vy_target - self.prev_vy_cmd, SLEW_VY * dt_cmd)
        yaw_rate = self.prev_yaw_cmd + _clamp(yaw_rate_target - self.prev_yaw_cmd,
                                              SLEW_YAW * dt_cmd)
        self.prev_vx_cmd  = vx
        self.prev_vy_cmd  = vy
        self.prev_yaw_cmd = yaw_rate

        return vx, vy, vz_target, yaw_rate


tracker = TargetTracker()

nf_tracker = NFTracker(
    distance_function="euclidean",
    distance_threshold=NF_DIST_THRESHOLD,
    hit_counter_max=NF_HIT_COUNTER_MAX,
    initialization_delay=NF_INIT_DELAY,
    past_detections_length=NF_PAST_DETS,
)
last_depth_seen = None       # depth fresca más reciente — persiste durante oclusión
last_depth_t    = 0.0
DEPTH_STALE_S   = 2.5        # s — si depth no se actualiza, no controlar vx (solo yaw)


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

        # ── Construir lista de Detection para norfair ──
        nf_dets = []
        if r_boxes is not None and len(r_boxes) > 0:
            cls_arr = r_boxes.cls.cpu().numpy().astype(int)
            conf_arr = r_boxes.conf.cpu().numpy()
            xyxy_arr = r_boxes.xyxy.cpu().numpy()
            keep = np.isin(cls_arr, list(allowed)) & (conf_arr >= CONF_LOCK)
            for j in np.where(keep)[0]:
                x1, y1, x2, y2 = map(int, xyxy_arr[j])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(annotated, f"{_name(int(cls_arr[j]))} {conf_arr[j]:.2f}",
                            (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                cx_d = (x1 + x2) / 2.0
                cy_d = (y1 + y2) / 2.0
                # Depth en el centro de la bbox (NaN si fuera de rango)
                err_d, dv_d = depth.get_value(int(cx_d), int(cy_d))
                nf_dets.append(Detection(
                    points=np.array([[cx_d, cy_d]]),
                    scores=np.array([conf_arr[j]]),
                    data={
                        "conf": float(conf_arr[j]),
                        "depth": float(dv_d) if np.isfinite(dv_d) else None,
                        "cls": int(cls_arr[j]),
                    },
                ))

        # Update norfair tracker (extrapola Kalman aunque nf_dets esté vacío)
        tracked_objects = nf_tracker.update(detections=nf_dets)

        # Pick best track: prefer live (fresh detection) > extrapolado, max conf
        target_obj = None
        live_target = None
        for obj in tracked_objects:
            if obj.live_points.any():
                if live_target is None or _last_conf(obj) > _last_conf(live_target):
                    live_target = obj
            if target_obj is None:
                target_obj = obj
        if live_target is not None:
            target_obj = live_target

        with alt_lock:
            cur_alt = rel_alt_m
        with modo_lock:
            en_guided = (modo_actual == 'GUIDED')

        t_now = time.time()

        if target_obj is not None:
            cx_est, cy_est = target_obj.estimate[0]
            cx_est = int(round(cx_est))
            cy_est = int(round(cy_est))
            is_live = bool(target_obj.live_points.any())
            conf_show = _last_conf(target_obj)
            cls_show = _last_cls(target_obj) or 0
            clase = _name(cls_show)

            # Depth: si hay detección fresca, refresca cache; si no, usa cache
            dv_obj = None
            if is_live and target_obj.last_detection is not None:
                dv_cand = target_obj.last_detection.data.get("depth")
                if dv_cand is not None and np.isfinite(dv_cand):
                    last_depth_seen = float(dv_cand)
                    last_depth_t = t_now
                    dv_obj = last_depth_seen
            if dv_obj is None and last_depth_seen is not None:
                if (t_now - last_depth_t) <= DEPTH_STALE_S:
                    dv_obj = last_depth_seen

            color = (0, 255, 0) if is_live else (0, 165, 255)
            cv2.circle(annotated, (cx_est, cy_est), 7, color, -1)
            tag = "TRK" if is_live else "PRED"
            label = (f"{tag} id={target_obj.id} {clase} c={conf_show:.2f}"
                     + (f" d={dv_obj:.1f}m" if dv_obj is not None else " d=?"))
            cv2.putText(annotated, label,
                        (cx_est, cy_est - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if dv_obj is not None:
                vehiculo = True
                vx, vy, vz, yr = tracker.calcular(cx_est, cy_est, dv_obj, cur_alt, t_now)
                # Si predicción (sin live), reduce yaw_rate y vy para no derivar
                # demasiado por errores de Kalman; mantén vx (avance).
                if not is_live:
                    yr *= 0.5
                    vy *= 0.3
                # Safety: don't descend if already too low
                if cur_alt < MIN_ALT_M and vz > 0:
                    vz = -0.5
                if en_guided and (t_now - ultimo_comando) >= (1.0 / CMD_RATE_HZ):
                    send_velocity_body(vx, vy, vz, yaw_rate=yr)
                    ultimo_comando = t_now
                    px_off = cx_est - CX_ZED
                    age = t_now - last_depth_t if last_depth_t else 0.0
                    icon = "🎯" if is_live else "🔮"
                    print(f"{icon} id={target_obj.id} {clase} c={conf_show:.2f} "
                          f"d={dv_obj:.1f}m vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f} "
                          f"yawR={math.degrees(yr):+.0f}°/s alt={cur_alt:.1f}m "
                          f"px_off={px_off:+.0f} "
                          f"{'LIVE' if is_live else f'KF age={age:.1f}s'} "
                          f"lat={(t_now-t_ini)*1000:.0f}ms")
                elif not en_guided:
                    print(f"👁 {clase} c={conf_show:.2f} d={dv_obj:.1f}m (MANUAL — no command)")

        # Sin target válido (sin tracks o depth stale):
        # decay vx/vy/yaw a cero respetando slew, mantener altitud.
        # (norfair ya maneja la extrapolación interna; aquí solo cae si TODO se perdió.)
        if not vehiculo and en_guided:
            if (t_now - ultimo_comando) >= (1.0 / CMD_RATE_HZ):
                alt_err = TAKEOFF_ALT - cur_alt
                vz_hold = _clamp(-alt_err * KP_VZ, V_VERT_MAX)
                dt_cmd = 1.0 / CMD_RATE_HZ
                vx_d = tracker.prev_vx_cmd + _clamp(-tracker.prev_vx_cmd, SLEW_VX_DOWN * dt_cmd)
                vy_d = tracker.prev_vy_cmd + _clamp(-tracker.prev_vy_cmd, SLEW_VY * dt_cmd)
                yr_d = tracker.prev_yaw_cmd + _clamp(-tracker.prev_yaw_cmd, SLEW_YAW * dt_cmd)
                tracker.prev_vx_cmd  = vx_d
                tracker.prev_vy_cmd  = vy_d
                tracker.prev_yaw_cmd = yr_d
                send_velocity_body(vx_d, vy_d, vz_hold, yaw_rate=yr_d)
                ultimo_comando = t_now
                n_tracks = len(tracked_objects)
                print(f"⏳ no-target n_tracks={n_tracks} alt={cur_alt:.1f}m")

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
