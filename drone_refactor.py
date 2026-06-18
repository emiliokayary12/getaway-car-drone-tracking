import math
import pyzed.sl as sl
from ultralytics import YOLO
import cv2
import numpy as np
import threading
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pymavlink import mavutil
import time

# Flag global para coordinar shutdown limpio
shutdown_flag = threading.Event()
emergency_kill = threading.Event()   # segundo Ctrl+C → force-disarm sin esperar LAND

# ── CLASES ─────────────────────────────────────────────────────
CLASS_NAMES = {0: 'ambulance', 1: 'fire_truck', 2: 'police_car'}

# ── CARGAR MODELO PRIMERO (reserva GPU antes que ZED) ──────────
model = YOLO("/home/getaway/Desktop/reto/best_run_21abril.engine", task="detect")

# ── INICIALIZAR ZED 2 ──────────────────────────────────────────
cam = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_units = sl.UNIT.METER

status = cam.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"Camera failed to open: {status}")
    exit(1)

# Intrínsecos reales de la ZED (cámara izquierda) — necesarios para
# convertir píxeles de YOLO a un vector 3D en frame cámara/body.
calib = cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
FX_ZED, FY_ZED = calib.fx, calib.fy
CX_ZED, CY_ZED = calib.cx, calib.cy
print(f"✅ ZED intrínsecos: fx={FX_ZED:.1f} fy={FY_ZED:.1f} "
      f"cx={CX_ZED:.1f} cy={CY_ZED:.1f}")

DIST_OBJETIVO = 5.0   # m — mantener 5 m de distancia al vehículo
TAKEOFF_ALT   = 2.0   # m — debe coincidir con MAV_CMD_NAV_TAKEOFF abajo

# ── Velocity-based PID (P-only) — backport del sim ─────────────
# Reemplaza el controlador de position-target original. El sim mostró que
# enviar `SET_POSITION_TARGET_LOCAL_NED` con offsets X/Y absolutos genera
# overshoot agresivo y el target sale del FOV. Con velocity-target el FC
# aplica su cascade PID interno (PSC_*/VEL_*/ATC_*) y queda mucho más estable.
# Caps conservadores vs. sim: hardware necesita margen para primeras pruebas.
KP_VX = 0.4           # m/s por m de error de profundidad
V_FWD_MAX = 4.0       # m/s — cap forward (vs 8.0 en sim)
KP_VY = 1.5           # m/s por px_err normalizado
V_LAT_MAX = 1.5       # m/s — cap lateral (vs 2.0 en sim)
KP_YAW_RATE = 0.8     # rad/s por rad de error angular
YAW_RATE_MAX = 0.5    # rad/s — cap yaw (~28°/s)
KP_VZ = 0.4
V_VERT_MAX = 0.8      # m/s — cap vertical (vs 1.0 en sim)

DEAD_ZONE_PX = 60     # ignorar |px_err| < N para no perseguir ruido
MIN_ALT_M    = 1.0    # piso de seguridad — no descender por debajo
CMD_RATE_HZ  = 10.0   # rate de setpoints de velocidad

runtime  = sl.RuntimeParameters()
image    = sl.Mat()
depth    = sl.Mat()

# ── VARIABLES COMPARTIDAS ──────────────────────────────────────
latest_frame = None
frame_lock   = threading.Lock()

# Heading actual del dron (rad, world frame) — leído del stream ATTITUDE.
heading_actual = 0.0
heading_lock   = threading.Lock()

# Modo de vuelo actual — para ceder control al piloto via RC.
# Si el piloto flippa el switch (LOITER/ALT_HOLD/STABILIZE/etc.), el script
# detecta el cambio y deja de mandar comandos de posición.
modo_actual = None
modo_lock   = threading.Lock()

# Altitud relativa actual (m) — leída de GLOBAL_POSITION_INT.
# Necesaria para el canal vz del controlador de velocidad (mantener TAKEOFF_ALT).
rel_alt_m = 0.0
alt_lock  = threading.Lock()

# ── CONECTAR PIXHAWK ───────────────────────────────────────────
print("Conectando con PixHawk...")
master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
master.wait_heartbeat()
print("✅ PixHawk conectado")

# ── PARÁMETROS: setup GPS-enabled + geofence ───────────────────
params = {
    # Fuentes EKF: XY/VEL desde GPS, altitud desde barómetro
    b'GPS_TYPE':       1,           # AUTO-detect
    b'EK3_SRC1_POSXY': 3,           # GPS
    b'EK3_SRC1_VELXY': 3,           # GPS
    b'EK3_SRC1_POSZ':  1,           # Barómetro (decisión del usuario)
    # Failsafes
    b'FS_EKF_ACTION':  1,           # LAND si EKF falla
    b'FS_THR_ENABLE':  0,           # Sin radio failsafe (no hay RC conectado)
    b'FS_GCS_ENABLE':  0,           # Sin GCS failsafe (heartbeats GCS arrancan tras takeoff)
    b'DISARM_DELAY':   10,
    # Pre-arm checks DESACTIVADOS:
    #   - no hay RC conectado → falla "RC not found"
    #   - el barómetro al arrancar puede diferir mucho del GPS
    # El GPS lock ya lo validamos manualmente antes de armar.
    b'ARMING_CHECK':   0,
    # Geofence — protección durante pruebas
    b'FENCE_ENABLE':   1,
    b'FENCE_TYPE':     3,           # alt-max + circle
    b'FENCE_RADIUS':   30.0,
    b'FENCE_ALT_MAX':  10.0,
    b'FENCE_ACTION':   4,           # Brake (4.1+); usar 0 (Report) si es más antiguo
}
for param, valor in params.items():
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        param, float(valor),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    time.sleep(0.3)

time.sleep(3)

# ── PEDIR STREAM DE GPS_RAW_INT (no se envía por default vía USB) ─
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
    0,
    mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,
    200000,   # us → 5 Hz
    0, 0, 0, 0, 0
)
time.sleep(0.5)

# ── ESPERAR GPS LOCK ───────────────────────────────────────────
print("Esperando GPS lock...")
gps_timeouts = 0
while True:
    msg = master.recv_match(type='GPS_RAW_INT', blocking=True, timeout=2)
    if msg and msg.fix_type >= 3 and msg.satellites_visible >= 6:
        print(f"✅ GPS lock — fix={msg.fix_type}, "
              f"sats={msg.satellites_visible}, "
              f"HDOP={msg.eph/100:.2f}")
        break
    if msg:
        gps_timeouts = 0
        print(f"  ...fix={msg.fix_type}, sats={msg.satellites_visible}, "
              f"HDOP={msg.eph/100:.2f}")
    else:
        gps_timeouts += 1
        print(f"  ...timeout esperando GPS_RAW_INT ({gps_timeouts})")
        if gps_timeouts == 5:
            print("  ⚠️  Re-solicitando stream GPS_RAW_INT...")
            master.mav.command_long_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,
                200000, 0, 0, 0, 0, 0
            )
        if gps_timeouts >= 15:
            print("  ❌ GPS no responde después de 30s.")
            print("     Verifica: módulo GPS conectado, SERIAL3_PROTOCOL=5,")
            print("     y reinicia la PixHawk si acabas de cambiar GPS_TYPE.")
            sys.exit(1)

# ── CALIBRACIÓN DE BARÓMETRO (ground pressure) ─────────────────
# Toma la presión actual como altitud 0. Sin esto, baro y GPS pueden
# diferir varios km (lo que rompe la referencia para FENCE_ALT_MAX y takeoff).
print("Calibrando barómetro al suelo...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION,
    0,
    0, 0, 1, 0, 0, 0, 0   # param3=1 → ground pressure calibration
)
time.sleep(3)   # el FC tarda ~2s en estabilizar la nueva referencia

# ── SOLICITAR STREAM DE ATTITUDE (para heading actual) ─────────
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
    0,
    mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
    100000,   # us → 10 Hz
    0, 0, 0, 0, 0
)

# ── SOLICITAR STREAM DE GLOBAL_POSITION_INT (para altitud relativa) ─
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
    0,
    mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
    200000,   # us → 5 Hz
    0, 0, 0, 0, 0
)

# ── MODO GUIDED ────────────────────────────────────────────────
modos = master.mode_mapping()
modos_inv = {v: k for k, v in modos.items()}   # id → nombre, para detectar override RC
modo  = 'GUIDED'
master.mav.set_mode_send(
    master.target_system,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    modos[modo]
)
time.sleep(2)
print(f"✅ Modo: {modo}")

def drenar_statustext(prefijo="  FC:", duracion=1.5):
    """Imprime cualquier STATUSTEXT pendiente del FC durante `duracion` seg."""
    t_end = time.time() + duracion
    while time.time() < t_end:
        m = master.recv_match(type='STATUSTEXT', blocking=False)
        if m:
            print(f"{prefijo} {m.text}")
        else:
            time.sleep(0.05)

def confirmar_armado(timeout=3.0):
    """Lee HEARTBEATs hasta confirmar (o descartar) el bit SAFETY_ARMED."""
    t_end = time.time() + timeout
    armado = False
    while time.time() < t_end:
        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=0.5)
        if hb:
            armado = bool(hb.base_mode &
                          mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            if armado:
                return True
    return armado

# Drenar mensajes residuales antes de armar
drenar_statustext(prefijo="  [pre-arm FC]:", duracion=0.5)

# ── ARMAR (sin force) ──────────────────────────────────────────
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    1, 0, 0, 0, 0, 0, 0
)
msg = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
ack_arm = msg.result if msg else None
print(f"  ACK arm: {ack_arm}")
drenar_statustext(prefijo="  FC:", duracion=1.0)

# Confirmar con HEARTBEAT que realmente está armado (no solo aceptado)
if confirmar_armado(timeout=2.0):
    print("✅ Motores armados (confirmado por HEARTBEAT)\n")
else:
    print("❌ ARM aceptado pero HEARTBEAT reporta DESARMADO.")
    print("   Causas típicas:")
    print("     - Safety switch sin presionar (LED rojo parpadeando)")
    print("     - Pre-arm check falló — revisa STATUSTEXT arriba")
    print("   Abortando antes de takeoff.")
    sys.exit(1)

# ── TAKEOFF ────────────────────────────────────────────────────
print("Enviando comando de takeoff...")
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    0,
    0, 0, 0, 0, 0, 0,
    2.0  # altura objetivo en metros
)
msg = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
if msg:
    if msg.result == 0:
        print("✅ Takeoff aceptado — motores activos")
    else:
        print(f"⚠️ Takeoff código: {msg.result}")
        drenar_statustext(prefijo="  FC:", duracion=1.5)

print("Esperando 3 segundos para que suban los motores...")
time.sleep(3)

# ── HEARTBEAT + WATCHDOG + LECTOR DE ATTITUDE + ALTITUDE ───────
def heartbeat_y_watchdog():
    global heading_actual, modo_actual, rel_alt_m
    while not shutdown_flag.is_set():
        master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0
        )

        # Drenar la cola y despachar por tipo
        while True:
            msg = master.recv_match(blocking=False)
            if msg is None:
                break
            t = msg.get_type()
            if t == 'HEARTBEAT':
                # Detectar cambios de modo (RC mode switch del piloto) y armado
                nuevo_modo = modos_inv.get(
                    msg.custom_mode, f"unknown({msg.custom_mode})"
                )
                armado = bool(msg.base_mode &
                              mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                with modo_lock:
                    if nuevo_modo != modo_actual:
                        print(f"📻 Modo: {modo_actual} → {nuevo_modo} "
                              f"({'armado' if armado else 'desarmado'})")
                        modo_actual = nuevo_modo
            elif t == 'ATTITUDE':
                with heading_lock:
                    heading_actual = msg.yaw
            elif t == 'GLOBAL_POSITION_INT':
                with alt_lock:
                    rel_alt_m = msg.relative_alt / 1000.0

        time.sleep(0.1)

threading.Thread(target=heartbeat_y_watchdog, daemon=True).start()

# ── COMANDO DE VELOCIDAD (BODY_OFFSET_NED) ─────────────────────
def send_velocity_body(vx, vy, vz, yaw_rate=0.0):
    """
    Setpoint de velocidad en frame body (NED).
    vx: + adelante (m/s)
    vy: + derecha  (m/s)
    vz: + abajo    (m/s)
    yaw_rate: + sentido horario visto desde arriba (rad/s)

    type_mask 0x07C7:
      bits 0-2 (pos)        = 1  → ignorar
      bits 3-5 (vel)        = 0  → usar
      bits 6-8 (accel)      = 1  → ignorar
      bit  9   (force)      = 1  → ignorar
      bit 10   (yaw abs)    = 1  → ignorar
      bit 11   (yaw_rate)   = 0  → usar
    """
    type_mask = 0b0000011111000111  # = 0x07C7
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        type_mask,
        0, 0, 0,            # pos ignorado
        vx, vy, vz,         # velocidad
        0, 0, 0,            # accel ignorado
        0,                  # yaw absoluto ignorado
        yaw_rate            # rad/s
    )

def hover():
    """Setpoint zero — el FC mantiene altitud/posición vía PSC_*."""
    send_velocity_body(0.0, 0.0, 0.0, 0.0)

def _clamp(x, lim):
    return max(-lim, min(lim, x))

# ── CÁLCULO DE VELOCIDAD AL VEHÍCULO OBJETIVO ──────────────────
def calcular_velocidad(cx, cy, depth_val, cur_alt):
    """
    A partir del centro del bbox + profundidad ZED + altitud relativa,
    devuelve setpoint de velocidad en frame body (vx fwd, vy right,
    vz down, yaw_rate). Asume ZED mirando hacia +X body.

    Canales:
      - vx: P sobre (depth - DIST_OBJETIVO). Saturado a V_FWD_MAX.
      - vy: P sobre (px_err / CX) — error lateral normalizado.
      - yaw_rate: P sobre ángulo (px_err / FX) — centra el target.
      - vz: P sobre (TAKEOFF_ALT - cur_alt) — mantiene altura nominal.
    Dead-zone en px_err evita perseguir ruido cuando el target ya está centrado.
    """
    px_err = cx - CX_ZED
    if abs(px_err) < DEAD_ZONE_PX:
        px_err = 0.0

    # Forward: cerrar gap hacia DIST_OBJETIVO
    vx = _clamp((depth_val - DIST_OBJETIVO) * KP_VX, V_FWD_MAX)

    # Lateral: error de pixel normalizado [-1, +1]
    x_norm = px_err / CX_ZED
    vy = _clamp(x_norm * KP_VY, V_LAT_MAX)

    # Yaw rate: small-angle aprox
    yaw_err  = math.atan2(px_err / FX_ZED, 1.0)
    yaw_rate = _clamp(yaw_err * KP_YAW_RATE, YAW_RATE_MAX)

    # Altitud: vz positivo = abajo, por eso negamos el error
    alt_err = TAKEOFF_ALT - cur_alt
    vz = _clamp(-alt_err * KP_VZ, V_VERT_MAX)

    return vx, vy, vz, yaw_rate

# ── SERVIDOR HTTP PARA VER VIDEO EN VIVO ───────────────────────
class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()
            _, jpg = cv2.imencode('.jpg', frame)
            self.wfile.write(b'--frame\r\n')
            self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
            self.wfile.write(jpg.tobytes())
            self.wfile.write(b'\r\n')

server = HTTPServer(('0.0.0.0', 8080), StreamHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print("Stream en http://localhost:8080")

# ── SHUTDOWN SEGURO (LAND in-place) ────────────────────────────
def shutdown_seguro(signum=None, frame=None):
    """Aterrizaje controlado: LAND → esperar touchdown → desarmar.
    Si llaman dos veces (segundo Ctrl+C), salta a force-disarm inmediato."""
    if shutdown_flag.is_set():
        # Segundo Ctrl+C → kill switch
        if not emergency_kill.is_set():
            print("\n⚠️ Segundo Ctrl+C — saltando a FORCE-DISARM inmediato")
            emergency_kill.set()
        return
    shutdown_flag.set()
    print("\n🛑 Shutdown — iniciando descenso controlado...")
    print("   (presiona Ctrl+C otra vez para force-disarm inmediato)")

    # 1. Hover unos frames para cancelar cualquier trayectoria
    for _ in range(3):
        try:
            hover()
        except Exception:
            pass
        time.sleep(0.05)

    # 2. Cambiar a LAND
    en_land = False
    try:
        if 'LAND' in master.mode_mapping():
            master.mav.set_mode_send(
                master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                master.mode_mapping()['LAND']
            )
            print("  ✅ Modo LAND activado — descendiendo...")
            en_land = True
        else:
            print("  ⚠️ Modo LAND no disponible — desarme directo")
    except Exception as e:
        print(f"  ⚠️ No se pudo entrar a LAND: {e}")

    # 3. Esperar auto-desarme al tocar suelo (o señal de kill)
    auto_desarmado = False
    if en_land:
        TIMEOUT_LAND = 8.0   # bajado de 25s — LAND real nunca tarda más
        t0 = time.time()
        ultimo_log = 0.0
        while time.time() - t0 < TIMEOUT_LAND:
            if emergency_kill.is_set():
                print("  ⚠️ Kill switch activado — abortando LAND")
                break
            msg = master.recv_match(type='HEARTBEAT', blocking=False)
            if msg:
                armado = bool(msg.base_mode &
                              mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                if not armado:
                    print(f"  ✅ Touchdown detectado — auto-desarmado "
                          f"({time.time()-t0:.1f}s)")
                    auto_desarmado = True
                    break
            master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0
            )
            if time.time() - ultimo_log > 2.0:
                print(f"  ...descendiendo ({time.time()-t0:.0f}s)")
                ultimo_log = time.time()
            time.sleep(0.2)
        else:
            print(f"  ⚠️ Timeout {TIMEOUT_LAND:.0f}s — forzando desarme")

    # 4. Desarme final con force=21196 si LAND no completó por sí solo
    #    21196 es el magic number de ArduPilot para "desarma aunque vueles"
    #    (contraparte de 2989 del force-arm). Necesario cuando el FC piensa
    #    que sigue volando (p.ej. dron sujetado en mano).
    if not auto_desarmado:
        try:
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0, 21196, 0, 0, 0, 0, 0    # disarm + force magic
            )
            print("  ✅ Force-disarm enviado (21196)")
            # Confirmar con HEARTBEAT que los motores realmente pararon
            t_end = time.time() + 2.0
            confirmado = False
            while time.time() < t_end:
                hb = master.recv_match(type='HEARTBEAT', blocking=False)
                if hb and not (hb.base_mode &
                               mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                    print("  ✅ DESARMADO confirmado por HEARTBEAT")
                    confirmado = True
                    break
                time.sleep(0.05)
            if not confirmado:
                print("  ❌ El FC NO confirmó desarme — DESCONECTA LA BATERÍA")
        except Exception as e:
            print(f"  ⚠️ Error desarmando: {e}")
    else:
        # Aún así enviar disarm sin force como red de seguridad
        try:
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 0, 0, 0, 0, 0, 0
            )
        except Exception:
            pass

    # 5. Cerrar cámara
    try:
        cam.close()
        print("  ✅ ZED cerrada")
    except Exception:
        pass

    time.sleep(0.5)
    print("👋 Salida limpia")
    sys.exit(0)

signal.signal(signal.SIGINT,  shutdown_seguro)
signal.signal(signal.SIGTERM, shutdown_seguro)

# ── LOOP PRINCIPAL ─────────────────────────────────────────────
ultimo_comando = time.time()
vehiculo_detectado = False

print("Sistema activo — buscando vehículos de emergencia...\n")

try:
    while not shutdown_flag.is_set():
        t_inicio = time.time()

        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(image, sl.VIEW.LEFT)
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
            frame = image.get_data()[:, :, :3]

            results   = model(frame, verbose=False)
            results[0].names = CLASS_NAMES
            annotated = results[0].plot()

            boxes = results[0].boxes
            vehiculo_detectado = False

            if boxes is not None and len(boxes) > 0:
                confianzas = boxes.conf.cpu().numpy()
                idx_mejor  = np.argmax(confianzas)
                box        = boxes[idx_mejor]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                confianza = float(box.conf[0])
                clase     = CLASS_NAMES[int(box.cls[0])]

                err, depth_val = depth.get_value(cx, cy)
                depth_ok = np.isfinite(depth_val)

                if depth_ok:
                    label = f"{clase} {confianza:.2f} | {depth_val:.1f}m"
                else:
                    label = f"{clase} {confianza:.2f}"

                cv2.putText(annotated, label, (cx, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.circle(annotated, (cx, cy), 8, (0, 255, 0), -1)

                if confianza > 0.60 and depth_ok:
                    vehiculo_detectado = True

                    with alt_lock:
                        cur_alt = rel_alt_m
                    vx, vy, vz, yaw_rate = calcular_velocidad(
                        cx, cy, depth_val, cur_alt
                    )
                    # Safety floor: si ya estamos muy bajo, no descender más
                    if cur_alt < MIN_ALT_M and vz > 0:
                        vz = -0.3
                    with modo_lock:
                        en_guided = (modo_actual == 'GUIDED')

                    # Throttle: setpoints a CMD_RATE_HZ máx
                    if en_guided and (time.time() - ultimo_comando) >= (1.0 / CMD_RATE_HZ):
                        send_velocity_body(vx, vy, vz, yaw_rate=yaw_rate)
                        ultimo_comando = time.time()
                        nota = ""
                    elif not en_guided:
                        nota = f" [piloto/{modo_actual}]"
                    else:
                        nota = " [throttled]"

                    t_fin = time.time()
                    latencia_ms = (t_fin - t_inicio) * 1000
                    print(f"🎯 {clase} | conf={confianza:.2f} | "
                          f"dist={depth_val:.1f}m | alt={cur_alt:.1f}m | "
                          f"vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f} | "
                          f"yawR={math.degrees(yaw_rate):+.0f}°/s | "
                          f"lat={latencia_ms:.1f}ms{nota}")

            if not vehiculo_detectado:
                with modo_lock:
                    en_guided = (modo_actual == 'GUIDED')
                if en_guided and (time.time() - ultimo_comando) >= (1.0 / CMD_RATE_HZ):
                    # Sin target: mantener altitud, sin velocidad horizontal
                    with alt_lock:
                        cur_alt = rel_alt_m
                    vz_hold = _clamp(-(TAKEOFF_ALT - cur_alt) * KP_VZ, V_VERT_MAX)
                    send_velocity_body(0.0, 0.0, vz_hold, yaw_rate=0.0)
                    ultimo_comando = time.time()

            with frame_lock:
                latest_frame = annotated
except Exception as e:
    print(f"❌ Error en loop principal: {e}")
finally:
    shutdown_seguro()
