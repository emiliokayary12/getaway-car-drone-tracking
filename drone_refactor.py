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

# ── HEARTBEAT + WATCHDOG + LECTOR DE ATTITUDE ──────────────────
def heartbeat_y_watchdog():
    global heading_actual, modo_actual
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

        time.sleep(0.1)

threading.Thread(target=heartbeat_y_watchdog, daemon=True).start()

# ── COMANDOS DE POSICIÓN ───────────────────────────────────────
def send_position_body_offset(forward, right, down, yaw=None):
    """
    Comando de posición relativa al cuerpo del dron (BODY_OFFSET_NED).
    forward: + adelante (m)
    right:   + derecha  (m)
    down:    + abajo    (m)
    yaw:     rad absoluto en frame mundial; None = no cambiar yaw.
    """
    type_mask = 0b0000111111111000   # solo posición
    if yaw is not None:
        type_mask &= ~0b010000000000  # también yaw
        yaw_val = yaw
    else:
        yaw_val = 0.0
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        type_mask,
        forward, right, down,
        0, 0, 0,
        0, 0, 0,
        yaw_val, 0
    )

def hover():
    """Mantén posición actual (offset 0 en body-frame)."""
    send_position_body_offset(0.0, 0.0, 0.0)

# ── CÁLCULO DEL OFFSET AL VEHÍCULO OBJETIVO ────────────────────
def calcular_offset_objetivo(cx, cy, depth_val):
    """
    A partir del centro del bbox y depth de la ZED, devuelve el offset
    en frame body (forward, right, down) al que ir para quedar a
    DIST_OBJETIVO del vehículo, y el yaw relativo necesario para
    centrar el objetivo horizontalmente.
    Asume la ZED montada mirando hacia adelante alineada con +X body.
    """
    dx_cam = (cx - CX_ZED) / FX_ZED   # tan(ángulo horizontal)
    dy_cam = (cy - CY_ZED) / FY_ZED   # tan(ángulo vertical)
    z_cam = depth_val                  # adelante
    x_cam = dx_cam * depth_val         # derecha
    y_cam = dy_cam * depth_val         # abajo
    forward = z_cam - DIST_OBJETIVO    # acercarse manteniendo distancia
    right   = x_cam
    down    = y_cam
    yaw_rel = math.atan2(x_cam, z_cam)
    return forward, right, down, yaw_rel

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

                    forward, right, down, yaw_rel = calcular_offset_objetivo(
                        cx, cy, depth_val
                    )
                    with heading_lock:
                        yaw_abs = heading_actual + yaw_rel
                    with modo_lock:
                        en_guided = (modo_actual == 'GUIDED')

                    if en_guided:
                        send_position_body_offset(
                            forward, right, down, yaw=yaw_abs
                        )
                        ultimo_comando = time.time()
                        nota = ""
                    else:
                        # Piloto tiene control via RC — no mandar comandos
                        nota = f" [piloto/{modo_actual}]"

                    t_fin = time.time()
                    latencia_ms = (t_fin - t_inicio) * 1000
                    print(f"🎯 {clase} | conf={confianza:.2f} | "
                          f"dist={depth_val:.1f}m | "
                          f"f={forward:+.2f} r={right:+.2f} d={down:+.2f} | "
                          f"yaw_rel={math.degrees(yaw_rel):+.1f}° | "
                          f"lat={latencia_ms:.1f}ms{nota}")

            if not vehiculo_detectado:
                with modo_lock:
                    en_guided = (modo_actual == 'GUIDED')
                if en_guided and time.time() - ultimo_comando > 0.5:
                    hover()
                    ultimo_comando = time.time()

            with frame_lock:
                latest_frame = annotated
except Exception as e:
    print(f"❌ Error en loop principal: {e}")
finally:
    shutdown_seguro()
