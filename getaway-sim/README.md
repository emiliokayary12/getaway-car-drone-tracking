# getaway-sim

Simulación end-to-end del proyecto **getaway-car**: dron F450 con cámara estéreo (stand-in de ZED 2) + modelo YOLO + ArduCopter SITL, todo en Docker. Permite iterar sobre lógica de detección y control sin tocar hardware.

Reproduce el flujo de `drone_refactor.py` (versión real, raíz del repo) corriendo contra **ArduCopter SITL** vía MAVLink UDP, recibiendo frames + depth de **Gazebo Harmonic** vía `ros_gz_bridge`. Conecta también con **QGroundControl** (UDP 14550) para tomar control manual y supervisar el vuelo.

## Arquitectura

```
┌──────────────── host (nvidia-docker) ────────────────┐
│                                                       │
│  ┌─────────────┐         ┌──────────────────────┐    │
│  │  getaway-   │ UDP 9002│   getaway-sim         │   │
│  │  sitl       │◄────────┤   Gazebo Harmonic     │   │
│  │             │ UDP 9003│   + ardupilot_gazebo  │   │
│  │ ArduCopter  ├────────►│   + ros_gz_bridge     │   │
│  │  4.5.7      │         │   + ROS 2 Humble      │   │
│  │ + MAVProxy  │ UDP14551│   + Ultralytics YOLO  │   │
│  └──────┬──────┘────────►│   + drone_refactor_sim│   │
│         │ UDP 14550      │                       │   │
│         ▼                └──────────────────────┘    │
│   QGroundControl (host)                              │
└───────────────────────────────────────────────────────┘
                              │
                              ▼
                  http://localhost:8080  (MJPEG con HUD + bbox)
```

## Layout

```
getaway-sim/
├── docker-compose.yml
├── docker/
│   ├── Dockerfile.sitl              # ArduCopter SITL (Copter-4.5.7) + MAVProxy
│   └── Dockerfile.sim               # Gazebo Harmonic + ROS 2 Humble + Ultralytics
├── gazebo/
│   ├── worlds/getaway.sdf           # Mundo: F450 + ambulance_mesh
│   └── models/
│       ├── iris_with_camera/        # F450 con cámara RGB + depth a 30 Hz
│       └── ambulance_mesh/          # GLB realista + VelocityControl plugin
├── scripts/
│   ├── entry_sim.sh                 # Lanza Xvfb + Gazebo + ros_gz_bridge + misión
│   ├── zed_sim.py                   # Shim pyzed.sl ← ROS 2 topics
│   └── drone_refactor_sim.py        # Misión: YOLO + control de velocidad + HUD
├── models/best.pt                   # YOLO custom: ambulance / fire_truck / police_car / other
└── Makefile
```

## Requisitos

- Linux con **nvidia-driver ≥ 535**, **nvidia-container-toolkit**.
- **Docker Engine ≥ 24** y `docker compose` v2.
- ~12 GB libres (imágenes + build de ArduPilot).
- GPU NVIDIA (Ampere o más reciente recomendado para YOLO en tiempo real).
- **QGroundControl** AppImage (opcional, para modo manual). No incluido en repo por tamaño; bájalo de <https://qgroundcontrol.com>.

## Build

```bash
cd getaway-sim
make build      # ~25 min la primera vez
```

## Run

```bash
make up         # arranca sitl + sim en background
# o en dos terminales:
make sitl       # ArduCopter SITL + MAVProxy
make sim        # Gazebo + bridge + misión + YOLO
```

Abre stream: `http://localhost:8080`. Verás bbox por clase + HUD (modo de vuelo, altitud, AUTO/MANUAL badge).

### Control manual con QGroundControl

```bash
./QGroundControl-x86_64.AppImage
```

QGC se conecta automáticamente al broker MAVProxy en UDP `127.0.0.1:14550`. Para que el script ceda control:

1. **Modo `GUIDED`** → script en control (autónomo, persigue ambulancia).
2. **Cualquier otro modo** (`LOITER`, `ALT_HOLD`, `STABILIZE`) → script suspende comandos. Tú vuelas.

Toggle libre entre auto/manual cambiando modo en QGC. HUD del stream muestra estado actual.

> **Tip:** Si "Guided" no aparece en el dropdown de QGC, abre *Vehicle Setup → Flight Modes* y activa **Edit Displayed Flight Modes** para mostrar todos. `Auto` no funciona sin misión cargada — usa siempre `Guided` para autónomo.

## Modelo YOLO

`scripts/drone_refactor_sim.py` carga `/sim/models/best.pt` (incluido en repo: 4 clases — `ambulance`, `fire_truck`, `police_car`, `other_vehicle`). Filtro `ALLOWED_CLASSES = {0, 1, 2}` ignora `other_vehicle`.

Si `best.pt` no existe, cae a `yolov8n.pt` (COCO) con filtro `COCO_VEHICLE_IDS = {2, 5, 7}` (car/bus/truck) para que pipeline siga validable.

## Controlador (velocity setpoints)

Hallazgo del sim: **position-target** (`SET_POSITION_TARGET_LOCAL_NED` con offsets X/Y) causaba overshoot agresivo + pérdida de target. Se migró a **velocity-target** (mismo MAVLink msg, type_mask `0x07C7`).

```python
KP_VX = 0.4   ;  V_FWD_MAX  = 8.0    # m/s   (~29 km/h)
KP_VY = 1.5   ;  V_LAT_MAX  = 2.0
KP_VZ = 0.4   ;  V_VERT_MAX = 1.0
KP_YAW_RATE = 0.8  ;  YAW_RATE_MAX = 0.5   # rad/s (~28°/s)
DIST_OBJETIVO = 12.0   # m, distancia objetivo
TAKEOFF_ALT   = 6.0
MIN_ALT_M     = 3.0
CMD_RATE_HZ   = 10.0
DEAD_ZONE_PX  = 60
```

P-only por canal:
- **vx**: usa profundidad ZED → `Kp * (depth - DIST_OBJETIVO)`, saturado a `V_FWD_MAX`.
- **vy / yaw_rate**: error pixel-x del bbox center, dead-zone para evitar cazar ruido.
- **vz**: error pixel-y, con floor `MIN_ALT_M` para no chocar suelo.

ArduCopter aplica su cascade PID interno (`PSC_*`, `VEL_*`, `ATC_*`) por encima → comportamiento estable sin tunear ganancias del FC.

## Ambulancia

Mesh GLB realista (`gazebo/models/ambulance_mesh/`). Visual rotado `+90° X` para corregir Y-up de glTF. Collision box `5.0 × 2.2 × 2.4 m`, masa `2500 kg`. Plugin `VelocityControl` recibe `gz.msgs.Twist` en `/model/ambulance/cmd_vel`.

`entry_sim.sh` la mueve a **5 m/s (18 km/h)** con curva suave (`angular.z = 0.05`). Spawn a `(12, 0, 1.2)` para evitar choque inicial con dron.

## Estabilización digital de cámara

Cámara está hard-mounted al frame del dron → roll/pitch transmiten directo y la imagen tiembla. Se aplica counter-rotación digital antes de servir el stream:

```python
# en drone_refactor_sim.py
M = cv2.getRotationMatrix2D((w/2, h/2), -roll_deg, 1.0)
M[1, 2] -= pitch_offset_px
frame = cv2.warpAffine(frame, M, (w, h))
```

Roll y pitch llegan vía heartbeat MAVLink (`ATTITUDE`). Resultado: horizonte estable bajo maniobras agresivas.

## Validaciones rápidas

1. **¿SITL armó?** Logs `make sitl`: `ARMED` y `Takeoff complete`.
2. **¿Frames llegan?** `make sim`: `✅ Intrínsecos sim: fx=...`.
3. **¿YOLO detecta?** `🎯 ambulance c=0.87 d=11.8m vx=-0.08 vy=0.02 ...`.
4. **¿FC obedece?** En QGC, posición del dron debe converger a `DIST_OBJETIVO` de la ambulancia.
5. **¿Manual funciona?** Cambia a `LOITER` en QGC; script imprime `⏸ modo manual`.

## Tuning rápido

| Quiero... | Editar | Valor actual |
|---|---|---|
| Distancia de chase | `drone_refactor_sim.py` ← `DIST_OBJETIVO` | `12.0 m` |
| Velocidad máxima | mismo ← `V_FWD_MAX` | `8.0 m/s` |
| Altura takeoff | mismo ← `TAKEOFF_ALT` | `6.0 m` |
| Rate de comandos | mismo ← `CMD_RATE_HZ` | `10 Hz` |
| Ganancias P | mismo ← `KP_VX/VY/VZ/YAW_RATE` | ver tabla arriba |
| Vel. ambulancia | `entry_sim.sh` ← `linear.x` en `cmd_vel` | `5.0 m/s` |
| Otra clase target | `drone_refactor_sim.py` ← `ALLOWED_CLASSES` | `{0, 1, 2}` |
| FOV cámara | `gazebo/models/iris_with_camera/model.sdf` ← `<horizontal_fov>` | `1.7453 rad` (100°) |

## Diferencias vs. vuelo real

| Aspecto | Real | Sim | Impacto |
|---|---|---|---|
| Depth | ZED estéreo, ruido + holes | Ground-truth float32 perfecto | Sim es **optimista** — añade ruido si quieres robustez probada |
| Latencia visión | Jetson Orin + TensorRT FP16 (~25 ms) | PC + PyTorch FP32 (variable) | No comparable; perfila aparte |
| GPS | Real, HDOP variable | SITL ideal | Sin efectos de multipath/dropout |
| Wind / IMU bias | Real | Cero por defecto | Configurable con `SIM_WIND_*` params |
| Controlador script | Position-target (overshoot) | **Velocity-target** (estable) | Backport pendiente a `drone_refactor.py` |
| Estabilización | Gimbal mecánico | Digital (warpAffine) | Real necesita gimbal o cámara con IBIS |

## Troubleshooting

**"connection refused" en SITL UDP 14551** → container `sitl` aún arrancando. Espera log "Frame: QUAD" antes de lanzar `sim`.

**Gazebo se queda en negro** → falta `xhost +local:docker` o falta runtime nvidia. Verifica `docker info | grep -i runtime`.

**`ArduPilotPlugin` no encontrado** → `GZ_SIM_SYSTEM_PLUGIN_PATH` no apunta a `/root/gz_ws/install/ardupilot_gazebo/lib`. Revisa `docker-compose.yml`.

**YOLO no usa GPU** → `docker compose exec sim python3 -c "import torch; print(torch.cuda.is_available())"` debe ser `True`. Si no, falta `runtime: nvidia` o nvidia-container-toolkit.

**Ambulancia rotada / dentro del suelo** → GLB es Y-up. SDF aplica `<pose>0 0 0 1.5708 0 0</pose>` en visual + spawn Z=1.2. Si cambias mesh, verifica orientación.

**`MAV_CMD_DO_SET_MODE failed` en QGC** → estás intentando `AUTO` sin misión. Usa `GUIDED` para autónomo o `LOITER`/`ALT_HOLD` para manual.

**Drone overshoot agresivo** → bajar `KP_VY` y `KP_YAW_RATE` en `drone_refactor_sim.py`, o subir `DEAD_ZONE_PX`.

**Cámara sigue temblorosa** → bajar `CMD_RATE_HZ` (menos comandos → menos pitch/roll oscilación), o aumentar fuerza de estabilización digital.

## Próximos pasos

- Backportar velocity-target controller a `drone_refactor.py` (hardware real).
- Aumentar `CMD_RATE_HZ` a 20 + extrapolación de posición ambulancia (Kalman 1D).
- Tunear `WPNAV_SPEED`, `WPNAV_ACCEL`, `PSC_VELXY_*` en SITL para match exact al perfil dinámico real.
- Inyectar ruido de depth y dropouts para acercar a comportamiento ZED real.
- Grabar bag de `/camera/image` + MAVLink para regresión automatizada.
- Múltiples target classes simultáneos con tracking (BoT-SORT, ya usado en hardware).
