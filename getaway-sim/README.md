# getaway-sim

Simulación end-to-end del proyecto **getaway-car**: dron F450 con cámara estéreo (stand-in de ZED 2) + modelo YOLO + ArduCopter SITL, todo en Docker.

Reproduce el flujo de `drone_refactor.py` sin tocar hardware: el script de misión corre contra **ArduCopter SITL** vía MAVLink UDP, recibe frames + depth de **Gazebo Harmonic** vía `ros_gz_bridge`, y ejecuta la misma lógica de detección → comando `SET_POSITION_TARGET_LOCAL_NED` que en vuelo real.

## Arquitectura

```
┌──────────────── host (nvidia-docker) ────────────────┐
│                                                       │
│  ┌─────────────┐         ┌──────────────────────┐    │
│  │  getaway-   │ UDP 9002│   getaway-sim         │   │
│  │  sitl       │◄────────┤   Gazebo Harmonic     │   │
│  │             │ UDP 9003│   + ardupilot_gazebo  │   │
│  │ ArduCopter  ├────────►│   + ros_gz_bridge     │   │
│  │ SITL        │         │   + ROS 2 Humble      │   │
│  └──────┬──────┘         │   + Ultralytics YOLO  │   │
│         │ UDP 14551      │   + drone_refactor_sim│   │
│         └───────────────►│                       │   │
│                          └──────────────────────┘    │
└───────────────────────────────────────────────────────┘
                              │
                              ▼
                  http://localhost:8080  (MJPEG stream con anotaciones)
```

## Layout

```
getaway-sim/
├── docker-compose.yml
├── docker/
│   ├── Dockerfile.sitl        # ArduCopter SITL (Copter-4.5.7)
│   └── Dockerfile.sim         # Gazebo + ROS 2 Humble + Ultralytics
├── gazebo/
│   ├── worlds/getaway.sdf     # Mundo con F450 + ambulancia
│   └── models/iris_with_camera/{model.config,model.sdf}
├── scripts/
│   ├── entry_sim.sh           # Lanza Gazebo + bridge + misión
│   ├── zed_sim.py             # Shim pyzed.sl ← ROS 2 topics
│   └── drone_refactor_sim.py  # Port simulado del script original
├── models/                    # Coloca aquí tu best.pt (o usa yolov8n.pt)
└── Makefile
```

## Requisitos

- Linux con **nvidia-driver ≥ 535**, **nvidia-container-toolkit**.
- **Docker Engine ≥ 24** y `docker compose` v2.
- ~12 GB libres (imágenes + build de ArduPilot).
- GPU NVIDIA (Ampere o más reciente recomendado para YOLO en tiempo real).

## Build

```bash
cd getaway-sim
make build      # ~25 min la primera vez
```

## Run

Dos terminales (o `make up`):

```bash
# Terminal 1 — SITL
make sitl
# Espera a "APM: AHRS: EKF3 IMU0 is using GPS"

# Terminal 2 — Gazebo + visión + misión
make sim
```

Abre el stream en `http://localhost:8080`. Verás bbox + depth + comandos enviados al FC.

## Modelo YOLO

Por defecto `drone_refactor_sim.py` busca `/sim/models/best.pt`. Si no existe, cae a `yolov8n.pt` (COCO) — útil para validar pipeline, pero **no detectará ambulancias específicas**. Para validación realista coloca tu modelo entrenado:

```bash
cp /ruta/a/best.pt models/best.pt
```

## Validaciones rápidas

1. **¿SITL armó?** En logs `make sitl` debe aparecer `ARMED` y `Takeoff complete`.
2. **¿Frames llegan?** En `make sim`: `✅ Intrínsecos sim: fx=...`.
3. **¿YOLO detecta?** Línea `🎯 ambulance c=... d=5.2m ...` cada vez que la bbox supera el umbral.
4. **¿FC obedece?** En `make sitl` (MAVProxy console) la posición XY del dron debe acercarse al objetivo (ubicado en `10, 0, 0`).

## Tuning rápido

| Quiero... | Editar |
|---|---|
| Otro modelo de objetivo (auto, persona) | `gazebo/worlds/getaway.sdf` ← `<model name="ambulance">` |
| Cambiar masa/inertia del F450 | `gazebo/models/iris_with_camera/model.sdf` ← `<inertial>` |
| Aumentar FOV de cámara | `model.sdf` ← `<horizontal_fov>` (1.7453 rad ≈ 100°) |
| Cambiar distancia objetivo | `scripts/drone_refactor_sim.py` ← `DIST_OBJETIVO` |
| Umbral de confianza | mismo archivo, `if conf > 0.40` |
| Altura de takeoff | mismo, `TAKEOFF_ALT` |

## Diferencias vs. vuelo real

| Aspecto | Real | Sim | Impacto |
|---|---|---|---|
| Depth | ZED estéreo, ruido + holes | Ground-truth float32 perfecto | Sim es **optimista** — añade ruido si quieres robustez probada |
| Latencia visión | Jetson Orin + TensorRT FP16 (~25 ms) | PC + PyTorch FP32 (variable) | No comparable; perfila aparte |
| GPS | Real, HDOP variable | SITL ideal | Sin efectos de multipath/dropout |
| Wind / IMU bias | Real | Cero por defecto | Configurable con `SIM_WIND_*` params |

## Troubleshooting

**"connection refused" en SITL UDP 14551** → el container `sitl` aún no terminó de arrancar. Espera al log "Frame: QUAD" antes de lanzar `sim`.

**Gazebo se queda en negro** → falta `xhost +local:docker` o falta el runtime nvidia. Verifica `docker info | grep -i runtime`.

**Plugin "ArduPilotPlugin" no encontrado** → `GZ_SIM_SYSTEM_PLUGIN_PATH` no apunta a `/root/gz_ws/install/ardupilot_gazebo/lib`. Revisa `docker-compose.yml`.

**YOLO no usa GPU** → `docker compose exec sim python3 -c "import torch; print(torch.cuda.is_available())"` debe ser `True`. Si no, falta `runtime: nvidia` o nvidia-container-toolkit.

**SITL arma pero no despega** → revisa que `ARMING_CHECK=0` se haya aplicado y que `MAV_FRAME_BODY_OFFSET_NED` no esté empujando hacia abajo (depth corruptos).

## Próximos pasos sugeridos

- Mover la ambulancia con un plugin `gz-sim-velocity-control-system` para probar tracking dinámico.
- Añadir múltiples target classes (police_car, fire_truck) con SDFs distintos.
- Inyectar ruido de depth y dropouts para acercar al comportamiento ZED real.
- Grabar bag de `/camera/image` + MAVLink para regresión.
