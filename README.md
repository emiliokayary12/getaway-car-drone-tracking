# Getaway Car Identification and Tracking in Drone Surveillance

Sistema autónomo embarcado para detección y seguimiento de vehículos de emergencia
(ambulancias, patrullas y camiones de bomberos) desde un dron. Visión por
computadora en tiempo real sobre **Jetson Orin Nano**, percepción 3D con
**ZED 2** y control de vuelo vía **MAVLink / ArduPilot** sobre un **PixHawk 6C**.

> Proyecto desarrollado en el curso **MR3002B.501 — Diseño e implementación de
> sistemas mecatrónicos** del Tecnológico de Monterrey, en colaboración con
> **Green Tech Innovation (GTI)** como socio formador.

---

## Resultados clave

| Métrica | Resultado | Meta del PSS |
|---|---|---|
| mAP@0.5 (media, 3 clases de emergencia) | **88.6 %** | ≥ 70 % |
| Precisión (media) | 90.4 % | ≥ 80 % |
| Recall (media) | 85.1 % | ≥ 75 % |
| F1-Score (media) | 87.7 % | ≥ 75 % |
| mAP@0.5:0.95 (media) | 74.5 % | — |
| Latencia de inferencia (TensorRT FP16) | **< 30 ms** | < 50 ms |
| Frecuencia del pipeline | **30 Hz** sostenidos | ≥ 20 Hz |
| Distancia de profundidad (ZED 2) | 0.2 – 10 m | — |
| Costo total del prototipo | **$1,421 USD** | — |

Comparado con el equivalente comercial más barato (Skydio Enterprise,
~$50 K USD), el prototipo logra funcionalidad equivalente con una **reducción
del 97.2 %** en costo de hardware.

> **Nota sobre las clases:** el modelo se entrenó con una cuarta clase negativa
> (`other_vehicle`) para reducir falsos positivos; no es clase objetivo y se
> excluye del cálculo de métricas reportadas.

---

## Arquitectura

```
┌────────────────┐   USB 3.0    ┌──────────────────────┐   MAVLink v2   ┌──────────────┐
│   ZED 2        │ ───────────► │  Jetson Orin Nano    │ ◄───────────►  │  PixHawk 6C  │
│  (estéreo 3D)  │              │  YOLO26 + TensorRT   │  /dev/ttyACM0  │  ArduCopter  │
└────────────────┘              │  control + pymavlink │                │     4.5      │
                                │  HTTP stream :8080   │                └──────┬───────┘
                                └──────────────────────┘                       │ PWM
                                                                                ▼
                                                                       ┌──────────────┐
                                                                       │  ESCs × 4    │
                                                                       │  Motores × 4 │
                                                                       └──────────────┘
```

**Pipeline por frame (≈ 30 ms total):**

1. **Captura** — ZED 2 a 30 fps en HD720 + mapa de profundidad
   (`DEPTH_MODE.PERFORMANCE`, unidades en metros).
2. **Detección** — YOLO26 fine-tuneado, exportado a TensorRT FP16, corriendo
   sobre la GPU 1024-core Ampere de la Jetson.
3. **Selección de objetivo** — selección single-target del bounding box con
   mayor confianza (`conf > 0.60`) por frame y consulta de profundidad en su
   centro. No se usa tracker multi-frame (ver Trabajo futuro).
4. **Control** — cálculo del offset 3D al objetivo en frame `BODY_OFFSET_NED`
   y envío de `SET_POSITION_TARGET_LOCAL_NED` al PixHawk (controlador
   proporcional, `Kp = 1.0`).
5. **Visualización** — frame anotado servido por HTTP MJPEG en
   `http://<jetson-ip>:8080`.

---

## Hardware (BOM)

Tipo de cambio: $17.50 MXN/USD (Banco de México, abril 2026).

| # | Componente | Especificación | Proveedor | USD |
|---|---|---|---|---|
| 1 | NVIDIA Jetson Orin Nano 8 GB | 67 TOPS, JetPack 6.2 | Arrow Electronics | $249 |
| 2 | Cámara estéreo ZED 2 | 110° FOV, IMU 6-DOF, 0.3–20 m | Stereolabs | $499 |
| 3 | PixHawk 6C + GPS M8N | ArduCopter 4.x, MAVLink v2, U-Blox M8N | Holybro | $299 |
| 4 | Airframe F450 quadcopter | — | — | $80 |
| 5 | ESCs BLHeli_32 40 A × 4 | DSHOT600 | Amazon | $60 |
| 6 | Batería LiPo 3S 4200 mAh × 2 | XT60, 35C | HobbyKing | $85 |
| 7 | Receptor FlySky FS-iA6B | iBUS | Amazon | $18 |
| 8 | Power Module GM v1.0 | Medición de corriente / 5 V regulado | Holybro | $25 |
| 9 | SSD NVMe 256 GB | Samsung M.2 2280 | Amazon | $69 |
| 10 | Cables, conectores | JST, XT60, Dupont, USB-C, HDMI | Digi-Key | $37 |
|   | **Total** | | | **$1,421** |

Esquema eléctrico, diagrama de software y planos del bracket impreso en 3D
para ZED + Jetson se documentan en el reporte de ingeniería de detalle
(no incluido en este repo).

---

## Software

| Componente | Versión | Licencia |
|---|---|---|
| Python | 3.10 | PSF |
| Ultralytics YOLO | YOLO26 | AGPL-3.0 |
| ZED SDK (`pyzed`) | 4.2 | Stereolabs Community |
| OpenCV | 4.x | Apache 2.0 |
| pymavlink | 2.x | LGPL-3.0 |
| ArduCopter | 4.5 | GPL-3.0 |
| TensorRT | 8.x (FP16) | NVIDIA EULA |

### Simulación (`getaway-sim/`)

Para iterar sin hardware se incluye una **simulación end-to-end en Docker**:
ArduCopter SITL + Gazebo Harmonic + ROS 2 Humble + Ultralytics, con dron F450,
cámara RGB-D (stand-in ZED 2) y un mesh GLB realista de ambulancia. QGroundControl
se conecta vía UDP 14550 para tomar control manual en cualquier momento.

```bash
cd getaway-sim
make build   # ~25 min primera vez
make up      # arranca sitl + sim
```

Stream MJPEG con bbox y HUD en <http://localhost:8080>. Ver
[`getaway-sim/README.md`](getaway-sim/README.md) para detalles completos
(controlador de velocidad, modo manual, estabilización digital, tuning).

**Hallazgos del sim que aplican a hardware real:**
- El controlador de `drone_refactor.py` (`position-target`) provoca overshoot
  agresivo y puede perder el target. La versión del sim usa **velocity-target**
  (`SET_POSITION_TARGET_LOCAL_NED` con `type_mask = 0x07C7`) con caps
  `V_FWD_MAX = 8 m/s`, `V_LAT_MAX = 2 m/s`, `YAW_RATE_MAX = 0.5 rad/s`.
  Pendiente backportar a `drone_refactor.py`.
- Cámara hard-mounted al frame transmite roll/pitch → se aplica counter-rotación
  digital (`cv2.warpAffine`) en el stream para estabilizar el horizonte.
- Modo manual ↔ autónomo se conmuta vía cambio de flight mode en QGC: `GUIDED` =
  script en control; `LOITER`/`ALT_HOLD`/`STABILIZE` = piloto en control. El script
  detecta el modo en el heartbeat y suspende sus comandos en automático.

**Modelo de IA:** YOLO26 fine-tuneado sobre un dataset propio de **2,261
imágenes** de vehículos de emergencia mexicanos (Cruz Roja, IMSS, ERUM, SSP,
Guardia Nacional, Bomberos CDMX y Monterrey), anotadas en Roboflow.
Entrenamiento en Kaggle con 2× T4 GPUs, 200 epochs (ver `train.ipynb`).
Exportado a TensorRT FP16 para inferencia en Jetson.

---

## Instalación (Jetson Orin Nano, JetPack 6.2)

```bash
# 1. Crear venv
python3.10 -m venv .venv && source .venv/bin/activate

# 2. Dependencias base
pip install ultralytics opencv-python numpy pymavlink

# 3. Wheel de pyzed precompilado para aarch64
pip install ./pyzed-4.2-cp310-cp310-linux_aarch64.whl

# 4. ZED SDK (instalador de Stereolabs, fuera de pip)
#    https://www.stereolabs.com/developers/release/
```

El PixHawk se conecta por USB y aparece como `/dev/ttyACM0` a 115200 baud.

---

## Uso

### Vuelo con GPS (recomendado en exterior)

```bash
python drone_refactor.py
```

Flujo:

1. Carga el engine TensorRT y abre la ZED.
2. Conecta con el PixHawk y aplica parámetros (EKF en GPS, geofence 30 m / 10 m,
   failsafes en off porque no hay RC conectado).
3. **Espera GPS lock** (fix ≥ 3D, ≥ 6 satélites) — aborta si no llega en 30 s.
4. Calibra el barómetro al suelo (`MAV_CMD_PREFLIGHT_CALIBRATION`).
5. Cambia a `GUIDED`, arma motores y despega a 2 m.
6. Entra al loop a 30 Hz: detección → cálculo de offset al objetivo →
   `SET_POSITION_TARGET_LOCAL_NED` en frame `BODY_OFFSET_NED`.
7. Stream HTTP de video disponible en `http://<jetson-ip>:8080`.
8. **Ctrl+C** → modo `LAND` → espera touchdown (auto-disarm) → cierre limpio.
   Un segundo Ctrl+C fuerza `disarm` inmediato con magic `21196`.


## Seguridad

- **Nunca volar sin hélices puestas en banco** — los tests cubren el
  comportamiento del FC sin alimentación a motores.
- **El botón de safety físico del PixHawk no se anula por software.** Los
  parámetros `BRD_SAFETYENABLE` y `MOT_SAFE_DISARM` se dejan en sus valores por
  defecto.
- En `drone_refactor.py`, `FENCE_ENABLE=1`, `FENCE_RADIUS=30 m` y
  `FENCE_ALT_MAX=10 m`. Con `FENCE_ACTION=4` (Brake), el dron frena al borde de
  la geofence en lugar de RTL/LAND.
- El handler de `SIGINT` ejecuta `LAND` y, si no hay touchdown en 8 s, envía
  force-disarm con el magic number `21196` de ArduPilot.

---

## Contexto académico

| Campo | Valor |
|---|---|
| Curso | MR3002B.501 — Diseño e implementación de sistemas mecatrónicos |
| Institución | Instituto Tecnológico y de Estudios Superiores de Monterrey |
| Semestre | Feb-Jun 2026 |
| Cliente / Socio formador | Green Tech Innovation (GTI) |
| Duración | 10 semanas (23/03/2026 — 12/06/2026) |
| Profesores | Dra. Aline Drivet · Dr. Luis E. Garza C. · Dr. Riccardo Cespi · Dr. Bernardino Castillo-Toledo |

---

## Equipo 6

- **Luis Fernando Sandoval Silva** — A01742322
- **Emilio Arturo Kayary Ramos** — A01552526
- **Juan Alberto Flores Mejía** — A00833988
- **Andrés Sojo Hinojosa** — A00573835

---

## Trabajo futuro

- Habilitar control completo del eje Z para seguimiento 3D pleno.
- Backportar a `drone_refactor.py` el controlador de velocidad con saturación
  validado en simulación, que mitiga el overshoot del esquema de posición.
- Incorporar filtro de Kalman en la estimación de profundidad.
- Extender a seguimiento multi-objetivo (multi-target tracking).
- Pruebas con vehículo objetivo en movimiento.

---

## Licencia

Código propio bajo MIT. Dependencias bajo sus respectivas licencias
(ver tabla de software arriba). El dataset de vehículos de emergencia
mexicanos es propiedad intelectual del equipo y del cliente GTI; consultar
antes de redistribuir.

## Referencias

Bibliografía técnica completa en el reporte de ingeniería de detalle.
Referencias principales:

- R. Sapkota, R. H. Cheppally, A. Sharda & M. Karkee — *YOLO26: Key
  Architectural Enhancements and Performance Benchmarking for Real-Time Object
  Detection*, arXiv:2509.25164, 2025.
- R. Khanam & M. Hussain — *YOLOv11: An Overview of the Key Architectural
  Enhancements*, arXiv:2410.17725, 2024.
- ArduPilot Dev Team — *ArduCopter Documentation: GUIDED Mode and MAVLink
  Interface*, 2024. <https://ardupilot.org/copter/docs/guided-mode.html>
- Stereolabs — *ZED 2 Stereo Camera Technical Documentation*, 2024.
  <https://www.stereolabs.com/docs/>
- M. Jovanovic — *Visual Navigation and Control of UAV Using MAVLink
  Protocol and PID Controller*, PeerJ Computer Science, 2025.
  DOI: 10.7717/peerj-cs.2587
