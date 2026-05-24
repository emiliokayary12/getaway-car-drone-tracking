"""
ZED 2 SDK shim — emulates the subset of `pyzed.sl` used by drone_refactor.py
by pulling frames + depth from ROS 2 topics published by the Gazebo bridge.

Topics (remapped by entry_sim.sh):
    /camera/image  — sensor_msgs/Image  (rgb8 or bgr8)
    /camera/depth  — sensor_msgs/Image  (32FC1, meters)
    /camera/info   — sensor_msgs/CameraInfo

Public API mimicked:
    sl.Camera, sl.InitParameters, sl.Mat, sl.RuntimeParameters,
    sl.ERROR_CODE, sl.RESOLUTION, sl.DEPTH_MODE, sl.UNIT,
    sl.VIEW, sl.MEASURE,
    Camera.open / grab / retrieve_image / retrieve_measure /
    get_camera_information / close,
    Mat.get_data / get_value.
"""
import threading
import time
from enum import IntEnum

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


# ── enums / constants mirrored from pyzed.sl ───────────────────
class ERROR_CODE(IntEnum):
    SUCCESS = 0
    FAILURE = 1


class RESOLUTION(IntEnum):
    HD720 = 0


class DEPTH_MODE(IntEnum):
    PERFORMANCE = 0


class UNIT(IntEnum):
    METER = 0


class VIEW(IntEnum):
    LEFT = 0


class MEASURE(IntEnum):
    DEPTH = 0


# ── ROS 2 bridge node (singleton) ──────────────────────────────
class _ZedBridge(Node):
    def __init__(self):
        super().__init__("zed_sim_bridge")
        self.br = CvBridge()
        self.rgb = None
        self.depth = None
        self.info = None
        self.lock = threading.Lock()
        self.create_subscription(Image, "/camera/image",
                                 self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/depth",
                                 self._on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/info",
                                 self._on_info, qos_profile_sensor_data)

    def _on_rgb(self, msg):
        img = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.rgb = img

    def _on_depth(self, msg):
        d = self.br.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        with self.lock:
            self.depth = d

    def _on_info(self, msg):
        with self.lock:
            self.info = msg


_bridge_singleton = None
_spin_thread = None


def _start_bridge():
    global _bridge_singleton, _spin_thread
    if _bridge_singleton is not None:
        return _bridge_singleton
    if not rclpy.ok():
        rclpy.init()
    _bridge_singleton = _ZedBridge()
    exe = rclpy.executors.SingleThreadedExecutor()
    exe.add_node(_bridge_singleton)
    _spin_thread = threading.Thread(target=exe.spin, daemon=True)
    _spin_thread.start()
    return _bridge_singleton


# ── Mat shim ───────────────────────────────────────────────────
class Mat:
    def __init__(self):
        self._data = None      # np.ndarray (HxWx3 BGR for RGB; HxW float32 for depth)
        self._kind = None      # "rgb" or "depth"

    def get_data(self):
        if self._data is None:
            return np.zeros((720, 1280, 4), dtype=np.uint8)
        if self._kind == "rgb":
            h, w, _ = self._data.shape
            out = np.zeros((h, w, 4), dtype=np.uint8)
            out[:, :, :3] = self._data
            out[:, :, 3] = 255
            return out
        return self._data

    def get_value(self, x, y):
        if self._kind != "depth" or self._data is None:
            return ERROR_CODE.FAILURE, float("nan")
        h, w = self._data.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return ERROR_CODE.FAILURE, float("nan")
        v = float(self._data[int(y), int(x)])
        return ERROR_CODE.SUCCESS, v


# ── InitParameters / RuntimeParameters ─────────────────────────
class InitParameters:
    def __init__(self):
        self.camera_resolution = RESOLUTION.HD720
        self.camera_fps = 30
        self.depth_mode = DEPTH_MODE.PERFORMANCE
        self.coordinate_units = UNIT.METER


class RuntimeParameters:
    pass


# ── Camera info structures ─────────────────────────────────────
class _CalibLeft:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


class _CalibParams:
    def __init__(self, left):
        self.left_cam = left


class _CameraConfig:
    def __init__(self, calib):
        self.calibration_parameters = calib


class _CameraInfo:
    def __init__(self, config):
        self.camera_configuration = config


# ── Main Camera class ──────────────────────────────────────────
class Camera:
    def __init__(self):
        self._bridge = None
        self._info_cached = None

    def open(self, init_params):
        self._bridge = _start_bridge()
        # Wait up to 10s for first frame + camera_info
        t0 = time.time()
        while time.time() - t0 < 30.0:
            with self._bridge.lock:
                ok_img = self._bridge.rgb is not None
                ok_info = self._bridge.info is not None
            if ok_img and ok_info:
                return ERROR_CODE.SUCCESS
            time.sleep(0.1)
        return ERROR_CODE.FAILURE

    def get_camera_information(self):
        with self._bridge.lock:
            ci = self._bridge.info
        if ci is None:
            # Fallback: derive from FOV 100° / 1280×720
            fx = 1280 / (2 * np.tan(np.deg2rad(100) / 2))
            calib = _CalibLeft(fx, fx, 640.0, 360.0)
        else:
            k = ci.k  # 3×3 row-major
            calib = _CalibLeft(fx=k[0], fy=k[4], cx=k[2], cy=k[5])
        return _CameraInfo(_CameraConfig(_CalibParams(calib)))

    def grab(self, runtime):
        with self._bridge.lock:
            ok = self._bridge.rgb is not None and self._bridge.depth is not None
        return ERROR_CODE.SUCCESS if ok else ERROR_CODE.FAILURE

    def retrieve_image(self, mat, view):
        with self._bridge.lock:
            img = self._bridge.rgb
        if img is None:
            return ERROR_CODE.FAILURE
        mat._data = img.copy()
        mat._kind = "rgb"
        return ERROR_CODE.SUCCESS

    def retrieve_measure(self, mat, measure):
        with self._bridge.lock:
            d = self._bridge.depth
        if d is None:
            return ERROR_CODE.FAILURE
        mat._data = d.copy()
        mat._kind = "depth"
        return ERROR_CODE.SUCCESS

    def close(self):
        try:
            if _bridge_singleton is not None:
                _bridge_singleton.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
