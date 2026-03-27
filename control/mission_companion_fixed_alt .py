# Run mission companion with fixed altitude (no adaptive DDQN control)
# python3 ~/ros2_ws/src/uav_iqa_ddqn/mission_companion_fixed_alt.py \
#   --connect_url udpin://0.0.0.0:14540 \
#   --camera_topic /iris/gazebo_distorted_preview/image_raw \
#   --wind_topic /uav/wind_speed \
#   --iqa_py ~/ros2_ws/src/uav_iqa_ddqn/iqa/iqa_model.py \
#   --save_dir ~/uav_results/fixed_alt/run1 \
#   --start_alt 21 \
#   --verbose_status

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, math, asyncio, argparse, threading, importlib.util
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32

from mavsdk import System

# ------------------------- utils -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_s() -> float:
    return time.time()

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

# ------------------------- IQA loader -------------------------
def load_iqa_process_bgr_image(iqa_py_path: str):
    iqa_py_path = os.path.expanduser(iqa_py_path)
    if not os.path.isfile(iqa_py_path):
        raise FileNotFoundError(f"IQA python file not found: {iqa_py_path}")
    spec = importlib.util.spec_from_file_location("iqa_model_dyn_fixed", iqa_py_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["iqa_model_dyn_fixed"] = m
    spec.loader.exec_module(m)
    if hasattr(m, "process_bgr_image"):
        return m.process_bgr_image
    raise AttributeError(f"{iqa_py_path} must define process_bgr_image(bgr)->dict")

def parse_iqa_dict(out: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    if not isinstance(out, dict):
        return float("nan"), {}
    if "quality_score_percent" in out:
        q = safe_float(out.get("quality_score_percent"), float("nan"))
    elif "quality_%" in out:
        q = safe_float(out.get("quality_%"), float("nan"))
    else:
        q = float("nan")
    metrics: Dict[str, float] = {}
    for k, v in out.items():
        if isinstance(v, (int, float)):
            metrics[k] = float(v)
    return q, metrics

# ------------------------- ROS2 cache -------------------------
class MultiCache(Node):
    def __init__(self, camera_topic: str, wind_topic: Optional[str]):
        super().__init__("uav_iqa_cache_fixed")
        self.br = CvBridge()
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._lock = threading.Lock()
        self._last_bgr = None
        self._last_cam_t = 0.0
        self._cam_frames = 0
        self._last_wind = None
        self._last_wind_t = 0.0

        self.create_subscription(Image, camera_topic, self._cam_cb, qos)
        if wind_topic:
            self.create_subscription(Float32, wind_topic, self._wind_cb, qos)

    def _cam_cb(self, msg: Image):
        try:
            bgr = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        with self._lock:
            self._last_bgr = bgr
            self._last_cam_t = now_s()
            self._cam_frames += 1

    def _wind_cb(self, msg: Float32):
        with self._lock:
            self._last_wind = float(msg.data)
            self._last_wind_t = now_s()

    def get_camera(self) -> Tuple[Optional[np.ndarray], float, int]:
        with self._lock:
            if self._last_bgr is None:
                return None, 0.0, self._cam_frames
            return self._last_bgr.copy(), self._last_cam_t, self._cam_frames

    def get_wind(self) -> Tuple[Optional[float], float]:
        with self._lock:
            return self._last_wind, self._last_wind_t

def start_ros_cache(camera_topic: str, wind_topic: Optional[str]):
    rclpy.init(args=None)
    node = MultiCache(camera_topic, wind_topic)
    stop_evt = threading.Event()

    def _spin():
        while rclpy.ok() and not stop_evt.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)

    thr = threading.Thread(target=_spin, daemon=True)
    thr.start()

    def stop():
        stop_evt.set()
        try: node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass

    return node, stop

# ------------------------- telemetry state -------------------------
@dataclass
class TelemetryState:
    armed: bool = False
    in_air: bool = False
    rel_alt_m: float = float("nan")
    abs_alt_m: float = float("nan")
    lat: float = float("nan")
    lon: float = float("nan")
    flight_mode: str = "UNKNOWN"
    waypoint_index: int = -1

# ------------------------- fixed companion -------------------------
class FixedAltCompanion:
    def __init__(self, args):
        self.args = args
        self.process_bgr_image = load_iqa_process_bgr_image(args.iqa_py)

        self.save_dir = os.path.expanduser(args.save_dir)
        ensure_dir(self.save_dir)

        self.cache, self.stop_ros = start_ros_cache(args.camera_topic, args.wind_topic)

        self.drone: Optional[System] = None
        self.tel = TelemetryState()

        self._shot_idx = 0
        self._last_shot_lat = None
        self._last_shot_lon = None
        self._last_warn_no_frame_t = 0.0

        self._rtl_seen = False

    async def _connect(self):
        self.drone = System()
        await self.drone.connect(system_address=self.args.connect_url)
        async for st in self.drone.core.connection_state():
            if st.is_connected:
                break
        print(f"[LINK] Connected via {self.args.connect_url}")

    async def _refresh_telemetry_once(self):
        try:
            async for a in self.drone.telemetry.armed():
                self.tel.armed = bool(a); break
        except Exception: pass

        try:
            async for ia in self.drone.telemetry.in_air():
                self.tel.in_air = bool(ia); break
        except Exception: pass

        try:
            async for pos in self.drone.telemetry.position():
                self.tel.lat = float(pos.latitude_deg)
                self.tel.lon = float(pos.longitude_deg)
                self.tel.abs_alt_m = float(pos.absolute_altitude_m)
                self.tel.rel_alt_m = float(pos.relative_altitude_m)
                break
        except Exception: pass

        try:
            async for fm in self.drone.telemetry.flight_mode():
                self.tel.flight_mode = str(fm).replace("FlightMode.", ""); break
        except Exception: pass

        try:
            async for idx in self.drone.mission_raw.current_mission_item():
                self.tel.waypoint_index = int(idx); break
        except Exception: pass

    def _get_wind(self) -> Optional[float]:
        w, t = self.cache.get_wind()
        if w is None:
            return None
        if (now_s() - t) > self.args.max_wind_age_s:
            return None
        return float(w)

    async def _wait_for_fresh_frame(self, max_wait: float = 5.0) -> Optional[np.ndarray]:
        start_time = now_s()
        while now_s() - start_time < max_wait:
            bgr, cam_t, _ = self.cache.get_camera()
            if bgr is not None:
                age = now_s() - cam_t
                if age <= self.args.max_frame_age_s:
                    return bgr
            await asyncio.sleep(0.1)
        return None

    def _should_take_shot_distance_only(self) -> bool:
        if self._last_shot_lat is None or self._last_shot_lon is None:
            return True
        d = haversine_m(self._last_shot_lat, self._last_shot_lon, self.tel.lat, self.tel.lon)
        return d >= self.args.min_shot_dist_m

    def _save_shot(self, bgr: np.ndarray, q: float, metrics: Dict[str, float], wind: Optional[float], note: str) -> str:
        self._shot_idx += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        q_str = "nan" if (q is None or not np.isfinite(q)) else f"{q:05.1f}"
        alt = self.tel.rel_alt_m if np.isfinite(self.tel.rel_alt_m) else -1.0
        img_name = f"shot_{self._shot_idx:05d}_Q{q_str}_alt{alt:04.1f}_{ts}_{note}.jpg"
        js_name = img_name.replace(".jpg", ".json")
        img_path = os.path.join(self.save_dir, img_name)
        js_path = os.path.join(self.save_dir, js_name)
        cv2.imwrite(img_path, bgr)

        meta = {
            "time": ts,
            "t_unix": time.time(),   # ✅ required
            "note": note,
            "quality": q,
            "metrics": metrics,
            "wind": wind,
            "telemetry": {
                "lat": self.tel.lat, "lon": self.tel.lon,
                "rel_alt_m": self.tel.rel_alt_m, "abs_alt_m": self.tel.abs_alt_m,
                "flight_mode": self.tel.flight_mode, "armed": self.tel.armed, "in_air": self.tel.in_air,
                "wp_index": self.tel.waypoint_index
            }
        }
        with open(js_path, "w") as f:
            json.dump(meta, f, indent=2)
        return img_path

    async def run(self):
        await self._connect()
        print(f"[INFO] Fixed-alt logger. Saving to: {self.save_dir}")
        print("[WAIT] Waiting for takeoff and reaching start altitude...")

        while True:
            await self._refresh_telemetry_once()

            if self.tel.flight_mode == "RETURN_TO_LAUNCH":
                self._rtl_seen = True

            if self._rtl_seen and (not self.tel.armed) and (not self.tel.in_air):
                print("[DONE] RTL complete -> exit")
                return

            if self.tel.in_air and np.isfinite(self.tel.rel_alt_m) and self.tel.rel_alt_m >= self.args.start_alt:
                print(f"[READY] Airborne at {self.tel.rel_alt_m:.1f}m - Logging")
                break

            if self.args.verbose_status:
                alt_str = f"{self.tel.rel_alt_m:.1f}" if np.isfinite(self.tel.rel_alt_m) else "nan"
                print(f"[WAIT] armed={self.tel.armed} in_air={self.tel.in_air} alt={alt_str}m mode={self.tel.flight_mode} wp={self.tel.waypoint_index}")

            await asyncio.sleep(0.5)

        print("[MISSION] Logging loop")
        while not self._rtl_seen:
            await self._refresh_telemetry_once()

            if self.tel.flight_mode == "RETURN_TO_LAUNCH":
                self._rtl_seen = True
                break

            if self.tel.flight_mode == "MISSION":
                if self._should_take_shot_distance_only():
                    bgr = await self._wait_for_fresh_frame(max_wait=3.0)
                    if bgr is None:
                        if now_s() - self._last_warn_no_frame_t > 2.0:
                            print("[WARN] No fresh camera frame (waited 3s)")
                            self._last_warn_no_frame_t = now_s()
                    else:
                        out = self.process_bgr_image(bgr)
                        Q, metrics = parse_iqa_dict(out)
                        wind = self._get_wind()
                        saved = self._save_shot(bgr, Q, metrics, wind, note="auto_fixed")
                        alt_str = f"{self.tel.rel_alt_m:5.1f}" if np.isfinite(self.tel.rel_alt_m) else " nan "
                        print(f"[SHOT] Q={Q:5.1f}% alt={alt_str}m mode={self.tel.flight_mode} wp={self.tel.waypoint_index} -> {saved}")
                        self._last_shot_lat = self.tel.lat
                        self._last_shot_lon = self.tel.lon

            await asyncio.sleep(0.3)

        print("[DONE] Mission complete")

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect_url", default="udpin://0.0.0.0:14540")
    ap.add_argument("--camera_topic", required=True)
    ap.add_argument("--wind_topic", default=None)
    ap.add_argument("--iqa_py", required=True)
    ap.add_argument("--save_dir", required=True)

    ap.add_argument("--start_alt", type=float, default=21.0)
    ap.add_argument("--min_shot_dist_m", type=float, default=4.0)
    ap.add_argument("--max_frame_age_s", type=float, default=1.0)
    ap.add_argument("--max_wind_age_s", type=float, default=2.0)
    ap.add_argument("--verbose_status", action="store_true")
    return ap

async def _run_app(args):
    app = FixedAltCompanion(args)
    try:
        await app.run()
    finally:
        try: app.stop_ros()
        except Exception: pass

def main():
    args = build_argparser().parse_args()
    asyncio.run(_run_app(args))

if __name__ == "__main__":
    main()
