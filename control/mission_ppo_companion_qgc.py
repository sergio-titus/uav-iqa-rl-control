#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QGC Mission Companion (PX4/MAVSDK + ROS2 camera IQA)
---------------------------------------------------
QGC flies the mission (path + yaw). We:
  - subscribe to a ROS2 camera topic (and optional wind topic)
  - run your IQA model (expects process_bgr_image(bgr)->dict with quality_% and distortions)
  - save shots with Q in filename + JSON metadata
  - if Q < target: HOLD, search altitude, lock best altitude by rewriting remaining mission waypoint altitudes, resume mission
  - if mode becomes RETURN_TO_LAUNCH: stop optimizing; just monitor until disarmed -> exit

Also reduces "overlap spam":
  - only take a shot if drone moved >= min_shot_dist_m OR max_shot_interval_s elapsed.

Note:
  - This does NOT control the mission path. "No overlap / no gaps" is mainly a QGC survey planning problem.
    This script only avoids re-shooting the same spot too often.
"""

import os
import sys
import time
import json
import math
import asyncio
import argparse
import threading
import importlib.util
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# wind supported types
from std_msgs.msg import Float32
try:
    from geometry_msgs.msg import Vector3Stamped
except Exception:
    Vector3Stamped = None

from mavsdk import System
from mavsdk.telemetry import FlightMode
from mavsdk.mission_raw import MissionRaw, MissionItem


# ------------------------- utils -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    # meters
    R = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def now_s() -> float:
    return time.time()


# ------------------------- IQA loader -------------------------
def load_iqa_process_bgr_image(iqa_py_path: str):
    """
    Expects iqa_model.py defines:
      process_bgr_image(bgr)->dict
    """
    iqa_py_path = os.path.expanduser(iqa_py_path)
    if not os.path.isfile(iqa_py_path):
        raise FileNotFoundError(f"IQA python file not found: {iqa_py_path}")

    spec = importlib.util.spec_from_file_location("iqa_model_dyn", iqa_py_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["iqa_model_dyn"] = m
    spec.loader.exec_module(m)

    if hasattr(m, "process_bgr_image"):
        return m.process_bgr_image

    raise RuntimeError(
        f"{iqa_py_path} must define process_bgr_image(bgr)->dict "
        f"(with keys like quality_% and distortions)."
    )


def parse_iqa_dict(out: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Returns (Q, distortions_dict).
    - Q from quality_% (preferred) or quality_score_percent.
    - distortions: any numeric key ending with '_%' except quality_%
    """
    q = out.get("quality_%", out.get("quality_score_percent", out.get("quality", 0.0)))
    Q = safe_float(q, 0.0)

    distort = {}
    for k, v in out.items():
        if not isinstance(k, str):
            continue
        if k == "quality_%":
            continue
        if k.endswith("%") or k.endswith("_%"):
            distort[k] = safe_float(v, 0.0)
    # also include common explicit names if present (even if not *_%)
    for k in ["blur_%", "lowres_%", "under_%", "over_%", "noise_%", "haze_%"]:
        if k in out and k not in distort:
            distort[k] = safe_float(out[k], 0.0)

    return Q, distort


# ------------------------- ROS2 cache node -------------------------
class MultiCache(Node):
    """
    Caches latest camera frame and optional wind (thread-safe).
    Supports:
      - wind topic Float32 (m/s)
      - wind topic Vector3Stamped (uses magnitude)
    If incompatible: logs warn, disables wind subscription.
    """
    def __init__(self, camera_topic: str, wind_topic: Optional[str]):
        super().__init__("iqa_multi_cache_node")
        self.br = CvBridge()

        self._lock = threading.Lock()

        self._last_bgr = None
        self._last_cam_t = 0.0
        self._cam_frames = 0

        self._last_wind = None
        self._last_wind_t = 0.0
        self._wind_type = "none"

        qos_cam = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.create_subscription(Image, camera_topic, self._cam_cb, qos_cam)
        self.get_logger().info(f"Subscribed camera: {camera_topic} (BEST_EFFORT)")

        if wind_topic:
            # try Float32 first (your working case)
            try:
                qos_wind = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=5
                )
                self.create_subscription(Float32, wind_topic, self._wind_cb_f32, qos_wind)
                self._wind_type = "std_msgs/msg/Float32"
                self.get_logger().info(f"Wind topic: {wind_topic} type: std_msgs/msg/Float32")
            except Exception as e:
                # fallback Vector3Stamped if available
                if Vector3Stamped is not None:
                    try:
                        self.create_subscription(Vector3Stamped, wind_topic, self._wind_cb_vec3, qos_wind)
                        self._wind_type = "geometry_msgs/msg/Vector3Stamped"
                        self.get_logger().info(f"Wind topic: {wind_topic} type: geometry_msgs/msg/Vector3Stamped")
                    except Exception as e2:
                        self.get_logger().warn(f"Wind subscribe failed ({e2}). Disabling wind.")
                else:
                    self.get_logger().warn(f"Wind subscribe failed ({e}). Disabling wind.")

    def _cam_cb(self, msg: Image):
        try:
            bgr = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        with self._lock:
            self._last_bgr = bgr
            self._last_cam_t = now_s()
            self._cam_frames += 1

    def _wind_cb_f32(self, msg: Float32):
        with self._lock:
            self._last_wind = float(msg.data)
            self._last_wind_t = now_s()

    def _wind_cb_vec3(self, msg):
        # magnitude
        v = msg.vector
        m = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
        with self._lock:
            self._last_wind = float(m)
            self._last_wind_t = now_s()

    def get_camera(self) -> Tuple[Optional[np.ndarray], float, int]:
        with self._lock:
            if self._last_bgr is None:
                return None, 0.0, self._cam_frames
            return self._last_bgr.copy(), self._last_cam_t, self._cam_frames

    def get_wind(self) -> Tuple[Optional[float], float, str]:
        with self._lock:
            return self._last_wind, self._last_wind_t, self._wind_type


def start_ros_cache(camera_topic: str, wind_topic: Optional[str]):
    rclpy.init(args=None)
    node = MultiCache(camera_topic, wind_topic)

    stop_evt = threading.Event()

    def _spin():
        while rclpy.ok() and not stop_evt.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)

    thr = threading.Thread(target=_spin, daemon=True)
    thr.start()

    def stop():
        stop_evt.set()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

    return node, stop


# ------------------------- MAVSDK telemetry state -------------------------
@dataclass
class TelemetryState:
    armed: bool = False
    in_air: bool = False
    rel_alt_m: float = float("nan")
    abs_alt_m: float = float("nan")
    lat: float = float("nan")
    lon: float = float("nan")
    flight_mode: str = "UNKNOWN"


class QGCMissionCompanion:
    def __init__(self, args):
        self.args = args

        self.process_bgr_image = load_iqa_process_bgr_image(args.iqa_py)

        self.save_dir = os.path.expanduser(args.save_dir)
        ensure_dir(self.save_dir)

        self.cache, self.stop_ros = start_ros_cache(args.camera_topic, args.wind_topic)

        self.drone: Optional[System] = None
        self.tel = TelemetryState()

        self._shot_idx = 0
        self._last_shot_t = 0.0
        self._last_shot_lat = None
        self._last_shot_lon = None

        self.best_alt = None
        self.best_q = -1.0

        self._rtl_seen = False

    async def _connect(self):
        self.drone = System()
        await self.drone.connect(system_address=self.args.connect_url)

        async for st in self.drone.core.connection_state():
            if st.is_connected:
                break
        print("[LINK] Connected")

    async def _refresh_telemetry_once(self):
        ar = await self.drone.telemetry.armed().__anext__()
        ia = await self.drone.telemetry.in_air().__anext__()
        pos = await self.drone.telemetry.position().__anext__()
        fm = await self.drone.telemetry.flight_mode().__anext__()

        self.tel.armed = bool(ar)
        self.tel.in_air = bool(ia)
        self.tel.rel_alt_m = float(pos.relative_altitude_m)
        self.tel.abs_alt_m = float(pos.absolute_altitude_m)
        self.tel.lat = float(pos.latitude_deg)
        self.tel.lon = float(pos.longitude_deg)
        self.tel.flight_mode = str(fm)

    def _get_fresh_frame(self, max_age_s: float = 2.0) -> Optional[np.ndarray]:
        bgr, t, frames = self.cache.get_camera()
        if bgr is None:
            return None
        age = now_s() - t
        if age > max_age_s:
            return None
        return bgr

    def _get_wind(self, max_age_s: float = 2.0) -> Tuple[Optional[float], float]:
        w, wt, _ = self.cache.get_wind()
        if w is None:
            return None, float("inf")
        age = now_s() - wt
        if age > max_age_s:
            return w, age
        return w, age

    def _should_take_shot(self) -> bool:
        """
        Avoid overlap spam:
          - take if moved >= min_shot_dist_m since last shot
          - OR if time since last shot >= max_shot_interval_s
        """
        t = now_s()
        if self._last_shot_t <= 0.0:
            return True

        if (t - self._last_shot_t) >= self.args.max_shot_interval_s:
            return True

        if self._last_shot_lat is None or self._last_shot_lon is None:
            return True

        if not (math.isfinite(self.tel.lat) and math.isfinite(self.tel.lon)):
            return True

        d = haversine_m(self._last_shot_lat, self._last_shot_lon, self.tel.lat, self.tel.lon)
        return d >= self.args.min_shot_dist_m

    def _save_shot(self, bgr: np.ndarray, Q: float, distort: Dict[str, float], wind: Optional[float], note: str):
        fn = f"shot_{self._shot_idx:06d}_Q{Q:05.1f}.jpg"
        path = os.path.join(self.save_dir, fn)
        self._shot_idx += 1

        cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        meta = {
            "t": now_s(),
            "note": note,
            "Q": Q,
            "distortions": distort,
            "wind_mps": wind,
            "alt_rel_m": self.tel.rel_alt_m,
            "lat": self.tel.lat,
            "lon": self.tel.lon,
            "flight_mode": self.tel.flight_mode,
            "camera_topic": self.args.camera_topic,
            "wind_topic": self.args.wind_topic,
        }
        with open(path.replace(".jpg", ".json"), "w") as f:
            json.dump(meta, f, indent=2)

        # update last shot position/time for anti-overlap
        self._last_shot_t = now_s()
        if math.isfinite(self.tel.lat) and math.isfinite(self.tel.lon):
            self._last_shot_lat = self.tel.lat
            self._last_shot_lon = self.tel.lon

        return os.path.basename(path)

    def _format_distortions(self, distort: Dict[str, float], max_items: int = 20) -> str:
        # stable order: common first, then the rest
        preferred = ["blur_%", "lowres_%", "under_%", "over_%", "noise_%", "haze_%"]
        items = []
        used = set()
        for k in preferred:
            if k in distort:
                items.append((k, distort[k]))
                used.add(k)
        for k in sorted(distort.keys()):
            if k in used:
                continue
            items.append((k, distort[k]))
        if len(items) > max_items:
            items = items[:max_items] + [("...", 0.0)]
        return " ".join([f"{k}={v:5.1f}" if k != "..." else "..." for k, v in items])

    async def _take_one_shot(self, note: str) -> Optional[float]:
        bgr, cam_t, frames = self.cache.get_camera()
        if bgr is None:
            if self.args.verbose_status:
                print(f"[WARN] No camera yet (frames={frames})")
            return None
        age = now_s() - cam_t
        if age > self.args.max_frame_age_s:
            if self.args.verbose_status:
                print(f"[WARN] No fresh camera frame (frames={frames}, age={age:.2f}s)")
            return None

        out = self.process_bgr_image(bgr)
        Q, distort = parse_iqa_dict(out)

        wind, wind_age = self._get_wind(max_age_s=self.args.max_wind_age_s)

        saved = self._save_shot(bgr, Q, distort, wind, note)

        wind_str = "wind=NA"
        if wind is not None:
            wind_str = f"wind={wind:.2f}m/s(age={wind_age:.2f}s)"

        distort_str = self._format_distortions(distort, max_items=self.args.print_max_distortions)

        print(
            f"[SHOT] Q={Q:5.1f}% alt={self.tel.rel_alt_m:5.1f}m mode={self.tel.flight_mode} "
            f"{wind_str} {distort_str} -> {saved}"
        )
        return Q

    async def _set_speed(self, mps: float) -> bool:
        """
        Try to slow down / restore cruise speed without mavlink_passthrough.
        PX4 usually supports mission.set_current_speed().
        """
        mps = max(0.1, float(mps))
        try:
            await self.drone.mission.set_current_speed(mps)
            return True
        except Exception as e:
            if self.args.verbose_status:
                print(f"[WARN] Could not change speed via mission.set_current_speed: {e}")
            # some MAVSDK versions might have action.set_maximum_speed; if not, ignore
            try:
                if hasattr(self.drone.action, "set_maximum_speed"):
                    await self.drone.action.set_maximum_speed(mps)
                    return True
            except Exception:
                pass
            return False

    async def _hold(self) -> bool:
        try:
            await self.drone.action.hold()
            return True
        except Exception as e:
            print(f"[WARN] HOLD failed: {e}")
            return False

    async def _resume_mission(self) -> bool:
        try:
            await self.drone.mission.start_mission()
            return True
        except Exception as e:
            print(f"[WARN] Mission resume failed: {e}")
            return False

    async def _goto_same_latlon_rel_alt(self, target_rel_alt_m: float):
        """
        Use goto_location(lat, lon, abs_alt, yaw=NaN) to request a new altitude.
        Works best in HOLD.
        """
        target_rel_alt_m = clamp(target_rel_alt_m, self.args.min_alt, self.args.max_alt)

        rel_now = self.tel.rel_alt_m
        abs_now = self.tel.abs_alt_m
        abs_target = abs_now + (target_rel_alt_m - rel_now)

        yaw_nan = float("nan")
        await self.drone.action.goto_location(self.tel.lat, self.tel.lon, abs_target, yaw_nan)

    async def _rewrite_mission_altitudes(self, new_rel_alt_m: float) -> int:
        """
        Rewrite waypoint altitudes in the vehicle mission (mission_raw).
        We do NOT touch TAKEOFF(22), LAND(21), RTL(20), etc. Only NAV_WAYPOINT(16) items.
        """
        if not self.args.rewrite_mission_alt:
            return 0

        try:
            mission_raw = self.drone.mission_raw  # IMPORTANT: avoid MissionRaw(System) channel crash
            items: List[MissionItem] = await mission_raw.download_mission()
        except Exception as e:
            print(f"[WARN] Could not download mission_raw: {e}")
            return 0

        changed = 0
        new_items: List[MissionItem] = []
        for it in items:
            # MAVLink commands:
            # 16 = NAV_WAYPOINT, 22 = NAV_TAKEOFF, 21 = NAV_LAND, 20 = NAV_RETURN_TO_LAUNCH
            if int(it.command) == 16:
                # it.z is altitude (meters) for PX4 mission
                if abs(float(it.z) - float(new_rel_alt_m)) > 0.05:
                    it = MissionItem(
                        seq=it.seq,
                        frame=it.frame,
                        command=it.command,
                        current=it.current,
                        autocontinue=it.autocontinue,
                        param1=it.param1, param2=it.param2, param3=it.param3, param4=it.param4,
                        x=it.x, y=it.y, z=float(new_rel_alt_m),
                        mission_type=it.mission_type
                    )
                    changed += 1
            new_items.append(it)

        if changed == 0:
            return 0

        try:
            await mission_raw.upload_mission(new_items)
            return changed
        except Exception as e:
            print(f"[WARN] Could not upload mission_raw: {e}")
            return 0

    async def _optimize_altitude_and_lock(self, q0: float):
        """
        When Q < target:
          - ignore if RTL
          - slow speed
          - HOLD
          - try alt steps (down then up) to reach target, choose best
          - rewrite mission altitude to best
          - resume mission
          - restore cruise speed
        """
        if self._rtl_seen:
            return

        print(f"[TRIGGER] Q={q0:.1f}% < {self.args.q_band_lo:.1f}% -> slow + altitude lock")
        await self._set_speed(self.args.slow_speed)

        await self._hold()
        await asyncio.sleep(self.args.hold_before_opt)

        await self._refresh_telemetry_once()

        rel_start = float(self.tel.rel_alt_m)
        best_q = q0
        best_alt = rel_start

        # Phase A: try down first
        rel = rel_start
        for k in range(1, self.args.max_retries + 1):
            rel = clamp(rel - self.args.alt_step, self.args.min_alt, self.args.max_alt)
            try:
                await self._goto_same_latlon_rel_alt(rel)
            except Exception as e:
                print(f"[WARN] goto_location failed: {e}")
                break

            await asyncio.sleep(self.args.settle_s)
            await self._refresh_telemetry_once()

            q = await self._take_one_shot(note=f"opt_down_{k}")
            if q is None:
                continue
            if q > best_q:
                best_q = q
                best_alt = float(self.tel.rel_alt_m)
            if q >= self.args.q_band_lo:
                # good enough
                break
            if rel <= self.args.min_alt + 1e-3:
                break

        # Phase B: if still below target, try up
        if best_q < self.args.q_band_lo:
            rel = best_alt
            for k in range(1, self.args.max_retries + 1):
                rel = clamp(rel + self.args.alt_step, self.args.min_alt, self.args.max_alt)
                try:
                    await self._goto_same_latlon_rel_alt(rel)
                except Exception as e:
                    print(f"[WARN] goto_location failed: {e}")
                    break

                await asyncio.sleep(self.args.settle_s)
                await self._refresh_telemetry_once()

                q = await self._take_one_shot(note=f"opt_up_{k}")
                if q is None:
                    continue
                if q > best_q:
                    best_q = q
                    best_alt = float(self.tel.rel_alt_m)
                if q >= self.args.q_band_lo:
                    break

        self.best_q = best_q
        self.best_alt = best_alt

        print(f"[OPT] Locked new altitude = {best_alt:.1f} m (best Q={best_q:.1f}%)")

        changed = await self._rewrite_mission_altitudes(best_alt)
        if changed > 0:
            print(f"[LOCK] Rewrote mission waypoint altitudes to {best_alt:.1f} m (changed {changed} items).")

        # resume mission
        await self._resume_mission()
        await asyncio.sleep(self.args.resume_delay_s)

        # restore speed
        await self._set_speed(self.args.cruise_speed)

    async def run(self):
        await self._connect()

        print(f"[INFO] Saving shots to: {self.save_dir}")
        print("[INFO] Start mission in QGC. I will save shots and intervene only when Q < target.")
        if self.args.rewrite_mission_alt:
            print("[INFO] Mission altitude lock enabled: will rewrite remaining waypoint altitudes when a new best alt is found.")
        print(f"[INFO] IQA will start only when altitude >= {self.args.start_alt:.1f} m")
        if self.args.wind_topic:
            print(f"[INFO] Wind topic: {self.args.wind_topic}")

        takeoff_seen = False
        while True:
            try:
                await self._refresh_telemetry_once()
            except Exception as e:
                print(f"[WARN] telemetry failed: {e} -> reconnecting")
                await asyncio.sleep(0.5)
                try:
                    await self._connect()
                except Exception as e2:
                    print(f"[WARN] reconnect failed: {e2}")
                continue

            # detect RTL
            if self.tel.flight_mode == "RETURN_TO_LAUNCH":
                if not self._rtl_seen:
                    self._rtl_seen = True
                    print("[RTL] RETURN_TO_LAUNCH detected -> disabling optimization. Will exit when disarmed.")
            # exit condition: after RTL or mission end, when disarmed and not in air
            if self._rtl_seen and (not self.tel.armed) and (not self.tel.in_air):
                print("[DONE] Mission finished (RTL complete). Vehicle disarmed and not in air. Exiting.")
                return

            # verbose status
            if self.args.verbose_status:
                cam, cam_t, frames = self.cache.get_camera()
                cam_age = (now_s() - cam_t) if cam_t else float("inf")
                wind, wind_t, wtype = self.cache.get_wind()
                wind_age = (now_s() - wind_t) if wind_t else float("inf")
                wstr = "wind=NA"
                if wind is not None:
                    wstr = f"wind={wind:.2f}({wtype}, age={wind_age:.2f}s)"
                print(f"[STATUS] armed={self.tel.armed} in_air={self.tel.in_air} alt={self.tel.rel_alt_m:.1f} mode={self.tel.flight_mode} cam_frames={frames} cam_age={cam_age:.2f}s {wstr}")

            # wait for takeoff
            if self.tel.armed and self.tel.in_air:
                takeoff_seen = True

            if not takeoff_seen:
                await asyncio.sleep(0.2)
                continue

            # start IQA only above start_alt
            if not (self.tel.armed and self.tel.in_air and (self.tel.rel_alt_m >= (self.args.start_alt - 0.2))):
                await asyncio.sleep(0.2)
                continue

            # if RTL: no more optimization, but still can take occasional shots if you want.
            # You said "no need good image when going back home", so we stop taking shots in RTL.
            if self._rtl_seen:
                await asyncio.sleep(0.5)
                continue

            # anti-overlap + rate limit
            if not self._should_take_shot():
                await asyncio.sleep(0.05)
                continue

            # take shot
            q = await self._take_one_shot(note="auto_mission")
            if q is None:
                await asyncio.sleep(0.1)
                continue

            # optimize only if below band low
            if q < self.args.q_band_lo:
                await self._optimize_altitude_and_lock(q)

            await asyncio.sleep(0.05)

    def close(self):
        self.stop_ros()


# ------------------------- CLI -------------------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect_url", default="udpin://0.0.0.0:14540")
    ap.add_argument("--camera_topic", default="/iris/gazebo_distorted_preview/image_raw")
    ap.add_argument("--wind_topic", default=None)
    ap.add_argument("--iqa_py", default="~/ros2_ws/src/uav_iqa_ddqn/iqa/iqa_model.py")
    ap.add_argument("--save_dir", default="~/uav_survey_shots")

    ap.add_argument("--start_alt", type=float, default=14.0)
    ap.add_argument("--min_alt", type=float, default=12.0)
    ap.add_argument("--max_alt", type=float, default=40.0)

    ap.add_argument("--q_band_lo", type=float, default=80.0)  # intervene if below this
    ap.add_argument("--q_band_hi", type=float, default=85.0)  # (kept for future, not used now)

    ap.add_argument("--alt_step", type=float, default=1.0)
    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--settle_s", type=float, default=0.8)

    ap.add_argument("--slow_speed", type=float, default=0.1)
    ap.add_argument("--cruise_speed", type=float, default=3.0)

    ap.add_argument("--rewrite_mission_alt", action="store_true")

    ap.add_argument("--resume_delay_s", type=float, default=0.5)
    ap.add_argument("--hold_before_opt", type=float, default=0.3)

    # anti-overlap shooting controls
    ap.add_argument("--min_shot_dist_m", type=float, default=2.0,
                    help="Minimum horizontal distance between shots to reduce overlap spam.")
    ap.add_argument("--max_shot_interval_s", type=float, default=3.0,
                    help="Force a shot if no shot was taken for this long (avoid gaps).")

    # camera/wind freshness
    ap.add_argument("--max_frame_age_s", type=float, default=2.0)
    ap.add_argument("--max_wind_age_s", type=float, default=2.0)

    # printing
    ap.add_argument("--print_max_distortions", type=int, default=20)
    ap.add_argument("--verbose_status", action="store_true")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    runner = QGCMissionCompanion(args)
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
