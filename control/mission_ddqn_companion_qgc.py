#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, math, asyncio, argparse, threading, importlib.util
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
from collections import deque

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32

from mavsdk import System
from mavsdk.mission_raw import MissionItem

try:
    import torch
except Exception:
    torch = None

# ------------------------- policy constants -------------------------
Q_OPT_LO = 80.0
Q_OPT_HI = 85.0
Q_UP_TRIGGER = 86.0

ACT_DESCEND = 0
ACT_HOVER   = 1
ACT_ASCEND  = 2

# ------------------------- utils -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_s() -> float:
    return time.time()

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_float(x, default=0.0) -> float:
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
    spec = importlib.util.spec_from_file_location("iqa_model_dyn", iqa_py_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["iqa_model_dyn"] = m
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
        super().__init__("uav_iqa_cache")
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

# ------------------------- DDQN loader -------------------------
def load_ddqn_policy(path: str, device: str = "cpu"):
    if torch is None:
        raise RuntimeError("torch is not available. Install PyTorch or run in environment with torch.")
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DDQN policy file not found: {path}")

    # ✅ If file is TorchScript, load it directly with torch.jit.load (PyTorch 2.6 safe).
    try:
        model = torch.jit.load(path, map_location=device)
        model.eval()
        return model
    except Exception:
        pass

    # Otherwise try normal torch.save(model, path) format (full module)
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

    raise RuntimeError(
        "Could not load DDQN model.\n"
        "Supported:\n"
        " - TorchScript: torch.jit.save(...)\n"
        " - Full torch module: torch.save(model, ...)\n"
        "Not supported: plain state_dict without architecture."
    )

    model.eval()
    return model

def _norm01(x, lo, hi):
    if not np.isfinite(x):
        return 0.0
    return float(clamp((float(x) - lo) / (hi - lo + 1e-9), 0.0, 1.0))

def build_ddqn_state(q: float, metrics: Dict[str, float], rel_alt_m: float, wind: Optional[float],
                     min_alt: float, max_alt: float) -> np.ndarray:
    """
    State vector with 12 features to match the trained DDQN model.
    Features: Q, blur, haze, noise, lowres, under, over, wind, rel_alt, 
              lat_velocity, lon_velocity, alt_velocity
    All normalized to 0..1.
    """
    # Core quality metrics (7 features)
    Q = _norm01(q, 0.0, 100.0)
    blur  = _norm01(metrics.get("blur_%", 0.0), 0.0, 100.0)
    haze  = _norm01(metrics.get("haze_%", 0.0), 0.0, 100.0)
    noise = _norm01(metrics.get("noise_%", 0.0), 0.0, 100.0)
    lowr  = _norm01(metrics.get("lowres_%", 0.0), 0.0, 100.0)
    und   = _norm01(metrics.get("under_%", 0.0), 0.0, 100.0)
    over  = _norm01(metrics.get("over_%", 0.0), 0.0, 100.0)
    
    # Environmental (1 feature)
    w     = _norm01(wind if wind is not None else 0.0, 0.0, 12.0)
    
    # Altitude (1 feature)
    alt   = _norm01(rel_alt_m if np.isfinite(rel_alt_m) else min_alt, min_alt, max_alt)
    
    # Velocity features (3 features) - placeholders since not in current telemetry
    lat_vel = 0.0  # placeholder
    lon_vel = 0.0  # placeholder
    alt_vel = 0.0  # placeholder
    
    return np.array([Q, blur, haze, noise, lowr, und, over, w, alt, 
                     lat_vel, lon_vel, alt_vel], dtype=np.float32)

def ddqn_choose_action(model, state: np.ndarray, device: str = "cpu") -> int:
    """
    Returns action in {0,1,2} for {DESCEND, HOVER, ASCEND}.
    Accepts models that output:
      - shape (3,) or (1,3) Q-values
      - logits/probs (we argmax anyway)
    """
    if torch is None:
        return ACT_HOVER
    x = torch.from_numpy(state).to(device=device).float().unsqueeze(0)  # (1, D)
    with torch.no_grad():
        y = model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if hasattr(y, "detach"):
            y = y.detach()
        y = y.squeeze(0)
        if y.numel() < 3:
            # unexpected output
            return ACT_HOVER
        a = int(torch.argmax(y[:3]).item())
        return a

# ------------------------- main companion -------------------------
class MissionCompanion:
    def __init__(self, args):
        self.args = args
        self.process_bgr_image = load_iqa_process_bgr_image(args.iqa_py)

        self.device = args.ddqn_device
        self.ddqn = load_ddqn_policy(args.ddqn_policy, device=self.device)

        self.save_dir = os.path.expanduser(args.save_dir)
        ensure_dir(self.save_dir)

        self.cache, self.stop_ros = start_ros_cache(args.camera_topic, args.wind_topic)

        self.drone: Optional[System] = None
        self.tel = TelemetryState()

        self._shot_idx = 0
        self._last_shot_lat = None
        self._last_shot_lon = None
        self._last_metrics: Dict[str, float] = {}
        self._last_warn_no_frame_t = 0.0

        self._rtl_seen = False
        self._did_initial_scan = False
        self._locked_altitude: Optional[float] = None
        self._mission_in_progress = False
        self._correction_cooldown = 0.0

        # anti-pingpong
        self._max_wp_seen = -1

        # high-quality 4-shot trigger
        self._recent_q = deque(maxlen=4)

    # ---------- MAVSDK helpers ----------
    async def _connect(self):
        if self.args.connect_url.strip().lower().startswith("grpc://"):
            raise RuntimeError("Do NOT use grpc:// in --connect_url. Use udp://, udpin://, udpout:// etc.")
        self.drone = System()
        await self.drone.connect(system_address=self.args.connect_url)
        async for st in self.drone.core.connection_state():
            if st.is_connected:
                break
        print(f"[LINK] Connected via {self.args.connect_url}")

    async def _safe(self, coro_fn, *args, **kwargs):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if ("StatusCode.UNAVAILABLE" in msg) or ("failed to connect to all addresses" in msg) or ("Connection refused" in msg):
                print("[LINK] MAVSDK backend disconnected. Reconnecting and retrying once...")
                await self._connect()
                await asyncio.sleep(0.3)
                return await coro_fn(*args, **kwargs)
            raise

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
                self.tel.waypoint_index = int(idx)
                if self.tel.waypoint_index > self._max_wp_seen:
                    self._max_wp_seen = self.tel.waypoint_index
                break
        except Exception: pass

    # ---------- motion control ----------
    async def _goto_same_latlon_rel_alt(self, target_rel_alt_m: float):
        target_rel_alt_m = clamp(target_rel_alt_m, self.args.min_alt, self.args.max_alt)

        if not np.isfinite(self.tel.rel_alt_m) or not np.isfinite(self.tel.abs_alt_m):
            return

        if abs(float(self.tel.rel_alt_m) - target_rel_alt_m) < 0.25:
            return

        rel_now = float(self.tel.rel_alt_m)
        abs_now = float(self.tel.abs_alt_m)
        abs_target = abs_now + (target_rel_alt_m - rel_now)

        print(f"[GOTO] Altitude: {rel_now:.1f}m -> {target_rel_alt_m:.1f}m")

        # Simple approach: use goto_location but handle it carefully
        try:
            # Save current flight mode
            current_mode = self.tel.flight_mode
            
            # Use goto_location
            await self._safe(self.drone.action.goto_location, 
                            float(self.tel.lat), float(self.tel.lon), 
                            float(abs_target), float("nan"))
            
            # Wait to reach altitude
            start_time = now_s()
            while now_s() - start_time < 6.0:
                await asyncio.sleep(0.5)
                await self._refresh_telemetry_once()
                if np.isfinite(self.tel.rel_alt_m) and abs(float(self.tel.rel_alt_m) - target_rel_alt_m) < 0.6:
                    print(f"[GOTO] Reached ~{float(self.tel.rel_alt_m):.1f}m")
                    break
            
            # If we were in mission mode, resume it
            if current_mode == "MISSION" and self._mission_in_progress:
                await asyncio.sleep(0.5)
                await self._continue_mission_anti_pingpong()
                
        except Exception as e:
            print(f"[GOTO] Error: {e}")

    # ---------- speed control ----------
    async def _slow_down_for_correction(self, speed_m_s: float = 0.2):
        """Slow down the drone without pausing mission."""
        try:
            await self._safe(self.drone.action.set_current_speed, float(speed_m_s))
            print(f"[SLOW] Speed reduced to {speed_m_s} m/s")
            return True
        except Exception as e:
            print(f"[SLOW] Could not set speed: {e}")
            return False

    async def _restore_normal_speed(self, speed_m_s: float = 5.0):
        """Restore normal mission speed."""
        try:
            await self._safe(self.drone.action.set_current_speed, float(speed_m_s))
            print(f"[RESTORE] Speed restored to {speed_m_s} m/s")
            return True
        except Exception as e:
            print(f"[RESTORE] Could not restore speed: {e}")
            return False

    # ---------- sensors ----------
    def _get_wind(self) -> Optional[float]:
        w, t = self.cache.get_wind()
        if w is None:
            return None
        if (now_s() - t) > self.args.max_wind_age_s:
            return None
        return float(w)

    async def _wait_for_fresh_frame(self, max_wait: float = 5.0) -> Optional[Tuple[np.ndarray, float]]:
        start_time = now_s()
        while now_s() - start_time < max_wait:
            bgr, cam_t, _ = self.cache.get_camera()
            if bgr is not None:
                age = now_s() - cam_t
                if age <= self.args.max_frame_age_s:
                    return bgr, cam_t
            await asyncio.sleep(0.1)
        return None

    # ---------- shooting ----------
    def _format_metrics(self, m: Dict[str, float], max_items: int) -> str:
        if not m:
            return ""
        preferred = ["blur_%", "lowres_%", "under_%", "over_%", "noise_%", "haze_%"]
        out = []
        for k in preferred:
            if k in m:
                out.append(f"{k}={m[k]:.1f}")
            if len(out) >= max_items:
                break
        return " ".join(out)

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
            "time": ts, "note": note, "quality": q, "metrics": metrics, "wind": wind,
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

    def _should_take_shot_distance_only(self) -> bool:
        if self._last_shot_lat is None or self._last_shot_lon is None:
            return True
        d = haversine_m(self._last_shot_lat, self._last_shot_lon, self.tel.lat, self.tel.lon)
        return d >= self.args.min_shot_dist_m

    async def _take_one_shot(self, note: str, force: bool) -> Optional[float]:
        if (not force) and (not self._should_take_shot_distance_only()):
            return None

        frame_data = await self._wait_for_fresh_frame(max_wait=3.0)
        if frame_data is None:
            if now_s() - self._last_warn_no_frame_t > 2.0:
                print("[WARN] No fresh camera frame (waited 3s)")
                self._last_warn_no_frame_t = now_s()
            return None

        bgr, _ = frame_data
        out = self.process_bgr_image(bgr)
        Q, metrics = parse_iqa_dict(out)
        self._last_metrics = metrics or {}

        wind = self._get_wind()
        saved = self._save_shot(bgr, Q, metrics, wind, note)

        m_str = self._format_metrics(metrics, self.args.print_max_metrics)
        wind_str = "wind=NA" if wind is None else f"wind={wind:.2f}m/s"
        alt_str = f"{self.tel.rel_alt_m:5.1f}" if np.isfinite(self.tel.rel_alt_m) else " nan "
        print(f"[SHOT] Q={Q:5.1f}% alt={alt_str}m mode={self.tel.flight_mode} wp={self.tel.waypoint_index} {wind_str} {m_str} -> {saved}")

        self._last_shot_lat = self.tel.lat
        self._last_shot_lon = self.tel.lon

        if Q is not None and np.isfinite(Q):
            self._recent_q.append(float(Q))

        return Q

    # ---------- mission control ----------
    async def _rewrite_mission_altitudes(self, new_rel_alt_m: float) -> int:
        if not self.args.rewrite_mission_alt:
            return 0

        try:
            items: List[MissionItem] = await self._safe(self.drone.mission_raw.download_mission)
        except Exception:
            return 0

        changed = 0
        new_items: List[MissionItem] = []
        for it in items:
            if int(it.command) == 16:  # NAV_WAYPOINT
                if abs(float(it.z) - float(new_rel_alt_m)) > 0.05:
                    it = MissionItem(
                        seq=it.seq, frame=it.frame, command=it.command, current=it.current,
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
            await self._safe(self.drone.mission_raw.upload_mission, new_items)
            return changed
        except Exception:
            return 0

    async def _continue_mission_anti_pingpong(self):
        await self._refresh_telemetry_once()
        wp_now = int(self.tel.waypoint_index) if self.tel.waypoint_index is not None else -1

        if wp_now >= 0 and self._max_wp_seen >= 0 and wp_now < self._max_wp_seen:
            print(f"[CONTINUE] anti-pingpong: wp_now={wp_now} < max_seen={self._max_wp_seen} -> set_current({self._max_wp_seen})")
            try:
                await self._safe(self.drone.mission_raw.set_current_mission_item, int(self._max_wp_seen))
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f"[CONTINUE] set_current failed (ignored): {e}")

        print("[CONTINUE] start_mission()")
        try:
            await self._safe(self.drone.mission.start_mission)
        except Exception as e:
            print(f"[CONTINUE] start_mission failed: {e}")

        await asyncio.sleep(0.8)
        await self._refresh_telemetry_once()

    # ---------- initial scan ----------
    async def _initial_scan_find_lock_altitude(self):
        print("[INIT-SCAN] Go up 1m steps until Q < 80, then lock at (bad_alt - 1m)")

        current_alt = max(self.args.start_alt, self.args.min_alt)
        
        # Go to start altitude
        await self._goto_same_latlon_rel_alt(current_alt)
        await asyncio.sleep(1.5)
        await self._refresh_telemetry_once()

        last_good_alt = float(self.tel.rel_alt_m) if np.isfinite(self.tel.rel_alt_m) else current_alt
        found_bad = False
        consecutive_no_image = 0

        while current_alt <= self.args.max_alt and not found_bad:
            print(f"[INIT-SCAN] Testing alt={current_alt:.1f}m")
            
            # Take shot at current altitude
            q = await self._take_one_shot(note=f"scan_alt{current_alt:.1f}", force=True)

            if q is None or not np.isfinite(q):
                consecutive_no_image += 1
                print("[INIT-SCAN] No valid image")
                if consecutive_no_image >= 2:
                    current_alt += 1.0
                    if current_alt <= self.args.max_alt:
                        await self._goto_same_latlon_rel_alt(current_alt)
                        await asyncio.sleep(1.0)
                continue

            consecutive_no_image = 0

            if q >= Q_OPT_LO:
                last_good_alt = current_alt
                current_alt += 1.0
                if current_alt <= self.args.max_alt:
                    await self._goto_same_latlon_rel_alt(current_alt)
                    await asyncio.sleep(1.0)
            else:
                found_bad = True
                target_alt = max(current_alt - 1.0, self.args.min_alt)
                self._locked_altitude = target_alt
                print(f"[INIT-SCAN] Found bad Q={q:.1f}% at {current_alt:.1f}m -> lock {target_alt:.1f}m")
                await self._goto_same_latlon_rel_alt(target_alt)
                await asyncio.sleep(1.0)
                await self._take_one_shot(note=f"locked_alt{target_alt:.1f}", force=True)

        if not found_bad:
            self._locked_altitude = clamp(last_good_alt, self.args.min_alt, self.args.max_alt)
            print(f"[INIT-SCAN] Never found Q<80. Locking at {self._locked_altitude:.1f}m")

        changed = await self._rewrite_mission_altitudes(self._locked_altitude)
        if changed > 0:
            print(f"[LOCK] Rewrote {changed} mission waypoints to alt={self._locked_altitude:.1f}m")

        await self._continue_mission_anti_pingpong()

    # ---------- DDQN correction ----------
    async def _handle_low_quality_ddqn(self, q_bad: float) -> bool:
        print(f"[LOW-Q] Q={q_bad:.1f}% < 80 -> slowing down + retry")
        
        # Try to slow down instead of pausing
        speed_success = await self._slow_down_for_correction(0.2)
        await asyncio.sleep(1.0)
        await self._refresh_telemetry_once()
        
        await asyncio.sleep(0.6)
        q2 = await self._take_one_shot(note="lowq_retry_slow", force=True)
        
        # If retry is good, restore speed and continue
        if q2 is not None and np.isfinite(q2) and q2 >= Q_OPT_LO:
            print(f"[LOW-Q] Second shot ok: Q={q2:.1f}% -> restore speed")
            if speed_success:
                await self._restore_normal_speed()
            await self._continue_mission_anti_pingpong()
            return True
        
        # DDQN decision
        q_for_state = q2 if (q2 is not None and np.isfinite(q2)) else q_bad
        metrics = self._last_metrics or {}
        wind = self._get_wind()
        
        act = self._ddqn_decide(q_for_state, metrics, wind)
        act_name = {0:"DESCEND",1:"HOVER",2:"ASCEND"}.get(act, "HOVER")
        print(f"[DDQN] action={act} ({act_name}) from state(Q={q_for_state:.1f}%)")
        
        cur_alt = float(self.tel.rel_alt_m) if np.isfinite(self.tel.rel_alt_m) else (self._locked_altitude or self.args.start_alt)
        
        if act == ACT_DESCEND:
            target_alt = clamp(cur_alt - self.args.step_m, self.args.min_alt, self.args.max_alt)
            note = "ddqn_descend"
        elif act == ACT_ASCEND:
            target_alt = clamp(cur_alt + self.args.step_m, self.args.min_alt, self.args.max_alt)
            note = "ddqn_ascend"
        else:
            target_alt = cur_alt
            note = "ddqn_hover"
        
        # Try altitude change while maintaining slow speed
        if abs(target_alt - cur_alt) >= 0.2:
            await self._goto_same_latlon_rel_alt(target_alt)
            await asyncio.sleep(0.6)
            await self._refresh_telemetry_once()
        else:
            await asyncio.sleep(0.5)
        
        q3 = await self._take_one_shot(note=note, force=True)
        
        # If improved, restore speed and rewrite mission
        if q3 is not None and np.isfinite(q3) and q3 >= Q_OPT_LO:
            print(f"[LOCK] Improved Q={q3:.1f}% at alt={float(self.tel.rel_alt_m):.1f}m -> lock & rewrite")
            self._locked_altitude = float(self.tel.rel_alt_m)
            changed = await self._rewrite_mission_altitudes(self._locked_altitude)
            if changed > 0:
                print(f"[LOCK] Updated {changed} mission waypoints to alt={self._locked_altitude:.1f}m")
            
            if speed_success:
                await self._restore_normal_speed()
            await self._continue_mission_anti_pingpong()
            return True
        
        # Not improved -> return to locked altitude if set
        if self._locked_altitude is not None:
            await self._goto_same_latlon_rel_alt(self._locked_altitude)
        
        # Always restore speed before continuing
        if speed_success:
            await self._restore_normal_speed()
        await self._continue_mission_anti_pingpong()
        return False

    def _ddqn_decide(self, q: float, metrics: Dict[str, float], wind: Optional[float]) -> int:
        state = build_ddqn_state(
            q=q,
            metrics=metrics,
            rel_alt_m=float(self.tel.rel_alt_m) if np.isfinite(self.tel.rel_alt_m) else self.args.start_alt,
            wind=wind,
            min_alt=self.args.min_alt,
            max_alt=self.args.max_alt,
        )
        a = ddqn_choose_action(self.ddqn, state, device=self.device)

        # Optional safety override: your rule says Q<80 should descend.
        if self.args.force_descend_when_lowq and (q is not None) and np.isfinite(q) and (q < Q_OPT_LO):
            a = ACT_DESCEND

        return a

    # ---------- optional high-quality ascend trigger ----------
    async def _maybe_highq_ascend_search(self):
        if not self.args.enable_highq_ascent:
            return
        if len(self._recent_q) < 4:
            return
        if not all((q is not None and np.isfinite(q) and q >= Q_UP_TRIGGER) for q in self._recent_q):
            return
        
        cur_alt = float(self.tel.rel_alt_m) if np.isfinite(self.tel.rel_alt_m) else (self._locked_altitude or self.args.start_alt)
        target_alt = clamp(cur_alt + self.args.step_m, self.args.min_alt, self.args.max_alt)
        if abs(target_alt - cur_alt) < 0.2:
            return
        
        print(f"[HIGH-Q] Last 4 shots >= {Q_UP_TRIGGER:.1f}% -> slow down and try ascend to {target_alt:.1f}m")
        
        # Slow down instead of pause/hold
        speed_success = await self._slow_down_for_correction(0.2)
        await asyncio.sleep(1.0)
        await self._refresh_telemetry_once()
        
        await self._goto_same_latlon_rel_alt(target_alt)
        await asyncio.sleep(0.6)
        await self._refresh_telemetry_once()
        
        q_test = await self._take_one_shot(note="highq_ascend_test", force=True)
        if q_test is not None and np.isfinite(q_test) and q_test >= Q_UP_TRIGGER:
            print(f"[HIGH-Q] Still >= {Q_UP_TRIGGER:.1f}% at {target_alt:.1f}m -> lock & rewrite")
            self._locked_altitude = float(self.tel.rel_alt_m)
            changed = await self._rewrite_mission_altitudes(self._locked_altitude)
            if changed > 0:
                print(f"[LOCK] Updated {changed} mission waypoints to alt={self._locked_altitude:.1f}m")
        
        # Restore normal speed and resume
        if speed_success:
            await self._restore_normal_speed()
        await self._continue_mission_anti_pingpong()
        
        # Reset deque
        self._recent_q.clear()

    # ---------- main loop ----------
    async def run(self):
        await self._connect()
        print(f"[INFO] Saving shots to: {os.path.expanduser(self.save_dir)}")
        print(f"[INFO] DDQN policy: {os.path.expanduser(self.args.ddqn_policy)}  device={self.device}")
        print("[INFO] Initial scan: up 1m steps until Q < 80, lock at (bad_alt - 1m), rewrite mission")
        print("[INFO] During mission: if Q < 80 -> slow down to 0.2 m/s -> retry -> DDQN decides +/- step_m -> restore speed")
        if self.args.wind_topic:
            print(f"[INFO] Wind topic: {self.args.wind_topic}")

        # Wait for takeoff and reaching start altitude
        print("[WAIT] Waiting for takeoff and reaching start altitude...")
        while True:
            await self._refresh_telemetry_once()

            if self.tel.flight_mode == "RETURN_TO_LAUNCH":
                self._rtl_seen = True

            if self._rtl_seen and (not self.tel.armed) and (not self.tel.in_air):
                print("[DONE] RTL complete -> exit")
                return

            if self.tel.in_air and np.isfinite(self.tel.rel_alt_m) and self.tel.rel_alt_m >= self.args.start_alt:
                print(f"[READY] Airborne at {self.tel.rel_alt_m:.1f}m - Starting")
                break

            if self.args.verbose_status:
                alt_str = f"{self.tel.rel_alt_m:.1f}" if np.isfinite(self.tel.rel_alt_m) else "nan"
                print(f"[WAIT] armed={self.tel.armed} in_air={self.tel.in_air} alt={alt_str}m mode={self.tel.flight_mode} wp={self.tel.waypoint_index}")

            await asyncio.sleep(0.5)

        # Initial scan + lock altitude + rewrite mission
        if not self._did_initial_scan:
            self._did_initial_scan = True
            await self._initial_scan_find_lock_altitude()
            self._mission_in_progress = True

        print("[MISSION] Main loop")
        while not self._rtl_seen:
            await self._refresh_telemetry_once()

            if self.tel.flight_mode == "RETURN_TO_LAUNCH":
                self._rtl_seen = True
                break

            # If mission index goes backwards, try to correct (anti-pingpong)
            if self.tel.waypoint_index >= 0 and self._max_wp_seen >= 0 and self.tel.waypoint_index < self._max_wp_seen:
                await self._continue_mission_anti_pingpong()

            if self.tel.flight_mode == "MISSION":
                q_auto = await self._take_one_shot(note="auto", force=False)

                # Optional high-Q ascend trigger
                if q_auto is not None and np.isfinite(q_auto):
                    await self._maybe_highq_ascend_search()

                # Low-Q correction with cooldown
                if (q_auto is not None and np.isfinite(q_auto) and q_auto < Q_OPT_LO and
                    self._mission_in_progress and now_s() > self._correction_cooldown):
                    ok = await self._handle_low_quality_ddqn(q_auto)
                    self._correction_cooldown = now_s() + float(self.args.correction_cooldown_s)
                    if not ok:
                        print("[MISSION] Could not improve altitude this time; continuing.")
            else:
                await asyncio.sleep(0.8)

            await asyncio.sleep(0.4)

        print("[DONE] Mission complete")

# ------------------------- args -------------------------
def build_argparser():
    ap = argparse.ArgumentParser()

    # For your SITL: PX4 is on 14580. udp:// works (warning is OK).
    ap.add_argument("--connect_url", default="udp://127.0.0.1:14580",
                    help="MAVSDK vehicle URL (examples: udp://127.0.0.1:14580, udpout://127.0.0.1:14580, udpin://0.0.0.0:14540)")

    ap.add_argument("--camera_topic", required=True)
    ap.add_argument("--wind_topic", default=None)
    ap.add_argument("--iqa_py", required=True)
    ap.add_argument("--save_dir", default="~/uav_survey_shots")

    ap.add_argument("--start_alt", type=float, default=18.0)
    ap.add_argument("--min_alt", type=float, default=12.0)
    ap.add_argument("--max_alt", type=float, default=40.0)

    ap.add_argument("--rewrite_mission_alt", action="store_true")

    ap.add_argument("--min_shot_dist_m", type=float, default=4.0)
    ap.add_argument("--max_frame_age_s", type=float, default=1.0)
    ap.add_argument("--max_wind_age_s", type=float, default=2.0)

    ap.add_argument("--correction_cooldown_s", type=float, default=30.0)

    # DDQN
    ap.add_argument("--ddqn_policy", required=True, help="Path to DDQN model (.pth). Must be full Module or TorchScript.")
    ap.add_argument("--ddqn_device", default="cpu", help="cpu or cuda")
    ap.add_argument("--step_m", type=float, default=1.0, help="Altitude step for DDQN actions")

    # Behavior toggles
    ap.add_argument("--force_descend_when_lowq", action="store_true",
                    help="Override DDQN: if Q<80 always descend (matches your rule).")
    ap.add_argument("--enable_highq_ascent", action="store_true",
                    help="If last 4 shots >= 86%, pause+hold then test ascend by 1m and lock if still >=86%")

    ap.add_argument("--verbose_status", action="store_true")
    ap.add_argument("--print_max_metrics", type=int, default=6)
    return ap

async def _run_app(args):
    app = MissionCompanion(args)
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
