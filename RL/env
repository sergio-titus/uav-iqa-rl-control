import time
import threading
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
import yaml

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

from grpc.aio import AioRpcError

from iqa.iqa_model import process_bgr_image

def load_yaml_config(filename: str):
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "configs" / filename
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class UAVIQAEnv(Node):
    def __init__(
    self,
    env_config_path: str = "env.yaml",
    reward_config_path: str = "reward.yaml",
):
        super().__init__("uav_iqa_env_node")

env_cfg = load_yaml_config(env_config_path)["env"]
reward_cfg = load_yaml_config(reward_config_path)["reward"]

image_topic = env_cfg["image_topic"]
wind_speed_topic = env_cfg["wind_speed_topic"]
wind_burst_topic = env_cfg["wind_burst_topic"]

self.step_dt = float(env_cfg["step_dt"])

self.alt_min = float(env_cfg["altitude"]["min"])
self.alt_max = float(env_cfg["altitude"]["max"])
self.start_alt = float(env_cfg["altitude"]["start"])

self.home_n = float(env_cfg["home"]["north"])
self.home_e = float(env_cfg["home"]["east"])
self.yaw_deg = float(env_cfg["home"]["yaw_deg"])

connect_url = env_cfg["mavsdk"]["connect_url"]

self.reward_cfg = reward_cfg
self.target_quality = float(reward_cfg["target_quality"])

self.bridge = CvBridge()
        # latest signals
        self.last_img_bgr = None
        self.last_iqa: Optional[Dict[str, float]] = None
        self.last_wind_speed = 0.0
        self.last_wind_burst = False

        # ROS subscriptions
        self.create_subscription(Image, image_topic, self._img_cb, 10)
        self.create_subscription(Float32, wind_speed_topic, self._wind_speed_cb, 10)
        self.create_subscription(Bool, wind_burst_topic, self._wind_burst_cb, 10)

        # MAVSDK async thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        self._drone: Optional[System] = None
        self._connected = False
        self._offboard_started = False

        self._ned_d = 0.0
        self._alt_m = 0.0

        self._target_alt = self.start_alt
        self._target_speed = 0.6

        # start connection in background (no hard crash if it retries)
        self._run_async(self._connect_forever(connect_url))

        self.n_actions = 3
        self.obs_dim = 12

    # ---------------- ROS callbacks ----------------
    def _img_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_img_bgr = bgr
            self.last_iqa = process_bgr_image(bgr)
        except Exception as e:
            self.get_logger().warn(f"IQA image cb failed: {e}")

    def _wind_speed_cb(self, msg: Float32):
        self.last_wind_speed = float(msg.data)

    def _wind_burst_cb(self, msg: Bool):
        self.last_wind_burst = bool(msg.data)

    # ---------------- Async helpers ----------------
    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut

    async def _connect_forever(self, url: str):
        """
        Keep trying to connect. If gRPC closes, recreate System() and reconnect.
        """
        while True:
            try:
                self._drone = System()
                await self._drone.connect(system_address=url)

                async for state in self._drone.core.connection_state():
                    if state.is_connected:
                        self._connected = True
                        break

                # start telemetry
                self._loop.create_task(self._telemetry_task())

                # arm + takeoff + offboard
                await self._drone.action.arm()
                await self._drone.action.set_takeoff_altitude(self.start_alt)
                await self._drone.action.takeoff()
                await asyncio.sleep(3.0)

                self._target_alt = float(np.clip(self.start_alt, self.alt_min, self.alt_max))
                await self._send_alt_setpoint(self._target_alt)
                try:
                    await self._drone.offboard.start()
                    self._offboard_started = True
                except OffboardError:
                    self._offboard_started = False

                # if we got here, stay connected until something breaks
                while self._connected:
                    await asyncio.sleep(0.5)

            except (AioRpcError, ConnectionError, RuntimeError) as e:
                self._connected = False
                self._offboard_started = False
                self.get_logger().warn(f"MAVSDK disconnected, retrying... ({type(e).__name__}: {e})")
                await asyncio.sleep(1.0)
            except Exception as e:
                self._connected = False
                self._offboard_started = False
                self.get_logger().warn(f"MAVSDK error, retrying... ({type(e).__name__}: {e})")
                await asyncio.sleep(1.0)

    async def _telemetry_task(self):
        while True:
            try:
                async for pv in self._drone.telemetry.position_velocity_ned():
                    self._ned_d = float(pv.position.down_m)
                    self._alt_m = float(-self._ned_d)
            except AioRpcError:
                self._connected = False
                self._offboard_started = False
                return
            except Exception:
                self._connected = False
                self._offboard_started = False
                return

    async def _send_alt_setpoint(self, alt_m: float):
        alt_m = float(np.clip(alt_m, self.alt_min, self.alt_max))
        down = -alt_m
        sp = PositionNedYaw(self.home_n, self.home_e, down, self.yaw_deg)
        await self._drone.offboard.set_position_ned(sp)

    async def _ensure_offboard(self):
        if not self._connected or self._drone is None:
            return
        try:
            await self._send_alt_setpoint(self._target_alt)
            if not self._offboard_started:
                await self._drone.offboard.start()
                self._offboard_started = True
        except (AioRpcError, OffboardError):
            self._offboard_started = False

    # ---------------- Public API ----------------
    def set_target_speed(self, v: float):
        self._target_speed = float(max(0.3, min(3.0, v)))

    def reset(self):
        # wait for mavsdk connection + at least one IQA sample
        t0 = time.time()
        while (not self._connected) and (time.time() - t0 < 20.0):
            rclpy.spin_once(self, timeout_sec=0.1)

        t1 = time.time()
        while self.last_iqa is None and time.time() - t1 < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        self._target_alt = float(np.clip(self.start_alt, self.alt_min, self.alt_max))
        return self._get_obs()

    def step(self, action: int):
        if action == 0:
            self._target_alt -= 1.0
        elif action == 2:
            self._target_alt += 1.0
        self._target_alt = float(np.clip(self._target_alt, self.alt_min, self.alt_max))

        inner_hz = int(np.clip(self._target_speed / 0.6, 1, 6))
        inner_dt = self.step_dt / inner_hz

        for _ in range(inner_hz):
            asyncio.run_coroutine_threadsafe(self._ensure_offboard(), self._loop)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(inner_dt)

        obs = self._get_obs()
        info = self._get_info()

        q = info["quality"]
alt = info["altitude"]

target_q = self.target_quality

if q >= target_q:
    success_cfg = self.reward_cfg["success"]
    reward = float(success_cfg["base"])
    reward += float(success_cfg["altitude_weight"]) * (alt / self.alt_max)
    reward += float(success_cfg["tolerance_bonus"]) * max(
        0.0,
        1.0 - abs(q - target_q) / float(success_cfg["tolerance_range"])
    )
else:
    failure_cfg = self.reward_cfg["failure"]
    reward = float(failure_cfg["base_penalty"]) - (
        (target_q - q) * float(failure_cfg["quality_penalty_scale"])
    )

        return obs, float(reward), False, info

    def _get_obs(self):
        rclpy.spin_once(self, timeout_sec=0.0)
        iqa = self.last_iqa or {
            "quality_%": 0.0, "blur_%": 100.0, "lowres_%": 100.0,
            "under_%": 0.0, "over_%": 0.0, "noise_%": 0.0, "haze_%": 0.0,
            "edge_density": 0.0, "fft_energy": 0.0
        }

        return np.array([
            float(self._alt_m),
            float(self.last_wind_speed),
            1.0 if self.last_wind_burst else 0.0,
            float(iqa["quality_%"]),
            float(iqa["blur_%"]),
            float(iqa["lowres_%"]),
            float(iqa["under_%"]),
            float(iqa["over_%"]),
            float(iqa["noise_%"]),
            float(iqa["haze_%"]),
            float(iqa["edge_density"]),
            float(iqa["fft_energy"]),
        ], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        iqa = self.last_iqa or {}
        q = float(iqa.get("quality_%", 0.0))
        return {
            "altitude": float(self._alt_m),
            "wind": float(self.last_wind_speed),
            "wind_burst": bool(self.last_wind_burst),
            "quality": q,
            "blur": float(iqa.get("blur_%", 0.0)),
            "lowres": float(iqa.get("lowres_%", 0.0)),
            "under": float(iqa.get("under_%", 0.0)),
            "over": float(iqa.get("over_%", 0.0)),
            "noise": float(iqa.get("noise_%", 0.0)),
            "haze": float(iqa.get("haze_%", 0.0)),
            "edge_density": float(iqa.get("edge_density", 0.0)),
            "fft_energy": float(iqa.get("fft_energy", 0.0)),
            "is_ge80": bool(q >= 80.0),
            "abs_err_to_80": float(abs(q - 80.0)),
        }
