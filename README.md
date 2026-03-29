# UAV-IQA-RL-Control

Perception-driven UAV altitude control using reinforcement learning and image quality assessment for precision agriculture.

---

## Overview

This repository implements a closed-loop aerial perception and control framework in which a UAV dynamically adapts its altitude based on real-time image quality.

The objective is to maintain an optimal trade-off between:
- image clarity
- flight stability
- safety
- area coverage

---

## Training

Run any module from a single entry point:

```bash
python main.py --task ddqn
python main.py --task ppo
python main.py --task yolo
python main.py --task cnn

## Simulation Workflow (PX4 + Gazebo + ROS2)

# 1. Launch QGroundControl
~/apps/qgc/QGroundControl.AppImage

# 2. Kill previous simulation
pkill -9 -f 'gzserver|gzclient|gazebo|px4' 2>/dev/null || true

# 3. Set Gazebo environment
export GAZEBO_PLUGIN_PATH=~/ros2_ws/install/gazebo_env_plugins/lib:\
~/PX4-Autopilot/build/px4_sitl_default/build_gazebo-classic:\
/usr/lib/x86_64-linux-gnu/gazebo-11/plugins

export GAZEBO_MODEL_PATH=~/ros2_ws/src/RRT_DQN_PATH_PLANNING/models:\
~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models

# 4. Start PX4 SITL
cd ~/PX4-Autopilot
make px4_sitl gazebo-classic

# 5. Run DDQN mission
python3 ~/ros2_ws/src/uav_iqa_ddqn/mission_companion_ddqn.py \
  --connect_url udpin://0.0.0.0:14540 \
  --camera_topic /iris/gazebo_distorted_preview/image_raw \
  --wind_topic /uav/wind_speed \
  --iqa_py ~/ros2_ws/src/uav_iqa_ddqn/iqa/iqa_model.py \
  --ddqn_policy ~/ros2_ws/src/uav_iqa_ddqn/checkpoints/DDQN_training/best_policy.pt \
  --save_dir ~/uav_survey_shots \
  --start_alt 18 --min_alt 12 --max_alt 40 \
  --rewrite_mission_alt \
  --min_shot_dist_m 4.0 \
  --max_frame_age_s 1.0 \
  --force_descend_when_lowq \
  --verbose_status
