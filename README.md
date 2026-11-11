# SO-101 Embodied Mac Studio 🦾🍎

A high-performance, full-stack **Embodied AI & Imitation Learning (ACT)** workbench running natively on **Apple Silicon (Mac M-Series)** for the **SO-ARM101** robotic arm.

Designed to remove the friction of robotic teleoperation, dataset recording, local training, and policy inference without requiring expensive Linux/NVIDIA workstation setups.

---

## ✨ Highlights

- **🍎 100% Native Apple Silicon Acceleration**:
  - ACT (Action Chunking with Transformers) training and inference accelerated by **Metal Performance Shaders (MPS)**.
  - End-to-end 10Hz closed-loop control running smoothly on Mac mini / MacBook (M1/M2/M3/M4).
- **🖥️ Full-Featured Web Studio UI (`:8000`)**:
  - **Live Camera Feed**: 30 FPS video streaming with hardware release management.
  - **Multi-Version Waypoint Teaching**: 4-point teaching interface with fine-grained parameter editing, trajectory preview, and cross-version cloning.
  - **Automated Demonstration Harvester**: Execute smooth interpolated trajectories to record high-quality imitation learning datasets automatically.
  - **One-Click Local Training**: Trigger local MPS ACT training with live step/loss monitoring directly in the browser.
  - **ACT Autonomous Evaluation**: Integrated model catalog, single-trial autonomous runs, and real-time motion safety interlocks.
- **🛡️ Built-in Hardware Safety & Motion Guard**:
  - Safe torque preloading (never drop arms unexpectedly).
  - Anti-regrasp guard to prevent closed-loop loop-locking or table damage.
  - Independent Standby Recovery to ensure identical initial boundary conditions.
- **👁️ Multimodal Perception (VLM) Integration Ready**:
  - Seamless bridge to **Grounding-Pi** with **Rejection Lab LoRA** for natural language target grounding and false-positive target rejection.

---

## 🏛️ System Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │             Browser Web Studio UI (Port 8000)          │
       │   - Live Video HUD           - Waypoint Teaching & Edit│
       │   - Harvester & Recorder     - Local MPS ACT Training  │
       │   - Model Catalog & Eval     - Grounding-Pi VLM Panel  │
       └───────────────────────────┬────────────────────────────┘
                                   │ HTTP / WebSockets / MJPEG
       ┌───────────────────────────▼────────────────────────────┐
       │              FastAPI Embodied Backend Server           │
       │   [ scripts/teleop_studio.py ]                         │
       ├───────────────────────────┬────────────────────────────┤
       │                           │                            │
       ▼                           ▼                            ▼
┌──────────────┐          ┌───────────────────┐       ┌──────────────────┐
│  Perception  │          │   Policy Engine   │       │  Hardware Bus    │
│  OpenCV Cam  │          │  LeRobot ACT MPS  │       │  Rustypot STS    │
│  GroundingPi │          │  Chunk:50, Step:20│       │  Follower (120)  │
│  (Port 8001) │          │  Adaptive Offset  │       │  Leader   (130)  │
└──────────────┘          └───────────────────┘       └──────────────────┘
```

---

## 🚀 Quick Start

### 1. Requirements & Hardware

- **Host**: Mac with Apple Silicon (M1/M2/M3/M4), macOS 14+.
- **Robot Arm**: [SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) with Feetech STS3215 serial bus servos.
- **USB Serial Controller**: Dedicated USB-to-RS485/TTL adapter (e.g. `/dev/cu.usbserial-120`).
- **Camera**: Standard UVC USB camera or integrated Mac webcam.

### 2. Environment Setup

```bash
git clone https://github.com/kongzhuww/so-arm101-embodied-mac-studio.git
cd so-arm101-embodied-mac-studio

# Create virtual environment and install dependencies
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-installed.txt
```

### 3. Launching the Web Studio

```bash
bash run python -B -u scripts/teleop_studio.py
```
Open **http://localhost:8000** in your browser.

---

## 📖 Typical Workflow

### Phase 1: Waypoint Teaching & Demonstration Harvesting
1. In the Web Studio, pose the arm by hand and record the 4 core points:
   - `P1`: Standby
   - `P2`: Grasp
   - `P3`: Lift
   - `P4`: Box Release
2. Click **"▶️ 跑一次当前示教脚本"** to test trajectory smoothness.
3. Click **"🎬 自动抓取并录制一条"** to collect demonstration episodes with synchronized camera frames and servo telemetry.

### Phase 2: Local ACT Training on Apple Silicon (MPS)
1. Select the training scope (current version or merged versions).
2. Click **"训练所选示范 (MPS)"**.
3. Training runs completely locally on your Mac GPU with live loss updates.

### Phase 3: Autonomous ACT Policy Deployment
1. Click **"切换 ACT 模型"** to switch into autonomous testing mode.
2. Select your trained checkpoint from the dropdown.
3. Click **"回到示教待机位"** to standardize initial posture.
4. Click **"▶️ 跑一次所选 ACT 模型"** to observe real-time vision-based closed-loop manipulation!

---

## 📂 Project Structure

```text
├── calibration/             # Servo offsets and kinematic calibration
├── config/                  # Waypoints and studio configurations
├── lerobot/                 # LeRobot ACT policy modeling and control utils
├── reports/                 # Offline diagnostic logs and trial reports
├── scripts/
│   ├── teleop_studio.py     # Core Web Studio server & interactive GUI
│   ├── studio_act.py        # Autonomous ACT policy runner & safety monitor
│   ├── studio_standby.py    # Non-destructive standby recovery controller
│   ├── studio_visual.py     # Background normalization & image alignment
│   ├── train_act_repair.py  # Isolated offline training & validation pipeline
│   ├── train_act_visual.py  # Visual domain fine-tuning with data augmentation
│   └── repair_gripper_zero.py # High-precision servo zero-point correction
└── requirements-installed.txt
```

---

## 🛡️ License

This project is licensed under the Apache 2.0 License.
