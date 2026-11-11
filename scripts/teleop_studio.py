"""Complete SO-101 Web Studio:
- Camera Live Feed (30 FPS)
- Twin-Teleop (Real-time Leader-Follower Mirroring)
- Free-Hand Waypoint Teaching (Record 4 Points directly in Browser UI)
- Auto-Run & Record (Execute Perfected Trajectory & Save Dataset)
- Spacebar / One-click Record
- Real-time Episode Management (Delete Last, Clear All)
- Local ACT Training Trigger
"""
import os
import sys
import time
import json
import threading
import subprocess
from pathlib import Path
from contextlib import closing, contextmanager
from teaching_versions import TeachingVersions, POINT_NAMES, derive_waypoints, atomic_json
from datetime import datetime, timezone

import cv2
import numpy as np
import rustypot
from fastapi import FastAPI, Request
from pydantic import BaseModel, StrictInt
from typing import Literal
from studio_act import ActModels
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "calibration/twin_mirror_profile.json"
WP_FILE = ROOT / "config/auto_waypoints.json"
PORT_FOLLOWER = "/dev/cu.usbserial-120"
PORT_LEADER = "/dev/cu.usbserial-130"

_, definition = rustypot.find_model('sts3215')
bus_dict = {i: definition for i in range(1, 7)}

app = FastAPI(title="SO-101 Embodied Studio")

# Global State
state = {
    "teleop_active": False,
    "recording": False,
    "current_episode_frames": 0,
    "total_episodes": 0,
    "gripper_pct": 0.0,
    "training": False,
    "workbench_mode": "teach",
    "act_models": [],
    "selected_act_model": None,
    "act_running": False,
    "act_step": 0,
    "standby_running": False,
    "standby_status": "尚未复位；复位不会自动启动ACT",
    "act_status": "尚未运行模型",
    "running_act_model": None,
    "train_step": 0,
    "train_loss": 0.0,
    "camera_enabled": os.environ.get('STUDIO_CAMERA_START','0')=='1',
    "camera_active": False,
    "camera_error": "",
    "status_msg": "就绪",
    "leader_port": PORT_LEADER,
    "follower_port": PORT_FOLLOWER,
    "waypoints": {
        "p1": None,
        "p2": None,
        "p3": None,
        "p4": None
    }
}

# Load saved waypoints if exist
if WP_FILE.exists():
    try:
        wp_data = json.loads(WP_FILE.read_text())
        state["waypoints"]["p1"] = wp_data.get("p1_standby")
        state["waypoints"]["p2"] = wp_data.get("p2_grasp")
        state["waypoints"]["p3"] = wp_data.get("p3_lift")
        state["waypoints"]["p4"] = wp_data.get("p4_release")
    except Exception:
        pass

latest_jpeg = None
latest_frame_time = 0.0
lock = threading.Lock()
record_lock = threading.Lock()
motion_lock = threading.Lock()
version_store = None  # Initialized at service startup, not on module import.
act_library = None
act_thread = None
act_stop_event = threading.Event()
standby_stop_event = threading.Event()
standby_thread = None


def refresh_act_models():
    if act_library:
        selection = act_library.current()
        state.update(workbench_mode=selection['mode'],act_models=selection['models'],
                     selected_act_model=selection['model'])


@contextmanager
def configuration_edit(require_teach=True):
    if require_teach and state.get('workbench_mode','teach') != 'teach':
        raise ValueError('当前是 ACT 测试模式，请先切回示教 / 录制')
    if state['training'] or state['recording'] or state['teleop_active'] or (teleop_thread and teleop_thread.is_alive()):
        raise ValueError('请先结束当前动作、录制或训练')
    if not motion_lock.acquire(blocking=False):
        raise ValueError('机械臂正在运行，不能切换或修改版本')
    try:
        yield
    finally:
        motion_lock.release()


def sync_version_state():
    global DATASET_DIR
    v = version_store.active()
    DATASET_DIR = version_store.directory(v)
    state['active_version'] = {k: v[k] for k in ('id', 'name', 'revision', 'open_gripper', 'updated_at')}
    state['versions'] = version_store.summaries()
    state['dataset_dir'] = str(DATASET_DIR)
    state['total_episodes'] = len(version_store.episode_files(v))
    state['waypoints'] = {key: v['waypoints'].get(name) for key, name in POINT_NAMES.items()}


def active_waypoints():
    return version_store.active()['waypoints'] if version_store else json.loads(WP_FILE.read_text())


from studio_samples import validate_episode
import base64


def capture_sample(bus, target):
    # Observation precedes the corresponding command; follower, never leader.
    with lock:
        jpeg, image_time = latest_jpeg, latest_frame_time
    now = time.time()
    if jpeg is None or not 0 <= now - image_time <= 0.25:
        raise RuntimeError('摄像头帧不可用/过期，本条录制不入库')
    positions = []
    for motor in range(1, 7):
        value, error = bus.read_register_with_error(motor, 'present_position', retries=1)
        if error:
            raise RuntimeError(f'从臂 {motor} 号读取失败: {error}')
        positions.append(int(value) & 0x0FFF)
    stamp = time.time()
    if stamp - image_time > 0.25:
        raise RuntimeError('串口读取耗时过长，图像与关节不同步')
    return {'schema_version': 2, 'timestamp': stamp,
            'image_timestamp': image_time, 'state': positions,
            'action': [int(v) for v in target],
            'image_jpeg': base64.b64encode(jpeg).decode('ascii')}


def save_episode(rows):
    validate_episode(rows)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    files = list(DATASET_DIR.glob('episode_*.json'))
    index = max((int(f.stem.split('_')[-1]) for f in files), default=-1) + 1
    path = DATASET_DIR / f'episode_{index:04d}.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(rows))
    temp.replace(path)
    state['total_episodes'] = len(files) + 1
    if version_store:
        v = version_store.active()
        atomic_json(path.parent / 'metadata' / path.name, {'version': v, 'frames': len(rows),
                    'recorded_at': datetime.now(timezone.utc).isoformat()})
        sync_version_state()
    state['status_msg'] = f'已保存真实图像示范：{len(rows)} 帧'

teleop_thread = None
camera_thread = None
stop_teleop = threading.Event()
stop_camera = threading.Event()
camera_wake = threading.Event()
camera_released = threading.Event()
camera_released.set()
camera_control_lock = threading.Lock()
current_episode_data = []

# Reuse latest studio_dataset if available, otherwise create a new one
existing_dirs = sorted((ROOT / "datasets").glob("studio_dataset_v2_*"))
if existing_dirs and list(existing_dirs[-1].glob("episode_*.json")):
    DATASET_DIR = existing_dirs[-1]
    state["total_episodes"] = len(list(DATASET_DIR.glob("episode_*.json")))
else:
    DATASET_DIR = ROOT / "datasets" / f"studio_dataset_v2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SO-101 具身工作台</title>
  <style>
    :root { --bg: #090d16; --panel: #162032; --card: #0f172a; --border: #293548; --accent: #3b82f6; --danger: #ef4444; --success: #22c55e; --purple: #8b5cf6; }
    body { margin: 0; background: var(--bg); color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
    
    #top-bar { height: 52px; background: #0c1322; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; padding: 0 20px; box-sizing: border-box; }
    .hw-tag { display: flex; align-items: center; gap: 8px; background: var(--panel); padding: 5px 12px; border-radius: 6px; border: 1px solid var(--border); font-size: 12px; font-family: monospace; }
    .dot { width: 8px; height: 8px; border-radius: 50%; }
    .dot-green { background: var(--success); box-shadow: 0 0 6px var(--success); }
    .dot-blue { background: var(--accent); box-shadow: 0 0 6px var(--accent); }

    #container { flex: 1; display: flex; overflow: hidden; }
    #sidebar { width: 380px; background: var(--panel); padding: 16px; display: flex; flex-direction: column; gap: 12px; border-right: 1px solid var(--border); overflow-y: auto; }
    #main { flex: 1; padding: 16px; display: flex; flex-direction: column; gap: 12px; align-items: center; justify-content: center; }
    
    .card { background: var(--card); border-radius: 10px; padding: 12px; border: 1px solid var(--border); }
    .card-title { font-size: 13px; font-weight: bold; color: #94a3b8; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
    .btn { background: var(--accent); color: white; border: none; border-radius: 6px; padding: 10px; font-size: 13px; font-weight: bold; cursor: pointer; transition: all 0.15s; width: 100%; box-sizing: border-box; }
    .btn:hover { filter: brightness(1.1); transform: translateY(-1px); }
    .btn:disabled { opacity: 0.35; cursor: not-allowed; transform: none; }
    .btn-danger { background: var(--danger); }
    .btn-success { background: var(--success); }
    .btn-purple { background: var(--purple); }
    .btn-secondary { background: #334155; }
    
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .wp-btn { text-align: left; padding: 10px; font-size: 12px; background: #1e293b; border: 1px solid var(--border); color: #cbd5e1; border-radius: 6px; cursor: pointer; display: flex; flex-direction: column; gap: 4px; }
    .wp-btn:hover { background: #334155; }
    .wp-btn b { color: white; font-size: 13px; }
    .wp-done { border-color: var(--success); background: #14532d22; }
    
    #video-box { width: 640px; height: 480px; background: black; border-radius: 12px; overflow: hidden; border: 2px solid var(--border); position: relative; box-shadow: 0 10px 25px rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; }
    #live-feed { width: 100%; height: 100%; object-fit: cover; }
    #vision-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
    .overlay-hud { position: absolute; top: 12px; left: 12px; background: rgba(0,0,0,0.7); padding: 5px 10px; border-radius: 6px; font-family: monospace; font-size: 12px; backdrop-filter: blur(4px); }
    .log-box { width: 640px; background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; font-family: monospace; font-size: 12px; color: #94a3b8; }
  </style>
</head>
<body>

  <div id="top-bar">
    <div style="font-weight:bold; font-size:15px; color:#f8fafc;">SO-101 具身操作工作台</div>
    <div style="display:flex; gap:12px;">
      <div class="hw-tag"><span class="dot dot-blue"></span>主控 (Leader): <b style="color:#60a5fa;" id="hw-leader">130</b> (4.1V 示教)</div>
      <div class="hw-tag"><span class="dot dot-green"></span>从动 (Follower): <b style="color:#4ade80;" id="hw-follower">120</b> (12.3V 动力)</div>
    </div>
  </div>

  <div id="container">
    <div id="sidebar">
      
      <div class="card" style="border-color:#64748b;">
        <div class="grid-2">
          <button id="mode-teach" class="btn" onclick="changeWorkbenchMode('teach')">示教 / 录制</button>
          <button id="mode-act" class="btn btn-secondary" onclick="changeWorkbenchMode('act')">切换 ACT 模型</button>
        </div>
      </div>

      <div id="act-panel" class="card" style="display:none;border-color:#a78bfa;">
        <div class="card-title">🧠 ACT 模型测试</div>
        <select id="act-model-select" style="width:100%;padding:8px;" onchange="chooseActModel(this.value)"></select>
        <div id="act-model-info" style="font-size:11px;color:#c4b5fd;margin-top:8px;word-break:break-all;"></div>
        <div style="font-size:11px;color:#94a3b8;margin:8px 0;">只选择模型不会移动。点下方跑一次会驱动机械臂，不使用示教版本的点位，不录制素材。</div>
        <button id="act-stop" class="btn btn-danger" onclick="versionRequest('/api/act/stop')" disabled>停止 ACT · 保持当前位置</button>
        <div id="act-status" style="font-size:11px;color:#fbbf24;margin-top:8px;">尚未运行模型</div>
      </div>

      <div id="versions-card" data-teach-panel class="card" style="border-color:#a78bfa;">
        <div class="card-title">🗂 示教版本 <span id="version-badge"></span></div>
        <select id="version-select" data-version-edit onchange="switchVersion(this.value)" style="width:100%; padding:8px;"></select>
        <div id="version-info" title="已有素材后再改点位，会另存新修订；原素材保留在历史中，不会被覆盖。" style="font-size:11px; color:#94a3b8; margin:8px 0;"></div>
        <div class="grid-2">
          <button class="btn btn-secondary" data-version-edit onclick="createVersion(true)">复制成下一版本</button>
          <button class="btn btn-secondary" data-version-edit onclick="createVersion(false)">新增空白版本</button>
        </div>
        <div style="font-size:11px;color:#94a3b8;margin-top:7px;">自动编号，不用取名。切换不会移动机械臂。</div>
      </div>

      <!-- 网页示教打点 (Waypoint Teaching) -->
      <div id="waypoints-card" data-teach-panel class="card" style="border-color:#38bdf844;">
        <div class="card-title">
          <span style="color:#38bdf8;">📍 示教打点 (手掰从臂记录)</span>
          <button class="btn btn-secondary" style="width:auto; padding:3px 8px; font-size:11px;" onclick="disableTorque()">⚡️ 卸下力矩</button>
        </div>
        <div style="font-size:11px; color:#94a3b8; margin-bottom:8px;">按 1～4 保存当前姿态。2 点轻夹方块，4 点在盒上张爪释放。</div>
        
        <div class="grid-2">
          <div id="wp-1" class="wp-btn" onclick="saveWaypoint('p1')">
            <b>1. 缩回待机点</b>
            <span id="wp-1-val" style="font-size:10px; color:#64748b;">未录制</span>
          </div>
          <div id="wp-2" class="wp-btn" onclick="saveWaypoint('p2')">
            <b>2. 咬合红块点</b>
            <span id="wp-2-val" style="font-size:10px; color:#64748b;">未录制</span>
          </div>
          <div id="wp-3" class="wp-btn" onclick="saveWaypoint('p3')">
            <b>3. 垂直抬升点</b>
            <span id="wp-3-val" style="font-size:10px; color:#64748b;">未录制</span>
          </div>
          <div id="wp-4" class="wp-btn" onclick="saveWaypoint('p4')">
            <b>4. 盒上释放点</b>
            <span id="wp-4-val" style="font-size:10px; color:#64748b;">未录制</span>
          </div>
        </div>
        <details id="point-tools" style="margin-top:10px;" ontoggle="if(this.open)loadPointEditor()">
          <summary style="cursor:pointer;font-size:12px;color:#7dd3fc;">复制 / 编辑点位参数</summary>
          <div style="font-size:11px;color:#94a3b8;margin:8px 0;">保存到 <b id="point-tools-version"></b>，不会移动机械臂；相关过渡点自动更新。</div>
          <div style="padding:8px;background:#1e293b;border-radius:6px;">
            <label style="font-size:12px;">从版本复制指定点位</label>
            <select id="point-copy-source" data-parameter-edit style="width:100%;padding:6px;margin:6px 0;" onchange="updateCopyAvailability()"></select>
            <div class="grid-2" style="font-size:11px;margin-bottom:7px;">
              <label><input id="copy-p1" data-parameter-edit type="checkbox" checked>1 待机</label>
              <label><input id="copy-p2" data-parameter-edit type="checkbox">2 抓取</label>
              <label><input id="copy-p3" data-parameter-edit type="checkbox">3 抬升</label>
              <label><input id="copy-p4" data-parameter-edit type="checkbox" checked>4 放盒</label>
            </div>
            <button class="btn btn-secondary" data-parameter-edit style="font-size:11px;" onclick="copySelectedPoints()">复制所选点位到当前版本</button>
          </div>
          <div style="padding-top:10px;">
            <select id="edit-point" data-parameter-edit style="width:100%;padding:6px;" onchange="loadPointEditor()">
              <option value="p1">编辑 1 · 待机点</option><option value="p2">编辑 2 · 抓取点</option>
              <option value="p3">编辑 3 · 抬升点</option><option value="p4">编辑 4 · 放盒点</option>
            </select>
            <div class="grid-2" style="font-size:11px;gap:7px;margin:8px 0;">
              <label>1 底座<input id="joint-1" data-parameter-edit type="number" min="0" max="4095" step="1" oninput="parameterDirty=true" style="box-sizing:border-box;width:100%;"></label>
              <label>2 肩部<input id="joint-2" data-parameter-edit type="number" min="0" max="4095" step="1" oninput="parameterDirty=true" style="box-sizing:border-box;width:100%;"></label>
              <label>3 肘部<input id="joint-3" data-parameter-edit type="number" min="0" max="4095" step="1" oninput="parameterDirty=true" style="box-sizing:border-box;width:100%;"></label>
              <label>4 腕俯仰<input id="joint-4" data-parameter-edit type="number" min="0" max="4095" step="1" oninput="parameterDirty=true" style="box-sizing:border-box;width:100%;"></label>
              <label>5 腕旋转<input id="joint-5" data-parameter-edit type="number" min="0" max="4095" step="1" oninput="parameterDirty=true" style="box-sizing:border-box;width:100%;"></label>
              <label>6 夹爪<input id="joint-6" data-parameter-edit type="number" min="0" max="4095" step="1" oninput="parameterDirty=true" style="box-sizing:border-box;width:100%;"></label>
            </div>
            <div class="grid-2">
              <button class="btn btn-secondary" data-parameter-edit style="font-size:11px;" onclick="loadPointEditor()">重新载入</button>
              <button class="btn" data-parameter-edit style="font-size:11px;" onclick="saveParameterEdit()">保存这个点位</button>
            </div>
            <div id="point-edit-note" style="font-size:10px;color:#94a3b8;margin-top:6px;">原始舵机刻度 0～4095，必须是整数。参数修改后先试跑验证。</div>
          </div>
        </details>
      </div>

      <!-- 自动化轨迹执行 -->
      <div class="card">
        <div id="run-title" class="card-title">🤖 示教脚本试跑</div>
        <div id="run-context" style="font-size:11px;color:#94a3b8;margin-bottom:8px;"></div>
        <div class="grid-2" style="margin-bottom:6px;">
          <button id="btn-standby" class="btn btn-secondary" onclick="returnToStandby()">回到示教待机位</button>
          <button id="btn-standby-stop" class="btn btn-danger" onclick="versionRequest('/api/standby/stop')" disabled>停止复位 · 保持</button>
        </div>
        <div id="standby-status" style="font-size:11px;color:#fbbf24;margin-bottom:8px;">只复位，不抓放，不启动ACT。请确保夹爪无物体、运动路径清空。</div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <button id="btn-auto-test" class="btn" style="background:#0284c7;" onclick="runOnce()">▶️ 跑一次示教脚本</button>
          <button id="btn-auto-record" class="btn btn-success" onclick="runAutoPlayback(true)">🔴 自动抓取放盒 (并录制入库)</button>
        </div>
      </div>

      <!-- Optional manual teleoperation stays collapsed during script teaching. -->
      <details id="teleop-card" data-teach-panel class="card">
        <summary style="cursor:pointer;font-size:12px;margin-bottom:6px;">🎮 手动遥操作（可选） <span id="teleop-badge" style="color:#64748b;">已断开</span></summary>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <button id="btn-teleop" class="btn btn-secondary" onclick="toggleTeleop()">启动双臂实时跟随</button>
          <button id="btn-record-manual" class="btn btn-secondary" onclick="toggleManualRecord()">手动录制动作 (空格键切换)</button>
        </div>
      </details>

      <!-- 数据集与训练 -->
      <div id="dataset-card" data-teach-panel class="card" style="margin-top:auto;">
        <div class="card-title">
          <span>📦 当前版本素材</span>
          <b id="episodes-count" style="color:var(--success); font-size:14px;">0 条</b>
        </div>
        <div style="display:flex; gap:6px; margin-bottom:8px;">
          <button class="btn btn-secondary" style="font-size:11px; padding:6px;" onclick="deleteLastEpisode()">🗑️ 废弃上一条</button>
          <button class="btn btn-secondary" style="font-size:11px; padding:6px; color:#f87171;" onclick="clearAllEpisodes()">清空本版素材</button>
        </div>
        <select id="train-scope" style="width:100%;padding:6px;margin-bottom:7px;">
          <option value="current">只训练当前版本</option>
          <option value="all">合并所有版本</option>
        </select>
        <button id="btn-train" class="btn btn-purple" onclick="startTraining()">训练所选示范 (MPS)</button>
        <div id="train-status" style="font-size:11px; color:#94a3b8; margin-top:6px; font-family:monospace;">模型就绪</div>
      </div>

    </div>

    <div id="main">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
        <button id="btn-camera" class="btn btn-secondary" style="width:auto;" onclick="toggleCamera()">关闭摄像头</button>
        <span id="camera-status" style="font-size:12px;color:#94a3b8;">摄像头启动中</span>
      </div>
      <div id="video-box">
        <div id="camera-placeholder" style="display:none;color:#94a3b8;">摄像头已关闭（设备已释放）</div>
        <img id="live-feed" src="/video_feed" alt="Camera Stream">
        <canvas id="vision-canvas" width="640" height="480"></canvas>
        <div class="overlay-hud">
          <span id="hud-status">● 待机</span> | 640x480 @ 30fps
        </div>
      </div>

      <!-- Grounding-Pi Vision Recognition Control Panel -->
      <div class="card" style="width: 640px; box-sizing: border-box; border-color: #38bdf8;">
        <div class="card-title">
          <span>🔍 Grounding-Pi 目标定位与拒识实验对比 (端口 8001)</span>
          <span id="vlm-status-indicator" style="font-size:12px; color:#94a3b8;">● 服务状态检查中...</span>
        </div>
        <div style="display:flex; gap:8px; margin-bottom:8px;">
          <input id="vlm-input-phrase" type="text" placeholder="输入识别指令，如 'the red cube', 'the clear container', 'the green apple'" value="the red cube" style="flex:1; padding:8px 10px; border-radius:6px; border:1px solid var(--border); background:#0f172a; color:#f8fafc; font-size:13px;" onkeydown="if(event.key==='Enter')runGroundingDetection()">
          <button id="btn-vlm-detect" class="btn" style="width:140px; background:#0284c7;" onclick="runGroundingDetection()" disabled>🎯 识别与比对</button>
          <button class="btn btn-secondary" style="width:70px;" onclick="clearVisionCanvas()">清除</button>
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
          <div style="background:#0f172a; border-radius:6px; padding:8px; border:1px solid #334155;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
              <span style="font-weight:bold; color:#f87171; font-size:12px;">🔴 MLX-GroundingPI (原版)</span>
              <button id="btn-toggle-base" class="btn" style="width:auto; padding:3px 8px; font-size:11px; background:#334155;" onclick="toggleSpecificModel('base')">加载此模型</button>
            </div>
            <div id="vlm-base-res" style="font-size:11px; color:#cbd5e1; font-family:monospace; min-height:36px; word-break:break-all;">未加载（点击上方按钮开启）</div>
          </div>
          <div style="background:#0f172a; border-radius:6px; padding:8px; border:1px solid #10b981;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
              <span style="font-weight:bold; color:#34d399; font-size:12px;">🟢 Rejection-Lab (LoRA拒识版)</span>
              <button id="btn-toggle-lab" class="btn" style="width:auto; padding:3px 8px; font-size:11px; background:#059669;" onclick="toggleSpecificModel('lab')">加载此模型</button>
            </div>
            <div id="vlm-lab-res" style="font-size:11px; color:#cbd5e1; font-family:monospace; min-height:36px; word-break:break-all;">未加载（点击上方按钮开启）</div>
          </div>
        </div>
      </div>
      <div id="log" class="log-box">状态就绪。建议：先用手摆好姿势，点击左侧【1~4 示教打点】完成定位。</div>
    </div>
  </div>

  <script>
    let isTeleop = false;
    let isManualRecording = false;
    let versionsSignature = '';
    let latestState = null;
    let pointEditorContext = null;
    let parameterDirty = false;
    let copySourceSignature = '';
    let actModelsSignature = '';
    let workbenchPending = false;
    let cameraPending = false;

    async function toggleCamera() {
      if (!latestState || cameraPending) return;
      cameraPending = true;
      document.getElementById('btn-camera').disabled = true;
      try {
        await versionRequest('/api/camera?enabled=' + !latestState.camera_enabled);
      } finally { cameraPending = false; await syncState(); }
    }

    async function workbenchRequest(path) {
      workbenchPending = true;
      document.getElementById('btn-auto-test').disabled = true;
      document.getElementById('act-model-select').disabled = true;
      try { await versionRequest(path); }
      finally { workbenchPending = false; await syncState(); }
    }

    async function changeWorkbenchMode(mode) {
      await workbenchRequest('/api/workbench/mode?mode=' + mode);
    }

    async function chooseActModel(ident) {
      await workbenchRequest('/api/act/select?ident=' + encodeURIComponent(ident));
    }

    // Vision Canvas & Grounding-Pi UI Functions
    function clearVisionCanvas() {
      const canvas = document.getElementById('vision-canvas');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      document.getElementById('vlm-base-res').innerText = vlmStatus.base ? '就绪 (待识别)' : '未加载';
      document.getElementById('vlm-lab-res').innerText = vlmStatus.lab ? '就绪 (待识别)' : '未加载';
    }

    async function runGroundingDetection() {
      const phrase = document.getElementById('vlm-input-phrase').value.trim();
      if (!phrase) return alert('请输入想要定位的目标描述！');
      
      const btn = document.getElementById('btn-vlm-detect');
      btn.disabled = true;
      btn.innerText = '⚡️ 识别中...';
      
      try {
        // Fetch current snapshot frame directly from Studio backend
        const frameBlob = await (await fetch('/api/camera/snapshot')).blob();
        
        const formData = new FormData();
        formData.append('phrase', phrase);
        formData.append('model_type', 'both');
        formData.append('file', frameBlob, 'snapshot.jpg');
        
        const resp = await fetch('http://localhost:8001/detect', {
          method: 'POST',
          body: formData
        });
        
        const data = await resp.json();
        if (!data.ok) throw new Error(data.error || '定位失败');
        
        const canvas = document.getElementById('vision-canvas');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // 1. Process Base Model (Red Box)
        const base = data.results && data.results.base;
        const baseRes = document.getElementById('vlm-base-res');
        if (base) {
          if (base.found) {
            baseRes.innerHTML = '<b>已定位 (' + base.elapsed_seconds + 's)</b><br>中心: <code>[' + base.center_xy.join(', ') + ']</code><br>框: <code>' + JSON.stringify(base.box) + '</code>';
            const p = base.pixels; // [ymin, xmin, ymax, xmax]
            const x = p[1] * 2, y = p[0] * 2, w = (p[3] - p[1]) * 2, h = (p[2] - p[0]) * 2;
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = '#ef4444';
            ctx.font = '11px monospace';
            ctx.fillText('🔴 原版: ' + phrase, x, Math.max(12, y - 4));
          } else {
            baseRes.innerHTML = '<b>拒识 (无目标)</b><br>未识别到目标 / 已过滤';
          }
        } else if (!vlmStatus.base) {
          baseRes.innerText = '未加载此模型 (点击上方按钮加载)';
        }
        
        // 2. Process Lab LoRA Model (Green Box)
        const lab = data.results && data.results.lab;
        const labRes = document.getElementById('vlm-lab-res');
        if (lab) {
          if (lab.found) {
            labRes.innerHTML = '<b style="color:#34d399;">已定位 (' + lab.elapsed_seconds + 's)</b><br>中心: <code>[' + lab.center_xy.join(', ') + ']</code><br>框: <code>' + JSON.stringify(lab.box) + '</code>';
            const p = lab.pixels;
            const x = p[1] * 2, y = p[0] * 2, w = (p[3] - p[1]) * 2, h = (p[2] - p[0]) * 2;
            ctx.strokeStyle = '#10b981';
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
            ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = '#10b981';
            ctx.font = 'bold 12px monospace';
            ctx.fillText('🟢 Lab LoRA: ' + phrase, x, Math.max(14, y - 6));
            ctx.fillStyle = '#f59e0b';
            ctx.beginPath();
            ctx.arc(lab.center_xy[0] * 2, lab.center_xy[1] * 2, 4, 0, Math.PI * 2);
            ctx.fill();
          } else {
            labRes.innerHTML = '<b style="color:#10b981;">🛡️ 成功拒识 (None)</b><br><span style="color:#34d399;">画面未发现目标，防止盲抓！</span>';
          }
        } else if (!vlmStatus.lab) {
          labRes.innerText = '未加载此模型 (点击上方按钮加载)';
        }
        
      } catch (err) {
        alert('视觉识别服务通信失败: ' + err.message + ' 请确认 8001 端口服务已开启。');
      } finally {
        btn.disabled = false;
        btn.innerText = '🎯 识别与比对';
      }
    }

    let vlmStatus = { base: false, lab: false };
    let isVlmPending = false;

    async function checkVlmHealth() {
      try {
        const res = await fetch('http://localhost:8001/health');
        const data = await res.json();
        vlmStatus.base = !!data.base_loaded;
        vlmStatus.lab = !!data.lab_loaded;
        
        const ind = document.getElementById('vlm-status-indicator');
        const btnBase = document.getElementById('btn-toggle-base');
        const btnLab = document.getElementById('btn-toggle-lab');
        const detectBtn = document.getElementById('btn-vlm-detect');
        
        if (isVlmPending) return;
        
        // Update Base Button UI
        if (vlmStatus.base) {
          btnBase.innerText = '🔴 卸载释放';
          btnBase.style.background = '#7f1d1d';
          btnBase.style.color = '#fca5a5';
        } else {
          btnBase.innerText = '加载原版';
          btnBase.style.background = '#334155';
          btnBase.style.color = 'white';
        }
        
        // Update Lab Button UI
        if (vlmStatus.lab) {
          btnLab.innerText = '🟢 卸载释放';
          btnLab.style.background = '#064e3b';
          btnLab.style.color = '#6ee7b7';
        } else {
          btnLab.innerText = '加载拒识版';
          btnLab.style.background = '#059669';
          btnLab.style.color = 'white';
        }
        
        const anyLoaded = vlmStatus.base || vlmStatus.lab;
        detectBtn.disabled = !anyLoaded;
        
        if (vlmStatus.base && vlmStatus.lab) {
          ind.innerHTML = '<span style="color:#4ade80;">● 双模型已就绪 (可对比)</span>';
        } else if (anyLoaded) {
          ind.innerHTML = '<span style="color:#38bdf8;">● 单模型已就绪 (省显存)</span>';
        } else {
          ind.innerHTML = '<span style="color:#94a3b8;">● 模型未加载 (0显存占用)</span>';
        }
      } catch (e) {
        document.getElementById('vlm-status-indicator').innerHTML = '<span style="color:#f87171;">● 视觉服务(8001)未启动</span>';
        document.getElementById('btn-toggle-base').disabled = true;
        document.getElementById('btn-toggle-lab').disabled = true;
        document.getElementById('btn-vlm-detect').disabled = true;
      }
    }

    async function toggleSpecificModel(target) {
      if (isVlmPending) return;
      isVlmPending = true;
      const btn = target === 'base' ? document.getElementById('btn-toggle-base') : document.getElementById('btn-toggle-lab');
      btn.disabled = true;
      const willLoad = !vlmStatus[target];
      btn.innerText = willLoad ? '加载中...' : '卸载中...';
      
      try {
        const formData = new FormData();
        formData.append('target', target);
        const res = await fetch('http://localhost:8001/models/toggle', { method: 'POST', body: formData });
        const data = await res.json();
        setLog(data.msg || '操作成功');
      } catch (err) {
        alert('模型切换失败: ' + err.message);
      } finally {
        isVlmPending = false;
        btn.disabled = false;
        await checkVlmHealth();
      }
    }

    setInterval(checkVlmHealth, 2000);
    checkVlmHealth();

    async function returnToStandby() {
      if (workbenchPending || !latestState) return;
      workbenchPending = true;
      document.getElementById('btn-standby').disabled = true;
      try {
        const info = await (await fetch('/api/standby')).json();
        if (!info.ok) return setLog(info.msg);
        if (!confirm('将移动机械臂并调整夹爪，仅回到待机位。\\n来源：' + info.source + '\\n目标：' + info.target.join(', ') + '\\n请确认夹爪没有夹住物体、整条移动路径无障碍。不会自动启动ACT。')) return;
        await versionRequest('/api/standby/start?context=' + encodeURIComponent(info.context));
      } catch (e) { setLog('待机复位请求失败：' + e.message); }
      finally { workbenchPending = false; await syncState(); }
    }

    async function runOnce() {
      if (workbenchPending || !latestState) return;
      if (latestState.workbench_mode === 'act') {
        const ident = latestState.selected_act_model;
        if (!ident) return setLog('请先选择一个已训练模型');
        await workbenchRequest('/api/act/start?ident=' + encodeURIComponent(ident));
      } else {
        await runAutoPlayback(false);
      }
    }
    const pointKeys = ['p1', 'p2', 'p3', 'p4'];
    const storedPointKeys = {p1:'p1_standby',p2:'p2_grasp',p3:'p3_lift',p4:'p4_release'};

    function updateCopyAvailability() {
      if (!latestState) return;
      const source = latestState.versions.find(v => v.id === document.getElementById('point-copy-source').value);
      const busy = latestState.training || latestState.recording || latestState.teleop_active || latestState.motion_active;
      for (const key of pointKeys) {
        const box = document.getElementById('copy-' + key);
        const available = !!(source && source.waypoints && source.waypoints[storedPointKeys[key]]);
        box.disabled = busy || !available;
        if (!available) box.checked = false;
      }
    }

    function loadPointEditor() {
      if (!latestState || !latestState.active_version) return;
      const v = latestState.active_version;
      const point = document.getElementById('edit-point').value || 'p1';
      document.getElementById('edit-point').value = point;
      const values = latestState.waypoints[point];
      for (let i=1; i<=6; i++) document.getElementById('joint-' + i).value = values ? String(values[i-1]) : '';
      pointEditorContext = {target_version_id:v.id,expected_updated_at:v.updated_at,point};
      parameterDirty = false;
      document.getElementById('point-tools-version').textContent = v.name;
      document.getElementById('point-edit-note').textContent = values ? '已载入保存值；修改后点保存。仅改参数，不移动机械臂。' : '这个点还没保存，可以填写完整的 6 个整数后保存。';
    }

    async function postPointParameters(path, payload) {
      try {
        const res = await fetch(path, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
        const data = await res.json();
        const detail = Array.isArray(data.detail) ? data.detail.map(x => x.msg).join('；') : data.detail;
        setLog(data.msg || detail || (data.ok ? '参数已保存' : '保存失败'));
        await syncState();
        return !!data.ok;
      } catch(e) { setLog('请求失败：' + e.message); return false; }
    }

    async function saveParameterEdit() {
      if (!pointEditorContext) return setLog('请先展开编辑区并载入一个点位');
      const values = [];
      for (let i=1; i<=6; i++) {
        const raw = document.getElementById('joint-' + i).value.trim();
        if (!/^[0-9]+$/.test(raw) || Number(raw)>4095) return setLog('6 个关节值都必须是 0～4095 的整数');
        values.push(Number(raw));
      }
      if (await postPointParameters('/api/waypoints/edit', {...pointEditorContext,values})) loadPointEditor();
    }

    async function copySelectedPoints() {
      if (!latestState) return;
      const source = latestState.versions.find(v => v.id === document.getElementById('point-copy-source').value);
      const points = pointKeys.filter(key => {const box=document.getElementById('copy-'+key);return box.checked && !box.disabled;});
      if (!source || !points.length) return setLog('请选择来源版本，并勾选已保存的点位');
      const v = latestState.active_version;
      const payload = {target_version_id:v.id,expected_updated_at:v.updated_at,
        source_version_id:source.id,source_updated_at:source.updated_at,points};
      if (await postPointParameters('/api/waypoints/copy',payload)) {
        if (!parameterDirty) loadPointEditor();
        else document.getElementById('point-edit-note').textContent = '复制已保存；编辑框仍有未保存修改，请重新载入核对。';
      }
    }

    async function versionRequest(path) {
      try {
        const res = await fetch(path, {method:'POST'});
        const data = await res.json();
        setLog(data.msg || (data.ok ? '已保存' : '操作失败'));
      } catch (e) { setLog('请求失败：' + e.message); }
      await syncState();
    }

    async function switchVersion(id) {
      await versionRequest('/api/versions/select?ident=' + encodeURIComponent(id));
    }

    async function createVersion(copy) {
      await versionRequest('/api/versions/create?copy_current=' + copy);
    }

    async function setLog(text) {
      document.getElementById('log').innerText = text;
    }

    async function saveWaypoint(name) {
      setLog(`正在抓取从臂姿态: ${name}...`);
      const res = await fetch(`/api/waypoint/save?name=${name}`, { method: 'POST' });
      const data = await res.json();
      if (data.ok) {
        setLog(`✅ 点位 [${name}] 记录成功: ${JSON.stringify(data.pos)}`);
        await syncState();
      } else {
        alert("记录失败: " + data.msg);
      }
    }

    async function disableTorque() {
      const res = await fetch('/api/follower/torque_off', { method: 'POST' });
      const data = await res.json();
      setLog(data.msg || '卸力矩请求已处理');
    }

    async function runAutoPlayback(record) {
      if (latestState && latestState.workbench_mode === 'act') return setLog('当前是 ACT 模式，不能运行示教脚本或录制');
      const modeText = record ? "自动抓取并录制示范" : "自动抓取测试";
      setLog(`⏳ 正在执行: ${modeText}...`);
      const res = await fetch(`/api/auto_playback?record=${record}`, { method: 'POST' });
      const data = await res.json();
      setLog(data.msg);
      await syncState();
    }

    async function toggleTeleop() {
      const btn = document.getElementById('btn-teleop');
      btn.disabled = true;
      if (!isTeleop) {
        const res = await fetch('/api/teleop/start', { method: 'POST' });
        const data = await res.json();
        if (data.ok) isTeleop = true;
        else setLog(data.msg || '无法启动遥操作');
      } else {
        await fetch('/api/teleop/stop', { method: 'POST' });
        isTeleop = false;
      }
      btn.disabled = false;
      await syncState();
    }

    async function toggleManualRecord() {
      const endpoint = isManualRecording ? '/api/record/stop' : '/api/record/start';
      const res = await fetch(endpoint, { method: 'POST' });
      const data = await res.json();
      if (!data.ok) setLog(data.msg);
      await syncState();
    }

    async function deleteLastEpisode() {
      if (!confirm("确定废弃并删除刚刚录制的这一条吗？")) return;
      await versionRequest('/api/record/delete_last');
    }

    async function clearAllEpisodes() {
      if (!confirm('只清空当前版本当前修订的示范？其他版本、历史修订和旧数据不受影响。')) return;
      await versionRequest('/api/record/clear_all');
    }

    async function startTraining() {
      const scope = document.getElementById('train-scope').value;
      const label = scope === 'all' ? '合并所有版本的当前修订' : '当前版本的当前修订';
      if (!confirm('确定训练' + label + '？不包含历史修订或旧的20条。')) return;
      document.getElementById('btn-train').disabled = true;
      const res = await fetch('/api/train/start?scope=' + scope, { method: 'POST' });
      const data = await res.json();
      setLog(data.msg || '训练请求已处理');
      await syncState();
    }

    async function syncState() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();
        latestState = data;
        isTeleop = data.teleop_active;
        isManualRecording = data.recording;
        document.getElementById('btn-train').disabled = data.training;
        
        const busy = data.training || data.recording || data.teleop_active || data.motion_active || data.act_running || workbenchPending;
        document.getElementById('btn-standby').disabled = busy;
        document.getElementById('btn-standby-stop').disabled = !data.standby_running;
        document.getElementById('standby-status').textContent = data.standby_status || '只复位，不抓放，不启动ACT。请确保夹爪无物体、运动路径清空。';
        const cameraEnabled = data.camera_enabled !== false;
        const cameraClosing = !cameraEnabled && data.camera_released === false;
        document.getElementById('btn-camera').innerText = cameraEnabled ? '关闭摄像头' : '开启摄像头';
        document.getElementById('btn-camera').disabled = cameraPending || cameraClosing || data.recording || data.teleop_active || data.motion_active || data.act_running;
        document.getElementById('camera-status').textContent = data.camera_error || (cameraClosing ? '正在释放摄像头…' : cameraEnabled ? (data.camera_active ? '摄像头已开启' : '摄像头启动中…') : '摄像头已关闭，设备已释放');
        const feed = document.getElementById('live-feed');
        feed.style.display = cameraEnabled ? '' : 'none';
        if (cameraEnabled && feed.getAttribute('src') !== '/video_feed') feed.setAttribute('src','/video_feed');
        if (!cameraEnabled) feed.removeAttribute('src');
        document.getElementById('camera-placeholder').style.display = cameraEnabled ? 'none' : '';
        document.getElementById('camera-placeholder').textContent = cameraClosing ? '正在释放摄像头…' : '摄像头已关闭（设备已释放）';
        const mode = data.workbench_mode || 'teach';
        const isActMode = mode === 'act';
        document.querySelectorAll('[data-teach-panel]').forEach(el => {el.style.display = isActMode ? 'none' : '';});
        document.getElementById('act-panel').style.display = isActMode ? '' : 'none';
        document.getElementById('mode-teach').className = isActMode ? 'btn btn-secondary' : 'btn';
        document.getElementById('mode-act').className = isActMode ? 'btn' : 'btn btn-secondary';
        document.getElementById('mode-teach').disabled = busy;
        document.getElementById('mode-act').disabled = busy;
        const models = data.act_models || [];
        const modelSelect = document.getElementById('act-model-select');
        const modelSignature = JSON.stringify(models);
        if (modelSignature !== actModelsSignature) {
          modelSelect.replaceChildren();
          for (const model of models) modelSelect.add(new Option(model.label,model.id));
          actModelsSignature = modelSignature;
        }
        modelSelect.value = data.selected_act_model || '';
        modelSelect.disabled = busy;
        const selected = models.find(model => model.id === data.selected_act_model);
        document.getElementById('act-model-info').textContent = selected ? selected.path : '没有完整的已保存 ACT 模型';
        document.getElementById('act-stop').disabled = !data.act_running;
        document.getElementById('act-status').textContent = data.act_status || '尚未运行模型';
        document.getElementById('run-title').textContent = isActMode ? '🧠 ACT 单次测试' : '🤖 示教脚本试跑';
        document.getElementById('run-context').textContent = isActMode ? ('所选模型：' + (selected ? selected.label : '未选择')) : ('点位来源：' + (data.active_version ? data.active_version.name : '当前示教版本'));
        document.getElementById('btn-auto-test').innerText = isActMode ? '▶️ 跑一次所选 ACT 模型' : '▶️ 跑一次当前示教脚本';
        document.getElementById('btn-auto-record').style.display = isActMode ? 'none' : '';
        document.querySelectorAll('[data-version-edit]').forEach(el => {el.disabled = busy;});
        document.querySelectorAll('[data-parameter-edit]').forEach(el => {el.disabled = busy;});
        document.querySelectorAll('.wp-btn').forEach(el => {el.style.pointerEvents = busy ? 'none' : '';});
        document.getElementById('btn-auto-test').disabled = busy || (isActMode && !selected);
        document.getElementById('btn-auto-record').disabled = busy;
        document.getElementById('train-scope').disabled = busy;
        document.getElementById('btn-train').disabled = busy;
        if (data.active_version) {
          const v = data.active_version;
          const select = document.getElementById('version-select');
          const signature = JSON.stringify(data.versions);
          if (signature !== versionsSignature) {
            select.replaceChildren();
            for (const item of data.versions) {
              select.add(new Option(item.name, item.id));
            }
            versionsSignature = signature;
          }
          select.value = v.id;
          if (pointEditorContext && pointEditorContext.target_version_id !== v.id) {
            document.getElementById('point-tools').open = false;
            pointEditorContext = null;
            parameterDirty = false;
          }
          document.getElementById('point-tools-version').textContent = v.name;
          const sourceSelect = document.getElementById('point-copy-source');
          const sourceSignature = JSON.stringify(data.versions.map(item => [item.id,item.name]));
          if (sourceSignature !== copySourceSignature) {
            const previousSource = sourceSelect.value;
            sourceSelect.replaceChildren();
            for (const item of data.versions) sourceSelect.add(new Option(item.name,item.id));
            const preferred = data.versions.find(item => item.id !== v.id && item.complete) || data.versions.find(item => item.complete) || data.versions[0];
            sourceSelect.value = data.versions.some(item => item.id === previousSource) ? previousSource : preferred.id;
            copySourceSignature = sourceSignature;
          }
          updateCopyAvailability();
          document.getElementById('version-badge').textContent = v.name;
          const summary = data.versions.find(x => x.id === v.id);
          document.getElementById('version-info').textContent = `${data.total_episodes} 条素材` + (summary.history_episodes ? ` · 历史保留 ${summary.history_episodes} 条` : '') + ' · 点位自动保存';
        }
        document.getElementById('episodes-count').innerText = data.total_episodes + ' 条';
        
        // Waypoints UI
        for (let i = 1; i <= 4; i++) {
          const key = 'p' + i;
          const box = document.getElementById(`wp-${i}`);
          const val = document.getElementById(`wp-${i}-val`);
          if (data.waypoints[key]) {
            box.className = 'wp-btn wp-done';
            val.innerText = '已保存';
            val.title = data.waypoints[key].join(', ');
            val.style.color = '#4ade80';
          } else {
            box.className = 'wp-btn';
            val.innerText = '未录制';
            val.style.color = '#64748b';
          }
        }
        
        // Teleop button UI
        const btnTeleop = document.getElementById('btn-teleop');
        const teleBadge = document.getElementById('teleop-badge');
        if (isTeleop) {
          btnTeleop.innerText = '断开遥操作 (Torque OFF)';
          btnTeleop.className = 'btn btn-danger';
          teleBadge.innerText = '同步中';
          teleBadge.style.color = '#22c55e';
        } else {
          btnTeleop.innerText = '启动双臂实时跟随';
          btnTeleop.className = 'btn btn-secondary';
          teleBadge.innerText = '已断开';
          teleBadge.style.color = '#64748b';
        }

        // Manual record button UI
        const btnRec = document.getElementById('btn-record-manual');
        if (isManualRecording) {
          btnRec.innerText = '🔴 录制中... 点击/空格保存';
          btnRec.className = 'btn btn-danger';
        } else {
          btnRec.innerText = '开始手动录制 (空格键切换)';
          btnRec.className = 'btn btn-secondary';
        }

        if (data.training) {
          document.getElementById('train-status').innerText = `正在训练: Step ${data.train_step}/3000 (Loss: ${data.train_loss.toFixed(3)})`;
        } else {
          document.getElementById('train-status').innerText = data.status_msg;
        }
      } catch (e) {}
    }

    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space' && !['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(e.target.tagName)) {
        e.preventDefault();
        toggleManualRecord();
      }
    });

    setInterval(syncState, 250);
    syncState();
  </script>
</body>
</html>
"""

def camera_worker():
    global latest_jpeg, latest_frame_time
    while not stop_camera.is_set():
        with camera_control_lock:
            enabled=state['camera_enabled']
            if enabled:camera_released.clear()
        if not enabled:
            camera_wake.wait(.2)
            camera_wake.clear()
            continue
        cap=None
        try:
            cap=cv2.VideoCapture(0)
            if not cap.isOpened():raise RuntimeError('无法打开摄像头，请检查连接后重新开启')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            cap.set(cv2.CAP_PROP_FPS,30)
            state['camera_active']=True
            failures=0
            while state['camera_enabled'] and not stop_camera.is_set():
                ret,frame=cap.read()
                if not ret:
                    failures+=1
                    if failures>=30:raise RuntimeError('摄像头画面读取失败，请重新开启')
                    stop_camera.wait(.033)
                    continue
                failures=0
                captured_at=time.time()
                small=cv2.resize(frame,(320,240),interpolation=cv2.INTER_AREA)
                ok,buf=cv2.imencode('.jpg',small,[cv2.IMWRITE_JPEG_QUALITY,90])
                if ok:
                    with lock:
                        if state['camera_enabled']:
                            latest_jpeg=buf.tobytes()
                            latest_frame_time=captured_at
                stop_camera.wait(.033)
        except Exception as exc:
            state['camera_error']=str(exc)
            state['camera_enabled']=False
        finally:
            released=True
            try:
                if cap is not None:cap.release()
            except Exception as exc:
                released=False
                state['camera_enabled']=False
                state['camera_error']=f'无法确认摄像头释放，请重启网页服务：{exc}'
            with lock:
                latest_jpeg=None
                latest_frame_time=0
            state['camera_active']=not released
            if released:camera_released.set()
            else:return

def move_interp(bus, start_pos, target_pos, duration_s, hz=30):
    steps = max(10, int(duration_s * hz))
    dt = duration_s / steps
    for s in range(1, steps + 1):
        alpha = s / steps
        smooth_alpha = (1.0 - np.cos(alpha * np.pi)) / 2.0
        cur = start_pos * (1 - smooth_alpha) + target_pos * smooth_alpha
        for i in range(1, 7):
            bus.write_register_with_error(i, "goal_position", int(cur[i-1]), retries=0)
        time.sleep(dt)

def teleop_loop():
    global current_episode_data
    profile = json.loads(PROFILE_PATH.read_text())
    l_port = profile["leader_port"]
    f_port = profile["follower_port"]
    l_ref = np.array(profile["leader_ref"])
    f_ref = np.array(profile["follower_ref"])
    signs = np.array(profile["signs"])
    limits = profile["follower_limits"]
    
    try:
        with closing(rustypot.Bus(l_port, 1000000, 0.05, bus_dict)) as l_bus, \
             closing(rustypot.Bus(f_port, 1000000, 0.05, bus_dict)) as f_bus:
            
            for i in range(1, 7):
                l_bus.write_register_with_error(i, "torque_enable", 0, retries=1)
                
            torque_limits = [300, 500, 500, 350, 250, 400]
            for i in range(1, 7):
                f_bus.write_register_with_error(i, "torque_limit", torque_limits[i-1], retries=1)
                f_bus.write_register_with_error(i, "goal_speed", 350, retries=1)
                f_bus.write_register_with_error(i, "acceleration", 5, retries=1)
                f_bus.write_register_with_error(i, "goal_time", 0, retries=1)
                
            l_curr = np.array([l_bus.read_register_with_error(i, 'present_position', retries=2)[0] & 0x0FFF for i in range(1, 7)])
            f_curr = np.array([f_bus.read_register_with_error(i, 'present_position', retries=2)[0] & 0x0FFF for i in range(1, 7)])
            target = f_ref + signs * (l_curr - l_ref)
            for i in range(6):
                target[i] = max(limits[i][0], min(limits[i][1], target[i]))
                
            for i in range(1, 7):
                f_bus.write_register_with_error(i, "torque_enable", 1, retries=1)
                
            for s in range(1, 31):
                alpha = s / 30
                interp = f_curr * (1 - alpha) + target * alpha
                for i in range(1, 7):
                    f_bus.write_register_with_error(i, "goal_position", int(interp[i-1]), retries=0)
                time.sleep(1.0 / 30)
                
            state["teleop_active"] = True
            
            hz = 30
            dt = 1.0 / hz
            while not stop_teleop.is_set():
                t0 = time.perf_counter()
                l_pos = np.array([l_bus.read_register_with_error(i, 'present_position', retries=1)[0] & 0x0FFF for i in range(1, 7)])
                f_target = f_ref + signs * (l_pos - l_ref)
                for i in range(6):
                    f_target[i] = max(limits[i][0], min(limits[i][1], f_target[i]))
                    
                with record_lock:
                    sample = None
                    if state['recording'] and (not current_episode_data or
                            time.time() - current_episode_data[-1]['timestamp'] >= 0.09):
                        sample = capture_sample(f_bus, f_target)
                    for i in range(1, 7):
                        error = f_bus.write_register_with_error(i, "goal_position", int(f_target[i-1]), retries=0)
                        if error:
                            raise RuntimeError(f'从臂 {i} 号指令失败: {error}')
                    if sample is not None:
                        sample['command_timestamp'] = time.time()
                        current_episode_data.append(sample)
                        state['current_episode_frames'] = len(current_episode_data)
                    
                grip_pct = (f_target[5] - limits[5][0]) * 100 / (limits[5][1] - limits[5][0])
                state["gripper_pct"] = float(np.clip(grip_pct, 0, 100))
                
                elapsed = time.perf_counter() - t0
                if dt > elapsed:
                    time.sleep(dt - elapsed)
                    
    except Exception as exc:
        state['status_msg'] = f'遥操作/录制失败（未入库）：{exc}'
        with record_lock:
            state['recording'] = False
            current_episode_data = []
    finally:
        with closing(rustypot.Bus(f_port, 1000000, 0.05, bus_dict)) as f_bus:
            for i in range(1, 7):
                try: f_bus.write_register_with_error(i, "torque_enable", 0, retries=1)
                except: pass
        state["teleop_active"] = False

@app.get("/")
def get_index():
    return HTMLResponse(HTML_PAGE)

@app.get("/api/state")
def get_state():
    return {**state, 'motion_active': motion_lock.locked(), 'camera_released':camera_released.is_set()}

@app.post('/api/camera')
def set_camera_enabled(enabled: bool):
    global latest_jpeg,latest_frame_time
    if not motion_lock.acquire(blocking=False):
        return {'ok':False,'msg':'动作正在进行，不能切换摄像头'}
    try:
        if state['recording'] or state['teleop_active'] or state.get('act_running'):
            return {'ok':False,'msg':'请先停止动作或录制，再切换摄像头'}
        with camera_control_lock:
            state['camera_enabled']=enabled
            state['camera_error']=''
            if not enabled:
                with lock:
                    latest_jpeg=None
                    latest_frame_time=0
            camera_wake.set()
        if not enabled and not camera_released.wait(3):
            return {'ok':False,'msg':'关闭请求已发出，但设备仍在释放中，请稍候；未确认关闭'}
        return {'ok':True,'msg':'摄像头正在开启' if enabled else '摄像头已关闭，设备已释放'}
    finally:
        motion_lock.release()


@app.get("/api/camera/snapshot")
def get_camera_snapshot():
    with lock:
        jpeg = latest_jpeg
    if not jpeg:
        # Fallback to last recorded current image if camera is paused
        fallback = ROOT / "reports/act-start-stall-current.jpg"
        if fallback.exists():
            return Response(content=fallback.read_bytes(), media_type="image/jpeg")
        return JSONResponse(status_code=400, content={"ok": False, "msg": "摄像头未就绪"})
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/video_feed")
def video_feed():
    def gen():
        while state['camera_enabled'] and not stop_camera.is_set():
            with lock:jpeg=latest_jpeg
            if jpeg:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            time.sleep(0.033)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

class PointEditRequest(BaseModel):
    target_version_id: str
    expected_updated_at: str
    point: Literal['p1', 'p2', 'p3', 'p4']
    values: list[StrictInt]


class PointCopyRequest(BaseModel):
    target_version_id: str
    expected_updated_at: str
    source_version_id: str
    source_updated_at: str
    points: list[Literal['p1', 'p2', 'p3', 'p4']]


def check_edit_target(ident, updated_at):
    current = version_store.active()
    if current['id'] != ident:
        raise ValueError('当前版本已经切换，请重新载入参数后再保存')
    if current['updated_at'] != updated_at:
        raise ValueError('点位已被修改，请重新载入最新参数后再保存')


@app.post('/api/waypoints/edit')
def edit_waypoint_values(request: PointEditRequest):
    try:
        with configuration_edit():
            check_edit_target(request.target_version_id, request.expected_updated_at)
            v = version_store.set_points({POINT_NAMES[request.point]: request.values})
            sync_version_state()
            state['status_msg'] = f"已保存到 {v['name']} 的 {request.point[1:]} 号点位；没有移动机械臂"
        return {'ok': True, 'msg': state['status_msg']}
    except Exception as exc:
        return {'ok': False, 'msg': str(exc)}


@app.post('/api/waypoints/copy')
def copy_waypoint_values(request: PointCopyRequest):
    try:
        with configuration_edit():
            check_edit_target(request.target_version_id, request.expected_updated_at)
            source = version_store.get(request.source_version_id)
            if source['updated_at'] != request.source_updated_at:
                raise ValueError('来源版本已被修改，请刷新后重新复制')
            v = version_store.copy_points(request.source_version_id, request.points)
            sync_version_state()
            state['status_msg'] = f"已从 {source['name']} 复制到 {v['name']}；未勾选的示教点不变，没有移动机械臂"
        return {'ok': True, 'msg': state['status_msg']}
    except Exception as exc:
        return {'ok': False, 'msg': str(exc)}


def act_frame_rgb():
    if not state['camera_enabled']:
        raise ValueError('摄像头已关闭，请先开启再运行 ACT')
    with lock:
        jpeg, captured = latest_jpeg, latest_frame_time
    if jpeg is None or not 0 <= time.time()-captured <= .25:
        raise ValueError('实时摄像头未就绪或画面过期，未启动 ACT')
    frame=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('实时画面无法解码')
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


@app.get('/api/act/models')
def get_act_models():
    refresh_act_models()
    return {'models':state['act_models'],'selected':state['selected_act_model']}


@app.post('/api/workbench/mode')
def set_workbench_mode(mode: Literal['teach','act']):
    try:
        with configuration_edit(require_teach=False):
            act_library.save(mode=mode)
            refresh_act_models()
        return {'ok':True,'msg':'已切换到 ACT 模型测试' if mode=='act' else '已切换到示教 / 录制'}
    except Exception as exc:
        return {'ok':False,'msg':str(exc)}


@app.post('/api/act/select')
def select_act_model(ident: str):
    try:
        with configuration_edit(require_teach=False):
            if state.get('workbench_mode')!='act':
                raise ValueError('请先切换到 ACT 模型测试模式')
            act_library.save(model=ident)
            refresh_act_models()
            state['act_status']='模型已选定；点击下方跑一次才会驱动机械臂'
        return {'ok':True,'msg':state['act_status']}
    except Exception as exc:
        return {'ok':False,'msg':str(exc)}


def act_web_worker(model_path):
    logfile=None
    def on_event(**event):
        state['act_status']=event.get('message','')
        state['status_msg']=state['act_status']
        if 'step' in event:state['act_step']=event['step']
        with logfile.open('a') as log:
            log.write(json.dumps({'time':time.time(),'model':str(model_path),**event},ensure_ascii=False)+'\n')
    try:
        from studio_act import run_once
        tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        logfile=ROOT/'reports'/f'act-web-{tag}.jsonl'
        logfile.parent.mkdir(parents=True,exist_ok=True)
        result=run_once(model_path,ROOT,act_frame_rgb,act_stop_event,on_event)
        state['act_result']=result
        state['act_status']=result.get('message','模型测试结束')
        atomic_json(logfile.with_suffix('.json'),result)
    except Exception as exc:
        state['act_status']=f'ACT 后台错误：{exc}'
    finally:
        state['status_msg']=state['act_status']
        state['act_running']=False
        motion_lock.release()


@app.post('/api/act/start')
def start_act_once(ident: str):
    global act_thread
    if state.get('workbench_mode')!='act':
        return {'ok':False,'msg':'当前是示教模式，请先切换 ACT 模型'}
    if state['training'] or state['recording'] or state['teleop_active'] or (teleop_thread and teleop_thread.is_alive()):
        return {'ok':False,'msg':'请先结束其他动作、录制或训练'}
    if not motion_lock.acquire(blocking=False):
        return {'ok':False,'msg':'已有动作正在运行，不能重复启动'}
    try:
        if state.get('workbench_mode')!='act' or ident!=state.get('selected_act_model'):
            raise ValueError('所选模式或模型已变化，请重新确认后点击运行')
        model_path=act_library.resolve(ident)
        act_frame_rgb()  # Read cached image only; no serial/model opened here.
        act_stop_event.clear()
        state.update(act_running=True,act_step=0,running_act_model=ident,
                     act_status='正在加载所选 ACT 模型',act_result=None)
        act_thread=threading.Thread(target=act_web_worker,args=(model_path,),daemon=True)
        act_thread.start()
        return {'ok':True,'msg':f'已启动 ACT 单次测试：{ident}，不录制、不自动重试'}
    except Exception as exc:
        state['act_running']=False
        motion_lock.release()
        return {'ok':False,'msg':str(exc)}


def standby_spec():
    from studio_standby import model_standby,validate_target
    if state.get('workbench_mode')=='act':
        ident=state.get('selected_act_model')
        target=model_standby(act_library.resolve(ident))
        source=f'ACT模型 {ident} 的训练待机点'
        context=f'act:{ident}:{target}'
    else:
        target=validate_target(active_waypoints().get('p1_standby'))
        version=state.get('active_version',{})
        source=version.get('name','当前示教版本')+' 的1号待机点'
        context=f'teach:{version.get("id")}:{version.get("updated_at")}:{target}'
    return {'ok':True,'target':target,'source':source,'context':context}


@app.get('/api/standby')
def get_standby():
    try:return standby_spec()
    except Exception as exc:return {'ok':False,'msg':str(exc)}


def standby_web_worker(spec):
    try:
        from studio_standby import run_standby
        def progress(message):
            state['standby_status']=message
            state['status_msg']=message
        result=run_standby(spec['target'],lambda:rustypot.Bus(PORT_FOLLOWER,1000000,.05,bus_dict),standby_stop_event,progress)
        state['standby_result']=result
        state['standby_status']=result.get('message','复位结束')
        tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        atomic_json(ROOT/'reports'/f'standby-{tag}.json',{'source':spec['source'],**result})
    except Exception as exc:
        state['standby_status']=f'待机复位错误：{exc}'
    finally:
        state['status_msg']=state['standby_status']
        state['standby_running']=False
        motion_lock.release()


@app.post('/api/standby/start')
def start_standby(context: str):
    global standby_thread
    if not motion_lock.acquire(blocking=False):
        return {'ok':False,'msg':'已有动作正在运行，请先停止'}
    try:
        if state['training'] or state['recording'] or state['teleop_active'] or state.get('act_running') or (teleop_thread and teleop_thread.is_alive()):
            raise ValueError('请先结束动作、录制或训练，再复位')
        spec=standby_spec()
        if context!=spec['context']:raise ValueError('模式、模型或待机点已变化，请重新点击确认')
        standby_stop_event.clear()
        state.update(standby_running=True,standby_result=None,standby_status='正在独立复位；不会启动ACT')
        standby_thread=threading.Thread(target=standby_web_worker,args=(spec,),daemon=True)
        standby_thread.start()
        return {'ok':True,'msg':'已启动独立待机复位，不抓放、不录制、不自动启动ACT'}
    except Exception as exc:
        state['standby_running']=False
        motion_lock.release()
        return {'ok':False,'msg':str(exc)}


@app.post('/api/standby/stop')
def stop_standby():
    if not state['standby_running']:return {'ok':True,'msg':'当前没有待机复位任务，未发送动作'}
    standby_stop_event.set()
    return {'ok':True,'msg':'已请求停止复位，将保持实测位置与力矩'}


@app.post('/api/act/stop')
def stop_act_once():
    if not state['act_running']:
        return {'ok':True,'msg':'当前没有 ACT 推理在运行；没有发送动作'}
    act_stop_event.set()
    state['act_status']='正在停止 ACT，将保持当前位置，不释放力矩'
    return {'ok':True,'msg':state['act_status']}


@app.post('/api/versions/select')
def select_version(ident: str):
    try:
        with configuration_edit():
            v = version_store.select(ident)
            sync_version_state()
            state['status_msg'] = f"已切换到 {v['name']}；仅载入点位，没有移动机械臂"
        return {'ok': True, 'msg': state['status_msg']}
    except Exception as exc:
        return {'ok': False, 'msg': str(exc)}


@app.post('/api/versions/create')
def create_version(name: str = '', copy_current: bool = False):
    try:
        with configuration_edit():
            old = version_store.active()
            version_store.create(name, old['waypoints'] if copy_current else {}, old['open_gripper'])
            sync_version_state()
        return {'ok': True, 'msg': '新版本已保存并选中；录制数据不会复制'}
    except Exception as exc:
        return {'ok': False, 'msg': str(exc)}


@app.post('/api/versions/opening')
def save_opening(value: int):
    try:
        with configuration_edit():
            if not 0 <= value <= 4095:
                raise ValueError('张爪刻度需要在 0～4095 内')
            v = version_store.active()
            version_store.update(derive_waypoints(v['waypoints'], value), value)
            sync_version_state()
        return {'ok': True, 'msg': '当前版本张爪目标已保存；未移动夹爪，请先试跑验证'}
    except Exception as exc:
        return {'ok': False, 'msg': str(exc)}


@app.post('/api/versions/opening/capture')
def capture_opening():
    try:
        with configuration_edit():
            with closing(rustypot.Bus(PORT_FOLLOWER, 1000000, 0.05, bus_dict)) as bus:
                value, error = bus.read_register_with_error(6, 'present_position', retries=2)
                if error:
                    raise ValueError(f'夹爪读取失败: {error}')
                value = int(value) & 0x0FFF
            v = version_store.active()
            version_store.update(derive_waypoints(v['waypoints'], value), value)
            sync_version_state()
        return {'ok': True, 'msg': f'已记录当前夹爪开口 {value}；没有发送运动指令'}
    except Exception as exc:
        return {'ok': False, 'msg': str(exc)}


@app.post('/api/waypoint/save')
def api_save_waypoint(name: str):
    if name not in POINT_NAMES:
        return {'ok': False, 'msg': 'Invalid point name'}
    try:
        with configuration_edit():
            with closing(rustypot.Bus(PORT_FOLLOWER, 1000000, 0.05, bus_dict)) as bus:
                pos = []
                for i in range(1, 7):
                    value, error = bus.read_register_with_error(i, 'present_position', retries=2)
                    if error:
                        raise ValueError(f'从臂 {i} 号读取失败: {error}')
                    pos.append(int(value) & 0x0FFF)
            version_store.set_points({POINT_NAMES[name]: pos})
            sync_version_state()
            state['status_msg'] = '点位已保存到当前示教版本'
        return {'ok': True, 'pos': pos}
    except Exception as exc:
        return {'ok': False, 'msg': str(exc)}

@app.post("/api/follower/torque_off")
def force_torque_off():
    if state.get('standby_running'):
        standby_stop_event.set()
        return {'ok':True,'msg':'已请求停止复位并保持力矩；停止后再次点击卸下力矩可释放'}
    if state.get('act_running'):
        act_stop_event.set()
        return {'ok':True,'msg':'已请求停止 ACT 并保持力矩；停止后再次点击卸下力矩可释放'}
    with closing(rustypot.Bus(PORT_FOLLOWER, 1000000, 0.05, bus_dict)) as bus:
        for i in range(1, 7):
            try: bus.write_register_with_error(i, "torque_enable", 0, retries=1)
            except: pass
    state["teleop_active"] = False
    return {"ok": True, "msg": "所有电机扭矩已释放，手臂已变软"}

@app.post("/api/auto_playback")
def api_auto_playback(record: bool = False):
    if state.get('workbench_mode','teach')!='teach':
        return {'ok':False,'msg':'当前是 ACT 模式，不能执行示教脚本；请使用 ACT 跑一次'}
    if state['teleop_active'] or state['recording'] or state['training'] or (teleop_thread and teleop_thread.is_alive()) or not motion_lock.acquire(blocking=False):
        return {'ok': False, 'msg': '另一个操作正在进行'}
    try:
        return auto_playback_impl(record)
    except Exception as exc:
        state['status_msg'] = f'自动执行/录制失败（未入库）：{exc}'
        return {'ok': False, 'msg': state['status_msg']}
    finally:
        motion_lock.release()


def auto_playback_impl(record=False):
    if record:
        with lock:
            if latest_jpeg is None or time.time() - latest_frame_time > 0.25:
                raise RuntimeError('摄像头未就绪，不启动动作')
    if version_store is None and not WP_FILE.exists():
        return {"ok": False, "msg": "请先在网页上完成 1~4 示教打点！"}
    
    wp = active_waypoints()
    for k in ("p1_standby", "p2_grasp", "p3_lift", "p4_release"):
        if k not in wp:
            return {"ok": False, "msg": f"缺少关键点位 {k}，请先打点！"}
            
    p1 = np.array(wp["p1_standby"])
    p2_app = np.array(wp["p2_approach"])
    p2_open = np.array(wp["p2_open"])
    p2_grasp = np.array(wp["p2_grasp"])
    # Use the exact taught grip position; never silently tighten a saved point.
    
    p3_lift = np.array(wp["p3_lift"])
    # 3点抬起与4点运送过程中，夹爪必须严格保持2点的捏合度，绝不回弹
    p3_lift[5] = p2_grasp[5]
    
    p4_box = np.array(wp["p4_box"])
    p4_box[5] = p2_grasp[5]
    
    p4_rel = np.array(wp["p4_release"])
    p4_exit = np.array(wp["p4_exit"])
    p1_transit = np.array(wp["p1_transit"])

    ep_data = []
    capture_enabled = False  # Exclude initial repositioning/idle from demonstration.

    with closing(rustypot.Bus(PORT_FOLLOWER, 1000000, 0.05, bus_dict)) as bus:
        # 进一步调低夹爪出力上限 (从 200 降到 130，进一步减轻死捏力道)
        torque_limits = [300, 450, 450, 300, 200, 130]
        for i in range(1, 7):
            bus.write_register_with_error(i, "torque_limit", torque_limits[i-1], retries=1)
            bus.write_register_with_error(i, "goal_speed", 300, retries=1)
            bus.write_register_with_error(i, "acceleration", 5, retries=1)
            bus.write_register_with_error(i, "goal_time", 0, retries=1)
            bus.write_register_with_error(i, "torque_enable", 1, retries=1)
            
        cur = np.array([bus.read_register_with_error(i, 'present_position', retries=2)[0] & 0x0FFF for i in range(1, 7)])
        
        def run_seg(start, target, dur):
            steps = max(5, int(dur * 20))
            dt = dur / steps
            for s in range(1, steps + 1):
                alpha = s / steps
                sm_alpha = (1.0 - np.cos(alpha * np.pi)) / 2.0
                c = start * (1 - sm_alpha) + target * sm_alpha
                sample = None
                if record and capture_enabled and (not ep_data or time.time() - ep_data[-1]['timestamp'] >= 0.09):
                    sample = capture_sample(bus, c)
                for i in range(1, 7):
                    error = bus.write_register_with_error(i, 'goal_position', int(c[i-1]), retries=0)
                    if error:
                        raise RuntimeError(f'从臂 {i} 号指令失败: {error}')
                if sample is not None:
                    sample['command_timestamp'] = time.time()
                    ep_data.append(sample)
                time.sleep(dt)

        def hold_record(target, duration):
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                if record and capture_enabled:
                    row = capture_sample(bus, target)
                    row['command_timestamp'] = row['timestamp']
                    row['holding'] = True  # Previous target remains active, no new command.
                    ep_data.append(row)
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))

        # 1. 待机位
        run_seg(cur, p1, 1.2)
        hold_record(p1, 0.2)
        capture_enabled = True
        # 2. 安全高空飞临红块上方 (不贴地斜切，极其从容稳健)
        run_seg(p1, p2_app, 1.4)
        # 3. 正上方高空优雅张爪
        run_seg(p2_app, p2_open, 0.7)
        hold_record(p2_open, 0.2)
        # 4. 垂直精准下潜并死死咬住红块！
        run_seg(p2_open, p2_grasp, 1.2)
        hold_record(p2_grasp, 0.5)
        run_seg(p2_grasp, p3_lift, 1.2)
        run_seg(p3_lift, p4_box, 1.5)
        run_seg(p4_box, p4_rel, 0.8)
        hold_record(p4_rel, 0.3)
        run_seg(p4_rel, p4_exit, 1.0)
        run_seg(p4_exit, p1_transit, 1.2)
        run_seg(p1_transit, p1, 1.2)
        # Keep observing the final commanded pose while the servos settle.
        hold_record(p1, 0.8)
        
        for i in range(1, 7):
            bus.write_register_with_error(i, "goal_position", int(p1[i-1]), retries=0)
            
    if record and len(ep_data) > 10:
        save_episode(ep_data)
        return {"ok": True, "msg": f"✅ 自动抓取完成！第 {state['total_episodes']} 条示范已成功入库保存！"}
        
    return {"ok": True, "msg": "自动抓取测试执行完成，已完完整整回到待机位！"}

@app.post("/api/teleop/start")
def start_teleop():
    global teleop_thread
    if state.get('workbench_mode','teach')!='teach':
        return {'ok':False,'msg':'请先切回示教 / 录制模式'}
    if state["teleop_active"]: return {"ok": True}
    if motion_lock.locked() or state['training']:
        return {'ok': False, 'msg': '自动动作或训练正在进行'}
    stop_teleop.clear()
    teleop_thread = threading.Thread(target=teleop_loop, daemon=True)
    teleop_thread.start()
    time.sleep(1.2)
    return {"ok": True}

@app.post("/api/teleop/stop")
def stop_teleop_route():
    stop_teleop.set()
    return {"ok": True}

@app.post("/api/record/start")
def start_record():
    global current_episode_data
    if state.get('workbench_mode','teach')!='teach' or not state['teleop_active'] or state['training'] or motion_lock.locked():
        return {'ok': False, 'msg': '请先在示教模式启动遥操作，且不要同时训练'}
    with lock:
        if latest_jpeg is None or time.time() - latest_frame_time > 0.25:
            return {'ok': False, 'msg': '摄像头未就绪'}
    with record_lock:
        if state['recording']:
            return {'ok': False, 'msg': '已在录制'}
        current_episode_data = []
        state['current_episode_frames'] = 0
        state['recording'] = True
    return {'ok': True}

@app.post("/api/record/stop")
def stop_record_route():
    global current_episode_data
    with record_lock:
        state['recording'] = False
        rows, current_episode_data = current_episode_data, []
    try:
        save_episode(rows)
        return {'ok': True}
    except Exception as exc:
        state['status_msg'] = f'未保存：{exc}'
        return {'ok': False, 'msg': state['status_msg']}

@app.post("/api/record/delete_last")
def delete_last_episode():
    try:
        with configuration_edit():
            files = sorted(DATASET_DIR.glob('episode_*.json'))
            if files:
                last_file = files[-1]
                last_file.unlink()
                (DATASET_DIR / 'metadata' / last_file.name).unlink(missing_ok=True)
            if version_store:
                sync_version_state()
            else:
                state['total_episodes'] = max(0, len(files)-1)
        return {'ok': True, 'total': state['total_episodes']}
    except Exception as exc:
        return {'ok': False, 'msg': str(exc)}

@app.post("/api/record/clear_all")
def clear_all_episodes():
    import shutil
    try:
        with configuration_edit():
            if DATASET_DIR.exists():
                shutil.rmtree(DATASET_DIR)
            state['total_episodes'] = 0
            if version_store:
                sync_version_state()
        return {'ok': True}
    except Exception as exc:
        return {'ok': False, 'msg': str(exc)}

@app.post("/api/train/start")
def train_start(scope: str = 'current'):
    if state['training']:
        return {'ok': True, 'msg': '训练已在进行中'}
    try:
        with configuration_edit():
            from studio_samples import resample_episode
            if version_store:
                entries = version_store.training_files(scope)
                files = [f for v, f in entries]
                training_dirs = list(dict.fromkeys(f.parent for f in files))
                selection = [{'id': v['id'], 'name': v['name'], 'revision': v['revision']} for v in
                             {v['id']: v for v, f in entries}.values()]
            else:
                files = sorted(DATASET_DIR.glob('episode_*.json'))
                training_dirs, selection = [DATASET_DIR], []
            if not files:
                raise ValueError('所选版本没有新版真实图像示范，旧数据未自动混入')
            for path in files:
                resample_episode(json.loads(path.read_text()))
            state.update(training=True, train_step=0, train_loss=0.0,
                         training_versions=selection,
                         status_msg=f'正在转换 {len(files)} 条真实示范')
    except Exception as exc:
        state['status_msg'] = f'不能训练：{exc}'
        return {'ok': False, 'msg': state['status_msg']}
    
    def background_train():
        try:
            from train_studio_pipeline import run_training_pipeline
            def on_progress(step, loss):
                state["train_step"] = step
                state["train_loss"] = loss
                
            final_dir = run_training_pipeline(training_dirs, progress_callback=on_progress,
                                              chunk_size=50, n_action_steps=20)
            atomic_json(final_dir.parent / 'teaching-versions.json', selection)
            state['last_model'] = str(final_dir)
            refresh_act_models()
            state['status_msg'] = '训练完成（尚未验证实机成功率）'
        except Exception as e:
            state['status_msg'] = f'训练失败：{e}'
            print("训练发生错误:", e, flush=True)
        finally:
            state["training"] = False
            
    threading.Thread(target=background_train, daemon=True).start()
    return {'ok': True, 'msg': f'已启动训练：{len(files)} 条示范，5 秒预测窗口；不会驱动机械臂'}

if __name__ == "__main__":
    version_store = TeachingVersions(ROOT)
    act_library = ActModels(ROOT)
    sync_version_state()
    refresh_act_models()
    state['status_msg'] = '已启用示教版本；切换版本不会移动机械臂，旧数据保留'
    t_cam = threading.Thread(target=camera_worker, daemon=True)
    t_cam.start()
    print("🚀 SO-101 具身操作全功能工作台已启动: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
