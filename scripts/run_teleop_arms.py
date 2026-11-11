"""Direct Leader -> Follower teleoperation for SO-101.
Reads Leader angles (degrees) and maps to Follower targets at 30Hz.
Features:
- Safe soft-start: ramps from follower's current pose to leader's pose over 2 seconds.
- Speed and torque limit protections on follower.
- Graceful shutdown: disables torque on Ctrl+C.
"""
import sys
import time
import json
from pathlib import Path
from contextlib import closing
import numpy as np

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode

ROOT = Path(__file__).resolve().parents[1]
LEADER_PORT = "/dev/cu.usbserial-120"
FOLLOWER_PORT = "/dev/cu.usbserial-130"
BAUDRATE = 1000000

NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Latest calibration files
LEADER_CALIB_FILE = ROOT / "calibration/so101-leader-20261106T024101-001458Z.json"
FOLLOWER_CALIB_FILE = ROOT / "calibration/so101-follower-20261031T144217-014032Z.json"

def load_calib(path):
    d = json.loads(path.read_text())
    return {
        name: MotorCalibration(
            id=d[name]["id"],
            drive_mode=d[name]["drive_mode"],
            homing_offset=d[name]["homing_offset"],
            range_min=d[name]["range_min"],
            range_max=d[name]["range_max"],
        )
        for name in NAMES
    }

def main():
    print("=== SO-101 双臂主从实时遥操作 ===")
    leader_calib = load_calib(LEADER_CALIB_FILE)
    follower_calib = load_calib(FOLLOWER_CALIB_FILE)
    
    # Leader: read-only, norm_mode DEGREES for body, 0..100 for gripper
    leader_motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    
    # Follower: controlled, same norm modes
    follower_motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    
    print("正在连接主控臂 (Leader)...")
    leader_bus = FeetechMotorsBus(port=LEADER_PORT, motors=leader_motors, calibration=leader_calib)
    leader_bus.connect()
    for name in NAMES:
        leader_bus.write("Torque_Enable", name, 0)
        
    print("正在连接从动臂 (Follower)...")
    follower_bus = FeetechMotorsBus(port=FOLLOWER_PORT, motors=follower_motors, calibration=follower_calib)
    follower_bus.connect()
    
    # Set safe running profile on follower
    torque_limits = {1: 300, 2: 500, 3: 500, 4: 350, 5: 250, 6: 400}
    for i, name in enumerate(NAMES, 1):
        follower_bus.write("Torque_Limit", name, torque_limits[i])
        follower_bus.write("Goal_Speed", name, 400)
        follower_bus.write("Acceleration", name, 5)
        follower_bus.write("Goal_Time", name, 0)
    
    try:
        # Read initial poses
        leader_pose = leader_bus.sync_read("Present_Position", NAMES, normalize=True)
        follower_start = follower_bus.sync_read("Present_Position", NAMES, normalize=True)
        
        print(f"\n主控臂初始角度: {leader_pose}")
        print(f"从动臂初始角度: {follower_start}")
        print("\n>>> 开始缓慢平滑对齐 (2秒软启动)... <<<")
        
        # Enable torque on follower
        for name in NAMES:
            follower_bus.write("Torque_Enable", name, 1)
            
        # Ramp from follower_start to leader_pose
        ramp_steps = 60
        for step in range(1, ramp_steps + 1):
            alpha = step / ramp_steps
            interp_target = {
                name: (1 - alpha) * follower_start[name] + alpha * leader_pose[name]
                for name in NAMES
            }
            follower_bus.sync_write("Goal_Position", interp_target, normalize=True)
            time.sleep(2.0 / ramp_steps)
            
        print("\n==================================================")
        print("🎉 实时遥操作运行中！在手里转动主臂，从动臂即时跟随！")
        print("按 Ctrl+C 安全退出并关闭从动臂扭矩。")
        print("==================================================")
        
        hz = 30
        dt = 1.0 / hz
        last_log = time.time()
        while True:
            t0 = time.perf_counter()
            # Read Leader normalized position
            target = leader_bus.sync_read("Present_Position", NAMES, normalize=True)
            
            # Send directly to Follower
            follower_bus.sync_write("Goal_Position", target, normalize=True)
            
            if time.time() - last_log > 1.0:
                print(f"实时跟随 [肩:{target['shoulder_pan']:+.1f}° 大臂:{target['shoulder_lift']:+.1f}° 肘:{target['elbow_flex']:+.1f}° 腕:{target['wrist_flex']:+.1f}° 爪:{target['gripper']:.0f}%]")
                last_log = time.time()
                
            elapsed = time.perf_counter() - t0
            if dt > elapsed:
                time.sleep(dt - elapsed)
                
    except KeyboardInterrupt:
        print("\n\n正在安全退出遥操作...")
    finally:
        print("关闭从动臂扭矩 (Torque OFF)...")
        for name in NAMES:
            try:
                follower_bus.write("Torque_Enable", name, 0)
            except Exception:
                pass
        leader_bus.disconnect(disable_torque=False)
        follower_bus.disconnect(disable_torque=True)
        print("已完全断开并安全退出。")

if __name__ == "__main__":
    main()
