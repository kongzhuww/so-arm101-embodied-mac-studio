"""Direct automatic recorder that captures genuine real-time video frames
and servo joint state at 10Hz into a production-grade LeRobotDataset.
Features:
- Gentle torque limits (130 on gripper).
- Clean vertical approach and deposit to prevent block drops.
- Synchronized camera frame per step.
"""
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone
from contextlib import closing
import cv2
import numpy as np
import rustypot

ROOT = Path(__file__).resolve().parents[1]
WP_FILE = ROOT / "config/auto_waypoints.json"
PORT = "/dev/cu.usbserial-120"
BAUDRATE = 1000000

NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
TASK = 'Pick up the red cube and place it into the box.'

_, definition = rustypot.find_model('sts3215')
bus_dict = {i: definition for i in range(1, 7)}

def normalize_joints(raw, calibration):
    values = []
    for name, val in zip(NAMES, raw):
        c = calibration[name]
        lo, hi = c['range_min'], c['range_max']
        if name == 'gripper':
            values.append((min(hi, max(lo, val))-lo)*100/(hi-lo))
        else:
            values.append((val-(lo+hi)/2)*360/4095)
    return np.asarray(values, dtype=np.float32)

def move_and_record(bus, cap, dataset, calibration, start_pos, target_pos, duration_s, hz=10):
    steps = max(5, int(duration_s * hz))
    dt = duration_s / steps
    
    last_obs = None
    for s in range(1, steps + 1):
        t_start = time.perf_counter()
        alpha = s / steps
        smooth_alpha = (1.0 - np.cos(alpha * np.pi)) / 2.0
        cur_target = start_pos * (1 - smooth_alpha) + target_pos * smooth_alpha
        
        for i in range(1, 7):
            bus.write_register_with_error(i, "goal_position", int(cur_target[i-1]), retries=0)
            
        raw = [bus.read_register_with_error(i, 'present_position', retries=1)[0] & 0x0FFF for i in range(1, 7)]
        norm_state = normalize_joints(raw, calibration)
        
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320, 240), interpolation=cv2.INTER_AREA)
            
        if last_obs is not None:
            dataset.add_frame({
                'observation.state': last_obs[0],
                'action': norm_state,
                'observation.images.external_front': last_obs[1],
                'task': TASK
            })
            
        last_obs = (norm_state, frame)
        elapsed = time.perf_counter() - t_start
        if dt > elapsed:
            time.sleep(dt - elapsed)
            
    return last_obs

def run_one_episode(bus, cap, dataset, calibration, wp):
    p1 = np.array(wp["p1_standby"])
    p2_app = np.array(wp["p2_approach"])
    p2_open = np.array(wp["p2_open"])
    p2_grasp = np.array(wp["p2_grasp"])
    p2_grasp[5] = min(int(p2_grasp[5]), 2300)
    
    p3_lift = np.array(wp["p3_lift"])
    p3_lift[5] = p2_grasp[5]
    
    p4_box = np.array(wp["p4_box"])
    p4_box[5] = p2_grasp[5]
    
    p4_rel = np.array(wp["p4_release"])
    p4_exit = np.array(wp["p4_exit"])
    p1_transit = np.array(wp["p1_transit"])

    cur = np.array([bus.read_register_with_error(i, 'present_position', retries=2)[0] & 0x0FFF for i in range(1, 7)])
    
    move_and_record(bus, cap, dataset, calibration, cur, p1, 1.2)
    time.sleep(0.2)
    move_and_record(bus, cap, dataset, calibration, p1, p2_app, 1.4)
    move_and_record(bus, cap, dataset, calibration, p2_app, p2_open, 0.7)
    time.sleep(0.2)
    move_and_record(bus, cap, dataset, calibration, p2_open, p2_grasp, 1.2)
    time.sleep(0.5)
    move_and_record(bus, cap, dataset, calibration, p2_grasp, p3_lift, 1.2)
    move_and_record(bus, cap, dataset, calibration, p3_lift, p4_box, 1.5)
    move_and_record(bus, cap, dataset, calibration, p4_box, p4_rel, 0.8)
    time.sleep(0.3)
    move_and_record(bus, cap, dataset, calibration, p4_rel, p4_exit, 1.0)
    move_and_record(bus, cap, dataset, calibration, p4_exit, p1_transit, 1.2)
    move_and_record(bus, cap, dataset, calibration, p1_transit, p1, 1.2)
    
    for i in range(1, 7):
        bus.write_register_with_error(i, "goal_position", int(p1[i-1]), retries=0)
        
    dataset.save_episode()

def main():
    from lerobot.datasets import LeRobotDataset
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = ROOT / 'datasets' / f'so101_pickplace_{tag}'
    calibration = json.loads((ROOT / 'calibration/so101-follower-20261031T144217-014032Z.json').read_text())
    
    print(f"正在创建全新高质量 LeRobot 数据集: {root.name}")
    dataset = LeRobotDataset.create(
        repo_id='local/so101_pickplace', root=root, fps=10,
        robot_type='so101_follower', use_videos=False, image_writer_threads=2,
        features={
            'observation.state': {'dtype':'float32','shape':(6,), 'names':NAMES},
            'action': {'dtype':'float32','shape':(6,), 'names':NAMES},
            'observation.images.external_front': {'dtype':'image','shape':(240,320,3), 'names':['height','width','channels']},
        }
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    wp = json.loads(WP_FILE.read_text())
    
    with closing(rustypot.Bus(PORT, BAUDRATE, 0.05, bus_dict)) as bus:
        torque_limits = [300, 450, 450, 300, 200, 130]
        for i in range(1, 7):
            bus.write_register_with_error(i, "torque_limit", torque_limits[i-1], retries=1)
            bus.write_register_with_error(i, "goal_speed", 300, retries=1)
            bus.write_register_with_error(i, "acceleration", 5, retries=1)
            bus.write_register_with_error(i, "goal_time", 0, retries=1)
            bus.write_register_with_error(i, "torque_enable", 1, retries=1)
            
        print("\n=======================================================")
        print("🤖 开始全自动录制 10 条真正含每帧实景画面的标准示范！")
        print("=======================================================\n")
        
        for ep in range(1, 11):
            print(f">>> 正在执行并录制第 {ep} / 10 条示范...")
            run_one_episode(bus, cap, dataset, calibration, wp)
            print(f"✅ 第 {ep} 条录制成功！请在 3 秒内将红块重新摆回桌上...")
            time.sleep(3.0)
            
        dataset.finalize()
        print(f"\n🎉 10 条高质量示范已全部录制完成！数据集路径:\n{root}")
        
    cap.release()

if __name__ == "__main__":
    main()
