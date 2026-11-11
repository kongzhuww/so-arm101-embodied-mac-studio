"""Run autonomous inference using the newly trained ACT model.
Connects directly to follower arm, reads camera at 10Hz, predicts joint actions,
applies safe torque/speed limits, and executes the sequence.
"""
import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import rustypot

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
PORT_FOLLOWER = "/dev/cu.usbserial-120"
CALIB_FILE = ROOT / "calibration/so101-follower-20261031T144217-014032Z.json"

# Gentle safe dynamics
TORQUE_LIMITS = [300, 450, 450, 300, 200, 150]
SPEED_LIMITS = [500, 500, 500, 500, 500, 500]
ACCEL_LIMITS = [15, 15, 15, 15, 15, 15]

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

def denormalize_joints(norm, calibration):
    raw = []
    for i, name in enumerate(NAMES):
        c = calibration[name]
        lo, hi = c['range_min'], c['range_max']
        if name == 'gripper':
            v = lo + (norm[i] / 100.0) * (hi - lo)
        else:
            v = ((lo + hi) / 2.0) + (norm[i] * 4095.0 / 360.0)
        raw.append(int(np.clip(round(v), lo, hi)))
    return raw

def recorded_box_poses(model_path):
    """Completion observation only: does not give version IDs/goals to policy."""
    manifest = model_path.parent / 'run-manifest.json'
    if not manifest.exists():
        return []
    sources = json.loads(manifest.read_text()).get('source', [])
    if isinstance(sources, str):
        sources = [sources]
    poses = []
    for source in sources:
        snapshots = sorted((Path(source) / 'metadata').glob('episode_*.json'))
        if snapshots:
            points = json.loads(snapshots[0].read_text())['version']['waypoints']
            pose = points.get('p4_release')
            if pose is not None and pose[:5] not in poses:
                poses.append(pose[:5])
    return poses


def main():
    # 1. Locate the latest trained model
    out_dirs = sorted((ROOT / "outputs").glob("act_*"))
    if not out_dirs:
        print("未找到任何训练输出目录！")
        return
    model_path = out_dirs[-1] / "final"
    print(f"📦 加载最新 ACT 策略权重: {model_path}")

    from lerobot.policies.act import ACTPolicy
    from lerobot.policies import make_pre_post_processors

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    policy = ACTPolicy.from_pretrained(str(model_path)).to(device).eval()
    policy.reset()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=model_path,
                                         preprocessor_overrides={'device_processor': {'device': str(device)}})

    calibration = json.loads(CALIB_FILE.read_text())
    box_poses = recorded_box_poses(model_path)
    print(f'回程检测使用 {len(box_poses)} 个训练快照中的盒位；模型不接收版本编号', flush=True)

    # 2. Open Camera
    print("📷 打开摄像头...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    # 3. Connect Follower Arm
    print(f"🦾 连接从动臂: {PORT_FOLLOWER}...")
    _, definition = rustypot.find_model('sts3215')
    bus_dict = {i: definition for i in range(1, 7)}
    bus = rustypot.Bus(PORT_FOLLOWER, 1000000, 0.05, bus_dict)

    def read_positions():
        positions = []
        for i in range(1, 7):
            value, status = bus.read_register_with_error(i, 'present_position', retries=1)
            if status:
                raise RuntimeError(f'Joint {i} read status: {status}')
            positions.append(int(value) & 0x0FFF)
        return positions

    def write_register(i, register, value):
        status = bus.write_register_with_error(i, register, int(value), retries=0)
        if status:
            raise RuntimeError(f'Joint {i} {register} write status: {status}')

    normal_exit = False
    try:
        # Preload measured position so enabling never recalls an old goal.
        start = read_positions()
        print('Start measured:', start, flush=True)
        torque_limits = [300, 450, 450, 300, 200, 130]
        for i in range(1, 7):
            write_register(i, 'goal_position', start[i-1])
            write_register(i, 'torque_limit', torque_limits[i-1])
            write_register(i, 'goal_speed', 300)
            write_register(i, 'acceleration', 5)
            write_register(i, 'goal_time', 0)
            write_register(i, 'torque_enable', 1)

        print('ACT 单次测试：最多 60 秒；到盒区后返回待机并稳定 1 秒即结束，正常结束保持力矩', flush=True)
        dt = 0.1
        steps = 600
        deadline = time.monotonic() + 60
        standby = np.array(json.loads((ROOT / 'config/auto_waypoints.json').read_text())['p1_standby'])
        seen_box = False
        returning = False
        return_since = None
        for step in range(steps):
            if time.monotonic() >= deadline:
                break
            t_start = time.perf_counter()
            
            # Read current arm joint positions
            cur_raw = read_positions()
            at_box = any(np.all(np.abs(np.asarray(cur_raw[:5])-pose) <= np.array([80,110,110,100,80]))
                         for pose in box_poses) if box_poses else (cur_raw[1] > 2500 and cur_raw[2] < 1650)
            seen_box = seen_box or at_box
            returning = returning or (seen_box and cur_raw[1] < 1900 and cur_raw[2] > 2100)
            returned = seen_box and np.all(np.abs(np.asarray(cur_raw) - standby) <= np.array([40, 80, 80, 60, 40, 100]))
            return_since = (return_since or time.monotonic()) if returned else None
            if return_since is not None and time.monotonic() - return_since >= 1.0:
                for i in range(1, 7):
                    write_register(i, 'goal_position', cur_raw[i-1])
                print(f'回程完成，实测位置={cur_raw}；保持当前位置与力矩，不再循环', flush=True)
                break
            norm_state = normalize_joints(cur_raw, calibration)

            # Read camera frame
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError('摄像头读取失败，结束本次测试')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)

            # Prepare observation batch and predict using predict_action
            from lerobot.common.control_utils import predict_action
            action_pt = predict_action(
                {
                    'observation.state': norm_state,
                    'observation.images.external_front': rgb_small
                },
                policy, device, pre, post, use_amp=False, task='Pick up the red cube and place it into the box.',
                robot_type='so101_follower'
            )
            action_np = action_pt.detach().cpu().numpy().reshape(-1)

            if action_np.shape != (6,) or not np.isfinite(action_np).all():
                raise RuntimeError('模型输出无效，结束本次测试')
            goal_raw = denormalize_joints(action_np, calibration)
            if returning and goal_raw[1] > 2000 and goal_raw[2] < 2200:
                for i in range(1, 7):
                    write_register(i, 'goal_position', cur_raw[i-1])
                print(f'回程后模型又转向抓取，终止本轮并保持：{cur_raw}；未完成正常回程', flush=True)
                break

            for i in range(1, 7):
                write_register(i, 'goal_position', goal_raw[i-1])

            if step % 20 == 0:
                print(f'Step {step}/{steps}: measured={cur_raw} target={goal_raw}', flush=True)

            elapsed = time.perf_counter() - t_start
            if dt > elapsed:
                time.sleep(dt - elapsed)

        else:
            print('达到步数上限，停止预测但保持最后目标与力矩', flush=True)
        normal_exit = True
    except KeyboardInterrupt:
        print("\n用户手动中止推理")
    finally:
        if normal_exit:
            print('正常退出：不释放力矩；若达到时间上限，不代表回程成功', flush=True)
        else:
            print('异常/手动中止：释放电机力矩', flush=True)
            for i in range(1, 7):
                try:
                    write_register(i, 'torque_enable', 0)
                except Exception as exc:
                    print(f'WARNING: joint {i} torque-off failed: {exc}', flush=True)
        bus.close()
        cap.release()
        print("推理测试结束", flush=True)

if __name__ == "__main__":
    main()
