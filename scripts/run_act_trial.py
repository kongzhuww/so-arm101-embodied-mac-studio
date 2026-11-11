"""One authorized 60-second ACT run; default plan only. No auto-home/return.

Uses saved policy/preprocessors and current RGB/joints. Targets stay within
recorded demonstration/calibration ranges, with a 64-count relative step at
10Hz. Explicit run profile: raw speed300/acceleration5; torque limits per joint
250/500/500/300/200/100. These are raw settings, not calibrated force/speed.
No file logging or snapshots. Finally disables all six motors.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import time

import cv2
import numpy as np
import torch

import gripper_microstep as m
from record_act_web import NAMES, normalize, TASK

MODEL = m.ROOT/'outputs/act_20261031T173416Z/final'
DATA = m.ROOT/'datasets/so101_pickplace_20261031T173416Z'
TORQUE_LIMITS = {1:250, 2:500, 3:500, 4:300, 5:200, 6:300}


class QuietEvents:
    def emit(self, **kwargs): pass
    def close(self): pass


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--ack-act-motion', action='store_true')
    args = p.parse_args(argv)
    if args.execute != args.ack_act_motion:
        p.error('Both flags require current real ACT-motion consent')
    if not args.execute:
        print('PLAN ONLY: no camera/serial/model opened. Saved ACT -> all6 joints, 60s, 10Hz; then torque OFF. No homing.')
        return 0
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies import make_pre_post_processors
    from lerobot.common.control_utils import predict_action

    calibration = m.load_calibration()
    cal_dict = {name: vars(c) for name, c in calibration.items()}
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    policy = ACTPolicy.from_pretrained(str(MODEL)).to(device).eval()
    policy.reset()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=MODEL,
        preprocessor_overrides={'device_processor': {'device': str(device)}})
    stats = json.loads((DATA/'meta/stats.json').read_text())['action']
    action_lo, action_hi = np.array(stats['min']), np.array(stats['max'])
    raw_lo = np.array([calibration[n].range_min for n in NAMES])
    raw_hi = np.array([calibration[n].range_max for n in NAMES])
    midpoint = (raw_lo+raw_hi)/2
    journal = QuietEvents()
    camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, fps=60, width=640, height=480, backend=1200))
    frames = 0
    last_rgb = None

    def predict(raw_positions):
        nonlocal last_rgb
        rgb = camera.async_read(timeout_ms=500)
        if rgb.shape != (480,640,3):
            raise m.Stop('Unexpected camera frame shape')
        last_rgb = rgb
        image = cv2.resize(rgb, (320,240), interpolation=cv2.INTER_AREA)
        action = predict_action(
            {'observation.state': normalize(raw_positions, cal_dict),
             'observation.images.external_front': image},
            policy, device, pre, post, use_amp=False, task=TASK,
            robot_type='so101_follower')
        action = action.detach().cpu().numpy().reshape(-1)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise m.Stop('Invalid ACT action')
        bounded = np.clip(action, action_lo, action_hi)
        raw_goal = bounded*4095/360+midpoint
        raw_goal[5] = bounded[5]*(raw_hi[5]-raw_lo[5])/100+raw_lo[5]
        return action, np.rint(np.clip(raw_goal, raw_lo, raw_hi)).astype(int)

    try:
        camera.connect()
        with closing(m.open_raw()) as raw:
            gate = m.Motor6Gate(raw, journal)
            for i in range(1, 7):
                if gate.read(i, 'model_number') != 777:
                    raise m.Stop('Motor model mismatch')
            temperature_limits = {}
            for n in NAMES:
                c = calibration[n]
                i = c.id
                if gate.read(i, 'torque_enable') != 0 or gate.read(i, 'operating_mode') != 0 or gate.read(i, 'phase') & 16:
                    raise m.Stop('Requires torque-off position mode with unchanged phase')
                actual = tuple(gate.read(i, k) for k in ('homing_offset','min_position_limit','max_position_limit'))
                if actual != (c.homing_offset,c.range_min,c.range_max):
                    raise m.Stop('Calibration differs from hardware')
                temperature_limit, read_status = raw.read_register_with_error(i,'max_temperature_limit',retries=0)
                if read_status or temperature_limit <= 0:
                    raise m.Stop(f'Motor{i} temperature protection configuration unreadable or disabled')
                temperature_limits[i] = temperature_limit
                if gate.read(i, 'status') or gate.read(i, 'present_temperature') >= temperature_limits[i] or not 100 <= gate.read(i,'present_voltage') <= 130:
                    raise m.Stop(f'Motor{i} alarm/temperature/voltage outside configured protection')
            positions = np.array([gate.read(i, 'present_position') for i in range(1,7)])
            # Real camera/model warm-up BEFORE the first actuation, not random data.
            predict(positions)
            policy.reset()
            actuated = False
            completed = False
            try:
                print('ACT 60s RUN: speed300; torque limits250/500/500/300/200/100; no homing.', flush=True)
                actuated = True  # Also clean up after uncertain configuration/goal writes.
                for i in range(1,7):
                    settings = {'torque_limit':TORQUE_LIMITS[i], 'acceleration':5,
                                'goal_speed':300, 'goal_time':0}
                    for key, value in settings.items():
                        status = raw.write_register_with_error(i,key,value,retries=0)
                        if status or gate.read(i,'torque_enable') != 0 or gate.read(i,key) != value:
                            raise m.Stop(f'Motor{i} run-profile setup failed; no goal retry')
                for i in range(1,7):
                    # This hold goal is an ACTUATION, never a passive preload.
                    current = gate.read(i, 'present_position')
                    status = raw.write_register_with_error(i, 'goal_position', current, retries=0)
                    journal.emit(event='initial_hold_is_actuation', id=i, target=current, status=status)
                    if status:
                        raise m.Stop('Initial goal alarm; no retry')
                deadline = time.monotonic()+60
                previous = positions
                while time.monotonic() < deadline:
                    tick = time.monotonic()
                    positions = np.array([gate.read(i, 'present_position') for i in range(1,7)])
                    if np.max(np.abs(positions-previous)) > 64:
                        raise m.Stop('Unexpected measured joint displacement')
                    action, proposed = predict(positions)
                    if time.monotonic() >= deadline:
                        break
                    targets = np.clip(proposed, positions-64, positions+64)
                    targets = np.clip(targets, raw_lo, raw_hi).astype(int)
                    for i in range(1,7):
                        alarm = gate.read(i,'status')
                        temperature = gate.read(i,'present_temperature')
                        output_limit = gate.read(i,'torque_limit')
                        if alarm or temperature >= temperature_limits[i] or not 0 <= output_limit <= TORQUE_LIMITS[i]:
                            raise m.Stop(f'Motor{i}: status={alarm}, temperature={temperature}, configured_temperature_limit={temperature_limits[i]}, torque_limit={output_limit}, configured_output_limit={TORQUE_LIMITS[i]}')
                    for i, value in enumerate(targets,1):
                        journal.emit(event='write_attempt', id=i, register='goal_position', value=int(value))
                        status = raw.write_register_with_error(i, 'goal_position', int(value), retries=0)
                        if status:
                            raise m.Stop(f'Motor{i} goal alarm; no retry')
                    frames += 1
                    journal.emit(event='act_frame', frame=frames, positions=positions.tolist(),
                                 action=action.tolist(), proposed=proposed.tolist(), sent=targets.tolist())
                    previous = positions
                    time.sleep(max(0,.1-(time.monotonic()-tick)))
                completed = True
                journal.emit(event='trial_duration_complete', frames=frames, task_success='not_assessed')
            finally:
                if actuated:
                    off = []
                    for i in range(1,7):
                        verified = False
                        for _ in range(2):
                            try:
                                status = raw.write_register_with_error(i,'torque_enable',0,retries=0)
                                if status == 0 and gate.read(i,'torque_enable') == 0:
                                    verified = True
                                    break
                            except Exception as exc:
                                journal.emit(event='disable_problem', id=i, error=type(exc).__name__)
                        off.append(verified)
                    journal.emit(event='cleanup', all_torque_off_verified=all(off), per_motor=off)
                    if not all(off):
                        raise m.Stop('无法确认六轴扭矩关闭，请立即物理断电')
            print('ACT run finished:',completed,'frames:',frames,'; all torque OFF verified.',flush=True)
    finally:
        if camera.is_connected:
            camera.disconnect()
        journal.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
