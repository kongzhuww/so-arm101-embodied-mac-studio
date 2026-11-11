"""NO HARDWARE ACCESS: phone -> tiny translation -> joint target preview.

This is a mathematical preview, not a safety controller. There is no serial,
robot construction, torque switch, motor write, auto-home or automatic return.
Baseline is a saved read, NOT live feedback. Physical model alignment and
collision/cable limits remain unverified. Defaults to a synthetic 1mm test;
--live observes the phone status file for up to 180s, without moving any motor.
"""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from lerobot.model.kinematics import RobotKinematics

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
CAL = ROOT / 'calibration/so101-follower-20261031T144217-014032Z.json'
URDF = ROOT / 'models/SO101/so101_new_calib.urdf'
STATUS = ROOT / 'reports/phone-input-status.json'
REPORT = ROOT / 'reports/phone-arm-preview.json'


class Preview:
    def __init__(self):
        self.cal = json.loads(CAL.read_text())
        self.raw = np.asarray(json.loads((ROOT/'config/arm-preview-baseline.json').read_text())['positions'], dtype=float)
        self.mid = np.array([(self.cal[n]['range_min']+self.cal[n]['range_max'])/2 for n in NAMES])
        self.q0 = (self.raw[:5]-self.mid)*360/4095
        self.kin = RobotKinematics(str(URDF), joint_names=NAMES)
        self.t0 = self.kin.forward_kinematics(self.q0).copy()
        self.urdf_limits = {}
        for joint in ET.parse(URDF).getroot().findall('joint'):
            if joint.get('name') in NAMES:
                limit = joint.find('limit')
                self.urdf_limits[joint.get('name')] = np.rad2deg([
                    float(limit.get('lower')), float(limit.get('upper'))])
        self.reference = None
        self.enabled = False
        self.saw_release = False

    def propose(self, delta):
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError('Invalid translation')
        if np.linalg.norm(delta) > .002000001:
            raise ValueError('Preview limited to 2mm from recorded pose')
        target = self.t0.copy()
        target[:3, 3] += delta
        # Five body joints cannot satisfy arbitrary 3D position plus a fixed
        # 3D orientation. Translation-only preview explicitly leaves orientation
        # unconstrained; do NOT relax residual or joint/position bounds.
        q = self.kin.inverse_kinematics(
            self.q0, target, orientation_weight=0.0, max_iters=16)
        if not np.isfinite(q).all():
            raise ValueError('Nonfinite IK')
        raw = np.rint(q*4095/360+self.mid).astype(int)
        reasons = []
        if np.max(np.abs(q-self.q0)) > 1:
            reasons.append('joint change exceeds 1 degree')
        for i, name in enumerate(NAMES):
            c = self.cal[name]
            if not c['range_min']+64 <= raw[i] <= c['range_max']-64:
                reasons.append(f'motor{i+1} outside recorded inward interval')
            lo, hi = self.urdf_limits[name]
            if not lo <= q[i] <= hi:
                reasons.append(f'motor{i+1} outside URDF interval')
        residual = float(np.linalg.norm(self.kin.forward_kinematics(q)[:3, 3]-target[:3, 3]))
        if residual > .0001:
            reasons.append('IK position residual exceeds 0.1mm')
        return {
            'mode': 'DRY_RUN_NO_HARDWARE', 'motion_authorized': False,
            'math_checks_passed': not reasons, 'rejections': reasons,
            'ee_delta_m': delta.tolist(), 'position_residual_m': residual,
            'orientation_policy': 'unconstrained_translation_only',
            'seed_degrees': self.q0.tolist(), 'target_degrees': q.tolist(),
            'proposed_raw_ids1_to6': [int(v) for v in raw]+[int(self.raw[5])],
            'baseline_is_live': False,
            'unverified': ['physical pose and directions', 'self/table collision',
                           'motor5 cable-safe interval', 'powered body-joint control'],
        }

    def sample(self, sample, now):
        stamp = sample.get('last_pose_at', 0)
        if not 0 <= now-stamp <= 1:
            raise ValueError('Phone stream stale')
        pose = np.asarray(sample.get('phone_pose_flu'), dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('Fresh phone matrix unavailable; refresh input service')
        if not sample.get('move_pressed', False):
            self.enabled = False
            self.reference = None
            self.saw_release = True
            return None
        if not self.saw_release:
            return None
        # Same reference rotation and x/y remap as the official Android phone
        # processors. This still needs a physical forward/left/up direction test.
        if not self.enabled:
            self.reference = pose.copy()
            self.enabled = True
        local = self.reference[:3, :3].T @ (pose[:3, 3]-self.reference[:3, 3])
        delta = np.array([local[1], -local[0], local[2]]) * .05
        length = np.linalg.norm(delta)
        if length > .002:
            delta *= .002/length
        return self.propose(delta)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args(argv)
    planner = Preview()
    if not args.live:
        report = planner.propose([0, 0, .001])
        REPORT.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 0 if report['math_checks_passed'] else 1
    deadline = time.monotonic()+180
    last_stamp = None
    count = 0
    print('DRY RUN ONLY: release Move, then hold Move and translate phone slowly.', flush=True)
    while time.monotonic() < deadline:
        sample = json.loads(STATUS.read_text())
        if not 0 <= time.time()-sample.get('last_pose_at', 0) <= 1:
            # Read-only: emit no target from stale data and require a new
            # release/reference after tracking resumes. Wait for user input.
            planner.enabled = False
            planner.reference = None
            planner.saw_release = False
            time.sleep(.05)
            continue
        if sample.get('last_pose_at') == last_stamp:
            time.sleep(.05)
            continue
        last_stamp = sample.get('last_pose_at')
        report = planner.sample(sample, time.time())
        if report:
            count += 1
            report['live_phone_preview_count'] = count
            REPORT.write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        time.sleep(.05)
    return 0 if count else 1


if __name__ == '__main__':
    raise SystemExit(main())
