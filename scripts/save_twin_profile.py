"""Calibrated Direct Leader-Follower Mirroring Profile.
Calculated directly from human reference posture matching:
Leader Reference:   [2285, 2061, 1997, 2027, 1886, 1531]
Follower Reference: [2050, 2058, 2039, 2108, 1922, 1995]
"""
import json
from pathlib import Path

PROFILE = {
    "leader_port": "/dev/cu.usbserial-120",
    "follower_port": "/dev/cu.usbserial-130",
    "baudrate": 1000000,
    "leader_ref": [2285, 2061, 1997, 2027, 1886, 1531],
    "follower_ref": [2050, 2058, 2039, 2108, 1922, 1995],
    # Direction signs:
    # 1: base (pan)同向
    # 2: shoulder同向
    # 3: elbow同向
    # 4: wrist flex同向
    # 5: wrist roll同向
    # 6: gripper: leader松开数值变大(+1231)，从动臂开合方向
    "signs": [1, 1, 1, 1, 1, 1],
    # Follower safe limits
    "follower_limits": [
        [817, 3411],
        [860, 3207],
        [874, 3102],
        [917, 3171],
        [0, 4095],
        [1995, 3438]  # 1995是完全闭合，3438完全张开
    ]
}

p = Path(__file__).resolve().parents[1] / "calibration" / "twin_mirror_profile.json"
p.write_text(json.dumps(PROFILE, indent=2))
print("Saved mirror profile to", p)
