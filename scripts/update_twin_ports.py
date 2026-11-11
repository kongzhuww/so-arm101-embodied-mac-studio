"""Update twin mirror profile with auto-detected port roles by voltage.
Leader: 4.1V (USB powered, no 12V) -> /dev/cu.usbserial-130
Follower: 12.3V (12V powered)     -> /dev/cu.usbserial-120
"""
import json
from pathlib import Path

PROFILE_PATH = Path(__file__).resolve().parents[1] / "calibration/twin_mirror_profile.json"
profile = json.loads(PROFILE_PATH.read_text())

# Swap ports to match actual current physical plug state
profile["leader_port"] = "/dev/cu.usbserial-130"
profile["follower_port"] = "/dev/cu.usbserial-120"

PROFILE_PATH.write_text(json.dumps(profile, indent=2))
print("Updated twin_mirror_profile.json with correct swapped ports:")
print(f"  Leader:   {profile['leader_port']}")
print(f"  Follower: {profile['follower_port']}")
