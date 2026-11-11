import json
from pathlib import Path
from datetime import datetime, timezone
from contextlib import closing
import rustypot

PORT = "/dev/cu.usbserial-120"
BAUDRATE = 1000000
ROOT = Path(__file__).resolve().parents[1]

NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# We captured the real human operation ranges during your 30s session:
# shoulder_pan: rotated from ~100 to 1439
# shoulder_lift: rotated from 29 to 4094
# elbow_flex: 2217 to 3203
# wrist_flex: ~500 to 1242
# wrist_roll: full turn 0..4095
# gripper: 34 to 1240

def read_raw_u16(bus, motor_id):
    val, status = bus.read_register_with_error(motor_id, "present_position", retries=2)
    # unsigned 12-bit value (0..4095)
    return val & 0x0FFF

def finalize_leader_calib():
    _, definition = rustypot.find_model('sts3215')
    with closing(rustypot.Bus(PORT, BAUDRATE, 0.1, {i: definition for i in range(1, 7)})) as bus:
        for i in range(1, 7):
            bus.write_register_with_error(i, "torque_enable", 0, retries=1)
            bus.write_register_with_error(i, "lock", 0, retries=1)
        
        cur = [read_raw_u16(bus, i) for i in range(1, 7)]
        print(f"Current leader positions: {cur}")
        
        # Valid physical ranges (0..4095 clamped):
        ranges = [
            (50, 1500),      # shoulder_pan
            (100, 3900),     # shoulder_lift
            (2100, 3300),    # elbow_flex
            (400, 1400),     # wrist_flex
            (0, 4095),       # wrist_roll
            (30, 1300),      # gripper
        ]
        
        calib = {}
        for idx, name in enumerate(NAMES):
            ref = cur[idx]
            offset = 2048 - ref
            mi, ma = ranges[idx]
            calib[name] = {
                "id": idx + 1,
                "drive_mode": 0,
                "homing_offset": offset,
                "range_min": mi,
                "range_max": ma,
            }
            raw_offset = (abs(offset) | 0x0800) if offset < 0 else offset
            bus.write_register_with_error(idx + 1, "homing_offset", raw_offset, retries=2)
            bus.write_register_with_error(idx + 1, "min_position_limit", mi, retries=2)
            bus.write_register_with_error(idx + 1, "max_position_limit", ma, retries=2)
        
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
        out_path = ROOT / "calibration" / f"so101-leader-{tag}.json"
        out_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        print(f"\n✅ 主臂标定成功写入并保存到: {out_path}")
        print(json.dumps(calib, indent=2))
        return out_path

if __name__ == "__main__":
    finalize_leader_calib()
