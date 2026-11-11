"""Finalize leader calibration using the captured ranges and the current held posture.
Uses official FeetechMotorsBus logic to properly compute and clamp sign-magnitude homing offsets.
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode

ROOT = Path(__file__).resolve().parents[1]
PORT = "/dev/cu.usbserial-120"
BAUDRATE = 1000000

NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Ranges captured directly during your manual motion test:
CAPTURED_RANGES = {
    "shoulder_pan": (100, 1439),
    "shoulder_lift": (29, 4094),
    "elbow_flex": (2217, 3203),
    "wrist_flex": (561, 1242),
    "wrist_roll": (0, 4095),
    "gripper": (34, 1240),
}

def finalize():
    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    bus = FeetechMotorsBus(port=PORT, motors=motors)
    
    print(f"Connecting to leader on {PORT}...")
    bus.connect()
    try:
        # Unlock EEPROM
        for name in NAMES:
            bus.write("Lock", name, 0)
            bus.write("Torque_Enable", name, 0)
        
        # Read current reference position held by user
        current_positions = bus.sync_read("Present_Position", NAMES, normalize=False, num_retry=5)
        print(f"Captured current reference posture: {current_positions}")
        
        homing_offsets = bus.set_half_turn_homings(NAMES)
        print(f"Computed homing offsets: {homing_offsets}")
        
        # Read positions again after homing offset applied to verify
        # Then create calibration object
        calib_dict = {}
        for name in NAMES:
            mi, ma = CAPTURED_RANGES[name]
            calib_dict[name] = MotorCalibration(
                id=bus.motors[name].id,
                drive_mode=0,
                homing_offset=int(homing_offsets[name]),
                range_min=int(mi),
                range_max=int(ma),
            )
        
        print("\nWriting calibration position limits to EEPROM...")
        bus.write_calibration(calib_dict)
        
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
        out_path = ROOT / "calibration" / f"so101-leader-{tag}.json"
        
        data_to_save = {}
        for name, cal in calib_dict.items():
            data_to_save[name] = {
                "id": cal.id,
                "drive_mode": cal.drive_mode,
                "homing_offset": cal.homing_offset,
                "range_min": cal.range_min,
                "range_max": cal.range_max,
            }
        
        out_path.write_text(json.dumps(data_to_save, indent=2), encoding="utf-8")
        print(f"\n🎉 主臂标定彻底完成！文件已保存：\n{out_path}")
        print(json.dumps(data_to_save, indent=2))
        return out_path
    finally:
        bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    finalize()
