"""Use official LeRobot FeetechMotorsBus to calibrate leader arm on /dev/cu.usbserial-120.
Saves calibration JSON to so-arm101-act/calibration/
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from lerobot.motors.feetech import FeetechMotorsBus, FeetechMotorsBusConfig
from lerobot.motors import Motor, MotorCalibration

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

def calibrate():
    motors = {name: Motor(id=i+1, model="sts3215") for i, name in enumerate(NAMES)}
    config = FeetechMotorsBusConfig(port=PORT, baudrate=BAUDRATE, motors=motors)
    bus = FeetechMotorsBus(config)
    
    print(f"Connecting to leader on {PORT}...")
    bus.connect()
    try:
        # Unlock EEPROM
        for name in NAMES:
            bus.write("Lock", name, 0)
            bus.write("Torque_Enable", name, 0)
        
        print("\n>>> 开始标定！<<<")
        # record_calibration handles reference posture prompt and range recording properly
        # with Feetech sign-magnitude offset clamping!
        calib_dict = bus.record_calibration(
            mode="position",
            full_turn_motors=["wrist_roll"],
            inverted_motors=(),
        )
        
        print("\nWriting calibration to motors EEPROM...")
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
        print(f"\n🎉 主臂标定成功！文件已保存：\n{out_path}")
        print(json.dumps(data_to_save, indent=2))
        return out_path
    finally:
        bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    calibrate()
