import json
import time
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

def read_pos(bus, motor_id):
    # Retry up to 3 times on transient status glitch
    for attempt in range(3):
        try:
            val, status = bus.read_register_with_error(motor_id, "present_position", retries=1)
            # If status flag is set during fast manual move, ignore unless persistent
            if status != 0 and attempt < 2:
                time.sleep(0.01)
                continue
            if val & 0x8000:
                val = -(val & 0x7FFF)
            return val
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.01)

def run_range_recording(duration_s=30):
    _, definition = rustypot.find_model('sts3215')
    with closing(rustypot.Bus(PORT, BAUDRATE, 0.05, {i: definition for i in range(1, 7)})) as bus:
        for i in range(1, 7):
            bus.write_register_with_error(i, "torque_enable", 0, retries=1)
            bus.write_register_with_error(i, "lock", 0, retries=1)
        
        # Capture reference
        ref_positions = [read_pos(bus, i) for i in range(1, 7)]
        print(f"Reference captured: {ref_positions}")
        
        mins = list(ref_positions)
        maxs = list(ref_positions)
        
        # Merge previously captured good ranges for gripper and wrist
        mins[5] = min(mins[5], 34)
        maxs[5] = max(maxs[5], 1240)
        
        print(f"\n>>> 范围录制开始！你有 {duration_s} 秒时间 <<<")
        print("重点扳动：")
        print("  1. 底座 (1号) - 往左、往右转到正常工作角度")
        print("  2. 大臂 (2号) - 往前伸展、往上抬起")
        print("  3. 肘部、手腕、夹爪随意活动一下")
        
        start_t = time.time()
        last_print = 0
        while time.time() - start_t < duration_s:
            for idx in range(6):
                try:
                    pos = read_pos(bus, idx + 1)
                    if pos < mins[idx]:
                        mins[idx] = pos
                    if pos > maxs[idx]:
                        maxs[idx] = pos
                except Exception:
                    pass
            rem = int(duration_s - (time.time() - start_t))
            if rem != last_print and rem % 5 == 0:
                print(f"剩余时间: {rem} 秒... 当前读数范围:")
                for n, mi, ma in zip(NAMES, mins, maxs):
                    print(f"  {n}: min={mi}, max={ma}")
                last_print = rem
            time.sleep(0.02)
        
        print("\n>>> 录制完成！计算标定参数... <<<")
        mins[4] = 0
        maxs[4] = 4095
        
        calib = {}
        for idx, name in enumerate(NAMES):
            ref = ref_positions[idx]
            offset = 2048 - ref
            calib[name] = {
                "id": idx + 1,
                "drive_mode": 0,
                "homing_offset": offset,
                "range_min": mins[idx],
                "range_max": maxs[idx],
            }
            raw_offset = (abs(offset) | 0x0800) if offset < 0 else offset
            bus.write_register_with_error(idx + 1, "homing_offset", raw_offset, retries=2)
            bus.write_register_with_error(idx + 1, "min_position_limit", mins[idx], retries=2)
            bus.write_register_with_error(idx + 1, "max_position_limit", maxs[idx], retries=2)
        
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
        out_path = ROOT / "calibration" / f"so101-leader-{tag}.json"
        out_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        print(f"\n✅ 主臂标定成功保存到: {out_path}")
        print(json.dumps(calib, indent=2))
        return out_path

if __name__ == "__main__":
    run_range_recording(duration_s=30)
