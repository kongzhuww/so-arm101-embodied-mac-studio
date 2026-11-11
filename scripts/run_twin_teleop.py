"""Direct Twin-Mirroring Teleoperation.
Bypasses complex abstract normalizations and uses physical twin reference matching!
Leader posture maps directly to follower posture 1:1.
"""
import sys
import time
import json
from pathlib import Path
from contextlib import closing
import numpy as np
import rustypot

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "calibration/twin_mirror_profile.json"

def main():
    profile = json.loads(PROFILE_PATH.read_text())
    l_port = profile["leader_port"]
    f_port = profile["follower_port"]
    l_ref = np.array(profile["leader_ref"])
    f_ref = np.array(profile["follower_ref"])
    signs = np.array(profile["signs"])
    limits = profile["follower_limits"]
    
    _, definition = rustypot.find_model('sts3215')
    bus_dict = {i: definition for i in range(1, 7)}
    
    print("=== SO-101 孪生直映遥操作 (Twin Mirroring) ===")
    print("正在连接主臂与从臂...")
    
    with closing(rustypot.Bus(l_port, 1000000, 0.05, bus_dict)) as l_bus, \
         closing(rustypot.Bus(f_port, 1000000, 0.05, bus_dict)) as f_bus:
         
        # Ensure Leader torque is off
        for i in range(1, 7):
            l_bus.write_register_with_error(i, "torque_enable", 0, retries=1)
            
        # Configure follower output profile
        # Moderately safe speeds and torque limits
        torque_limits = [300, 500, 500, 350, 250, 350]
        for i in range(1, 7):
            f_bus.write_register_with_error(i, "torque_limit", torque_limits[i-1], retries=1)
            f_bus.write_register_with_error(i, "goal_speed", 350, retries=1)
            f_bus.write_register_with_error(i, "acceleration", 5, retries=1)
            f_bus.write_register_with_error(i, "goal_time", 0, retries=1)
            
        # Read starting positions
        l_curr = np.array([l_bus.read_register_with_error(i, 'present_position', retries=2)[0] & 0x0FFF for i in range(1, 7)])
        f_curr = np.array([f_bus.read_register_with_error(i, 'present_position', retries=2)[0] & 0x0FFF for i in range(1, 7)])
        
        # Calculate target based on leader offset from its reference
        target = f_ref + signs * (l_curr - l_ref)
        # Clamp to safe follower limits
        for i in range(6):
            target[i] = max(limits[i][0], min(limits[i][1], target[i]))
            
        print(f"\n主臂当前位置: {l_curr.tolist()}")
        print(f"从臂当前位置: {f_curr.tolist()}")
        print(f"从臂目标位置: {target.astype(int).tolist()}")
        
        print("\n>>> 开始 2 秒平滑启动对齐... <<<")
        # Turn torque ON on follower
        for i in range(1, 7):
            f_bus.write_register_with_error(i, "torque_enable", 1, retries=1)
            
        steps = 50
        for s in range(1, steps + 1):
            alpha = s / steps
            interp = f_curr * (1 - alpha) + target * alpha
            for i in range(1, 7):
                f_bus.write_register_with_error(i, "goal_position", int(interp[i-1]), retries=0)
            time.sleep(2.0 / steps)
            
        print("\n=======================================================")
        print("🎉 孪生镜像遥操作运行中！主臂动，从臂 1:1 同步跟随！")
        print("按 Ctrl+C 随时安全退出，从臂自动关闭扭矩。")
        print("=======================================================")
        
        hz = 30
        dt = 1.0 / hz
        last_log = time.time()
        
        try:
            while True:
                t0 = time.perf_counter()
                
                # 1. Read Leader 6 motors
                l_pos = np.array([l_bus.read_register_with_error(i, 'present_position', retries=1)[0] & 0x0FFF for i in range(1, 7)])
                
                # 2. Direct twin transform: Follower = F_ref + Signs * (Leader - L_ref)
                f_target = f_ref + signs * (l_pos - l_ref)
                
                # 3. Apply safety clamping
                for i in range(6):
                    f_target[i] = max(limits[i][0], min(limits[i][1], f_target[i]))
                    
                # 4. Write to Follower 6 motors
                for i in range(1, 7):
                    f_bus.write_register_with_error(i, "goal_position", int(f_target[i-1]), retries=0)
                    
                if time.time() - last_log > 1.0:
                    grip_pct = (f_target[5] - limits[5][0]) * 100 / (limits[5][1] - limits[5][0])
                    print(f"实时跟随中... 夹爪开合度: {grip_pct:.1f}%")
                    last_log = time.time()
                    
                elapsed = time.perf_counter() - t0
                if dt > elapsed:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n\n安全退出中...")
        finally:
            print("正在关闭从臂扭矩 (Torque OFF)...")
            for i in range(1, 7):
                try:
                    f_bus.write_register_with_error(i, "torque_enable", 0, retries=1)
                except Exception:
                    pass
            print("已完全断开并安全退出。")

if __name__ == "__main__":
    main()
