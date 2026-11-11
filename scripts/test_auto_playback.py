"""Test the freshly taught trajectory with exact smooth interpolation and hold.
"""
import sys
import time
import json
from pathlib import Path
from contextlib import closing
import numpy as np
import rustypot

ROOT = Path(__file__).resolve().parents[1]
WP_FILE = ROOT / "config/auto_waypoints.json"
PORT = "/dev/cu.usbserial-120"
BAUDRATE = 1000000

_, definition = rustypot.find_model('sts3215')
bus_dict = {i: definition for i in range(1, 7)}

def move_interp(bus, start_pos, target_pos, duration_s):
    steps = max(10, int(duration_s * 30))
    dt = duration_s / steps
    for s in range(1, steps + 1):
        alpha = s / steps
        smooth_alpha = (1.0 - np.cos(alpha * np.pi)) / 2.0
        cur = start_pos * (1 - smooth_alpha) + target_pos * smooth_alpha
        for i in range(1, 7):
            bus.write_register_with_error(i, "goal_position", int(cur[i-1]), retries=0)
        time.sleep(dt)

def main():
    wp = json.loads(WP_FILE.read_text())
    
    p1 = np.array(wp["p1_standby"])
    p2_app = np.array(wp["p2_approach"])
    p2_open = np.array(wp["p2_open"])
    p2_grasp = np.array(wp["p2_grasp"])
    p3_lift = np.array(wp["p3_lift"])
    p4_box = np.array(wp["p4_box"])
    p4_rel = np.array(wp["p4_release"])
    p4_exit = np.array(wp["p4_exit"])
    p1_transit = np.array(wp["p1_transit"])

    print("=== 开始验证全新 4 点轨迹 ===")
    with closing(rustypot.Bus(PORT, BAUDRATE, 0.05, bus_dict)) as bus:
        torque_limits = [300, 550, 600, 350, 250, 400]
        for i in range(1, 7):
            bus.write_register_with_error(i, "torque_limit", torque_limits[i-1], retries=1)
            bus.write_register_with_error(i, "goal_speed", 300, retries=1)
            bus.write_register_with_error(i, "acceleration", 5, retries=1)
            bus.write_register_with_error(i, "goal_time", 0, retries=1)
            bus.write_register_with_error(i, "torque_enable", 1, retries=1)
            
        cur = np.array([bus.read_register_with_error(i, 'present_position', retries=2)[0] & 0x0FFF for i in range(1, 7)])
        
        # 1. 待机位
        print("1. 平稳移到待机起点 (p1)...")
        move_interp(bus, cur, p1, duration_s=1.2)
        time.sleep(0.3)
        
        # 2. 闭爪前伸到红块上方
        print("2. 闭爪平稳前伸到红块正上方...")
        move_interp(bus, p1, p2_app, duration_s=1.5)
        
        # 3. 正上方张开爪子
        print("3. 正上方张开爪子...")
        move_interp(bus, p2_app, p2_open, duration_s=0.6)
        time.sleep(0.2)
        
        # 4. 下潜死死咬合红块
        print("4. 下潜咬紧红块...")
        move_interp(bus, p2_open, p2_grasp, duration_s=1.2)
        time.sleep(0.5)
        
        # 5. 垂直拔起避障
        print("5. 垂直提空...")
        move_interp(bus, p2_grasp, p3_lift, duration_s=1.2)
        
        # 6. 转往盒子上方
        print("6. 移到盒子...")
        move_interp(bus, p3_lift, p4_box, duration_s=1.5)
        
        # 7. 大张爪落盒
        print("7. 大张开爪子放块...")
        move_interp(bus, p4_box, p4_rel, duration_s=0.8)
        time.sleep(0.4)
        
        # 8. 原地高空抬起脱盒
        print("8. 垂直抬离盒子...")
        move_interp(bus, p4_rel, p4_exit, duration_s=1.0)
        
        # 9. 顺畅闭爪收高位
        print("9. 闭爪收拢中转...")
        move_interp(bus, p4_exit, p1_transit, duration_s=1.2)
        
        # 10. 彻底深压回第1点并锁死待机！
        print("10. 强力彻底缩回第 1 点到底并保持支撑...")
        move_interp(bus, p1_transit, p1, duration_s=1.2)
        for i in range(1, 7):
            bus.write_register_with_error(i, "goal_position", int(p1[i-1]), retries=0)
            
        print("\n🎉 全新轨迹验证完成！")

if __name__ == "__main__":
    main()
