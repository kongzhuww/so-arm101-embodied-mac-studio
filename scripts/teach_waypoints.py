"""Interactive Waypoint Teaching Script for SO-101.
Reads human-held positions on Follower (/dev/cu.usbserial-120) with torque OFF.
Saves keypoints for autonomous trajectory playback & dataset recording.
"""
import sys
import time
import json
from pathlib import Path
from contextlib import closing
import rustypot

PORT = "/dev/cu.usbserial-120"
BAUDRATE = 1000000
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "config/auto_waypoints.json"

_, definition = rustypot.find_model('sts3215')
bus_dict = {i: definition for i in range(1, 7)}

def read_current_pos(bus):
    return [bus.read_register_with_error(i, 'present_position', retries=2)[0] & 0x0FFF for i in range(1, 7)]

def main():
    print("=== SO-101 示教打点工具 (Waypoint Teaching) ===")
    print("机械臂全程断电/无力状态，纯手动摆动！\n")
    
    with closing(rustypot.Bus(PORT, BAUDRATE, 0.05, bus_dict)) as bus:
        # Ensure all torque off
        for i in range(1, 7):
            bus.write_register_with_error(i, 'torque_enable', 0, retries=1)
            
        waypoints = {}
        
        # 1. Standby pose
        input("【第 1 步】：用手把机械臂摆在【上方待机姿态】（夹爪半开，悬在半空），摆好按回车...")
        p_ready = read_current_pos(bus)
        # Ensure gripper is open in standby (e.g., ~2800)
        p_ready[5] = max(p_ready[5], 2800)
        waypoints["ready"] = p_ready
        print(f"  --> 已记录【上方待机姿态】: {p_ready}\n")
        
        # 2. Grasp pose
        input("【第 2 步】：把红块放在你想抓的位置，用手把机械臂移过去，让夹爪正好把红块【包在中间捏紧】，摆好按回车...")
        p_grasp = read_current_pos(bus)
        # Ensure gripper is tightly closed (e.g., ~2050)
        p_grasp[5] = min(p_grasp[5], 2150)
        waypoints["grasp"] = p_grasp
        print(f"  --> 已记录【精准咬合姿态】: {p_grasp}\n")
        
        # 3. Lift pose
        input("【第 3 步】：保持捏住红块的手感，稍微把手臂垂直向上抬起 5~10 厘米（升空避障），摆好按回车...")
        p_lift = read_current_pos(bus)
        p_lift[5] = p_grasp[5] # Keep closed
        waypoints["lift"] = p_lift
        print(f"  --> 已记录【垂直抬升姿态】: {p_lift}\n")
        
        # 4. Box release pose
        input("【第 4 步】：把机械臂移到【盒子上方】（松手放开红块的位置），摆好按回车...")
        p_drop = read_current_pos(bus)
        # Gripper open to release
        p_drop[5] = 2900
        waypoints["drop"] = p_drop
        print(f"  --> 已记录【入盒释放姿态】: {p_drop}\n")
        
        OUTPUT_FILE.write_text(json.dumps(waypoints, indent=2))
        print(f"🎉 4 个黄金打点已全部保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
