"""Calibrate SO-101 leader arm on /dev/cu.usbserial-120.
Interactive via simple text interface or background recorder.
Only writes EEPROM offsets/limits; never enables torque or sends motion goals.
"""
import json
from pathlib import Path
import time
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
FULL_TURN = ["wrist_roll"]

def read_pos(bus, motor_id):
    val, status = bus.read_register_with_error(motor_id, "present_position", retries=2)
    if status != 0:
        raise RuntimeError(f"Motor {motor_id} alarm {status}")
    if val & 0x8000:
        val = -(val & 0x7FFF)
    return val

def record_baseline():
    _, definition = rustypot.find_model('sts3215')
    with closing(rustypot.Bus(PORT, BAUDRATE, 0.1, {i: definition for i in range(1, 7)})) as bus:
        for i in range(1, 7):
            bus.write_register_with_error(i, "torque_enable", 0, retries=1)
            bus.write_register_with_error(i, "lock", 0, retries=1)
        positions = [read_pos(bus, i) for i in range(1, 7)]
    return positions

if __name__ == "__main__":
    print(f"Connecting to leader on {PORT}...")
    pos = record_baseline()
    print(f"Captured reference posture positions: {pos}")
