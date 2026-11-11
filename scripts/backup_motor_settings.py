"""Separate READ-only register snapshot. Does NOT broaden read_only_probe's gate.

Default: print plan, no serial access. No writes or calibration, even on failure.
Snapshot values are raw register encodings, not LeRobot calibration or joint angles.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from contextlib import closing

import read_only_probe as base

REGISTERS = {
    "model_number": (3, 2), "id": (5, 1), "torque_enable": (40, 1),
    "lock": (55, 1), "operating_mode": (33, 1), "phase": (18, 1),
    "homing_offset": (31, 2), "min_position_limit": (9, 2),
    "max_position_limit": (11, 2), "firmware_major_version": (0, 1),
    "firmware_minor_version": (1, 1), "present_position": (56, 2),
}


def request_packet(motor_id, register):
    if type(motor_id) is not int or motor_id not in base.IDS or register not in REGISTERS:
        raise base.ProbeError("Outside backup's fixed ID/register READ allowlist")
    addr, size = REGISTERS[register]
    body = bytes((motor_id, 4, 2, addr, size))
    return b"\xff\xff" + body + bytes(((~sum(body)) & 255,))


ALLOWED = frozenset(request_packet(i, r) for i in base.IDS for r in REGISTERS)


def validate_packet(packet):
    if packet not in ALLOWED:
        raise base.ProbeError("Blocked non-allowlisted backup request")


class BackupLink(base.ReadOnlyLink):
    def read_register(self, motor_id, register):
        import time
        packet = request_packet(motor_id, register)
        validate_packet(packet)
        self._serial.reset_input_buffer()
        if self._serial.write(packet) != len(packet):
            raise base.ProbeError("Partial backup READ transmission; no retry")
        deadline = time.monotonic() + 0.5
        head = self._read_exact(4, deadline)
        size = REGISTERS[register][1]
        if head != bytes((255, 255, motor_id, size + 2)):
            raise base.ProbeError("Unexpected backup response header/ID/length")
        tail = self._read_exact(size + 2, deadline)
        if sum(head[2:] + tail) & 255 != 255:
            raise base.ProbeError("Backup response checksum mismatch")
        if tail[0] != 0:
            raise base.ProbeError(f"Motor {motor_id}: alarm/status {tail[0]}; stop")
        return int.from_bytes(tail[1:-1], "little")


def collect(link):
    motors = {}
    for motor_id in base.IDS:
        model = link.read_register(motor_id, "model_number")
        if model != base.EXPECTED_MODEL:
            raise base.ProbeError(f"Unexpected model at ID {motor_id}: {model}")
        torque = link.read_register(motor_id, "torque_enable")
        if torque != 0:
            raise base.ProbeError(f"ID {motor_id} torque is {torque}; stop without changing it")
        values = {"model_number": model, "torque_enable": torque}
        for register in REGISTERS:
            if register not in values:
                values[register] = link.read_register(motor_id, register)
        if values["id"] != motor_id:
            raise base.ProbeError("Servo ID register mismatch")
        # Recheck at end of each motor, do not assume a past read is current forever.
        if link.read_register(motor_id, "torque_enable") != 0:
            raise base.ProbeError("Torque changed during backup; stop without changing it")
        motors[str(motor_id)] = values
    return {
        "schema": "so101-raw-register-backup-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "port": base.PORT, "baudrate": base.BAUDRATE,
        "encoding": "Unsigned little-endian raw words. Homing offset uses sign-magnitude bit11; position bit15. NOT joint angles.",
        "registers": REGISTERS, "motors": motors,
        "safety": "Only READ(0x02) requests. No register writes/torque changes. Snapshot is not a calibration file or automatic restore command.",
    }


def save_snapshot(snapshot, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (datetime.now(timezone.utc).strftime("motor-settings-%Y%m%dT%H%M%S-%fZ.json"))
    with path.open("x", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
        f.write("\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ack-read-only", action="store_true")
    args = parser.parse_args(argv)
    if args.execute != args.ack_read_only:
        parser.error("Require BOTH --execute and --ack-read-only to open the port")
    if not args.execute:
        print(json.dumps({"mode": "PLAN_ONLY_NO_SERIAL_ACCESS", "port": base.PORT, "registers": REGISTERS}, indent=2))
        return 0
    try:
        with closing(base.open_verified_port()) as port:
            snapshot = collect(BackupLink(port))
        path = save_snapshot(snapshot, Path(__file__).resolve().parents[1] / "backups")
        print("Read-only snapshot saved:", path)
        for motor_id, values in snapshot["motors"].items():
            print(json.dumps({"id": motor_id, **values}))
        print("Port closed. No registers or torque state changed by this tool.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; port closed, no corrective writes.")
        return 130
    except Exception as e:
        print(f"STOP: {type(e).__name__}: {e}. No corrective writes.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
