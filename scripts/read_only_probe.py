"""STS3215 diagnostic: plan by default; never writes servo registers.

No robot SDK connect/configure calls. Serial transport does transmit READ request
bytes; that is distinct from a servo register WRITE. Power-up movement and USB
control-line transients are not prevented by this software.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import closing
from typing import Protocol

IDS = tuple(range(1, 7))
REGISTERS = {"model_number": (3, 2), "torque_enable": (40, 1), "present_position": (56, 2)}
EXPECTED_MODEL = 777  # rustypot v1.13.1 STS3215 definition; not a gear-ratio check.
PORT = "/dev/cu.usbserial-130"
USB_ID = (0x1A86, 0x7523)
BAUDRATE = 1_000_000


class ProbeError(RuntimeError):
    pass


class SerialLike(Protocol):
    def write(self, data: bytes) -> int: ...
    def read(self, size: int) -> bytes: ...
    def reset_input_buffer(self) -> None: ...
    def close(self) -> None: ...


def request_packet(motor_id: int, register: str) -> bytes:
    if type(motor_id) is not int or motor_id not in IDS:
        raise ProbeError("Only unicast motor IDs 1..6 are allowed")
    if register not in REGISTERS:
        raise ProbeError("Register is not on the read allowlist")
    address, size = REGISTERS[register]
    body = bytes((motor_id, 4, 0x02, address, size))
    return b"\xff\xff" + body + bytes(((~sum(body)) & 0xFF,))


# Complete allowed byte strings, not just a check of the instruction byte.
ALLOWED_PACKETS = frozenset(request_packet(i, r) for i in IDS for r in REGISTERS)


def validate_request(packet: bytes) -> None:
    if packet not in ALLOWED_PACKETS:
        raise ProbeError("Blocked outgoing packet outside the exact READ allowlist")


class ReadOnlyLink:
    def __init__(self, serial_port: SerialLike):
        self._serial = serial_port

    def _read_exact(self, size: int, deadline: float) -> bytes:
        data = bytearray()
        while len(data) < size:
            if time.monotonic() >= deadline:
                raise ProbeError("Response deadline exceeded; no retries")
            part = self._serial.read(size - len(data))
            if not part:
                raise ProbeError("Missing/truncated response; no retries")
            data.extend(part)
        return bytes(data)

    def read_register(self, motor_id: int, register: str) -> int:
        packet = request_packet(motor_id, register)
        validate_request(packet)  # Gate immediately before the only outgoing write call.
        self._serial.reset_input_buffer()
        if self._serial.write(packet) != len(packet):
            raise ProbeError("Partial request transmission; stopping without retry")
        deadline = time.monotonic() + 0.5
        header = self._read_exact(4, deadline)
        size = REGISTERS[register][1]
        if header[:2] != b"\xff\xff" or header[2] != motor_id or header[3] != size + 2:
            raise ProbeError("Unexpected response header/ID/length; no resynchronization")
        tail = self._read_exact(size + 2, deadline)
        response = header + tail
        if sum(response[2:]) & 0xFF != 0xFF:
            raise ProbeError("Response checksum mismatch")
        if tail[0] != 0:
            raise ProbeError(f"Motor {motor_id}: status/alarm 0x{tail[0]:02x}; stop")
        value = int.from_bytes(tail[1:-1], "little")
        if register == "present_position":
            # STS3215 uses sign-magnitude bit 15, not two's complement.
            value = -(value & 0x7FFF) if value & 0x8000 else value
        return value


def probe(link: ReadOnlyLink, emit=print) -> None:
    # Verify every identity before reading torque/position on any motor.
    for motor_id in IDS:
        model = link.read_register(motor_id, "model_number")
        emit(json.dumps({"id": motor_id, "model_number": model}))
        if model != EXPECTED_MODEL:
            raise ProbeError(f"Motor {motor_id}: expected model {EXPECTED_MODEL}, got {model}; no changes made")
    for motor_id in IDS:
        torque = link.read_register(motor_id, "torque_enable")
        emit(json.dumps({"id": motor_id, "torque_enable": torque}))
        if torque != 0:
            raise ProbeError(f"Motor {motor_id}: torque state is {torque}, not 0; stopping WITHOUT disabling it")
        position = link.read_register(motor_id, "present_position")
        emit(json.dumps({"id": motor_id, "position_raw_signed": position}))


def open_verified_port() -> SerialLike:
    # No import or device enumeration at all in the default plan-only mode.
    import serial
    from serial.tools import list_ports

    matches = [p for p in list_ports.comports() if p.device == PORT]
    if len(matches) != 1 or (matches[0].vid, matches[0].pid) != USB_ID:
        raise ProbeError("Expected USB serial path/VID/PID missing; do not guess another port")
    port = serial.Serial(
        port=None, baudrate=BAUDRATE, bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
        timeout=0.2, write_timeout=0.2, exclusive=True,
        xonxoff=False, rtscts=False, dsrdtr=False,
    )
    try:
        # Set requested inactive states before open. OS/adapter glitches remain possible.
        port.dtr = False
        port.rts = False
        port.port = PORT
        port.open()
        return port
    except BaseException:
        port.close()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually open the hardware port")
    parser.add_argument("--ack-power-risk", action="store_true", help="Acknowledge supported arm, clear workspace, accessible power cutoff, and possible power-up movement")
    args = parser.parse_args(argv)
    if args.execute != args.ack_power_risk:
        parser.error("Hardware access requires BOTH --execute and --ack-power-risk")
    if not args.execute:
        print(json.dumps({
            "mode": "PLAN_ONLY_NO_SERIAL_ACCESS", "port": PORT, "baudrate": BAUDRATE,
            "ids": IDS, "expected_model": EXPECTED_MODEL, "registers": REGISTERS,
            "on_error": "stop; close port only; never change torque or motor configuration",
            "warning": "Power-up motion cannot be prevented by this script. No scan or baudrate search.",
        }, indent=2))
        return 0
    try:
        with closing(open_verified_port()) as serial_port:
            probe(ReadOnlyLink(serial_port))
        print("Read-only probe complete; port closed; torque unchanged by this script.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; no torque-disable or motion command sent.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}. No automatic corrective writes.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
