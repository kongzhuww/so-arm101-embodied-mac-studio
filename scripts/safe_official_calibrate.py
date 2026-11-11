"""Official manual SO101 calibration through a restricted bus, NOT Robot.connect().

DEFAULT: plan only, no serial access. Execution requires real interactive terminal,
explicit flags, a matching raw backup, and typed consent before register writes.
No motor goal or torque-enable command is permitted. This is not a safety controller.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import rustypot
from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus
import read_only_probe as baseline

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
# Values are raw register encodings as seen by rustypot, not goals or degrees.
WRITE_LIMITS = {
    'torque_enable': (0, 0), 'lock': (0, 0), 'operating_mode': (0, 0),
    'homing_offset': (-2047, 2047), 'min_position_limit': (0, 4095),
    'max_position_limit': (0, 4095),
}
BASELINE_REGISTERS = ('model_number', 'id', 'torque_enable', 'lock', 'operating_mode',
                      'phase', 'homing_offset', 'min_position_limit', 'max_position_limit',
                      'firmware_major_version', 'firmware_minor_version')


class SafetyBlock(RuntimeError):
    pass


def check_write(motor_id, name, value):
    if type(motor_id) is not int or motor_id not in baseline.IDS:
        raise SafetyBlock('Only IDs 1..6 may be addressed')
    if name not in WRITE_LIMITS or type(value) is not int:
        raise SafetyBlock(f'Blocked register/value: {name!r}, {value!r}')
    lo, hi = WRITE_LIMITS[name]
    if not lo <= value <= hi:
        raise SafetyBlock(f'Blocked value for {name}: {value}')


class Journal:
    def __init__(self, path):
        self.file = path.open('x', encoding='utf-8')

    def emit(self, **event):
        self.file.write(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(), **event}) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


class RestrictedTransport:
    """Only exposes read, guarded single-register write, close. No raw bus escape API."""
    def __init__(self, raw, journal):
        self.__raw = raw
        self._journal = journal
        self.armed = False

    def read_register_with_error(self, motor_id, name, retries=0):
        if motor_id not in baseline.IDS or name not in (*BASELINE_REGISTERS, 'present_position'):
            raise SafetyBlock('Read outside calibration allowlist')
        value, status = self.__raw.read_register_with_error(motor_id, name, retries=0)
        if status != 0:
            raise SafetyBlock(f'Motor {motor_id}: status/alarm {status}; stopping')
        return value, status

    def sync_read_register_with_error(self, ids, name, retries=0):
        # Single-motor reads, no retry or broadcast; same interface for official logic.
        return [self.read_register_with_error(i, name) for i in ids]

    def write_register_with_error(self, motor_id, name, value, retries=0):
        check_write(motor_id, name, value)
        if not self.armed:
            raise SafetyBlock('Writes not armed by human confirmation')
        if name not in ('torque_enable', 'lock'):
            torque, _ = self.read_register_with_error(motor_id, 'torque_enable')
            if torque != 0:
                raise SafetyBlock('Torque is no longer off; refusing calibration write')
        self._journal.emit(event='write_attempt', id=motor_id, register=name, value=value)
        status = self.__raw.write_register_with_error(motor_id, name, value, retries=0)
        self._journal.emit(event='write_response', id=motor_id, register=name, status=status)
        if status != 0:
            raise SafetyBlock(f'Calibration write alarm/status {status}')
        return status

    def close(self):
        self.armed = False
        self.__raw.close()  # No automatic torque changes, on success or failure.


class SafeCalibrationBus(SerialMotorsBus):
    def __init__(self, journal):
        motors = {n: Motor(i, 'sts3215', MotorNormMode.RANGE_0_100 if n == 'gripper' else MotorNormMode.DEGREES)
                  for i, n in enumerate(NAMES, 1)}
        super().__init__(baseline.PORT, motors)
        self.journal = journal

    def _open_bus(self):
        from serial.tools import list_ports
        matches = [p for p in list_ports.comports() if p.device == baseline.PORT]
        if len(matches) != 1 or (matches[0].vid, matches[0].pid) != baseline.USB_ID:
            raise SafetyBlock('USB device path/VID/PID changed; refusing to guess')
        raw = rustypot.Bus(baseline.PORT, baseline.BAUDRATE, 0.2, self._definitions)
        return RestrictedTransport(raw, self.journal)

    def _set_torque(self, motors, enabled, num_retry):
        if enabled:
            raise SafetyBlock('Torque enabling is forbidden in this entrypoint')
        # Avoid rustypot.set_torque bypassing the single-write gate.
        for motor in self._get_motors_list(motors):
            self.write('Torque_Enable', motor, 0, normalize=False)
            self.write('Lock', motor, 0, normalize=False)

    def write(self, data_name, motor, value, *, normalize=True, num_retry=0):
        check_write(self._get_motor_id(motor), data_name.lower(), value)
        # Official calibration uses only these raw-valued settings, not goals.
        return super().write(data_name, motor, value, normalize=False, num_retry=0)

    def sync_write(self, *args, **kwargs):
        raise SafetyBlock('All sync writes (including goal positions) are forbidden')

    def configure_motors(self, *args, **kwargs):
        raise SafetyBlock('General motor configuration is forbidden')

    def disconnect(self, disable_torque=False):
        if disable_torque:
            raise SafetyBlock('Cleanup may only close the port, not change torque')
        return super().disconnect(disable_torque=False)


def load_backup(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != 'so101-raw-register-backup-v1' or data.get('port') != baseline.PORT:
        raise SafetyBlock('Unexpected backup schema/port')
    if set(data.get('motors', {})) != set(map(str, baseline.IDS)):
        raise SafetyBlock('Backup must cover exactly IDs 1..6')
    for i in baseline.IDS:
        v = data['motors'][str(i)]
        if v.get('model_number') != 777 or v.get('id') != i or v.get('torque_enable') != 0 or v.get('operating_mode') != 0:
            raise SafetyBlock('Backup is not the expected torque-off STS3215 position-mode baseline')
        if v.get('phase', 16) & 16:
            raise SafetyBlock('Phase bit4 must already be clear; do not auto-change it')
        if any(type(v.get(r)) is not int for r in BASELINE_REGISTERS):
            raise SafetyBlock('Backup register data missing or malformed')
    return data


def verify_baseline(bus, backup):
    for motor_id, motor in enumerate(NAMES, 1):
        expected = backup['motors'][str(motor_id)]
        for register in BASELINE_REGISTERS:
            actual = bus.read(register, motor, normalize=False)
            # rustypot returns decoded signed-magnitude offsets; backup stores raw u16.
            want = expected[register]
            if register == 'homing_offset':
                want = -(want & 2047) if want & 2048 else want
            if actual != want:
                raise SafetyBlock(f'Baseline changed at ID{motor_id} {register}: {want} -> {actual}; no auto-restore')


def validate_candidate(calibration):
    if set(calibration) != set(NAMES):
        raise SafetyBlock('Calibration joint set mismatch')
    for i, motor in enumerate(NAMES, 1):
        c = calibration[motor]
        if c.id != i or c.drive_mode != 0 or not -2047 <= c.homing_offset <= 2047:
            raise SafetyBlock('Unexpected calibration ID/direction/offset')
        if not 0 <= c.range_min < c.range_max <= 4095:
            raise SafetyBlock(f'Invalid recorded range for {motor}')
        if not c.range_min <= 2047 <= c.range_max:
            raise SafetyBlock(f'Reference midpoint outside recorded range for {motor}')


def run_session(bus, backup, journal, output, ask=input):
    try:
        bus.connect()  # MOTOR BUS only: handshake reads, no Robot.connect/configure.
        verify_baseline(bus, backup)
        journal.emit(event='waiting_calibration_authorization', writes_armed=False)
        print('\n【先停在这里】接下来会写校准参数，不会发送目标位置或使能扭矩。')
        print('请等参考姿态和人工操作说明确认后，再输入 CALIBRATE。')
        print('\nWARNING: next stage WRITES torque-off, lock, position mode, homing offsets and limits.')
        print('Support arm; no fingers between joints. Manual calibration, never powered positioning.')
        print('Do not force joints into stops or twist wrist cables. Wrist roll is not manually swept a full turn.')
        print('Reference diagram: https://huggingface.co/docs/lerobot/main/en/so101#calibrate')
        print('Official manual prompts follow. Press Ctrl+C to abort; partial writes are NOT auto-restored.')
        if ask('Type CALIBRATE to authorize these calibration writes, or anything else to cancel: ').strip().upper() != 'CALIBRATE':
            journal.emit(event='cancel_before_writes')
            return False
        bus._connected_bus.armed = True
        journal.emit(event='calibration_authorized')
        calibration = bus.record_calibration('position', full_turn_motors=['wrist_roll'])
        validate_candidate(calibration)
        candidate_path = output.with_suffix('.candidate.json')
        with candidate_path.open('x', encoding='utf-8') as f:
            json.dump({n: asdict(c) for n, c in calibration.items()}, f, indent=2)
            f.write('\n')
        journal.emit(event='candidate_checkpoint', path=str(candidate_path))
        print(json.dumps({n: asdict(c) for n, c in calibration.items()}, indent=2))
        print('Candidate checkpoint saved:', candidate_path)
        print('Review recorded ranges. These are not independently verified mechanical limits.')
        if ask('Type SAVE to write final limits and save JSON (case-insensitive); anything else aborts (earlier writes remain): ').strip().upper() != 'SAVE':
            journal.emit(event='cancel_after_reference_writes')
            return False
        bus.write_calibration(calibration)
        actual = bus.read_calibration()
        if actual != calibration:
            raise SafetyBlock('Final calibration readback differs; not saving as success')
        for motor in NAMES:
            if bus.read('Torque_Enable', motor, normalize=False) != 0:
                raise SafetyBlock('Final torque check is not zero')
        with output.open('x', encoding='utf-8') as f:
            json.dump({n: asdict(c) for n, c in calibration.items()}, f, indent=2)
            f.write('\n')
        journal.emit(event='saved', path=str(output))
        return True
    finally:
        bus.disconnect(disable_torque=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--ack-calibration-writes', action='store_true')
    parser.add_argument('--backup', type=Path)
    args = parser.parse_args(argv)
    if args.execute != args.ack_calibration_writes:
        parser.error('Execution requires BOTH --execute and --ack-calibration-writes')
    if not args.execute:
        print('PLAN ONLY: no serial access. Calls official bus.record_calibration(), not Robot.connect().')
        print('Manual TTY interaction required. Allowed writes:', WRITE_LIMITS)
        print('No goal, torque-on, ID, baudrate, PID or acceleration writes. See CALIBRATION-PLAN.md.')
        return 0
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print('STOP: real foreground interactive terminal required; no piped ENTER inputs.', file=sys.stderr)
        return 2
    if args.backup is None:
        parser.error('An explicit --backup path is required')
    journal = None
    try:
        backup = load_backup(args.backup)
        directory = ROOT / 'calibration'
        directory.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
        journal = Journal(directory / f'calibration-{stamp}.jsonl')
        output = directory / f'so101-follower-{stamp}.json'
        result = run_session(SafeCalibrationBus(journal), backup, journal, output)
        print('Calibration saved:' if result else 'Cancelled; inspect journal for any partial writes.', output if result else '')
        return 0 if result else 1
    except (Exception, KeyboardInterrupt) as exc:
        if journal is not None:
            journal.emit(event='stopped', error=type(exc).__name__, message=str(exc))
        print(f'STOP: {type(exc).__name__}: {exc}. No torque-on or automatic rollback.', file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        if journal is not None:
            journal.close()


if __name__ == '__main__':
    raise SystemExit(main())
