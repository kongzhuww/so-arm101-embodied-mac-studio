"""Motor1 base only, +16 raw counts once (~1.4deg), then torque OFF.

PLAN ONLY by default. Current explicit consent required before --execute
--ack-base-motion. Low RAM settings only; no EEPROM/ID/offset/limit changes,
no passive goal preload, explicit torque-on, retry of motion, return or ramp-up.
Other five motors are read only. Not a safety-rated controller.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import time

import gripper_microstep as m

STEP = 16
SETTINGS = {'torque_limit': 50, 'acceleration': 1, 'goal_speed': 50, 'goal_time': 0}


def perform(raw, calibration, journal, clock=time.monotonic, sleep=time.sleep):
    gate = m.Motor6Gate(raw, journal)  # Only its read allowlist is used.
    baseline = m.check_baseline(gate, calibration)
    start = baseline[1]
    c = calibration['shoulder_pan']
    target = start+STEP
    if not c.range_min+64 <= start < target <= c.range_max-64:
        raise m.Stop('Base target outside inward recorded interval')
    for name, value in [('firmware_major_version', 3), ('firmware_minor_version', 10),
                        ('velocity_unit_factor', 50), ('acceleration_multiplier', 1),
                        ('maximum_acceleration', 50)]:
        if gate.read(1, name) != value:
            raise m.Stop('Base firmware/scaling differs from reviewed configuration')
    if gate.read(1, 'status') or not 100 <= gate.read(1, 'present_voltage') <= 130 or gate.read(1, 'present_temperature') > 45:
        raise m.Stop('Base alarm, voltage or temperature outside test range')

    def stable_off():
        for i in range(1, 7):
            if gate.read(i, 'torque_enable') != 0 or abs(gate.read(i, 'present_position')-baseline[i]) > 2:
                raise m.Stop('Not stationary and torque-off before actuation')

    def write(name, value):
        if not ((name in SETTINGS and value == SETTINGS[name]) or
                (name == 'goal_position' and value == target) or
                (name == 'torque_enable' and value == 0)):
            raise m.Stop('Forbidden motor1 test write')
        journal.emit(event='write_attempt', id=1, register=name, value=value)
        status = raw.write_register_with_error(1, name, value, retries=0)
        journal.emit(event='write_response', id=1, register=name, status=status)
        if status:
            raise m.Stop('Base write alarm; no retry')

    stable_off()
    sleep(.1)
    stable_off()
    reached = False
    try:
        for name, value in SETTINGS.items():
            write(name, value)
            stable_off()
            if gate.read(1, name) != value:
                raise m.Stop('Low setting readback mismatch')
        for name, value in SETTINGS.items():
            if gate.read(1, name) != value:
                raise m.Stop('Low settings changed before actuation')
        print(f'Motor1 only: {start} -> {target}; first goal IS actuation', flush=True)
        write('goal_position', target)  # May implicitly enable torque.
        settled = 0
        deadline = clock()+2
        while clock() < deadline:
            pos = gate.read(1, 'present_position')
            torque = gate.read(1, 'torque_enable')
            alarm = gate.read(1, 'status')
            temp = gate.read(1, 'present_temperature')
            limit = gate.read(1, 'torque_limit')
            journal.emit(event='sample', position=pos, torque=torque,
                         status=alarm, temperature=temp, torque_limit=limit)
            if alarm or abs(pos-start) > STEP+4 or temp > 45 or not 0 <= limit <= 50 or torque not in (0, 1):
                raise m.Stop('Unexpected base state; stop')
            for i in range(2, 7):
                if gate.read(i, 'torque_enable') != 0 or abs(gate.read(i, 'present_position')-baseline[i]) > 16:
                    raise m.Stop('Other-axis state changed; stop')
            settled = settled+1 if abs(pos-target) <= 2 else 0
            if settled >= 3:
                reached = True
                break
            sleep(.05)
        journal.emit(event='result', start=start, target=target, reached=reached,
                     position=gate.read(1, 'present_position'))
    finally:
        off = False
        for _ in range(2):
            try:
                write('torque_enable', 0)
                if gate.read(1, 'torque_enable') == 0:
                    off = True
                    break
            except Exception as exc:
                journal.emit(event='disable_problem', error=type(exc).__name__)
        journal.emit(event='cleanup', motor1_torque_off_verified=off)
        if not off:
            raise m.Stop('无法确认1号扭矩关闭，请立即切断电源。')
    return reached


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--ack-base-motion', action='store_true')
    args = parser.parse_args(argv)
    if args.execute != args.ack_base_motion:
        parser.error('Both flags require current base-motion consent')
    if not args.execute:
        print('PLAN ONLY: motor1 +16 counts once, low RAM settings, <=2s then torque off. No serial opened.')
        return 0
    path = m.ROOT/'motion-tests'/('base1-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')+'.jsonl')
    journal = m.safety.Journal(path)
    try:
        with closing(m.open_raw()) as raw:
            done = perform(raw, m.load_calibration(), journal)
        journal.emit(event='port_closed', reached=done)
        print('Reached:', done, '; base torque OFF verified; port closed. Log:', path)
        return 0 if done else 1
    finally:
        journal.close()


if __name__ == '__main__':
    raise SystemExit(main())
