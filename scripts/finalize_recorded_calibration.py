"""Resume ONLY the final save after manual calibration; no reference reset/sweep.
Default is offline validation only. Execution requires TTY, flags, matching backup,
and a human SAVE. Uses the restricted bus and official write_calibration/readback.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unicodedata

from lerobot.motors import MotorCalibration
import safe_official_calibrate as s


def load_candidate(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if set(data) != set(s.NAMES):
        raise s.SafetyBlock('Candidate joint names mismatch')
    fields = {'id','drive_mode','homing_offset','range_min','range_max'}
    for values in data.values():
        if set(values) != fields or any(type(v) is not int for v in values.values()):
            raise s.SafetyBlock('Invalid candidate fields/types')
    candidate = {n: MotorCalibration(**v) for n,v in data.items()}
    s.validate_candidate(candidate)
    return candidate


def validate_resume_backup(backup, candidate):
    for motor_id,motor in enumerate(s.NAMES,1):
        v = backup['motors'][str(motor_id)]
        encoded = v['homing_offset']
        signed = -(encoded & 2047) if encoded & 2048 else encoded
        if signed != candidate[motor].homing_offset:
            raise s.SafetyBlock('Candidate offset does not match post-reference backup')
        if v['lock'] != 0 or v['min_position_limit'] != 0 or v['max_position_limit'] != 4095:
            raise s.SafetyBlock('Not the expected cancelled-before-final-save state; no automatic repair')
        pos = v.get('present_position')
        if type(pos) is not int or not candidate[motor].range_min <= pos <= candidate[motor].range_max:
            raise s.SafetyBlock('Backup position outside candidate range')


def confirm_save(ask, journal):
    """Never treat a typo/empty input as consent OR silently cancel on it.

    NFKC accepts fullwidth letters; invisible formatting/control characters remain
    invalid. Log only input shape, never the raw text (which could be private).
    """
    while True:
        raw = ask('输入 SAVE 或 保存 并回车；取消请输入 CANCEL（输错会重新询问）: ')
        token = unicodedata.normalize('NFKC', raw).strip().casefold()
        if token in ('save', '保存'):
            journal.emit(event='save_confirmation_recognized')
            return True
        if token in ('cancel', '取消'):
            journal.emit(event='cancel_confirmation_recognized')
            return False
        journal.emit(event='unrecognized_confirmation', length=len(raw),
                     has_control_or_format=any(unicodedata.category(c) in ('Cc','Cf') for c in raw),
                     has_non_ascii=any(ord(c)>127 for c in raw))
        print('未识别确认词，尚未写入。请手动键入 SAVE（不要粘贴整行提示），或 CANCEL 取消。')


def verify_positions(bus, candidate):
    for name,c in candidate.items():
        pos = bus.read('Present_Position',name,normalize=False)
        if not c.range_min <= pos <= c.range_max:
            raise s.SafetyBlock(f'{name} position outside recorded range; no automatic reposition')


def run_finalize(bus, backup, candidate, journal, output, ask=input):
    s.validate_candidate(candidate)
    validate_resume_backup(backup,candidate)
    try:
        bus.connect()  # MOTOR BUS only, reads/handshake; never Robot.connect.
        s.verify_baseline(bus,backup)
        # Recheck all actual positions before writing limits. Do not request motion.
        verify_positions(bus,candidate)
        print('已核对：参考偏移和备份匹配，扭矩关闭。不会重新采集或摆位。')
        print(json.dumps({n:asdict(c) for n,c in candidate.items()},indent=2))
        journal.emit(event='waiting_resume_save',writes_armed=False)
        if not confirm_save(ask,journal):
            journal.emit(event='resume_cancelled_before_writes')
            return False
        # Consent may take time: recheck state before arming the write gate.
        s.verify_baseline(bus,backup)
        verify_positions(bus,candidate)
        bus._connected_bus.armed=True
        journal.emit(event='resume_save_authorized')
        bus.write_calibration(candidate)  # Same offsets, recorded limits; no goal/torque writes.
        if bus.read_calibration() != candidate:
            raise s.SafetyBlock('Final readback mismatch; not saving as success')
        for name in s.NAMES:
            if bus.read('Torque_Enable',name,normalize=False) != 0:
                raise s.SafetyBlock('Final torque state is not zero')
        with output.open('x',encoding='utf-8') as f:
            json.dump({n:asdict(c) for n,c in candidate.items()},f,indent=2)
            f.write('\n')
        journal.emit(event='saved',path=str(output))
        return True
    finally:
        bus.disconnect(disable_torque=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--backup',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--ack-final-save-writes',action='store_true')
    args=parser.parse_args(argv)
    if args.execute != args.ack_final_save_writes:
        parser.error('Both hardware flags required')
    if args.execute and (not sys.stdin.isatty() or not sys.stdout.isatty()):
        print('STOP: foreground interactive terminal required.',file=sys.stderr)
        return 2
    journal=None
    try:
        candidate=load_candidate(args.candidate)
        backup=s.load_backup(args.backup)
        validate_resume_backup(backup,candidate)
        if not args.execute:
            print('PLAN ONLY: candidate/backup valid; no serial access, no reference reset or motion.')
            return 0
        directory=s.ROOT/'calibration'
        directory.mkdir(exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
        journal=s.Journal(directory/f'finalize-{stamp}.jsonl')
        journal.emit(event='resume_inputs',candidate=str(args.candidate.resolve()),backup=str(args.backup.resolve()))
        output=directory/f'so101-follower-{stamp}.json'
        done=run_finalize(s.SafeCalibrationBus(journal),backup,candidate,journal,output)
        print('Calibration saved:' if done else 'Cancelled; no final save.', output if done else '')
        return 0 if done else 1
    except (Exception,KeyboardInterrupt) as exc:
        if journal is not None:
            journal.emit(event='stopped',error=type(exc).__name__,message=str(exc))
        print(f'STOP: {type(exc).__name__}: {exc}. No torque-on or automatic rollback.',file=sys.stderr)
        return 130 if isinstance(exc,KeyboardInterrupt) else 1
    finally:
        if journal is not None:
            journal.close()


if __name__=='__main__':
    raise SystemExit(main())
