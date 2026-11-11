"""One phone-triggered motor6 step, not whole-arm teleoperation.

Default plan only. --execute --ack-motor6-motion requires current human consent.
Waits for fresh released A/B, then ONE new A or B press within 60 seconds.
A requests +32 raw counts; B requests -32. Physical opening direction must be
observed, not assumed. Never repeats, even if held, and never auto-returns.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import motor6_single_step as direct

STATUS = direct.m.ROOT / 'reports/phone-input-status.json'


class ButtonEdge:
    def __init__(self, armed_at):
        self.armed_at = armed_at
        self.released = False
        self.consumed = False

    def update(self, sample, now):
        if self.consumed:
            return None
        timestamp = sample.get('last_pose_at')
        if type(timestamp) not in (int, float) or not 0 <= now-timestamp <= 1.0:
            raise direct.m.Stop('No fresh phone pose; no movement')
        if timestamp < self.armed_at:
            return None
        a, b = sample.get('button_a'), sample.get('button_b')
        if type(a) is not bool or type(b) is not bool:
            raise direct.m.Stop('Phone A/B telemetry missing; no movement')
        if not a and not b:
            self.released = True
            return None
        if a and b:
            # A fresh release is required again after ambiguous input.
            self.released = False
            return None
        if self.released:
            self.consumed = True
            return 32 if a else -32
        return None


def wait_for_press(path=STATUS, clock=time.time, sleep=time.sleep):
    start = clock()
    edge = ButtonEdge(start)
    deadline = time.monotonic() + 60
    print('等待手机：先松开A/B，再按其中一个；只执行一次32格小步。', flush=True)
    while time.monotonic() < deadline:
        sample = json.loads(Path(path).read_text())
        step = edge.update(sample, clock())
        if step is not None:
            return step
        sleep(0.05)
    raise direct.m.Stop('No fresh new button press within 60s; stopped')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--ack-motor6-motion', action='store_true')
    args = parser.parse_args(argv)
    if args.execute != args.ack_motor6_motion:
        parser.error('Both hardware and current motion-consent flags required')
    if not args.execute:
        print('PLAN ONLY. No port opened. Phone A/B -> motor6 +/-32 counts once, '
              'existing low settings only, <=2s, then torque off. IDs1..5 not written.')
        return 0
    # No serial opening or writes while waiting for a new phone input.
    step = wait_for_press()
    # Refuse delayed actuation if the final input went stale before connection.
    sample = json.loads(STATUS.read_text())
    if not 0 <= time.time()-sample['last_pose_at'] <= 1.0:
        raise direct.m.Stop('Trigger became stale; no movement')
    path = direct.m.ROOT / 'motion-tests' / (
        'phone6-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ') + '.jsonl')
    journal = direct.m.safety.Journal(path)
    try:
        journal.emit(event='phone_single_step', step=step, input_at=sample['last_pose_at'])
        with closing(direct.m.open_raw()) as raw:
            reached = direct.perform(raw, direct.m.load_calibration(), journal, step=step)
        journal.emit(event='port_closed', reached=reached)
        print('Reached:', reached, '; torque OFF verified; port closed. Log:', path)
        return 0 if reached else 1
    finally:
        journal.close()


if __name__ == '__main__':
    raise SystemExit(main())
