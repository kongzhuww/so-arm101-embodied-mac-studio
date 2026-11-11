"""Only the already-authorized motor6 +8-count step. Goal write IS actuation.
No passive preload assumption, no explicit torque-on, no other-ID writes.
Default plan only; --execute means explicitly authorized first-goal actuation.
"""
import argparse
from contextlib import closing
from datetime import datetime,timezone
import time

import gripper_microstep as m

# Explicit operator authorization: one motor6-only approximately3degree step.
# No wider arm motion or power escalation is authorized.
HARDWARE_PAUSED_REASON=None
STEP=32


def perform(raw,calibration,journal,clock=time.monotonic,sleep=time.sleep,step=STEP):
    if type(step) is not int or not 0<abs(step)<=32:
        raise m.Stop('Single step must be a nonzero integer within +/-32 counts')
    gate=m.Motor6Gate(raw,journal)  # READ methods only; old gate is not armed.
    baseline=m.check_baseline(gate,calibration)
    for name,want in m.LOW_SETTINGS.items():
        if gate.read(6,name)!=want:
            raise m.Stop('Existing low settings differ; no automatic setting changes')
    start=baseline[6]
    for _ in range(3):
        if abs(gate.read(6,'present_position')-start)>2 or gate.read(6,'torque_enable')!=0 or gate.read(6,'status')!=0:
            raise m.Stop('Not stable and torque-off before first actuation')
        sleep(0.05)
    start=gate.read(6,'present_position')
    target=start+step
    c=calibration['gripper']
    if not c.range_min+64<=min(start,target)<=max(start,target)<=c.range_max-64:
        raise m.Stop('Outside inward gripper limits')
    journal.emit(event='first_goal_is_actuation',id=6,start=start,target=target,step=step)
    print('Motor6 only:',start,'->',target,flush=True)
    reached=False
    try:
        # Exactly one goal write; this can implicitly enable torque on this setup.
        journal.emit(event='write_attempt',id=6,register='goal_position',value=target)
        status=raw.write_register_with_error(6,'goal_position',target,retries=0)
        journal.emit(event='write_response',id=6,register='goal_position',status=status)
        if status!=0:
            raise m.Stop('Goal write alarm; do not retry')
        deadline=clock()+2.0
        settled=0
        while clock()<deadline:
            pos=gate.read(6,'present_position')
            torque=gate.read(6,'torque_enable')
            alarm=gate.read(6,'status')
            temperature=gate.read(6,'present_temperature')
            torque_limit=gate.read(6,'torque_limit')
            journal.emit(event='sample',position=pos,torque=torque,status=alarm,temperature=temperature,torque_limit=torque_limit)
            if alarm or abs(pos-start)>abs(step)+4 or torque not in (0,1) or temperature>45 or not 0<=torque_limit<=50:
                raise m.Stop(f'Unexpected state: position={pos},start={start},torque={torque},status={alarm},temperature={temperature},torque_limit={torque_limit}')
            for i in range(1,6):
                if gate.read(i,'torque_enable')!=0 or abs(gate.read(i,'present_position')-baseline[i])>16:
                    raise m.Stop('Other-axis state changed; stopping')
            settled=settled+1 if abs(pos-target)<=2 else 0
            if settled>=3:
                reached=True
                break
            sleep(0.05)
        journal.emit(event='result',reached=reached,start=start,target=target,position=gate.read(6,'present_position'))
    finally:
        off=False
        for _ in range(2):
            try:
                journal.emit(event='write_attempt',id=6,register='torque_enable',value=0)
                status=raw.write_register_with_error(6,'torque_enable',0,retries=0)
                if status==0 and gate.read(6,'torque_enable')==0:
                    off=True
                    break
            except Exception as exc:
                journal.emit(event='disable_problem',error=type(exc).__name__)
        journal.emit(event='cleanup',motor6_torque_off_verified=off)
        if not off:
            raise m.Stop('无法确认关闭：请立即手动切断电源。')
    return reached


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    if not a.execute:
        print('PLAN ONLY; no hardware. A goal write is treated as actuation: motor6 only,+8counts,<=2s then torque0.')
        return 0
    if HARDWARE_PAUSED_REASON:
        print('STOP:',HARDWARE_PAUSED_REASON)
        return 2
    path=m.ROOT/'motion-tests'/('direct6-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')+'.jsonl')
    journal=m.safety.Journal(path)
    try:
        with closing(m.open_raw()) as raw:
            done=perform(raw,m.load_calibration(),journal)
        journal.emit(event='port_closed',reached=done)
        print('Reached:',done,'; motor6 torque OFF verified; port closed. Log:',path)
        return 0 if done else 1
    finally:
        journal.close()


if __name__=='__main__':
    raise SystemExit(main())
