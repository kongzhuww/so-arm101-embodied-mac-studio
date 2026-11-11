"""GATED MOTOR-6 TEST. Default PLAN ONLY. Not a safety-rated controller.
Only after a human MOVE6: low raw output limit, current-position preload,
briefly enable motor6, request +8 encoder counts, then turn motor6 torque OFF.
No other motor is written. No automatic return, retry, or power escalation.
"""
import argparse
from contextlib import closing
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import rustypot
import finalize_recorded_calibration as calibration_loader
import inspect_gripper_readonly as diagnostic
import read_only_probe as base
import safe_official_calibrate as safety

ROOT=Path(__file__).resolve().parents[1]
CALIBRATION=ROOT/'calibration/so101-follower-20261031T144217-014032Z.json'
CALIBRATION_HASH='708f25225132a84be06fb11ea16ebb74df000612e13502edbcfcd21f5dd053ef'
STEP=8
MARGIN=64
LOW_SETTINGS={'goal_time':0,'goal_speed':50,'acceleration':1,'torque_limit':50}
# Real hardware exposed a false assumption: writing Goal_Position while torque=0
# was followed by torque=1. Do not retry by merely allowing that state through.
HARDWARE_DISABLED_REASON='实机写目标位置后观测到扭矩自动开启，原预装流程假设不成立；硬件执行暂停，禁止重试。'


class Stop(safety.SafetyBlock):
    pass


class Motor6Gate:
    def __init__(self,raw,journal):
        self._raw=raw
        self.journal=journal
        self.armed=False
        self.start=None
        self.target=None
        self.preloaded=False
        self.enable_requested=False
        self.move_requested=False
        self.ready=False

    def read(self,motor_id,name):
        if type(motor_id) is not int or motor_id not in base.IDS:
            raise Stop('Read ID out of range')
        if name not in ('model_number','torque_enable','present_position','homing_offset',
                        'min_position_limit','max_position_limit','firmware_major_version',
                        'firmware_minor_version',*diagnostic.REGISTERS):
            raise Stop('Read outside allowlist')
        value,status=self._raw.read_register_with_error(motor_id,name,retries=0)
        if status!=0:
            raise Stop(f'ID{motor_id} status/alarm {status}')
        return value

    def arm(self,start,lo,hi):
        if type(start) is not int or not lo<=start<start+STEP<=hi:
            raise Stop('Start/target outside inward soft range')
        self.start,self.target=start,start+STEP
        self.armed=True

    def write(self,motor_id,name,value):
        if not self.armed or type(motor_id) is not int or motor_id!=6 or type(value) is not int:
            raise Stop('Writes restricted to armed motor6')
        if name=='torque_enable':
            if value==1:
                if not self.ready or self.enable_requested:
                    raise Stop('Enable requires verified preload/limits, once only')
                self.enable_requested=True  # Never retry an uncertain enable.
            elif value!=0:
                raise Stop('Invalid torque flag')
        elif name in LOW_SETTINGS:
            if value!=LOW_SETTINGS[name] or self.enable_requested or self.read(6,'torque_enable')!=0:
                raise Stop('Only fixed low settings while torque off')
        elif name=='goal_position':
            if not self.enable_requested:
                if value!=self.start or self.preloaded or self.read(6,'torque_enable')!=0:
                    raise Stop('Only one preload matching measured start permitted')
                self.preloaded=True
            else:
                if value!=self.target or self.move_requested or self.read(6,'torque_enable')!=1:
                    raise Stop('Only one +8count goal permitted')
                self.move_requested=True  # Never retry an uncertain motion write.
        else:
            raise Stop('Register forbidden in motor6 test')
        self.journal.emit(event='micro_write_attempt',id=6,register=name,value=value)
        status=self._raw.write_register_with_error(6,name,value,retries=0)
        self.journal.emit(event='micro_write_response',id=6,register=name,status=status)
        if status!=0:
            raise Stop('Write returned alarm/status')

    def verify_prepared(self):
        if not self.preloaded:
            raise Stop('No current-position preload')
        # Read both conditions unconditionally and log actual values, not a generic
        # short-circuit boolean. Any nonzero status still blocks; no alarm clearing.
        readings={name:self.read(6,name) for name in (*LOW_SETTINGS,'goal_position','present_position','torque_enable','status')}
        self.journal.emit(event='micro_prepared_readback',readings=readings)
        for name,value in LOW_SETTINGS.items():
            if readings[name]!=value:
                raise Stop(f'Low setting readback mismatch: {name}={readings[name]} expected={value}')
        if readings['goal_position']!=self.start or abs(readings['present_position']-self.start)>2:
            raise Stop('Position changed during preload; no automatic reposition')
        if readings['torque_enable']!=0 or readings['status']!=0:
            raise Stop(f"Not torque-off/alarm-free before enabling: torque={readings['torque_enable']}, status=0x{readings['status']:02x}")
        self.ready=True


def check_baseline(gate,calibration):
    positions={}
    for name,c in calibration.items():
        i=c.id
        if gate.read(i,'model_number')!=777 or gate.read(i,'torque_enable')!=0:
            raise Stop('All six must be matching models with torque off')
        if gate.read(i,'operating_mode')!=0 or gate.read(i,'phase')&16:
            raise Stop('Unexpected mode/phase')
        if (gate.read(i,'homing_offset'),gate.read(i,'min_position_limit'),gate.read(i,'max_position_limit'))!=(c.homing_offset,c.range_min,c.range_max):
            raise Stop('Calibration does not match motors')
        pos=gate.read(i,'present_position')
        positions[i]=pos
        if name!='wrist_roll' and not c.range_min+MARGIN<=pos<=c.range_max-MARGIN:
            raise Stop(f'ID{i} too close to recorded endpoint; no automatic reposition')
    if (gate.read(6,'firmware_major_version'),gate.read(6,'firmware_minor_version'))!=(3,10):
        raise Stop('Firmware differs from reviewed diagnostic')
    for name,expected in (('velocity_unit_factor',50),('acceleration_multiplier',1),('maximum_acceleration',50)):
        if gate.read(6,name)!=expected:
            raise Stop('Control scaling differs from reviewed diagnostic')
    if gate.read(6,'status')!=0 or not 100<=gate.read(6,'present_voltage')<=130 or gate.read(6,'present_temperature')>45:
        raise Stop('Alarm, supply or starting temperature outside conservative test check')
    return positions


def run_test(gate,calibration,ask=input,clock=time.monotonic,sleep=time.sleep):
    positions=check_baseline(gate,calibration)
    original={name:gate.read(6,name) for name in ('goal_position',*LOW_SETTINGS)}
    gate.journal.emit(event='micro_preflight',positions=positions,original_motor6_settings=original)
    print('【会动】仅6号夹爪舵机：先对齐当前位置，低raw速度50/加速度1/扭矩限制50，再请求+8编码格（舵机轴约0.7度）。')
    print('这不是保证安全的夹持力。夹口必须空着，手远离夹口，其余臂段稳定支撑，随时可断电。')
    print('1..5号不使能、不写入。结束会关闭6号扭矩，不自动回位、不恢复旧目标0、不增加出力。')
    print('raw控制缩放按已核对的固件/参数限定；实际速度和力尚未测定，不能当作认证的物理上限。')
    gate.journal.emit(event='waiting_move6_confirmation',writes_armed=False)
    if ask('确认以上条件，输入 MOVE6 并回车才允许实际出力；其他输入取消: ').strip().upper()!='MOVE6':
        gate.journal.emit(event='micro_cancelled_before_writes')
        return False
    positions=check_baseline(gate,calibration)  # Fresh after user consent.
    start=positions[6]
    for _ in range(2):
        sleep(0.05)
        if abs(gate.read(6,'present_position')-start)>2:
            raise Stop('Motor6 is moving by hand; wait, do not enable')
    c=calibration['gripper']
    gate.arm(start,c.range_min+MARGIN,c.range_max-MARGIN)
    gate.journal.emit(event='micro_motion_authorized',start=start,target=start+STEP)
    reached=False
    try:
        for name,value in LOW_SETTINGS.items():
            gate.write(6,name,value)
        gate.write(6,'goal_position',start)
        # Require three consecutive clean samples. A bad sample aborts immediately;
        # this is NOT polling through/ignoring a transient alarm until it disappears.
        for sample in range(3):
            gate.verify_prepared()
            if sample<2:
                sleep(0.05)
        gate.write(6,'torque_enable',1)
        if gate.read(6,'torque_enable')!=1 or abs(gate.read(6,'present_position')-start)>4:
            raise Stop('Unexpected movement/state immediately after enable')
        gate.write(6,'goal_position',gate.target)
        deadline=clock()+2.0
        settled=0
        while clock()<deadline:
            pos=gate.read(6,'present_position')
            if abs(pos-start)>STEP+4:
                raise Stop('Observed displacement exceeds test envelope')
            if gate.read(6,'status')!=0 or gate.read(6,'present_temperature')>45:
                raise Stop('Alarm/temperature during test')
            if gate.read(6,'torque_enable')!=1 or gate.read(6,'torque_limit')!=50:
                raise Stop('Output restriction changed unexpectedly')
            for i in range(1,6):
                if gate.read(i,'torque_enable')!=0 or abs(gate.read(i,'present_position')-positions[i])>16:
                    raise Stop('Other axis changed state/position; check support')
            gate.journal.emit(event='micro_sample',position=pos,target=gate.target)
            settled=settled+1 if abs(pos-gate.target)<=2 else 0
            if settled>=3:
                reached=True
                break
            sleep(0.05)
        gate.journal.emit(event='micro_result',target_reached=reached)
    finally:
        # Only motor6: no object should be held, and the arm joints were never enabled.
        disabled=False
        for _ in range(2):
            try:
                gate.write(6,'torque_enable',0)
                if gate.read(6,'torque_enable')==0:
                    disabled=True
                    break
            except Exception as exc:
                gate.journal.emit(event='micro_disable_problem',error=type(exc).__name__)
        gate.armed=False
        gate.journal.emit(event='micro_cleanup',motor6_torque_off_verified=disabled)
        if not disabled:
            raise Stop('无法确认6号扭矩关闭：请立即手动切断电源。DO NOT retry motion.')
    return reached


def load_calibration():
    if hashlib.sha256(CALIBRATION.read_bytes()).hexdigest()!=CALIBRATION_HASH:
        raise Stop('Calibration file changed')
    return calibration_loader.load_candidate(CALIBRATION)


def open_raw():
    from serial.tools import list_ports
    matches=[p for p in list_ports.comports() if p.device==base.PORT]
    if len(matches)!=1 or (matches[0].vid,matches[0].pid)!=base.USB_ID:
        raise Stop('USB identity/path mismatch')
    _,definition=rustypot.find_model('sts3215')
    return rustypot.Bus(base.PORT,base.BAUDRATE,0.1,{i:definition for i in base.IDS})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--ack-motor6-motion',action='store_true')
    args=parser.parse_args(argv)
    if args.execute!=args.ack_motor6_motion:
        parser.error('Both hardware/motion flags required')
    if not args.execute:
        print('PLAN ONLY: no hardware access. Only6; +8counts; preload before enable; raw speed50/accel1/torque50; <=2s then torque off.')
        return 0
    if HARDWARE_DISABLED_REASON:
        print('STOP:',HARDWARE_DISABLED_REASON,file=sys.stderr)
        return 2
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print('STOP: foreground human terminal required.',file=sys.stderr)
        return 2
    journal=None
    try:
        calibration=load_calibration()
        directory=ROOT/'motion-tests'
        directory.mkdir(exist_ok=True)
        path=directory/('motor6-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')+'.jsonl')
        journal=safety.Journal(path)
        with closing(open_raw()) as raw:
            done=run_test(Motor6Gate(raw,journal),calibration)
        journal.emit(event='micro_session_closed',target_reached=done)
        print('Target reached.' if done else 'Cancelled or target not reached; no power escalation.')
        print('6号扭矩关闭已确认（如已执行），串口已关闭。Journal:',path)
        print('若执行了测试，低速/低扭矩RAM设置保留，不恢复旧目标0。')
        return 0 if done else 1
    except (Exception,KeyboardInterrupt) as exc:
        if journal:
            journal.emit(event='micro_stopped',error=type(exc).__name__,message=str(exc))
        print('STOP:',type(exc).__name__,str(exc),file=sys.stderr)
        return 130 if isinstance(exc,KeyboardInterrupt) else 1
    finally:
        if journal:
            journal.close()


if __name__=='__main__':
    raise SystemExit(main())
