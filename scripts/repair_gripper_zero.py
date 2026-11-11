"""Explicit one-motor zero correction. Default is OFFLINE PLAN, no serial.

User-approved shift +1038 reported counts on follower6; Feetech sign=-1.
Never writes a goal position or torque_enable=1. Never changes motors1..5.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import socket

ROOT=Path(__file__).resolve().parents[1]
REGISTERS=('model_number','id','torque_enable','lock','operating_mode','phase','homing_offset',
           'min_position_limit','max_position_limit','goal_position','present_position','present_voltage')
NAMES=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
SHIFT=1038


def repair(bus,calibration,backup_path,*,sign):
    if sign!=-1:raise ValueError('Unexpected encoder offset convention')
    journal=[]
    def read(motor,name):
        v,e=bus.read_register_with_error(motor,name,retries=0)
        if e:raise RuntimeError(f'ID{motor} {name}: status {e}')
        return int(v)
    def write(name,value):
        if not ((name=='torque_enable'and value==0)or(name=='lock'and value in (0,1))or(name=='homing_offset'and value==-1326)):
            raise ValueError('Outside one-motor repair write allowlist')
        journal.append({'motor':6,'register':name,'value':value,'status':'attempting'})
        log=Path(backup_path).with_suffix('.writes.json')
        log.write_text(json.dumps(journal,indent=2))
        e=bus.write_register_with_error(6,name,value,retries=0)
        journal[-1]['status']=e;log.write_text(json.dumps(journal,indent=2))
        if e:raise RuntimeError(f'ID6 {name} write status {e}; no retry')
    before={str(i):{name:read(i,name) for name in REGISTERS}for i in range(1,7)}
    backup={'created_at':datetime.now(timezone.utc).isoformat(),'port':'/dev/cu.usbserial-120',
            'motors':before,'calibration':calibration,'planned_reported_shift':SHIFT,'homing_offset_sign':sign}
    backup_path=Path(backup_path);backup_path.parent.mkdir(parents=True,exist_ok=True)
    with backup_path.open('x')as f:json.dump(backup,f,indent=2)
    for i,name in enumerate(NAMES,1):
        b=before[str(i)];c=calibration[name]
        if b['model_number']!=777 or b['id']!=i or b['operating_mode']!=0:
            raise RuntimeError('Unexpected device identity/mode; no writes')
        if (b['homing_offset'],b['min_position_limit'],b['max_position_limit'])!=(c['homing_offset'],c['range_min'],c['range_max']):
            raise RuntimeError(f'ID{i} calibration mismatch; no writes')
    six=before['6']
    if six['homing_offset']!=-288 or six['phase']!=12 or six['lock']not in (0,1):
        raise RuntimeError('Unexpected motor6 baseline; no writes')
    if not 110<=six['present_voltage']<=140:raise RuntimeError('Stable 12V supply required; no writes')
    new_offset=six['homing_offset']+sign*SHIFT
    if new_offset!=-1326:raise ValueError('Unexpected proposed offset')
    result={'status':'pending','backup':str(backup_path),'before':before,'writes':journal,
            'reported_shift':SHIFT,'new_homing_offset':new_offset}
    unlocked=False
    try:
        if six['torque_enable']!=0:write('torque_enable',0)
        if read(6,'torque_enable')!=0:raise RuntimeError('Torque-off not confirmed')
        if six['lock']!=0:
            # Mark before write so any uncertain response still leads to re-lock attempt.
            unlocked=True;write('lock',0)
            if read(6,'lock')!=0:raise RuntimeError('EEPROM unlock not confirmed')
        position_before=read(6,'present_position')
        write('homing_offset',new_offset)
        if read(6,'homing_offset')!=new_offset:raise RuntimeError('Offset readback mismatch; no automatic rollback/retry')
        position_after=read(6,'present_position')
        if abs((position_after-position_before)-SHIFT)>10:
            raise RuntimeError('Reported position shift mismatch; no automatic motion/rollback')
        result.update(position_before=position_before,position_after=position_after)
    finally:
        if unlocked:write('lock',six['lock'])
    after={str(i):{name:read(i,name)for name in REGISTERS}for i in range(1,7)}
    for i in range(1,6):
        for key in REGISTERS:
            if key not in ('present_position','present_voltage') and after[str(i)][key]!=before[str(i)][key]:
                raise RuntimeError(f'ID{i} {key} changed unexpectedly')
    for key in REGISTERS:
        if key not in ('homing_offset','torque_enable','present_position','present_voltage') and after['6'][key]!=six[key]:
            raise RuntimeError(f'ID6 {key} changed unexpectedly')
    if after['6']['torque_enable']!=0 or after['6']['homing_offset']!=new_offset:
        raise RuntimeError('Final verification failed')
    result.update(status='verified',after=after)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    if not args.apply:
        print('PLAN ONLY: follower120 motor6, offset -288 -> -1326; torque OFF, no goals, no other motor writes.');return
    with socket.socket()as sock:
        if sock.connect_ex(('127.0.0.1',8000))==0:raise RuntimeError('Stop Studio before applying to avoid concurrent serial access')
    import rustypot
    from serial.tools import list_ports
    ports=[p for p in list_ports.comports()if p.device=='/dev/cu.usbserial-120']
    if len(ports)!=1 or (ports[0].vid,ports[0].pid)!=(6790,29987):raise RuntimeError('USB/path identity mismatch')
    from contextlib import closing
    _,definition=rustypot.find_model('sts3215')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    cal=json.loads((ROOT/'calibration/so101-follower-20261031T144217-014032Z.json').read_text())
    backup=ROOT/'backups'/f'follower6-before-zero-{stamp}.json'
    with closing(rustypot.Bus('/dev/cu.usbserial-120',1000000,.2,{i:definition for i in range(1,7)}))as bus:
        result=repair(bus,cal,backup,sign=definition.homing_offset_sign)
    out=ROOT/'reports'/f'gripper-zero-repair-{stamp}.json';out.write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':result['status'],'before':result['position_before'],'after':result['position_after'],
        'offset':result['new_homing_offset'],'backup':str(backup),'report':str(out),'torque6':0},indent=2))

if __name__=='__main__':main()
