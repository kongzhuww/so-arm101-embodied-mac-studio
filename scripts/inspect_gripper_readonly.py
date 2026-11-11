"""Default offline plan. Hardware mode only reads six IDs and motor6 settings."""
import argparse
from contextlib import closing
from datetime import datetime,timezone
import json
from pathlib import Path

import rustypot
import read_only_probe as base

REGISTERS=('id','operating_mode','phase','lock','goal_position','goal_time','goal_speed',
           'torque_limit','max_torque_limit','acceleration','maximum_acceleration',
           'velocity_unit_factor','acceleration_multiplier','present_voltage',
           'present_temperature','present_load','present_current','status','moving')


def read(raw,motor_id,name):
    if motor_id not in base.IDS or name not in ('model_number','torque_enable','present_position',*REGISTERS):
        raise base.ProbeError('Outside diagnostic READ allowlist')
    value,status=raw.read_register_with_error(motor_id,name,retries=0)
    if status!=0:
        raise base.ProbeError(f'ID{motor_id} status/alarm {status}; stopping')
    return value


def collect(raw):
    baseline={}
    for i in base.IDS:
        model=read(raw,i,'model_number')
        if model!=777:
            raise base.ProbeError('Unexpected servo model')
        torque=read(raw,i,'torque_enable')
        if torque!=0:
            raise base.ProbeError('Torque is not off; stop without changing it')
        baseline[str(i)]={'model_number':model,'torque_enable':torque,'present_position':read(raw,i,'present_position')}
    settings={name:read(raw,6,name) for name in REGISTERS}
    if settings['id']!=6 or settings['operating_mode']!=0 or settings['phase']&16 or settings['status']!=0:
        raise base.ProbeError('Unexpected ID/mode/phase/status; no automatic correction')
    if read(raw,6,'torque_enable')!=0:
        raise base.ProbeError('Torque changed during reads')
    return {'mode':'READ_ONLY_NO_MOTION','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'baseline':baseline,'motor6_settings_raw':settings,'raw_unit_warning':'Raw register values; do not infer physical speed/current from names alone.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--ack-read-only',action='store_true')
    args=parser.parse_args(argv)
    if args.execute!=args.ack_read_only:
        parser.error('Both hardware flags required')
    if not args.execute:
        print('PLAN ONLY: no hardware access. Read motor6 control settings and six torque/position states.')
        return 0
    from serial.tools import list_ports
    matches=[p for p in list_ports.comports() if p.device==base.PORT]
    if len(matches)!=1 or (matches[0].vid,matches[0].pid)!=base.USB_ID:
        raise base.ProbeError('USB identity/path mismatch')
    _,definition=rustypot.find_model('sts3215')
    with closing(rustypot.Bus(base.PORT,base.BAUDRATE,0.2,{i:definition for i in base.IDS})) as raw:
        report=collect(raw)
    out=Path(__file__).resolve().parents[1]/'backups'/('gripper-settings-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')+'.json')
    with out.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))
    print('Saved:',out,'; port closed; no motor register writes.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
