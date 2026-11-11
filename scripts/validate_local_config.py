"""OFFLINE ONLY. Decode config/calibration and use official numeric conversions.
Does not construct a Robot or Camera, open serial, or send any device commands.
Reports stale-snapshot safety blockers separately from configuration validity.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import draccus
from lerobot.cameras.opencv import OpenCVCameraConfig  # registration only
from lerobot.motors import Motor, MotorCalibration, MotorNormMode, SerialMotorsBus
from lerobot.robots.config import RobotConfig
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

import safe_official_calibrate as safety

ROOT = Path(__file__).resolve().parents[1]


class OfflineBus(SerialMotorsBus):
    def connect(self, *args, **kwargs):
        raise safety.SafetyBlock('Offline normalizer cannot connect')

    def _open_bus(self):
        raise safety.SafetyBlock('Offline normalizer cannot open a port')

    def write(self, *args, **kwargs):
        raise safety.SafetyBlock('Offline normalizer cannot write')

    def sync_write(self, *args, **kwargs):
        raise safety.SafetyBlock('Offline normalizer cannot write')


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'so101-offline-validation-v1' or manifest.get('motion_authorized') is not False:
        raise safety.SafetyBlock('Expected offline-only manifest with motion_authorized=false')
    config = draccus.decode(RobotConfig, manifest['robot'])
    if not isinstance(config, SO101FollowerConfig) or config.use_degrees is not True:
        raise safety.SafetyBlock('Expected SO101 degree-mode config')
    if not config.id or Path(config.id).name != config.id or config.calibration_dir is None:
        raise safety.SafetyBlock('Missing or unsafe calibration path/id')
    calibration_path = config.calibration_dir / f'{config.id}.json'
    snapshot_path = Path(manifest['snapshot'])
    for path,key in ((calibration_path,'calibration_sha256'),(snapshot_path,'snapshot_sha256')):
        if fingerprint(path) != manifest[key]:
            raise safety.SafetyBlock(f'File hash changed: {path.name}; no automatic acceptance')
    with calibration_path.open() as stream, draccus.config_type('json'):
        calibration = draccus.load(dict[str,MotorCalibration], stream)
    safety.validate_candidate(calibration)
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get('schema') != 'so101-raw-register-backup-v1' or snapshot.get('port') != config.port:
        raise safety.SafetyBlock('Snapshot schema/port mismatch')
    if set(snapshot.get('motors',{})) != set(map(str,range(1,7))):
        raise safety.SafetyBlock('Snapshot must contain exactly six motors')
    margin = manifest['proposed_inward_margin_counts']
    if type(margin) is not int or margin <= 0:
        raise safety.SafetyBlock('Margin must be a positive integer')
    motors = {n:Motor(i,'sts3215',MotorNormMode.RANGE_0_100 if n=='gripper' else MotorNormMode.DEGREES)
              for i,n in enumerate(safety.NAMES,1)}
    bus = OfflineBus(config.port,motors,calibration)
    if bus.is_connected:
        raise safety.SafetyBlock('Offline bus unexpectedly connected')
    results = {}
    blockers = []
    for name,c in calibration.items():
        raw = snapshot['motors'][str(c.id)]
        encoded = raw['homing_offset']
        signed = -(encoded & 2047) if encoded & 2048 else encoded
        if (raw['id'],raw['model_number'],signed,raw['min_position_limit'],raw['max_position_limit']) != (c.id,777,c.homing_offset,c.range_min,c.range_max):
            raise safety.SafetyBlock(f'Snapshot/calibration mismatch: {name}')
        if raw['torque_enable'] != 0 or raw['operating_mode'] != 0 or raw['phase'] & 16:
            raise safety.SafetyBlock('Snapshot not the expected torque-off position-mode state')
        encoded_pos=raw['present_position']
        position=-(encoded_pos & 32767) if encoded_pos & 32768 else encoded_pos
        normal = bus._normalize({c.id:position})[c.id]
        back = bus._unnormalize({c.id:normal})[c.id]
        if abs(position-back)>1:
            raise safety.SafetyBlock(f'Numeric round-trip differs by more than one count: {name}')
        # Test conversion at internal points, not by driving any joint.
        for value in (c.range_min+1,(c.range_min+c.range_max)//2,c.range_max-1):
            converted=bus._normalize({c.id:value})
            if abs(bus._unnormalize(converted)[c.id]-value)>1:
                raise safety.SafetyBlock('Interior numeric conversion failed')
        row = {
            'id':c.id,'recorded_raw_range':[c.range_min,c.range_max],
            'snapshot_raw_position':position,'snapshot_normalized_position':normal,
            'units':'percent_0_100' if name=='gripper' else 'degrees',
            'numeric_roundtrip_error_counts':abs(position-back),
            'raw_midpoint_not_a_command':(c.range_min+c.range_max)/2,
        }
        if name=='wrist_roll':
            row['proposed_soft_range']=None
            row['powered_testing_allowed']=False
            blockers.append('wrist_roll: full encoder range is not a cable-safe range; independent manual limit approval required')
        else:
            lo,hi=c.range_min+margin,c.range_max-margin
            if lo>=hi:
                raise safety.SafetyBlock(f'Margin consumes entire range: {name}')
            row['proposed_soft_range']=[lo,hi]
            row['snapshot_inside_proposed_soft_range']=lo<=position<=hi
            if not lo<=position<=hi:
                blockers.append(f'{name}: historical position {position} outside proposed soft range [{lo},{hi}]')
        results[name]=row
    camera=config.cameras.get('external_front')
    if not isinstance(camera,OpenCVCameraConfig) or (camera.width,camera.height,camera.fps)!=(640,480,60):
        raise safety.SafetyBlock('Unexpected camera config; no device opened to auto-negotiate it')
    return {
        'mode':'OFFLINE_ONLY_NO_HARDWARE_ACCESS','configuration_valid':True,
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'calibration_file':str(calibration_path),'robot_id':config.id,
        'calibration_sha256':fingerprint(calibration_path),
        'historical_snapshot':str(snapshot_path),'snapshot_time_utc':snapshot.get('created_at_utc'),
        'snapshot_warning':'Historical readings only; NOT a fresh/live safety check.',
        'camera_config_only':{'name':'external_front','width':640,'height':480,'fps':60,'opened':False},
        'units_warning':'Body degrees use measured range midpoint as zero, NOT necessarily the captured2047 reference or URDF zero. Gripper is percent, not degrees.',
        'degree_clamping_warning':'Official degree conversions do not clamp targets to recorded limits. Future motion code must enforce raw soft limits independently.',
        'margin_counts_proposal':margin,'margin_not_written_or_validated':True,
        'joints':results,'motion_authorized':False,'motion_readiness':'BLOCKED_PENDING_LIVE_AND_PHYSICAL_CHECKS',
        'blockers_from_historical_data':blockers,
        'next_requirements':[
            'No automatic Robot.connect: it configures registers and enables torque.',
            'Human support/clearance/cutoff confirmation and a fresh torque-off read before any physical handling.',
            'Move away from recorded endpoints gently by hand only after state confirmation; never force or auto-position.',
            'Approve cable-safe wrist-roll interval; leave wrist roll excluded from initial powered test.',
            'Before torque-on: review/preload a supported-mode goal matching a fresh measured position; stale goals can cause a jump.',
            'Initial powered test requires separate explicit approval, conservative speed/torque planning and single-joint small step with monitoring.',
        ],
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'config/offline-validation.json')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv)
    report=build_report(args.manifest)
    text=json.dumps(report,indent=2,ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        # Reports are append-only; never overwrite calibration or another output file.
        with args.report.open('x',encoding='utf-8') as stream:
            stream.write(text+'\n')
        print('Offline report:',args.report)
    else:
        print(text)
    print('Configuration loading/numeric conversion: PASS. Motion remains NOT authorized.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
