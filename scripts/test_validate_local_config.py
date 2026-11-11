"""Offline config tests. All hardware constructors forbidden."""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lerobot.motors import Motor, MotorNormMode
import validate_local_config as v


@contextmanager
def fixture():
    template=json.loads((v.ROOT/'config/offline-validation.json').read_text())
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        cal_source=Path(template['robot']['calibration_dir'])/(template['robot']['id']+'.json')
        cal=root/cal_source.name
        cal.write_bytes(cal_source.read_bytes())
        snapshot=root/'snapshot.json'
        snapshot.write_bytes(Path(template['snapshot']).read_bytes())
        data=deepcopy(template)
        data['robot']['calibration_dir']=str(root)
        data['snapshot']=str(snapshot)
        manifest=root/'manifest.json'
        manifest.write_text(json.dumps(data))
        yield manifest,data,cal,snapshot


def dump(path,data):
    path.write_text(json.dumps(data))


class OfflineConfigTests(unittest.TestCase):
    def test_load_official_schema_and_report_blockers(self):
        with fixture() as (manifest,_,_,_):
            report=v.build_report(manifest)
        self.assertTrue(report['configuration_valid'])
        self.assertFalse(report['motion_authorized'])
        self.assertFalse(report['camera_config_only']['opened'])
        self.assertEqual(len(report['joints']),6)
        self.assertFalse(report['joints']['elbow_flex']['snapshot_inside_proposed_soft_range'])
        self.assertFalse(report['joints']['gripper']['snapshot_inside_proposed_soft_range'])
        self.assertIsNone(report['joints']['wrist_roll']['proposed_soft_range'])
        self.assertEqual(len(report['blockers_from_historical_data']),3)

    def test_no_robot_camera_or_serial_construction(self):
        with fixture() as (manifest,_,_,_), patch('rustypot.Bus',side_effect=AssertionError('no bus I/O')), patch('serial.Serial',side_effect=AssertionError('no serial I/O')), patch('lerobot.robots.so_follower.so_follower.SOFollower.__init__',side_effect=AssertionError('no robot object')), patch('cv2.VideoCapture',side_effect=AssertionError('no camera')):
            self.assertTrue(v.build_report(manifest)['configuration_valid'])

    def test_refuses_motion_manifest(self):
        with fixture() as (manifest,data,_,_):
            data['motion_authorized']=True
            dump(manifest,data)
            with self.assertRaises(v.safety.SafetyBlock):
                v.build_report(manifest)

    def test_refuses_hash_change(self):
        with fixture() as (manifest,_,cal,_):
            cal.write_text(cal.read_text()+' ')
            with self.assertRaisesRegex(v.safety.SafetyBlock,'hash changed'):
                v.build_report(manifest)

    def test_refuses_snapshot_torque_on_even_with_valid_hash(self):
        with fixture() as (manifest,data,_,snapshot):
            raw=json.loads(snapshot.read_text())
            raw['motors']['1']['torque_enable']=1
            dump(snapshot,raw)
            data['snapshot_sha256']=v.fingerprint(snapshot)
            dump(manifest,data)
            with self.assertRaises(v.safety.SafetyBlock):
                v.build_report(manifest)

    def test_refuses_snapshot_offset_mismatch(self):
        with fixture() as (manifest,data,_,snapshot):
            raw=json.loads(snapshot.read_text())
            raw['motors']['1']['homing_offset']=0
            dump(snapshot,raw)
            data['snapshot_sha256']=v.fingerprint(snapshot)
            dump(manifest,data)
            with self.assertRaisesRegex(v.safety.SafetyBlock,'mismatch'):
                v.build_report(manifest)

    def test_refuses_missing_or_unsafe_id(self):
        for identifier in ('','../bad'):
            with fixture() as (manifest,data,_,_):
                data['robot']['id']=identifier
                dump(manifest,data)
                with self.assertRaises(v.safety.SafetyBlock):
                    v.build_report(manifest)

    def test_refuses_other_normalization(self):
        with fixture() as (manifest,data,_,_):
            data['robot']['use_degrees']=False
            dump(manifest,data)
            with self.assertRaises(v.safety.SafetyBlock):
                v.build_report(manifest)

    def test_invalid_margin(self):
        for margin in (0,-1,True,3000):
            with fixture() as (manifest,data,_,_):
                data['proposed_inward_margin_counts']=margin
                dump(manifest,data)
                with self.assertRaises(v.safety.SafetyBlock):
                    v.build_report(manifest)

    def test_report_never_overwrites_existing_file(self):
        with fixture() as (manifest,_,cal,_):
            fingerprint=v.fingerprint(cal)
            with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
                v.main(['--manifest',str(manifest),'--report',str(cal)])
            self.assertEqual(v.fingerprint(cal),fingerprint)

    def test_report_new_file_only(self):
        with fixture() as (manifest,_,_,_):
            output=manifest.parent/'report.json'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(v.main(['--manifest',str(manifest),'--report',str(output)]),0)
            self.assertFalse(json.loads(output.read_text())['motion_authorized'])

    def test_offline_bus_cannot_connect_or_write(self):
        bus=v.OfflineBus('/dev/DO_NOT_OPEN',{'j':Motor(1,'sts3215',MotorNormMode.DEGREES)})
        for call in (bus.connect,bus._open_bus,lambda:bus.write('Goal_Position','j',2047),lambda:bus.sync_write('Goal_Position',{})):
            with self.assertRaises(v.safety.SafetyBlock):
                call()
        self.assertFalse(bus.is_connected)


if __name__=='__main__':
    unittest.main()
