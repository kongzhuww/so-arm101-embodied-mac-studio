"""Offline-only final-save recovery tests. No physical port opened."""
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lerobot.motors import MotorCalibration
import finalize_recorded_calibration as f
import safe_official_calibrate as s
from test_safe_official_calibrate import make_bus, backup_data, Log as safe_log


def fixture():
    bus,raw,proxy,log=make_bus()
    backup=backup_data()
    for i,v in backup['motors'].items():
        v['lock']=0
        v['present_position']=2047
        raw.values[int(i)]['lock']=0
    candidate={n:MotorCalibration(id=i,drive_mode=0,homing_offset=85,range_min=1000,range_max=3000)
               for i,n in enumerate(s.NAMES,1)}
    candidate['wrist_roll'].range_min=0
    candidate['wrist_roll'].range_max=4095
    return bus,raw,proxy,log,backup,candidate


class FinalizeTests(unittest.TestCase):
    def test_confirmation_accepts_case_spaces_chinese_and_fullwidth(self):
        for token in ('save  ', ' SAVE ', 'Save', '保存', ' ｓａｖｅ '):
            log=safe_log()
            self.assertTrue(f.confirm_save(lambda _:token,log))
            self.assertEqual(log.events[-1]['event'],'save_confirmation_recognized')

    def test_bad_or_invisible_input_reprompts_without_logging_content(self):
        log=safe_log()
        answers=iter(['', 'save\u200b', '\x1b[200~save\x1b[201~', 'save now', 'save'])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(f.confirm_save(lambda _:next(answers),log))
        bad=[e for e in log.events if e['event']=='unrecognized_confirmation']
        self.assertEqual(len(bad),4)
        self.assertTrue(any(e['has_control_or_format'] for e in bad))
        self.assertTrue(all(set(e)=={'event','length','has_control_or_format','has_non_ascii'} for e in bad))

    def test_only_explicit_cancel_cancels(self):
        for token in ('cancel',' CANCEL ','取消'):
            log=safe_log()
            answers=iter(['',token])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(f.confirm_save(lambda _:next(answers),log))

    def test_invalid_input_never_arms_before_save(self):
        bus,raw,proxy,log,backup,candidate=fixture()
        answers=iter(['save\u200b','save  '])
        def ask(_):
            self.assertEqual(raw.writes,[])
            self.assertFalse(proxy.armed)
            return next(answers)
        with tempfile.TemporaryDirectory() as d, patch.object(bus,'_open_bus',return_value=proxy), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(f.run_finalize(bus,backup,candidate,log,Path(d)/'out.json',ask=ask))
        self.assertTrue(raw.closed)
        self.assertEqual(len(raw.writes),18)

    def test_position_change_during_confirmation_blocks_writes(self):
        bus,raw,proxy,log,backup,candidate=fixture()
        original=raw.read_register_with_error
        confirmed=[False]
        def read(motor_id,name,retries=0):
            if confirmed[0] and name=='present_position':
                return 3500,0
            return original(motor_id,name,retries)
        def ask(_):
            confirmed[0]=True
            return 'SAVE'
        with tempfile.TemporaryDirectory() as d, patch.object(bus,'_open_bus',return_value=proxy), patch.object(raw,'read_register_with_error',side_effect=read), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(s.SafetyBlock):
                f.run_finalize(bus,backup,candidate,log,Path(d)/'out.json',ask=ask)
        self.assertEqual(raw.writes,[])
        self.assertTrue(raw.closed)

    def test_plan_no_hardware(self):
        _,_,_,_,backup,candidate=fixture()
        with tempfile.TemporaryDirectory() as d:
            c,b=Path(d)/'candidate.json',Path(d)/'backup.json'
            c.write_text(json.dumps({n:asdict(v) for n,v in candidate.items()}))
            b.write_text(json.dumps(backup))
            with patch.object(s,'SafeCalibrationBus',side_effect=AssertionError('no hardware')), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(f.main(['--candidate',str(c),'--backup',str(b)]),0)

    def test_noninteractive_refused_before_files_or_hardware(self):
        with patch.object(f.sys.stdin,'isatty',return_value=False), patch.object(s,'SafeCalibrationBus') as op, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(f.main(['--candidate','nonexistent','--backup','nonexistent','--execute','--ack-final-save-writes']),2)
            op.assert_not_called()

    def test_offset_mismatch_blocks_resume(self):
        _,_,_,_,backup,candidate=fixture()
        candidate['shoulder_pan'].homing_offset=86
        with self.assertRaises(s.SafetyBlock):
            f.validate_resume_backup(backup,candidate)

    def test_negative_encoded_offset(self):
        _,_,_,_,backup,candidate=fixture()
        backup['motors']['1']['homing_offset']=2048+138
        candidate['shoulder_pan'].homing_offset=-138
        f.validate_resume_backup(backup,candidate)

    def test_wrong_cancelled_state_or_position(self):
        for reg,value in [('lock',1),('min_position_limit',10),('max_position_limit',4000),('present_position',3500)]:
            _,_,_,_,backup,candidate=fixture()
            backup['motors']['1'][reg]=value
            with self.assertRaises(s.SafetyBlock):
                f.validate_resume_backup(backup,candidate)

    def test_lowercase_save_succeeds_without_reference_reset(self):
        bus,raw,proxy,log,backup,candidate=fixture()
        with tempfile.TemporaryDirectory() as d, patch.object(bus,'_open_bus',return_value=proxy), patch.object(bus,'record_calibration',side_effect=AssertionError('must not recalibrate')), contextlib.redirect_stdout(io.StringIO()):
            out=Path(d)/'final.json'
            self.assertTrue(f.run_finalize(bus,backup,candidate,log,out,ask=lambda _:' save '))
            self.assertEqual(json.loads(out.read_text())['shoulder_pan']['range_min'],1000)
        self.assertTrue(raw.closed)
        self.assertEqual(len(raw.writes),18)
        self.assertTrue(all(name in ('homing_offset','min_position_limit','max_position_limit') for _,name,_ in raw.writes))
        self.assertTrue(all(raw.values[i]['torque_enable']==0 for i in range(1,7)))

    def test_cancel_no_writes(self):
        bus,raw,proxy,log,backup,candidate=fixture()
        with tempfile.TemporaryDirectory() as d, patch.object(bus,'_open_bus',return_value=proxy), contextlib.redirect_stdout(io.StringIO()):
            out=Path(d)/'final.json'
            self.assertFalse(f.run_finalize(bus,backup,candidate,log,out,ask=lambda _:'cancel'))
            self.assertFalse(out.exists())
        self.assertEqual(raw.writes,[])
        self.assertTrue(raw.closed)

    def test_current_baseline_mismatch_no_writes(self):
        bus,raw,proxy,log,backup,candidate=fixture()
        raw.values[2]['homing_offset']=86
        with tempfile.TemporaryDirectory() as d, patch.object(bus,'_open_bus',return_value=proxy):
            with self.assertRaises(s.SafetyBlock):
                f.run_finalize(bus,backup,candidate,log,Path(d)/'final.json',ask=lambda _:'SAVE')
        self.assertEqual(raw.writes,[])
        self.assertTrue(raw.closed)

    def test_readback_failure_no_success_file(self):
        bus,raw,proxy,log,backup,candidate=fixture()
        with tempfile.TemporaryDirectory() as d, patch.object(bus,'_open_bus',return_value=proxy), patch.object(bus,'read_calibration',return_value={}), contextlib.redirect_stdout(io.StringIO()):
            out=Path(d)/'final.json'
            with self.assertRaises(s.SafetyBlock):
                f.run_finalize(bus,backup,candidate,log,out,ask=lambda _:'SAVE')
            self.assertFalse(out.exists())
        self.assertTrue(raw.closed)
        self.assertTrue(all(name!='torque_enable' for _,name,_ in raw.writes))

    def test_invalid_candidate_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json'
            p.write_text('{}')
            with self.assertRaises(s.SafetyBlock):
                f.load_candidate(p)


if __name__=='__main__':
    unittest.main()
