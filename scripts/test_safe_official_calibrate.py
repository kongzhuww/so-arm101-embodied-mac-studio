"""All hardware mocked. Fake ENTER events ONLY in unit tests, never hardware runs."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import safe_official_calibrate as s
import lerobot.motors.motors_bus as official


def backup_data():
    return {
        'schema': 'so101-raw-register-backup-v1', 'port': s.baseline.PORT,
        'motors': {str(i): {
            'model_number': 777, 'id': i, 'torque_enable': 0, 'lock': 1,
            'operating_mode': 0, 'phase': 12, 'homing_offset': 85,
            'min_position_limit': 0, 'max_position_limit': 4095,
            'firmware_major_version': 3, 'firmware_minor_version': 10,
        } for i in range(1, 7)},
    }


class Log:
    def __init__(self):
        self.events = []

    def emit(self, **event):
        self.events.append(event)


class FakeRaw:
    def __init__(self):
        self.values = {int(i): dict(v) for i, v in backup_data()['motors'].items()}
        self.writes = []
        self.position_reads = {}
        self.closed = False
        self.status = 0

    def read_register_with_error(self, motor_id, name, retries=0):
        assert not self.closed
        if name == 'present_position':
            index = self.position_reads.get(motor_id, 0)
            self.position_reads[motor_id] = index + 1
            return [2047, 2047, 1500, 2500][min(index, 3)], self.status
        return self.values[motor_id][name], self.status

    def write_register_with_error(self, motor_id, name, value, retries=0):
        assert not self.closed
        self.writes.append((motor_id, name, value))
        self.values[motor_id][name] = value
        return self.status

    def close(self):
        self.closed = True


def make_bus():
    log, raw = Log(), FakeRaw()
    proxy = s.RestrictedTransport(raw, log)
    bus = s.SafeCalibrationBus(log)
    return bus, raw, proxy, log


class SafeCalibrationTests(unittest.TestCase):
    def test_default_plan_never_creates_bus(self):
        with patch.object(s, 'SafeCalibrationBus', side_effect=AssertionError('must not instantiate')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(s.main([]), 0)

    def test_noninteractive_execution_refused_before_hardware(self):
        with patch.object(s.sys.stdin, 'isatty', return_value=False), patch.object(s, 'SafeCalibrationBus') as bus, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(s.main(['--execute', '--ack-calibration-writes']), 2)
            bus.assert_not_called()

    def test_both_flags_required(self):
        for args in (['--execute'], ['--ack-calibration-writes']):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    s.main(args)

    def test_bad_registers_and_values_blocked(self):
        for reg, value in [('goal_position',2047), ('id',1), ('baudrate',0), ('phase',12), ('p_coefficient',16), ('torque_enable',1), ('lock',1), ('operating_mode',1), ('homing_offset',2048), ('min_position_limit',-1), ('max_position_limit',4096), ('homing_offset',True), ('homing_offset',0.0)]:
            with self.assertRaises(s.SafetyBlock):
                s.check_write(1, reg, value)
        for motor_id in (0,7,254,True):
            with self.assertRaises(s.SafetyBlock):
                s.check_write(motor_id,'torque_enable',0)

    def test_unarmed_transport_never_writes(self):
        _, raw, proxy, _ = make_bus()
        with self.assertRaises(s.SafetyBlock):
            proxy.write_register_with_error(1,'homing_offset',0)
        self.assertEqual(raw.writes,[])

    def test_armed_transport_blocks_goal_and_torque_on(self):
        _, raw, proxy, _ = make_bus()
        proxy.armed=True
        for name,value in [('goal_position',1800),('torque_enable',1),('id',3)]:
            with self.assertRaises(s.SafetyBlock):
                proxy.write_register_with_error(1,name,value)
        self.assertEqual(raw.writes,[])

    def test_torque_rechecked_before_calibration_write(self):
        _, raw, proxy, _ = make_bus()
        proxy.armed=True
        raw.values[1]['torque_enable']=1
        with self.assertRaises(s.SafetyBlock):
            proxy.write_register_with_error(1,'homing_offset',0)
        self.assertEqual(raw.writes,[])

    def test_alarm_stops_read(self):
        _, raw, proxy, _ = make_bus()
        raw.status=4
        with self.assertRaises(s.SafetyBlock):
            proxy.read_register_with_error(1,'model_number')
        self.assertEqual(raw.writes,[])

    def test_no_transport_set_torque_or_sync_write_api(self):
        _, _, proxy, _ = make_bus()
        for name in ('set_torque','sync_write_register','change_id','change_baudrate'):
            self.assertFalse(hasattr(proxy,name))

    def test_bus_forbids_configuration_sync_write_and_enable(self):
        bus,raw,proxy,_=make_bus()
        bus._bus=proxy
        try:
            for call in (lambda:bus.enable_torque(), lambda:bus.sync_write('Goal_Position',2047), lambda:bus.configure_motors(), lambda:bus.write('ID','shoulder_pan',2)):
                with self.assertRaises(s.SafetyBlock):
                    call()
            self.assertEqual(raw.writes,[])
        finally:
            bus.disconnect()
        self.assertTrue(raw.closed)

    def test_baseline_mismatch_no_writes(self):
        bus,raw,proxy,log=make_bus()
        raw.values[1]['homing_offset']=86
        with tempfile.TemporaryDirectory() as d, patch.object(bus,'_open_bus',return_value=proxy):
            with self.assertRaises(s.SafetyBlock):
                s.run_session(bus,backup_data(),log,Path(d)/'out.json',ask=lambda _: 'CALIBRATE')
        self.assertEqual(raw.writes,[])
        self.assertTrue(raw.closed)

    def test_cancel_before_writes(self):
        bus,raw,proxy,log=make_bus()
        with tempfile.TemporaryDirectory() as d, patch.object(bus,'_open_bus',return_value=proxy), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(s.run_session(bus,backup_data(),log,Path(d)/'out.json',ask=lambda _: 'cancel'))
        self.assertEqual(raw.writes,[])
        self.assertTrue(raw.closed)

    def test_real_official_algorithm_with_fake_transport(self):
        bus,raw,proxy,log=make_bus()
        answers=iter([' calibrate ',' save '])
        with tempfile.TemporaryDirectory() as d, patch.object(bus,'_open_bus',return_value=proxy), patch('builtins.input',return_value=''), patch.object(official,'enter_pressed',side_effect=[False,True]), patch.object(official.time,'sleep'), contextlib.redirect_stdout(io.StringIO()):
            out=Path(d)/'out.json'
            self.assertTrue(s.run_session(bus,backup_data(),log,out,ask=lambda _: next(answers)))
            data=json.loads(out.read_text())
            self.assertEqual(set(data),set(s.NAMES))
            self.assertEqual(data['shoulder_pan']['range_min'],1500)
            self.assertEqual(data['shoulder_pan']['range_max'],2500)
            self.assertEqual(data['wrist_roll']['range_min'],0)
            self.assertEqual(data['wrist_roll']['range_max'],4095)
        self.assertTrue(raw.closed)
        self.assertTrue(raw.writes)
        for motor_id,name,value in raw.writes:
            s.check_write(motor_id,name,value)
            self.assertNotEqual((name,value),('torque_enable',1))
        self.assertTrue(all(raw.values[i]['torque_enable']==0 for i in range(1,7)))
        self.assertFalse(proxy.armed)
        self.assertTrue(any(e['event']=='saved' for e in log.events))

    def test_abort_after_reference_no_auto_rollback_or_enable(self):
        bus,raw,proxy,log=make_bus()
        answers=iter(['CALIBRATE','cancel'])
        with tempfile.TemporaryDirectory() as d, patch.object(bus,'_open_bus',return_value=proxy), patch('builtins.input',return_value=''), patch.object(official,'enter_pressed',side_effect=[False,True]), patch.object(official.time,'sleep'), contextlib.redirect_stdout(io.StringIO()):
            out=Path(d)/'out.json'
            self.assertFalse(s.run_session(bus,backup_data(),log,out,ask=lambda _:next(answers)))
            self.assertFalse(out.exists())
            self.assertTrue(out.with_suffix('.candidate.json').exists())
        self.assertTrue(raw.closed)
        self.assertTrue(any(e['event']=='cancel_after_reference_writes' for e in log.events))
        self.assertTrue(all(raw.values[i]['torque_enable']==0 for i in range(1,7)))

    def test_interruption_closes_without_enable(self):
        bus,raw,proxy,log=make_bus()
        def interrupt(_):
            raise KeyboardInterrupt
        with tempfile.TemporaryDirectory() as d, patch.object(bus,'_open_bus',return_value=proxy), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                s.run_session(bus,backup_data(),log,Path(d)/'out.json',ask=interrupt)
        self.assertTrue(raw.closed)
        self.assertEqual(raw.writes,[])

    def test_backup_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'backup.json'
            data=backup_data()
            path.write_text(json.dumps(data))
            self.assertEqual(s.load_backup(path),data)
            data['motors']['1']['phase']=28
            path.write_text(json.dumps(data))
            with self.assertRaises(s.SafetyBlock):
                s.load_backup(path)


if __name__=='__main__':
    unittest.main()
