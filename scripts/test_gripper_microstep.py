"""OFFLINE mock tests of all motor6 movement/cleanup branches."""
import contextlib
import io
import unittest
from unittest.mock import patch

import gripper_microstep as m
from test_safe_official_calibrate import Log


class FakeRaw:
    def __init__(self,stuck=False,overshoot=False,drift_other=False,disable_fails=False,enable_uncertain=False):
        self.cal=m.load_calibration()
        positions=[2029,1227,2528,2847,1962,2377]
        self.data={c.id:{'model_number':777,'torque_enable':0,'present_position':positions[c.id-1],
                        'homing_offset':c.homing_offset,'min_position_limit':c.range_min,'max_position_limit':c.range_max,
                        'operating_mode':0,'phase':12,'firmware_major_version':3,'firmware_minor_version':10,
                        'velocity_unit_factor':50,'acceleration_multiplier':1,'maximum_acceleration':50,
                        'present_voltage':123,'present_temperature':38,'status':0,
                        'goal_position':0,'goal_time':0,'goal_speed':0,'acceleration':0,'torque_limit':1000}
                   for c in self.cal.values()}
        self.writes=[]
        self.stuck=stuck
        self.overshoot=overshoot
        self.drift_other=drift_other
        self.disable_fails=disable_fails
        self.enable_uncertain=enable_uncertain

    def read_register_with_error(self,i,name,retries=0):
        assert retries==0
        return self.data[i][name],0

    def write_register_with_error(self,i,name,value,retries=0):
        assert i==6 and retries==0
        self.writes.append((i,name,value))
        if name=='torque_enable' and value==0 and self.disable_fails:
            raise OSError('simulated link fault during disable')
        self.data[i][name]=value
        if name=='torque_enable' and value==1:
            self.data[i]['present_position']=self.data[i]['goal_position']
            if self.enable_uncertain:
                raise OSError('enable applied but acknowledgement lost')
        if name=='goal_position' and self.data[i]['torque_enable']==1:
            if not self.stuck:
                self.data[i]['present_position']=value+(100 if self.overshoot else 0)
            if self.drift_other:
                self.data[2]['present_position']+=100
        return 0


class Clock:
    def __init__(self):
        self.now=0
    def __call__(self):
        self.now+=0.1
        return self.now


def run(raw,answer='MOVE6'):
    gate=m.Motor6Gate(raw,Log())
    with contextlib.redirect_stdout(io.StringIO()):
        result=m.run_test(gate,raw.cal,ask=lambda _:answer,clock=Clock(),sleep=lambda _:None)
    return result,gate


class MicrostepTests(unittest.TestCase):
    def test_default_no_hardware(self):
        with patch.object(m,'open_raw',side_effect=AssertionError('no hardware')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main([]),0)

    def test_noninteractive_execution_blocked(self):
        with patch.object(m.sys.stdin,'isatty',return_value=False),patch.object(m,'open_raw') as op,contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(m.main(['--execute','--ack-motor6-motion']),2)
            op.assert_not_called()

    def test_hardware_cli_disabled_even_with_tty_and_flags(self):
        with patch.object(m.sys.stdin,'isatty',return_value=True),patch.object(m.sys.stdout,'isatty',return_value=True),patch.object(m,'open_raw',side_effect=AssertionError('disabled must not open')),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(m.main(['--execute','--ack-motor6-motion']),2)

    def test_preload_auto_enable_reproduces_real_failure_and_cleans_up(self):
        raw=FakeRaw()
        original=raw.write_register_with_error
        def write(i,name,value,retries=0):
            result=original(i,name,value,retries)
            if name=='goal_position' and value==2377:
                raw.data[i]['torque_enable']=1
            return result
        gate=m.Motor6Gate(raw,Log())
        with patch.object(raw,'write_register_with_error',side_effect=write),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(m.Stop,'torque=1, status=0x00'):
                m.run_test(gate,raw.cal,ask=lambda _:'MOVE6',clock=Clock(),sleep=lambda _:None)
        self.assertNotIn((6,'torque_enable',1),raw.writes)
        self.assertNotIn((6,'goal_position',2385),raw.writes)
        self.assertEqual(raw.data[6]['torque_enable'],0)
        self.assertTrue(any(e['event']=='micro_cleanup' and e['motor6_torque_off_verified'] for e in gate.journal.events))

    def test_both_flags_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                m.main(['--execute'])

    def test_cancel_no_register_writes(self):
        raw=FakeRaw()
        result,gate=run(raw,'cancel')
        self.assertFalse(result)
        self.assertEqual(raw.writes,[])
        self.assertFalse(gate.armed)

    def test_success_only_motor6_plus8_preload_before_enable(self):
        raw=FakeRaw()
        result,gate=run(raw,' move6 ')
        self.assertTrue(result)
        self.assertEqual(raw.data[6]['present_position'],2385)
        self.assertEqual(raw.data[6]['torque_enable'],0)
        self.assertFalse(gate.armed)
        self.assertTrue(all(i==6 for i,_,_ in raw.writes))
        enable=raw.writes.index((6,'torque_enable',1))
        preload=raw.writes.index((6,'goal_position',2377))
        target=raw.writes.index((6,'goal_position',2385))
        self.assertLess(preload,enable)
        self.assertLess(enable,target)
        self.assertEqual(raw.writes[-1],(6,'torque_enable',0))
        self.assertEqual(sum(name=='goal_position' for _,name,_ in raw.writes),2)
        self.assertTrue(all(raw.data[i]['torque_enable']==0 for i in range(1,6)))
        self.assertEqual(raw.data[6]['torque_limit'],50)

    def test_block_other_motor_register_and_goal(self):
        raw=FakeRaw()
        gate=m.Motor6Gate(raw,Log())
        with self.assertRaises(m.Stop):
            gate.write(6,'torque_enable',1)
        gate.arm(2377,2071,3374)
        for i,name,value in ((1,'torque_enable',1),(6,'id',3),(6,'goal_position',0),(6,'goal_speed',0),(6,'torque_limit',1000),(6,'torque_enable',1)):
            with self.assertRaises(m.Stop):
                gate.write(i,name,value)
        self.assertEqual(raw.writes,[])

    def test_margin_and_max_delta_guards(self):
        gate=m.Motor6Gate(FakeRaw(),Log())
        for start in (2007,3370,4095):
            with self.assertRaises(m.Stop):
                gate.arm(start,2071,3374)

    def test_preflight_bad_torque_and_endpoint_no_writes(self):
        for i,name,value in ((2,'torque_enable',1),(3,'present_position',3103),(6,'velocity_unit_factor',99),(6,'present_temperature',60)):
            raw=FakeRaw()
            raw.data[i][name]=value
            with self.assertRaises(m.Stop):
                run(raw)
            self.assertEqual(raw.writes,[])

    def test_stuck_does_not_raise_power_or_retry_goal(self):
        raw=FakeRaw(stuck=True)
        result,_=run(raw)
        self.assertFalse(result)
        self.assertEqual(raw.data[6]['torque_enable'],0)
        self.assertEqual(sum(x==(6,'goal_position',2385) for x in raw.writes),1)
        self.assertEqual(raw.data[6]['torque_limit'],50)

    def test_overshoot_stops_and_disables(self):
        raw=FakeRaw(overshoot=True)
        with self.assertRaises(m.Stop):
            run(raw)
        self.assertEqual(raw.data[6]['torque_enable'],0)

    def test_other_axis_drift_stops_and_disables_only6(self):
        raw=FakeRaw(drift_other=True)
        with self.assertRaises(m.Stop):
            run(raw)
        self.assertTrue(all(i==6 for i,_,_ in raw.writes))
        self.assertEqual(raw.data[6]['torque_enable'],0)

    def test_uncertain_enable_not_retried_cleanup_runs(self):
        raw=FakeRaw(enable_uncertain=True)
        with self.assertRaises(OSError):
            run(raw)
        self.assertEqual(sum(x==(6,'torque_enable',1) for x in raw.writes),1)
        self.assertEqual(raw.data[6]['torque_enable'],0)
        self.assertNotIn((6,'goal_position',2385),raw.writes)

    def test_disable_failure_demands_manual_cutoff(self):
        raw=FakeRaw(disable_fails=True)
        with self.assertRaisesRegex(m.Stop,'手动切断电源'):
            run(raw)
        self.assertEqual(sum(x==(6,'torque_enable',0) for x in raw.writes),2)

    def test_failed_ready_sample_logs_exact_status_and_never_enables(self):
        raw=FakeRaw()
        original=raw.write_register_with_error
        def write(i,name,value,retries=0):
            result=original(i,name,value,retries)
            if name=='goal_position' and raw.data[i]['torque_enable']==0:
                raw.data[i]['status']=4
            return result
        gate=m.Motor6Gate(raw,Log())
        with patch.object(raw,'write_register_with_error',side_effect=write),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(m.Stop,'status=0x04'):
                m.run_test(gate,raw.cal,ask=lambda _:'MOVE6',clock=Clock(),sleep=lambda _:None)
        states=[e for e in gate.journal.events if e['event']=='micro_prepared_readback']
        self.assertEqual(states[-1]['readings']['status'],4)
        self.assertEqual(states[-1]['readings']['torque_enable'],0)
        self.assertNotIn((6,'torque_enable',1),raw.writes)
        self.assertNotIn((6,'goal_position',2385),raw.writes)
        self.assertEqual(raw.data[6]['torque_enable'],0)

    def test_all_three_ready_samples_clean_on_success(self):
        raw=FakeRaw()
        result,gate=run(raw)
        self.assertTrue(result)
        samples=[e for e in gate.journal.events if e['event']=='micro_prepared_readback']
        self.assertEqual(len(samples),3)
        self.assertTrue(all(e['readings']['torque_enable']==0 and e['readings']['status']==0 for e in samples))

    def test_preload_verification_blocks_unexpected_state(self):
        raw=FakeRaw()
        gate=m.Motor6Gate(raw,Log())
        gate.arm(2377,2071,3374)
        for name,value in m.LOW_SETTINGS.items():
            gate.write(6,name,value)
        gate.write(6,'goal_position',2377)
        raw.data[6]['present_position']=2387
        with self.assertRaises(m.Stop):
            gate.verify_prepared()
        with self.assertRaises(m.Stop):
            gate.write(6,'torque_enable',1)


if __name__=='__main__':
    unittest.main()
