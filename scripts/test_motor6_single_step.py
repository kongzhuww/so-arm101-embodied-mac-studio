import contextlib
import io
import unittest
from unittest.mock import patch
import motor6_single_step as direct
from test_gripper_microstep import FakeRaw,Clock
from test_safe_official_calibrate import Log


class RealBehaviorFake(FakeRaw):
    def __init__(self,overshoot=False,uncertain=False):
        super().__init__()
        self.data[6].update(direct.m.LOW_SETTINGS)
        self.overshoot=overshoot
        self.uncertain=uncertain
    def write_register_with_error(self,i,name,value,retries=0):
        assert i==6 and name in ('goal_position','torque_enable')
        self.writes.append((i,name,value))
        self.data[i][name]=value
        if name=='goal_position':
            self.data[i]['torque_enable']=1
            self.data[i]['present_position']=value+(100 if self.overshoot else 0)
            if self.uncertain:
                raise OSError('goal applied; lost ack')
        return 0


class DirectStepTests(unittest.TestCase):
    def run_fake(self,raw):
        with contextlib.redirect_stdout(io.StringIO()):
            return direct.perform(raw,raw.cal,Log(),clock=Clock(),sleep=lambda _:None)
    def test_default_no_hardware(self):
        with patch.object(direct.m,'open_raw',side_effect=AssertionError('no I/O')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(direct.main([]),0)
    def test_paused_hardware_entry_does_not_open(self):
        with patch.object(direct,'HARDWARE_PAUSED_REASON','paused'),patch.object(direct.m,'open_raw',side_effect=AssertionError('paused')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(direct.main(['--execute']),2)
    def test_goal_auto_enable_expected_and_only_one_step(self):
        raw=RealBehaviorFake()
        self.assertTrue(self.run_fake(raw))
        self.assertEqual(raw.writes,[(6,'goal_position',2377+direct.STEP),(6,'torque_enable',0)])
        self.assertEqual(raw.data[6]['torque_enable'],0)
    def test_overshoot_cleanup(self):
        raw=RealBehaviorFake(overshoot=True)
        with self.assertRaises(direct.m.Stop):
            self.run_fake(raw)
        self.assertEqual(raw.data[6]['torque_enable'],0)
    def test_uncertain_goal_no_retry_cleanup(self):
        raw=RealBehaviorFake(uncertain=True)
        with self.assertRaises(OSError):
            self.run_fake(raw)
        self.assertEqual(raw.writes,[(6,'goal_position',2377+direct.STEP),(6,'torque_enable',0)])
        self.assertEqual(raw.data[6]['torque_enable'],0)
    def test_wrong_settings_no_writes(self):
        raw=RealBehaviorFake()
        raw.data[6]['torque_limit']=1000
        with self.assertRaises(direct.m.Stop):
            self.run_fake(raw)
        self.assertEqual(raw.writes,[])
    def test_near_edge_no_writes(self):
        raw=RealBehaviorFake()
        raw.data[6]['present_position']=2019
        with self.assertRaises(direct.m.Stop):
            self.run_fake(raw)
        self.assertEqual(raw.writes,[])


if __name__=='__main__':
    unittest.main()
