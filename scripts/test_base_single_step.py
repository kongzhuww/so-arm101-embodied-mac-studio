import contextlib
import io
import unittest
from unittest.mock import patch

import base_single_step as base
from test_gripper_microstep import FakeRaw, Clock
from test_safe_official_calibrate import Log


class BaseFake(FakeRaw):
    def __init__(self, uncertain=False, overshoot=False, config_enables=False, drift=False):
        super().__init__()
        self.uncertain = uncertain
        self.overshoot = overshoot
        self.config_enables = config_enables
        self.drift = drift

    def write_register_with_error(self, i, name, value, retries=0):
        assert i == 1 and retries == 0
        self.writes.append((i, name, value))
        self.data[i][name] = value
        if name in base.SETTINGS and self.config_enables:
            self.data[i]['torque_enable'] = 1
        if name == 'goal_position':
            self.data[i]['torque_enable'] = 1
            self.data[i]['present_position'] = value+(100 if self.overshoot else 0)
            if self.drift:
                self.data[2]['present_position'] += 100
            if self.uncertain:
                raise OSError('goal applied, lost acknowledgement')
        return 0


class BaseTests(unittest.TestCase):
    def run_fake(self, raw):
        with contextlib.redirect_stdout(io.StringIO()):
            return base.perform(raw, raw.cal, Log(), clock=Clock(), sleep=lambda _: None)

    def test_plan_no_io(self):
        with patch.object(base.m, 'open_raw', side_effect=AssertionError('serial')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(base.main([]), 0)

    def test_only_one_goal_motor1_low_settings_then_off(self):
        raw = BaseFake()
        self.assertTrue(self.run_fake(raw))
        goals = [w for w in raw.writes if w[1] == 'goal_position']
        self.assertEqual(goals, [(1, 'goal_position', 2029+16)])
        self.assertTrue(all(w[0] == 1 for w in raw.writes))
        self.assertNotIn((1, 'torque_enable', 1), raw.writes)
        self.assertEqual(raw.writes[-1], (1, 'torque_enable', 0))
        for k, v in base.SETTINGS.items():
            self.assertEqual(raw.data[1][k], v)

    def test_uncertain_goal_never_retried_cleanup(self):
        raw = BaseFake(uncertain=True)
        with self.assertRaises(OSError):
            self.run_fake(raw)
        self.assertEqual(sum(w[1] == 'goal_position' for w in raw.writes), 1)
        self.assertEqual(raw.data[1]['torque_enable'], 0)

    def test_overshoot_and_other_axis_change_stop_cleanup(self):
        for kwargs in ({'overshoot': True}, {'drift': True}):
            raw = BaseFake(**kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(base.m.Stop):
                self.run_fake(raw)
            self.assertEqual(raw.data[1]['torque_enable'], 0)

    def test_configuration_enables_torque_blocks_goal(self):
        raw = BaseFake(config_enables=True)
        with self.assertRaises(base.m.Stop):
            self.run_fake(raw)
        self.assertFalse(any(w[1] == 'goal_position' for w in raw.writes))
        self.assertEqual(raw.data[1]['torque_enable'], 0)

    def test_wrong_scaling_no_writes(self):
        raw = BaseFake()
        raw.data[1]['velocity_unit_factor'] = 100
        with self.assertRaises(base.m.Stop):
            self.run_fake(raw)
        self.assertEqual(raw.writes, [])

    def test_endpoint_no_writes(self):
        raw = BaseFake()
        raw.data[1]['present_position'] = raw.cal['shoulder_pan'].range_max-65
        with self.assertRaises(base.m.Stop):
            self.run_fake(raw)
        self.assertEqual(raw.writes, [])


if __name__ == '__main__':
    unittest.main()
