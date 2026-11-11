import contextlib
import io
import unittest
from unittest.mock import patch

import phone_gripper_once as phone
import motor6_single_step as direct
from test_motor6_single_step import RealBehaviorFake
from test_gripper_microstep import Clock
from test_safe_official_calibrate import Log


def sample(t=100.1, a=False, b=False):
    return {'last_pose_at': t, 'button_a': a, 'button_b': b}


class PhoneButtonTests(unittest.TestCase):
    def test_plan_has_no_serial_or_input_polling(self):
        with patch.object(direct.m, 'open_raw', side_effect=AssertionError('serial')), \
             patch.object(phone, 'wait_for_press', side_effect=AssertionError('poll')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(phone.main([]), 0)

    def test_held_before_arm_cannot_trigger(self):
        edge = phone.ButtonEdge(100)
        self.assertIsNone(edge.update(sample(a=True), 100.2))
        self.assertIsNone(edge.update(sample(t=100.2), 100.3))
        self.assertEqual(edge.update(sample(t=100.3, a=True), 100.4), 32)
        self.assertIsNone(edge.update(sample(t=100.4, b=True), 100.5))

    def test_b_is_one_negative_step(self):
        edge = phone.ButtonEdge(100)
        edge.update(sample(), 100.2)
        self.assertEqual(edge.update(sample(t=100.2, b=True), 100.3), -32)

    def test_old_release_does_not_arm(self):
        edge = phone.ButtonEdge(100)
        self.assertIsNone(edge.update(sample(t=99.9), 100.2))
        self.assertIsNone(edge.update(sample(a=True), 100.2))

    def test_both_buttons_need_new_release(self):
        edge = phone.ButtonEdge(100)
        edge.update(sample(), 100.2)
        edge.update(sample(a=True, b=True), 100.2)
        self.assertIsNone(edge.update(sample(a=True), 100.2))
        edge.update(sample(), 100.2)
        self.assertEqual(edge.update(sample(a=True), 100.2), 32)

    def test_stale_future_or_missing_telemetry_rejected(self):
        for s, now in ((sample(), 102), (sample(t=101), 100),
                       ({'last_pose_at': 100.1}, 100.2)):
            with self.subTest(s=s), self.assertRaises(direct.m.Stop):
                phone.ButtonEdge(100).update(s, now)

    def test_negative_motion_uses_only_motor6_then_off(self):
        raw = RealBehaviorFake()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(direct.perform(raw, raw.cal, Log(), clock=Clock(),
                                          sleep=lambda _: None, step=-32))
        self.assertEqual(raw.writes, [(6, 'goal_position', 2377-32), (6, 'torque_enable', 0)])

    def test_oversized_or_zero_or_noninteger_steps_rejected(self):
        for step in (0, -33, 33, True, 1.5):
            raw = RealBehaviorFake()
            with self.subTest(step=step), self.assertRaises(direct.m.Stop):
                direct.perform(raw, raw.cal, Log(), step=step)
            self.assertEqual(raw.writes, [])

    def test_negative_endpoint_rejected_without_write(self):
        raw = RealBehaviorFake()
        raw.data[6]['present_position'] = raw.cal['gripper'].range_min+70
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(direct.m.Stop):
            direct.perform(raw, raw.cal, Log(), step=-32, sleep=lambda _: None)
        self.assertEqual(raw.writes, [])


if __name__ == '__main__':
    unittest.main()
