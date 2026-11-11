import contextlib
import io
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import joint_step as joint
import joint_control_server as web
from test_gripper_microstep import FakeRaw, Clock
from test_safe_official_calibrate import Log


class JointFake(FakeRaw):
    def __init__(self, selected=1, uncertain=False, config_enables=False):
        super().__init__()
        self.selected = selected
        self.uncertain = uncertain
        self.config_enables = config_enables
        self.closed = False

    def close(self):
        self.closed = True

    def write_register_with_error(self, i, name, value, retries=0):
        assert i == self.selected and retries == 0
        self.writes.append((i, name, value))
        self.data[i][name] = value
        if name in joint.SETTINGS and self.config_enables:
            self.data[i]['torque_enable'] = 1
        if name == 'goal_position':
            self.data[i]['torque_enable'] = 1
            self.data[i]['present_position'] = value
            if self.uncertain:
                raise OSError('goal applied, lost ack')
        return 0


def bounds(raw):
    return {i: (d['present_position']-64, d['present_position']+64) for i, d in raw.data.items()}


class JointTests(unittest.TestCase):
    def test_all_motors_and_directions_one_goal_then_off(self):
        for motor in range(1, 7):
            for step in (-16, 16):
                with self.subTest(motor=motor, step=step):
                    raw = JointFake(motor)
                    start = raw.data[motor]['present_position']
                    result = joint.perform(raw, raw.cal, Log(), motor, step, bounds(raw), clock=Clock(), sleep=lambda _: None)
                    self.assertTrue(result['reached'])
                    self.assertTrue(all(w[0] == motor for w in raw.writes))
                    self.assertEqual([w for w in raw.writes if w[1] == 'goal_position'], [(motor, 'goal_position', start+step)])
                    self.assertEqual(raw.writes[-1], (motor, 'torque_enable', 0))
                    self.assertNotIn((motor, 'torque_enable', 1), raw.writes)

    def test_unknown_input_and_session_boundary_no_write(self):
        for motor, step in ((0, 16), (7, 16), (True, 16), (1, 32), (1, 0)):
            raw = JointFake()
            with self.assertRaises(joint.m.Stop):
                joint.perform(raw, raw.cal, Log(), motor, step, bounds(raw))
            self.assertEqual(raw.writes, [])
        raw = JointFake()
        b = bounds(raw); b[1] = (2020, 2030)
        with self.assertRaises(joint.RangeStop):
            joint.perform(raw, raw.cal, Log(), 1, 16, b)
        self.assertEqual(raw.writes, [])

    def test_ambiguous_goal_no_retry_off(self):
        raw = JointFake(3, uncertain=True)
        with self.assertRaises(OSError):
            joint.perform(raw, raw.cal, Log(), 3, 16, bounds(raw), sleep=lambda _: None)
        self.assertEqual(sum(w[1] == 'goal_position' for w in raw.writes), 1)
        self.assertEqual(raw.data[3]['torque_enable'], 0)

    def test_implicit_config_enable_stops_before_goal(self):
        raw = JointFake(2, config_enables=True)
        with self.assertRaises(joint.m.Stop):
            joint.perform(raw, raw.cal, Log(), 2, 16, bounds(raw), sleep=lambda _: None)
        self.assertFalse(any(w[1] == 'goal_position' for w in raw.writes))
        self.assertEqual(raw.data[2]['torque_enable'], 0)

    def test_stop_before_or_during_motion(self):
        raw = JointFake()
        with self.assertRaises(joint.m.Stop):
            joint.perform(raw, raw.cal, Log(), 1, 16, bounds(raw), stopped=lambda: True)
        self.assertEqual(raw.writes, [])
        raw = JointFake()
        with self.assertRaises(joint.m.Stop):
            joint.perform(raw, raw.cal, Log(), 1, 16, bounds(raw),
                          stopped=lambda: any(w[1] == 'goal_position' for w in raw.writes), sleep=lambda _: None)
        self.assertEqual(raw.data[1]['torque_enable'], 0)


class WebTests(unittest.TestCase):
    def client(self, raw):
        positions = {i: d['present_position'] for i, d in raw.data.items()}
        app = web.build_app(raw.cal, positions, opener=lambda: raw)
        return TestClient(app, base_url=web.ORIGIN, client=('10.19.220.45', 12345))

    def headers(self):
        return {'Origin': web.ORIGIN}

    def test_plan_get_and_selection_have_no_writes(self):
        with patch.object(joint.m, 'open_raw', side_effect=AssertionError('I/O')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(web.main([]), 0)
        raw = JointFake()
        with self.client(raw) as c:
            self.assertEqual(c.get('/').status_code, 200)
            self.assertEqual(c.get('/status').status_code, 200)
        self.assertEqual(raw.writes, [])

    def test_post_once_duplicate_and_stop(self):
        raw = JointFake(4)
        with self.client(raw) as c:
            data = {'motor': 4, 'delta': 16, 'request_id': 'one-operation'}
            self.assertEqual(c.post('/step', json=data, headers=self.headers()).status_code, 200)
            count = len(raw.writes)
            self.assertTrue(raw.closed)
            self.assertEqual(c.post('/step', json=data, headers=self.headers()).status_code, 409)
            self.assertEqual(len(raw.writes), count)
            self.assertEqual(c.post('/stop', json={}, headers=self.headers()).status_code, 200)
            data['request_id'] = 'second-operation'
            self.assertEqual(c.post('/step', json=data, headers=self.headers()).status_code, 423)
            self.assertEqual(len(raw.writes), count)

    def test_cross_site_denied_and_invalid_motor_no_write(self):
        raw = JointFake()
        with self.client(raw) as c:
            data = {'motor': 1, 'delta': 16, 'request_id': 'one-operation'}
            self.assertEqual(c.post('/step', json=data, headers={'Origin': 'https://example.com'}).status_code, 403)
            data['motor'] = 9
            self.assertEqual(c.post('/step', json=data, headers=self.headers()).status_code, 400)
        self.assertEqual(raw.writes, [])


if __name__ == '__main__':
    unittest.main()
