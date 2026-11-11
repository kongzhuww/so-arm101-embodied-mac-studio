import unittest
from unittest.mock import patch
import numpy as np
import phone_arm_dryrun as dry


class ArmPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner = dry.Preview()

    def setUp(self):
        self.planner.reference = None
        self.planner.enabled = False
        self.planner.saw_release = False

    def sample(self, move=False, pose=None, stamp=100):
        return {'last_pose_at': stamp, 'move_pressed': move,
                'phone_pose_flu': (np.eye(4) if pose is None else pose).tolist()}

    def test_real_1mm_ik_math_only(self):
        report = self.planner.propose([0, 0, .001])
        self.assertTrue(report['math_checks_passed'])
        self.assertFalse(report['motion_authorized'])
        self.assertFalse(report['baseline_is_live'])
        self.assertEqual(report['proposed_raw_ids1_to6'][5], 2445)
        self.assertLess(report['position_residual_m'], .0001)

    def test_recorded_phone_failures_now_solve_without_relaxing_checks(self):
        recorded = [
            [-0.0014847522834451525, -0.001149266717018765, -0.0006889823437195725],
            [-0.0005114346540739647, -0.0019024940872500676, -0.00034489250874841597],
        ]
        for delta in recorded:
            with self.subTest(delta=delta):
                report = self.planner.propose(delta)
                self.assertTrue(report['math_checks_passed'], report['rejections'])
                self.assertLess(report['position_residual_m'], .0001)
                self.assertLessEqual(np.max(np.abs(np.array(report['target_degrees'])-self.planner.q0)), 1)
                self.assertEqual(report['orientation_policy'], 'unconstrained_translation_only')
                self.assertFalse(report['motion_authorized'])

    def test_all_six_translation_directions(self):
        for axis in range(3):
            for sign in (-1, 1):
                delta = np.zeros(3); delta[axis] = sign*.001
                with self.subTest(axis=axis, sign=sign):
                    report = self.planner.propose(delta)
                    self.assertTrue(report['math_checks_passed'], report['rejections'])
                    self.assertLess(report['position_residual_m'], .0001)

    def test_invalid_displacements_rejected(self):
        for delta in ([0, 0, .003], [float('nan'), 0, 0], [0, 0]):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                self.planner.propose(delta)

    def test_stale_and_missing_matrix_rejected(self):
        for sample in (self.sample(stamp=98), {'last_pose_at': 100}):
            with self.assertRaises(ValueError):
                self.planner.sample(sample, 100.1)

    def test_initial_hold_does_not_enable(self):
        self.assertIsNone(self.planner.sample(self.sample(move=True), 100.1))

    def test_release_reference_and_axis_map(self):
        self.assertIsNone(self.planner.sample(self.sample(), 100.1))
        with patch.object(self.planner, 'propose', side_effect=lambda d: np.array(d)):
            np.testing.assert_allclose(self.planner.sample(self.sample(move=True), 100.1), 0)
            p = np.eye(4); p[1, 3] = .02
            np.testing.assert_allclose(self.planner.sample(self.sample(move=True, pose=p), 100.1), [.001, 0, 0])
            self.assertIsNone(self.planner.sample(self.sample(), 100.1))
            np.testing.assert_allclose(self.planner.sample(self.sample(move=True, pose=p), 100.1), 0)

    def test_preview_total_translation_clamped(self):
        self.planner.sample(self.sample(), 100.1)
        with patch.object(self.planner, 'propose', side_effect=lambda d: np.array(d)):
            self.planner.sample(self.sample(move=True), 100.1)
            p = np.eye(4); p[0, 3] = 1
            delta = self.planner.sample(self.sample(move=True, pose=p), 100.1)
            self.assertAlmostEqual(np.linalg.norm(delta), .002)


if __name__ == '__main__':
    unittest.main()
