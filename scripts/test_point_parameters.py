"""Temporary-file tests for edit/copy endpoints; actual bus access forbidden."""
import copy
import unittest
from unittest.mock import patch
from pydantic import ValidationError

import test_teaching_versions as fixtures
from test_studio_recording import rows
from teaching_versions import derive_waypoints, POINT_NAMES
import teleop_studio as studio


class ParameterTests(unittest.TestCase):
    setUp = fixtures.VersionTests.setUp

    def edit_request(self, values=None, point='p2', **overrides):
        active=self.store.active()
        data=dict(target_version_id=active['id'],expected_updated_at=active['updated_at'],
                  point=point,values=values if values is not None else [2200]*6)
        data.update(overrides)
        return studio.PointEditRequest(**data)

    def copy_request(self, source, points=None, **overrides):
        active=self.store.active()
        data=dict(target_version_id=active['id'],expected_updated_at=active['updated_at'],
                  source_version_id=source['id'],source_updated_at=source['updated_at'],
                  points=points if points is not None else ['p1','p4'])
        data.update(overrides)
        return studio.PointCopyRequest(**data)

    def test_bulk_copy_leaves_other_points_and_source_unchanged(self):
        points={name:[2000+i]*6 for i,name in enumerate(POINT_NAMES.values())}
        source=self.store.set_points(points)
        target=self.store.doc['versions'][1]
        studio.select_version(target['id'])
        self.store.set_points({'p2_grasp':[2500]*6,'p3_lift':[2600]*6})
        studio.sync_version_state()
        with patch.object(studio.rustypot,'Bus',side_effect=AssertionError('NO HARDWARE')):
            result=studio.copy_waypoint_values(self.copy_request(source))
        self.assertTrue(result['ok'],result)
        v=self.store.active()['waypoints']
        self.assertEqual(v['p1_standby'],points['p1_standby'])
        self.assertEqual(v['p4_release'],points['p4_release'])
        self.assertEqual(v['p2_grasp'],[2500]*6)
        self.assertEqual(v['p3_lift'],[2600]*6)
        self.assertEqual(self.store.get(source['id']),source)

    def test_edit_keeps_exact_values_and_unrelated_transits(self):
        points={name:[2000+i*100]*6 for i,name in enumerate(POINT_NAMES.values())}
        wp=derive_waypoints(points)
        wp['p4_exit'][5]=2850
        wp['p1_transit'][5]=2100
        wp['p2_approach'][5]=2100
        self.store.update(wp)
        before=copy.deepcopy(wp)
        values=[2101,2202,2303,2404,2505,2343]
        with patch.object(studio.rustypot,'Bus',side_effect=AssertionError('NO HARDWARE')):
            result=studio.edit_waypoint_values(self.edit_request(values))
        self.assertTrue(result['ok'],result)
        after=self.store.active()['waypoints']
        self.assertEqual(after['p2_grasp'],values)
        for key in ('p1_standby','p3_lift','p4_release','p4_exit','p1_transit'):
            self.assertEqual(after[key],before[key],key)
        self.assertEqual(after['p2_approach'][5],2100)
        self.assertEqual(after['p4_box'][5],2343)

    def test_invalid_shape_and_range_do_not_write(self):
        original=self.store.path.read_bytes()
        for values in ([1]*5,[1]*7,[-1]*6,[4096]*6):
            with self.subTest(values=values):
                result=studio.edit_waypoint_values(self.edit_request(values))
                self.assertFalse(result['ok'])
                self.assertEqual(self.store.path.read_bytes(),original)
        for values in ([1.5]*6,[True]*6,['2000']*6):
            with self.assertRaises(ValidationError):
                self.edit_request(values)

    def test_missing_source_point_is_atomic(self):
        source=self.store.set_points({'p1_standby':[2000]*6})
        studio.select_version(self.store.doc['versions'][1]['id'])
        original=self.store.path.read_bytes()
        result=studio.copy_waypoint_values(self.copy_request(source))
        self.assertFalse(result['ok'])
        self.assertEqual(self.store.path.read_bytes(),original)
        self.assertEqual(self.store.active()['waypoints'],{})

    def test_stale_or_switched_target_rejected(self):
        request=self.edit_request()
        self.store.set_points({'p1_standby':[2000]*6})
        original=self.store.path.read_bytes()
        self.assertFalse(studio.edit_waypoint_values(request)['ok'])
        self.assertEqual(self.store.path.read_bytes(),original)
        request=self.edit_request()
        studio.select_version(self.store.doc['versions'][1]['id'])
        original=self.store.path.read_bytes()
        self.assertFalse(studio.edit_waypoint_values(request)['ok'])
        self.assertEqual(self.store.path.read_bytes(),original)

    def test_stale_source_and_empty_selection_rejected(self):
        source=self.store.set_points({'p1_standby':[2000]*6,'p4_release':[2400]*6})
        studio.select_version(self.store.doc['versions'][1]['id'])
        original=self.store.path.read_bytes()
        result=studio.copy_waypoint_values(self.copy_request(source,source_updated_at='stale'))
        self.assertFalse(result['ok'])
        self.assertFalse(studio.copy_waypoint_values(self.copy_request(source,points=[]))['ok'])
        self.assertEqual(self.store.path.read_bytes(),original)

    def test_edited_recorded_version_retains_old_samples(self):
        studio.save_episode(rows())
        original=studio.DATASET_DIR/'episode_0000.json'
        content=original.read_bytes()
        result=studio.edit_waypoint_values(self.edit_request([2343]*6))
        self.assertTrue(result['ok'],result)
        self.assertEqual(self.store.active()['revision'],2)
        self.assertEqual(studio.state['total_episodes'],0)
        self.assertEqual(original.read_bytes(),content)

    def test_identical_edit_does_not_fork_recordings(self):
        self.store.set_points({'p2_grasp':[2343]*6})
        studio.sync_version_state()
        studio.save_episode(rows())
        previous=self.store.active()
        result=studio.edit_waypoint_values(self.edit_request([2343]*6))
        self.assertTrue(result['ok'],result)
        self.assertEqual(self.store.active(),previous)
        self.assertEqual(studio.state['total_episodes'],1)

    def test_busy_endpoints_never_access_bus_or_write(self):
        request=self.edit_request()
        original=self.store.path.read_bytes()
        studio.motion_lock.acquire()
        try:
            with patch.object(studio.rustypot,'Bus',side_effect=AssertionError('NO HARDWARE')):
                self.assertFalse(studio.edit_waypoint_values(request)['ok'])
        finally:
            studio.motion_lock.release()
        studio.state['training']=True
        self.assertFalse(studio.edit_waypoint_values(request)['ok'])
        self.assertEqual(self.store.path.read_bytes(),original)


if __name__=='__main__':
    unittest.main()
