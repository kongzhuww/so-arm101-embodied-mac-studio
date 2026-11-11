"""Offline tests: all files temporary; all bus access mocked or forbidden."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from teaching_versions import TeachingVersions, derive_waypoints
from test_studio_recording import rows
import teleop_studio as studio


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        legacy = self.root/'config/auto_waypoints.json'
        legacy.parent.mkdir(parents=True)
        legacy.write_text(json.dumps({'p1_standby':[2000]*6}))
        self.store = TeachingVersions(self.root)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(studio, 'version_store', self.store))
        self.stack.enter_context(patch.object(studio, 'DATASET_DIR', self.store.directory()))
        self.stack.enter_context(patch.dict(studio.state, copy.deepcopy(studio.state)))
        studio.state.update(training=False, recording=False, teleop_active=False)
        studio.sync_version_state()

    def test_bootstrap_and_persistence(self):
        self.assertEqual(len(self.store.summaries()), 3)
        self.assertEqual(self.store.active()['name'], '版本1')
        self.assertEqual(studio.state['waypoints'], dict.fromkeys(('p1','p2','p3','p4')))
        backup=self.root/'config/teaching_versions/legacy_waypoints.json'
        self.assertEqual(json.loads(backup.read_text()), {'p1_standby':[2000]*6})
        v=self.store.doc['versions'][1]
        self.store.select(v['id'])
        self.assertEqual(TeachingVersions(self.root).active()['id'], v['id'])

    def test_create_assigns_sequential_names(self):
        fourth=self.store.create()
        fifth=self.store.create()
        self.assertEqual(fourth['name'],'版本4')
        self.assertEqual(fifth['name'],'版本5')
        self.assertEqual(self.store.active()['id'],fifth['id'])

    def test_switch_and_clone_never_access_hardware(self):
        points=derive_waypoints({'p1_standby':[2000]*6,'p2_grasp':[2343]*6},2700)
        self.store.update(points,2700)
        with patch.object(studio.rustypot,'Bus',side_effect=AssertionError('NO HARDWARE')):
            self.assertTrue(studio.create_version('复制测试',True)['ok'])
            self.assertEqual(self.store.active()['waypoints'], points)
            self.assertEqual(self.store.active()['open_gripper'],2700)
            self.assertEqual(studio.state['total_episodes'],0)
            self.assertTrue(studio.select_version(self.store.doc['versions'][1]['id'])['ok'])
        self.assertIsNone(studio.state['waypoints']['p2'])
        self.assertFalse(studio.select_version('../../escape')['ok'])

    def test_isolated_recordings_and_snapshots(self):
        first=self.store.active()
        self.store.update(derive_waypoints({'p1_standby':[2000]*6}))
        studio.sync_version_state()
        studio.save_episode(rows())
        original=studio.DATASET_DIR/'episode_0000.json'
        metadata=studio.DATASET_DIR/'metadata/episode_0000.json'
        self.assertTrue(metadata.exists())
        self.assertEqual(len(self.store.episode_files()),1)
        self.assertEqual(json.loads(metadata.read_text())['version']['id'],first['id'])
        studio.select_version(self.store.doc['versions'][1]['id'])
        self.assertEqual(studio.state['total_episodes'],0)
        studio.save_episode(rows())
        self.assertNotEqual(studio.DATASET_DIR, original.parent)
        self.assertEqual(len(self.store.training_files('all')),2)
        studio.clear_all_episodes()
        self.assertTrue(original.exists())
        self.assertEqual(len(self.store.training_files('all')),1)

    def test_edit_with_data_forks_revision(self):
        studio.save_episode(rows())
        original=studio.DATASET_DIR/'episode_0000.json'
        self.assertTrue(studio.save_opening(2700)['ok'])
        self.assertEqual(self.store.active()['revision'],2)
        self.assertEqual(studio.state['total_episodes'],0)
        self.assertEqual(studio.state['versions'][0]['history_episodes'],1)
        self.assertTrue(original.exists())
        self.assertEqual(self.store.training_files('all'),[])
        studio.save_opening(2650)
        self.assertEqual(self.store.active()['revision'],2)

    def test_point_capture_preserves_grip_no_writes(self):
        bus=MagicMock()
        bus.read_register_with_error.return_value=(2343,0)
        with patch.object(studio.rustypot,'Bus',return_value=bus):
            result=studio.api_save_waypoint('p2')
        self.assertTrue(result['ok'],result)
        self.assertEqual(self.store.active()['waypoints']['p2_grasp'],[2343]*6)
        bus.write_register_with_error.assert_not_called()

    def test_capture_opening_only_reads(self):
        bus=MagicMock()
        bus.read_register_with_error.return_value=(2650,0)
        with patch.object(studio.rustypot,'Bus',return_value=bus):
            result=studio.capture_opening()
        self.assertTrue(result['ok'],result)
        self.assertEqual(self.store.active()['open_gripper'],2650)
        bus.write_register_with_error.assert_not_called()

    def test_playback_uses_selected_version_not_legacy_file(self):
        from itertools import count
        points={'p1_standby':[2000]*6,'p2_grasp':[2343]*6,'p3_lift':[2400]*6,'p4_release':[2600]*6}
        self.store.update(derive_waypoints(points,2700),2700)
        studio.sync_version_state()
        bus=MagicMock()
        bus.read_register_with_error.return_value=(2000,0)
        bus.write_register_with_error.return_value=0
        clock=count(100,.2)
        with patch.object(studio.rustypot,'Bus',return_value=bus), \
             patch.object(studio.time,'sleep'), \
             patch.object(studio.time,'monotonic',side_effect=lambda:next(clock)):
            result=studio.api_auto_playback(False)
        self.assertTrue(result['ok'],result)
        goals=[call.args[2] for call in bus.write_register_with_error.call_args_list
               if call.args[:2]==(6,'goal_position')]
        self.assertIn(2700,goals)
        self.assertIn(2343,goals)
        self.assertNotIn(2850,goals)
        self.assertEqual(studio.state['total_episodes'],0)

    def test_busy_rejects_mutations(self):
        original=self.store.active()
        studio.motion_lock.acquire()
        try:
            self.assertFalse(studio.create_version('不能创建',False)['ok'])
            self.assertFalse(studio.save_opening(2600)['ok'])
            self.assertFalse(studio.api_save_waypoint('p2')['ok'])
        finally:
            studio.motion_lock.release()
        studio.state['training']=True
        self.assertFalse(studio.select_version(self.store.doc['versions'][1]['id'])['ok'])
        self.assertEqual(self.store.active(),original)

    def test_combined_training_uses_only_current_revisions(self):
        studio.save_episode(rows())
        studio.select_version(self.store.doc['versions'][1]['id'])
        studio.save_episode(rows())
        with patch.object(studio.threading,'Thread') as thread:
            result=studio.train_start('all')
        self.assertTrue(result['ok'],result)
        self.assertEqual(len(studio.state['training_versions']),2)
        self.assertTrue(studio.state['training'])
        thread.assert_called_once()
        thread.return_value.start.assert_called_once()

    def test_restore_keeps_original_data_and_existing_revision(self):
        from restore_successful_version import apply
        historical=self.root/'datasets/studio_dataset_20261106_155432'
        historical.mkdir(parents=True)
        segments=[[2000+i]*6 for i in range(9)]
        (historical/'episode_0000.json').write_text(json.dumps(
            [{'action':x} for x in segments+[segments[0]]]))
        source=self.root/'datasets/studio_dataset_v2_20261106_170913'
        source.mkdir()
        for i in range(10):
            data=rows()
            data[0]['action']=segments[3]
            data[1]['action']=segments[6]
            (source/f'episode_{i:04d}.json').write_text(json.dumps(data))
        original={p:p.read_bytes() for p in source.glob('*.json')}
        studio.save_episode(rows())
        existing=studio.DATASET_DIR/'episode_0000.json'
        previous=existing.read_bytes()
        report=apply(self.root)
        self.assertEqual(len(report['restored_episodes']),10)
        self.assertTrue(Path(report['original_store_backup']).exists())
        restored=TeachingVersions(self.root)
        self.assertEqual(restored.active()['name'],'版本1')
        self.assertEqual(restored.active()['waypoints']['p2_grasp'],segments[3])
        self.assertEqual(restored.active()['revision'],2)
        self.assertEqual(len(restored.episode_files()),10)
        self.assertEqual(existing.read_bytes(),previous)
        self.assertTrue(all(p.read_bytes()==content for p,content in original.items()))

    def test_auxiliary_generation_supports_partial_teaching(self):
        wp=derive_waypoints({'p3_lift':[2000]*6,'p4_release':[2100]*6})
        self.assertNotIn('p1_transit',wp)
        self.assertEqual(wp['p4_exit'][5],2100)
        points={'p1_standby':[2000]*6,'p2_grasp':[2343]*6,'p3_lift':[2400]*6,'p4_release':[2600]*6}
        wp=derive_waypoints(points,2700)
        self.assertEqual(wp['p2_grasp'][5],2343)
        self.assertEqual(wp['p2_open'][5],2700)
        self.assertEqual(wp['p4_box'][5],2343)
        self.assertEqual(wp['p4_exit'][5],2600)


if __name__=='__main__':
    unittest.main()
