"""ACT tests: temporary models, mocked camera/bus/policy; no real motion."""
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import numpy as np

from studio_act import ActModels, run_once
import test_teaching_versions as fixtures
import teleop_studio as studio


def fake_model(root, ident, episodes=10):
    final=Path(root)/'outputs'/ident/'final';final.mkdir(parents=True)
    for name in ('config.json','policy_preprocessor.json','policy_postprocessor.json'):
        (final/name).write_text(json.dumps({'type':'act'} if name=='config.json' else {'steps':[]}))
    (final/'model.safetensors').write_bytes(b'TEST FIXTURE ONLY')
    dataset=Path(root)/'datasets'/ident;dataset.mkdir(parents=True)
    (dataset/'meta').mkdir();(dataset/'meta/info.json').write_text(json.dumps({'total_episodes':episodes,'total_frames':100}))
    (final.parent/'run-manifest.json').write_text(json.dumps({'dataset':str(dataset)}))
    return final


class FakeClock:
    def __init__(self):self.now=0
    def __call__(self):return self.now
    def sleep(self,seconds):self.now+=max(seconds,.001)


class FakeBus:
    def __init__(self,poses=None):
        self.poses=poses or [[2081,1025,3064,2878,2093,2099]]
        self.reads=0;self.writes=[];self.closed=False
    def read_register_with_error(self,motor,register,retries):
        index=min(self.reads//6,len(self.poses)-1);self.reads+=1
        return self.poses[index][motor-1],0
    def write_register_with_error(self,motor,register,value,retries):
        self.writes.append((motor,register,value));return 0
    def close(self):self.closed=True


class FakePolicy:
    boxes=[[2080,2603,1438,2947,2094]]
    standbys=[[2081,1025,3064,2878,2093,2099]]
    def predict(self,positions,rgb):return [2082,1100,3000,2900,2093,2100]


class RunnerTests(unittest.TestCase):
    def run_fake(self,*,policy=None,stop=None,bus=None,emit=lambda **kw:None,max_seconds=.3):
        clock=FakeClock();bus=bus or FakeBus();stop=stop or threading.Event()
        result=run_once(Path('/TEST/model/final'),Path('/TEST'),lambda:np.zeros((240,320,3),np.uint8),
            stop,emit,backend_factory=lambda *args:policy or FakePolicy(),bus_factory=lambda:bus,
            max_seconds=max_seconds,clock=clock,sleep=clock.sleep)
        return result,bus

    def test_timeout_holds_and_closes_without_torque_off(self):
        result,bus=self.run_fake()
        self.assertEqual(result['status'],'timeout')
        self.assertTrue(result['held']);self.assertTrue(bus.closed)
        self.assertNotIn((6,'torque_enable',0),bus.writes)
        self.assertIn((6,'torque_limit',130),bus.writes)
        settings=[i for i,x in enumerate(bus.writes) if x[1]=='torque_limit']
        enables=[i for i,x in enumerate(bus.writes) if x[1]=='torque_enable']
        self.assertLess(max(settings),min(enables))

    def test_cancel_during_loading_never_opens_bus(self):
        stop=threading.Event()
        def backend(*args):stop.set();return FakePolicy()
        with patch('studio_act.PolicyBackend',side_effect=backend), \
             patch('rustypot.Bus',side_effect=AssertionError('NO HARDWARE')):
            result=run_once(Path('/test'),Path('/test'),lambda:None,stop)
        self.assertEqual(result['status'],'stopped')
        self.assertFalse(result['held'])

    def test_stop_after_prediction_does_not_send_predicted_goal(self):
        stop=threading.Event()
        class Policy(FakePolicy):
            calls=0
            def predict(self,positions,rgb):
                self.calls+=1
                if self.calls==2:stop.set()
                return [2222]*6
        result,bus=self.run_fake(policy=Policy(),stop=stop,max_seconds=1)
        self.assertEqual(result['status'],'stopped')
        predicted=[w for w in bus.writes if w[1]=='goal_position' and w[2]==2222]
        self.assertEqual(len(predicted),6)  # only the first prediction, not the stopped second
        self.assertTrue(result['held'])

    def test_bad_prediction_never_enables_torque(self):
        class Policy(FakePolicy):
            def predict(self,*args):return [float('nan')]*6
        result,bus=self.run_fake(policy=Policy())
        self.assertEqual(result['status'],'error')
        self.assertFalse(any(w[1]=='torque_enable' for w in bus.writes))
        self.assertTrue(bus.closed)

    def test_camera_failure_never_opens_bus(self):
        def broken():raise ValueError('stale camera')
        with patch('studio_act.PolicyBackend',return_value=FakePolicy()), \
             patch('rustypot.Bus',side_effect=AssertionError('NO HARDWARE')):
            result=run_once(Path('/test'),Path('/test'),broken,threading.Event())
        self.assertEqual(result['status'],'error')
        self.assertIn('stale camera',result['message'])

    def test_box_then_return_completes_without_repeating(self):
        start=FakePolicy.standbys[0]
        box=[2080,2603,1438,2947,2094,2400]
        bus=FakeBus([start,box]+[start]*30)
        result,bus=self.run_fake(bus=bus,max_seconds=3)
        self.assertEqual(result['status'],'completed')
        self.assertLess(result['steps'],30)
        self.assertTrue(result['held'])

    def test_regrasp_guard_stops_second_cycle(self):
        start=FakePolicy.standbys[0]
        box=[2080,2603,1438,2947,2094,2400]
        returning=[2080,1800,2300,2900,2093,2100]
        class Policy(FakePolicy):
            def predict(self,*args):return [2081,2200,2000,3000,2093,2200]
        result,bus=self.run_fake(policy=Policy(),bus=FakeBus([start,box,returning]),max_seconds=3)
        self.assertEqual(result['status'],'blocked')
        self.assertTrue(result['held'])


class CatalogAndApiTests(unittest.TestCase):
    setUp=fixtures.VersionTests.setUp

    def prepare(self):
        self.old=fake_model(self.root,'act_20261106T174035Z',10)
        self.new=fake_model(self.root,'act_20261107T070132Z',30)
        self.library=ActModels(self.root)
        self.stack.enter_context(patch.object(studio,'act_library',self.library))
        self.stack.enter_context(patch.object(studio,'ROOT',self.root))
        self.stack.enter_context(patch.object(studio,'act_stop_event',threading.Event()))
        self.stack.enter_context(patch.object(studio,'act_thread',None))
        studio.state.update(workbench_mode='teach',act_running=False,act_step=0)
        studio.refresh_act_models()

    def test_catalog_selection_and_modes_do_not_load_or_move(self):
        self.prepare()
        self.assertEqual([m['episodes'] for m in self.library.list()],[30,10])
        with patch('studio_act.PolicyBackend',side_effect=AssertionError('NO MODEL LOAD')), \
             patch.object(studio.rustypot,'Bus',side_effect=AssertionError('NO HARDWARE')):
            self.assertTrue(studio.set_workbench_mode('act')['ok'])
            self.assertTrue(studio.select_act_model(self.old.parent.name)['ok'])
        self.assertEqual(ActModels(self.root).current()['model'],self.old.parent.name)
        self.assertEqual(studio.state['workbench_mode'],'act')
        self.assertFalse(studio.state['act_running'])

    def test_incomplete_models_and_path_injection_rejected(self):
        self.prepare()
        (self.new/'model.safetensors').unlink()
        self.assertEqual(len(self.library.list()),1)
        with self.assertRaises(ValueError):self.library.resolve('../../secrets')
        self.assertEqual(self.library.current()['model'],self.old.parent.name)

    def test_cross_mode_motion_routes_blocked_without_hardware(self):
        self.prepare()
        with patch.object(studio.rustypot,'Bus',side_effect=AssertionError('NO HARDWARE')):
            self.assertFalse(studio.start_act_once(self.new.parent.name)['ok'])
            studio.set_workbench_mode('act')
            self.assertFalse(studio.api_auto_playback(False)['ok'])
            self.assertFalse(studio.start_teleop()['ok'])
            self.assertFalse(studio.select_version(self.store.doc['versions'][1]['id'])['ok'])
            self.assertFalse(studio.train_start('all')['ok'])

    def test_clicked_selected_model_is_the_worker_model(self):
        self.prepare();studio.set_workbench_mode('act');studio.select_act_model(self.old.parent.name)
        def thread(*,target,args,daemon):
            class Immediate:
                def start(self):target(*args)
            return Immediate()
        result={'status':'stopped','steps':1,'held':True,'message':'TEST STOP'}
        def fake_run(*args):
            args[4](phase='running',step=1,message='TEST RUN')
            args[4](phase='stopped',step=1,message='TEST STOP')
            return result
        with patch.object(studio,'act_frame_rgb',return_value=np.zeros((240,320,3),np.uint8)), \
             patch.object(studio.threading,'Thread',side_effect=thread), \
             patch('studio_act.run_once',side_effect=fake_run) as run, \
             patch.object(studio.rustypot,'Bus',side_effect=AssertionError('NO HARDWARE')):
            response=studio.start_act_once(self.old.parent.name)
        self.assertTrue(response['ok'],response)
        self.assertEqual(run.call_args.args[0].resolve(),self.old.resolve())
        self.assertFalse(studio.motion_lock.locked())
        self.assertFalse(studio.state['act_running'])
        self.assertEqual(studio.state['running_act_model'],self.old.parent.name)
        log=next((self.root/'reports').glob('act-web-*.jsonl'))
        events=[json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(events),2)
        self.assertEqual(events[-1]['phase'],'stopped')

    def test_stale_model_and_stale_camera_do_not_start_thread(self):
        self.prepare();studio.set_workbench_mode('act')
        with patch.object(studio.threading,'Thread') as thread:
            self.assertFalse(studio.start_act_once(self.old.parent.name)['ok'])
            with patch.object(studio,'act_frame_rgb',side_effect=ValueError('stale')):
                self.assertFalse(studio.start_act_once(self.new.parent.name)['ok'])
            thread.assert_not_called()
        self.assertFalse(studio.motion_lock.locked())

    def test_duplicate_start_and_switch_while_running_rejected_stop_is_event_only(self):
        self.prepare();studio.set_workbench_mode('act')
        with patch.object(studio,'act_frame_rgb',return_value=np.zeros((240,320,3),np.uint8)), \
             patch.object(studio.threading,'Thread'), \
             patch.object(studio.rustypot,'Bus',side_effect=AssertionError('NO HARDWARE')):
            try:
                self.assertTrue(studio.start_act_once(self.new.parent.name)['ok'])
                self.assertFalse(studio.start_act_once(self.new.parent.name)['ok'])
                self.assertFalse(studio.select_act_model(self.old.parent.name)['ok'])
                self.assertFalse(studio.set_workbench_mode('teach')['ok'])
                self.assertTrue(studio.stop_act_once()['ok'])
                self.assertTrue(studio.act_stop_event.is_set())
            finally:
                studio.state['act_running']=False
                if studio.motion_lock.locked():studio.motion_lock.release()

    def test_thread_start_failure_releases_lock(self):
        self.prepare();studio.set_workbench_mode('act')
        with patch.object(studio,'act_frame_rgb',return_value=np.zeros((240,320,3),np.uint8)), \
             patch.object(studio.threading,'Thread') as thread:
            thread.return_value.start.side_effect=RuntimeError('cannot start')
            self.assertFalse(studio.start_act_once(self.new.parent.name)['ok'])
        self.assertFalse(studio.motion_lock.locked())
        self.assertFalse(studio.state['act_running'])


if __name__=='__main__':unittest.main()
