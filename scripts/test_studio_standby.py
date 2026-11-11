"""Mock-only standby tests; never opens serial/camera or executes an ACT policy."""
import json
import threading
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
from studio_standby import run_standby,model_standby
import teleop_studio as studio
import test_teaching_versions as fixtures

TARGET=[2081,1025,3064,2878,2093,2099]


class Bus:
    def __init__(self,stuck=False):self.current=[2100,1200,2900,2800,2100,2200];self.writes=[];self.closed=False;self.stuck=stuck
    def read_register_with_error(self,motor,register,retries):return self.current[motor-1],0
    def write_register_with_error(self,motor,register,value,retries):
        self.writes.append((motor,register,value))
        if register=='goal_position' and not self.stuck:self.current[motor-1]=value
        return 0
    def close(self):self.closed=True


class RunnerTests(unittest.TestCase):
    def run_fake(self,stop=None,stuck=False):
        stop=stop or threading.Event();bus=Bus(stuck);now=[0]
        def wait(seconds):now[0]+=seconds;return stop.is_set()
        r=run_standby(TARGET,lambda:bus,stop,clock=lambda:now[0],wait=wait)
        return r,bus
    def test_completes_only_standby_and_keeps_torque(self):
        r,bus=self.run_fake();self.assertEqual(r['status'],'completed');self.assertTrue(r['held']);self.assertTrue(bus.closed)
        self.assertEqual(bus.current,TARGET)
        enables=[i for i,w in enumerate(bus.writes)if w[1]=='torque_enable']
        settings=[i for i,w in enumerate(bus.writes)if w[1]=='goal_speed']
        self.assertLess(max(settings),min(enables))
        self.assertFalse(any(w[1]=='torque_enable'and w[2]==0 for w in bus.writes))
        for motor in range(1,7):
            goals=[v for m,k,v in bus.writes if m==motor and k=='goal_position']
            self.assertLessEqual(max(abs(b-a)for a,b in zip(goals,goals[1:])),20)
    def test_stop_before_start_does_not_open_bus(self):
        stop=threading.Event();stop.set()
        r=run_standby(TARGET,lambda:self.fail('Must not open hardware'),stop)
        self.assertEqual(r['status'],'stopped')
    def test_stop_during_motion_holds_and_never_claims_arrived(self):
        stop=threading.Event();bus=Bus();now=[0]
        def wait(dt):now[0]+=dt;stop.set()
        r=run_standby(TARGET,lambda:bus,stop,clock=lambda:now[0],wait=wait)
        self.assertEqual(r['status'],'stopped');self.assertTrue(r['held']);self.assertNotEqual(bus.current,TARGET)
    def test_stuck_servos_timeout(self):
        r,bus=self.run_fake(stuck=True)
        self.assertEqual(r['status'],'timeout');self.assertTrue(r['held']);self.assertTrue(bus.closed)
    def test_model_target_from_training_not_current_teaching(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);final=root/'model/final';final.mkdir(parents=True);meta=root/'raw/metadata';meta.mkdir(parents=True)
            (final.parent/'run-manifest.json').write_text(json.dumps({'source':[str(meta.parent)]}))
            (meta/'episode_0.json').write_text(json.dumps({'version':{'waypoints':{'p1_standby':TARGET}}}))
            self.assertEqual(model_standby(final),TARGET)
            (meta/'episode_1.json').write_text(json.dumps({'version':{'waypoints':{'p1_standby':[2000]*6}}}))
            with self.assertRaises(ValueError):model_standby(final)


class ApiTests(unittest.TestCase):
    setUp=fixtures.VersionTests.setUp
    def prepare(self):
        self.stack.enter_context(patch.object(studio,'ROOT',self.root))
        self.stack.enter_context(patch.object(studio,'standby_stop_event',threading.Event()))
        studio.state.update(workbench_mode='teach',standby_running=False,act_running=False)
        self.store.set_points({'p1_standby':TARGET});studio.sync_version_state()
    def test_explicit_start_binds_context_and_only_calls_standby(self):
        self.prepare();spec=studio.get_standby();self.assertTrue(spec['ok'])
        def thread(*,target,args,daemon):
            class Immediate:
                def start(self):target(*args)
            return Immediate()
        with patch.object(studio.threading,'Thread',side_effect=thread), \
             patch('studio_standby.run_standby',return_value={'status':'completed','held':True,'message':'TEST'})as run, \
             patch.object(studio.rustypot,'Bus',side_effect=AssertionError('NO HARDWARE')), \
             patch.object(studio,'act_web_worker',side_effect=AssertionError('NO ACT')):
            self.assertTrue(studio.start_standby(spec['context'])['ok'])
        self.assertEqual(run.call_args.args[0],spec['target']);self.assertFalse(studio.motion_lock.locked());self.assertFalse(studio.state['act_running'])
    def test_context_busy_and_thread_failure_are_rejected(self):
        self.prepare();spec=studio.get_standby()
        with patch.object(studio.threading,'Thread')as thread:
            self.assertFalse(studio.start_standby('stale')['ok']);thread.assert_not_called()
            studio.state['recording']=True
            self.assertFalse(studio.start_standby(spec['context'])['ok']);studio.state['recording']=False
            thread.return_value.start.side_effect=RuntimeError('thread failed')
            self.assertFalse(studio.start_standby(spec['context'])['ok'])
        self.assertFalse(studio.motion_lock.locked())
    def test_stop_and_unload_request_dont_compete_for_bus(self):
        self.prepare();studio.state['standby_running']=True
        with patch.object(studio.rustypot,'Bus',side_effect=AssertionError('NO HARDWARE')):
            self.assertTrue(studio.stop_standby()['ok']);self.assertTrue(studio.standby_stop_event.is_set())
            self.assertIn('停止复位',studio.force_torque_off()['msg'])


if __name__=='__main__':unittest.main()
