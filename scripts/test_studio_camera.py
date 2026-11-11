"""No hardware: fake VideoCapture proves close releases, reopen creates a new handle."""
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
import teleop_studio as studio
import test_teaching_versions as fixtures


class CameraTests(unittest.TestCase):
    setUp=fixtures.VersionTests.setUp

    def prepare(self):
        for name in ('stop_camera','camera_wake','camera_released'):
            self.stack.enter_context(patch.object(studio,name,threading.Event()))
        studio.camera_released.set()
        self.stack.enter_context(patch.object(studio,'latest_jpeg',None))
        self.stack.enter_context(patch.object(studio,'latest_frame_time',0))
        studio.state.update(camera_enabled=False,camera_active=False,camera_error='',act_running=False)

    def wait_for(self,predicate):
        deadline=time.monotonic()+2
        while not predicate() and time.monotonic()<deadline:time.sleep(.01)
        self.assertTrue(predicate())

    def test_real_release_and_reopen_with_fake_device(self):
        self.prepare();devices=[]
        class Device:
            def __init__(self,*args):self.released=False;self.reads=0;devices.append(self)
            def isOpened(self):return True
            def set(self,*args):pass
            def read(self):self.reads+=1;return True,np.zeros((48,64,3),np.uint8)
            def release(self):self.released=True
        self.stack.enter_context(patch.object(studio.cv2,'VideoCapture',side_effect=Device))
        thread=threading.Thread(target=studio.camera_worker,daemon=True)
        def shutdown():
            studio.stop_camera.set();studio.camera_wake.set();thread.join(2)
            self.assertFalse(thread.is_alive())
        self.addCleanup(shutdown);thread.start()
        self.assertTrue(studio.set_camera_enabled(False)['ok'])
        self.assertEqual(devices,[])
        self.assertTrue(studio.set_camera_enabled(True)['ok'])
        self.wait_for(lambda:studio.latest_jpeg is not None)
        self.assertTrue(studio.state['camera_active'])
        self.assertTrue(studio.set_camera_enabled(False)['ok'])
        self.assertTrue(devices[0].released)
        self.assertIsNone(studio.latest_jpeg)
        self.assertEqual(studio.latest_frame_time,0)
        self.assertTrue(studio.get_state()['camera_released'])
        reads=devices[0].reads;time.sleep(.08)
        self.assertEqual(devices[0].reads,reads)
        with self.assertRaisesRegex(ValueError,'已关闭'):studio.act_frame_rgb()
        self.assertTrue(studio.set_camera_enabled(True)['ok'])
        self.wait_for(lambda:len(devices)==2 and studio.latest_jpeg is not None)
        self.assertFalse(devices[1].released)
        self.assertTrue(studio.set_camera_enabled(False)['ok'])
        self.assertTrue(devices[1].released)

    def test_motion_and_recording_block_toggle_but_training_does_not(self):
        self.prepare()
        for flag in ('recording','teleop_active','act_running'):
            studio.state[flag]=True
            self.assertFalse(studio.set_camera_enabled(True)['ok'])
            studio.state[flag]=False
        studio.motion_lock.acquire()
        try:self.assertFalse(studio.set_camera_enabled(True)['ok'])
        finally:studio.motion_lock.release()
        studio.state['training']=True
        self.assertTrue(studio.set_camera_enabled(False)['ok'])

    def test_open_failure_reports_error_and_releases(self):
        self.prepare()
        class Device:
            released=False
            def isOpened(self):return False
            def release(self):self.released=True
        device=Device()
        self.stack.enter_context(patch.object(studio.cv2,'VideoCapture',return_value=device))
        studio.state['camera_enabled']=True
        thread=threading.Thread(target=studio.camera_worker,daemon=True);thread.start()
        try:
            self.wait_for(lambda:bool(studio.state['camera_error']))
            self.wait_for(studio.camera_released.is_set)
            self.assertTrue(device.released)
            self.assertFalse(studio.state['camera_enabled'])
        finally:
            studio.stop_camera.set();studio.camera_wake.set();thread.join(2)
            self.assertFalse(thread.is_alive())


if __name__=='__main__':unittest.main()
