"""Offline tests: synthetic fixtures only; never opens camera or serial."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from studio_samples import decode_sample, resample_episode, trim_startup
import teleop_studio as studio


def rows():
    result = []
    for i in range(15):
        ok, jpg = cv2.imencode('.jpg', np.full((240, 320, 3), i * 10, np.uint8))
        assert ok
        result.append({'schema_version': 2, 'timestamp': 100 + i * .1,
                       'image_timestamp': 100 + i * .1 - .01,
                       'command_timestamp': 100 + i * .1 + .01,
                       'state': [2000 + i] * 6, 'action': [2010 + i] * 6,
                       'image_jpeg': base64.b64encode(jpg).decode('ascii')})
    return result


class RecordingTests(unittest.TestCase):
    def test_decode_and_resample(self):
        data = rows()
        self.assertEqual(len(resample_episode(data)), 15)
        self.assertEqual(decode_sample(data[0]).shape, (240, 320, 3))
        self.assertFalse(np.array_equal(decode_sample(data[0]), decode_sample(data[-1])))

    def test_reject_legacy_stale_corrupt_and_gaps(self):
        for kind in ('legacy', 'stale', 'corrupt', 'gap'):
            data = rows()
            if kind == 'legacy': del data[0]['schema_version']
            if kind == 'stale': data[0]['image_timestamp'] = 90
            if kind == 'corrupt': data[0]['image_jpeg'] = 'invalid'
            if kind == 'gap': data = data[:3] + data[8:]
            with self.subTest(kind=kind), self.assertRaises((ValueError, KeyError)):
                resample_episode(data)

    def test_capture_follower_and_exact_command(self):
        class Bus:
            def read_register_with_error(self, motor, register, retries):
                return 1000 + motor, 0
        with patch.object(studio, 'latest_jpeg', base64.b64decode(rows()[0]['image_jpeg'])), \
             patch.object(studio, 'latest_frame_time', 200), \
             patch.object(studio.time, 'time', return_value=200.01):
            row = studio.capture_sample(Bus(), [2100.9] * 6)
        self.assertEqual(row['state'], [1001, 1002, 1003, 1004, 1005, 1006])
        self.assertEqual(row['action'], [2100] * 6)
        decode_sample(row)

    def test_train_refuses_legacy_before_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'episode_0000.json').write_text(json.dumps([{'state': [0]*6}] * 12))
            with patch.object(studio, 'DATASET_DIR', path), \
                 patch.object(studio.threading, 'Thread') as thread:
                result = studio.train_start()
                self.assertFalse(result['ok'])
                thread.assert_not_called()

    def test_trim_keeps_real_alignment_and_source(self):
        data = rows()
        for i, r in enumerate(data):
            r['action'][1] = 1025 if i < 6 else 1200
        original = json.dumps(data)
        cropped = trim_startup(data, lead_seconds=0.3)
        self.assertAlmostEqual(cropped[0]['timestamp'], 100.3)
        self.assertIs(cropped[0], data[3])
        self.assertEqual(json.dumps(data), original)
        with self.assertRaises(ValueError):
            trim_startup(data, departure_counts=1000)

    def test_save_reload(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(studio, 'DATASET_DIR', Path(temp)):
                studio.save_episode(rows())
                data = json.loads((Path(temp) / 'episode_0000.json').read_text())
                self.assertEqual(data[0]['action'], [2010] * 6)
                self.assertEqual(len(resample_episode(data)), 15)


if __name__ == '__main__':
    unittest.main()
