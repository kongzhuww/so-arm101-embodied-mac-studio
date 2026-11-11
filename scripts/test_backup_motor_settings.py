"""Offline tests; all serial objects are fake or patched."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import backup_motor_settings as b
import read_only_probe as base
from test_read_only_probe import response


class Fake:
    def __init__(self, corrupt=None, value_override=None):
        self.pending = bytearray()
        self.sent = []
        self.closed = False
        self.corrupt = corrupt
        self.value_override = value_override or {}

    def close(self):
        self.closed = True

    def reset_input_buffer(self):
        self.pending.clear()

    def write(self, packet):
        assert packet[:2] == b'\xff\xff' and packet[3:5] == b'\x04\x02'
        assert packet in b.ALLOWED and not self.closed
        self.sent.append(packet)
        motor, addr, size = packet[2], packet[5], packet[6]
        value = {3: 777, 5: motor, 40: 0, 55: 1, 33: 0, 18: 0, 31: 0, 9: 0, 11: 4095, 0: 2, 1: 42, 56: 2048}[addr]
        value = self.value_override.get((motor, addr), value)
        data = response(motor, value, size)
        self.pending.extend(self.corrupt(data) if self.corrupt else data)
        return len(packet)

    def read(self, size):
        data = bytes(self.pending[:size])
        del self.pending[:size]
        return data


class BackupTests(unittest.TestCase):
    def test_default_no_device(self):
        with patch.object(base, 'open_verified_port', side_effect=AssertionError('no open')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(b.main([]), 0)

    def test_flags_required(self):
        for flags in (['--execute'], ['--ack-read-only']):
            with patch.object(base, 'open_verified_port') as op, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    b.main(flags)
                op.assert_not_called()

    def test_original_gate_not_broadened(self):
        self.assertEqual(len(base.ALLOWED_PACKETS), 18)
        with self.assertRaises(base.ProbeError):
            base.request_packet(1, 'homing_offset')
        with self.assertRaises(base.ProbeError):
            base.validate_request(b.request_packet(1, 'homing_offset'))

    def test_only_read_and_fixed_registers(self):
        self.assertEqual(len(b.ALLOWED), 72)
        for p in b.ALLOWED:
            b.validate_packet(p)
            self.assertEqual(p[4], 2)
            self.assertEqual(sum(p[2:]) & 255, 255)
        for reg in ('goal_position', 'acceleration', 'baudrate'):
            with self.assertRaises(base.ProbeError):
                b.request_packet(1, reg)
        for bad in (b'', bytes.fromhex('ff ff 01 04 03 28 01 ce')):
            with self.assertRaises(base.ProbeError):
                b.validate_packet(bad)

    def test_collect_and_save(self):
        fake = Fake()
        snapshot = b.collect(b.BackupLink(fake))
        self.assertEqual(len(fake.sent), 78)
        self.assertEqual(set(snapshot['motors']), {'1','2','3','4','5','6'})
        self.assertEqual(snapshot['motors']['1']['max_position_limit'], 4095)
        with tempfile.TemporaryDirectory() as d:
            path = b.save_snapshot(snapshot, Path(d))
            self.assertEqual(json.loads(path.read_text())['motors'], snapshot['motors'])

    def test_wrong_model_stops(self):
        fake = Fake(value_override={(1,3): 888})
        with self.assertRaises(base.ProbeError):
            b.collect(b.BackupLink(fake))
        self.assertEqual(len(fake.sent), 1)

    def test_torque_on_stops(self):
        fake = Fake(value_override={(1,40): 1})
        with self.assertRaises(base.ProbeError):
            b.collect(b.BackupLink(fake))
        self.assertEqual(len(fake.sent), 2)

    def test_corrupt_status_checksum_and_timeout(self):
        for corrupt in (lambda _: b'', lambda x: x[:-1], lambda x: x[:-1] + bytes([x[-1] ^ 1]), lambda _: response(1,777,2,4), lambda x: b'\x00' + x[1:]):
            with self.assertRaises(base.ProbeError):
                b.collect(b.BackupLink(Fake(corrupt=corrupt)))

    def test_id_register_mismatch(self):
        with self.assertRaisesRegex(base.ProbeError, 'ID register mismatch'):
            b.collect(b.BackupLink(Fake(value_override={(1,5): 2})))

    def test_main_failure_closes_without_saving(self):
        fake = Fake(value_override={(1,3): 0})
        with patch.object(base,'open_verified_port',return_value=fake), patch.object(b,'save_snapshot') as save, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(b.main(['--execute','--ack-read-only']), 1)
            save.assert_not_called()
        self.assertTrue(fake.closed)

    def test_main_success_closes_before_saving(self):
        fake = Fake()
        def save(snapshot, directory):
            self.assertTrue(fake.closed)
            return 'mock-snapshot.json'
        with patch.object(base,'open_verified_port',return_value=fake), patch.object(b,'save_snapshot',side_effect=save), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(b.main(['--execute','--ack-read-only']), 0)
        self.assertTrue(fake.closed)


if __name__ == '__main__':
    unittest.main()
