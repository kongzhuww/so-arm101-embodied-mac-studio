"""Offline only: never instantiates pyserial or opens a device."""
import contextlib
import io
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import read_only_probe as p


def response(motor_id, value, size, status=0):
    body = bytes((motor_id, size + 2, status)) + value.to_bytes(size, "little")
    return b"\xff\xff" + body + bytes(((~sum(body)) & 255,))


class FakeSerial:
    def __init__(self, override=None, mutate=None, partial_write=False, chunk=1):
        self.override = override or {}
        self.mutate = mutate
        self.partial_write = partial_write
        self.chunk = chunk
        self.sent = []
        self.pending = bytearray()
        self.closed = False

    def reset_input_buffer(self):
        self.pending.clear()

    def write(self, packet):
        # Independent assertions at the simulated transport boundary.
        assert not self.closed
        assert len(packet) == 8 and packet[:2] == b"\xff\xff"
        assert packet[2] in range(1, 7) and packet[3:5] == b"\x04\x02"
        assert (packet[5], packet[6]) in {(3, 2), (40, 1), (56, 2)}
        assert sum(packet[2:]) & 255 == 255
        self.sent.append(packet)
        motor_id, address, size = packet[2], packet[5], packet[6]
        value = self.override.get((motor_id, address), {3: 777, 40: 0, 56: 2048}[address])
        data = response(motor_id, value, size)
        if self.mutate:
            data = self.mutate(data)
        self.pending.extend(data)
        return len(packet) - 1 if self.partial_write else len(packet)

    def read(self, size):
        n = min(size, self.chunk, len(self.pending))
        data = bytes(self.pending[:n])
        del self.pending[:n]
        return data

    def close(self):
        self.closed = True


class SafetyTests(unittest.TestCase):
    def test_known_read_packet(self):
        self.assertEqual(p.request_packet(1, "model_number"), bytes.fromhex("ff ff 01 04 02 03 02 f3"))

    def test_exact_whitelist_only(self):
        self.assertEqual(len(p.ALLOWED_PACKETS), 18)
        for packet in p.ALLOWED_PACKETS:
            p.validate_request(packet)
            # Every one-byte mutation is blocked (unless it becomes another exact allowed READ).
            for i in range(len(packet)):
                for value in range(256):
                    changed = packet[:i] + bytes((value,)) + packet[i + 1:]
                    if changed not in p.ALLOWED_PACKETS:
                        with self.assertRaises(p.ProbeError):
                            p.validate_request(changed)

    def test_write_and_broadcast_blocked(self):
        for packet in (bytes.fromhex("ff ff 01 04 03 28 01 ce"), b"\xff\xff\xfe\x02\x01\xfe", b"", b"reset"):
            with self.assertRaises(p.ProbeError):
                p.validate_request(packet)

    def test_invalid_id_and_register(self):
        for motor_id in (0, 7, 254, -1, True, "1"):
            with self.assertRaises(p.ProbeError):
                p.request_packet(motor_id, "model_number")
        for register in ("goal_position", "id", "baudrate", "lock", "operating_mode"):
            with self.assertRaises(p.ProbeError):
                p.request_packet(1, register)

    def test_default_no_hardware(self):
        with patch.object(p, "open_verified_port", side_effect=AssertionError("must not open")), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(p.main([]), 0)
        self.assertIn("PLAN_ONLY_NO_SERIAL_ACCESS", out.getvalue())

    def test_both_execution_flags_required(self):
        for args in (["--execute"], ["--ack-power-risk"]):
            with patch.object(p, "open_verified_port", side_effect=AssertionError("must not open")), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    p.main(args)
                self.assertEqual(caught.exception.code, 2)

    def test_success_only_18_reads(self):
        fake = FakeSerial()
        lines = []
        p.probe(p.ReadOnlyLink(fake), emit=lines.append)
        self.assertEqual(len(fake.sent), 18)
        self.assertTrue(all(packet[5] == 3 for packet in fake.sent[:6]))
        self.assertEqual(len(lines), 18)

    def test_model_mismatch_stops_immediately(self):
        fake = FakeSerial(override={(2, 3): 999})
        with self.assertRaisesRegex(p.ProbeError, "expected model"):
            p.probe(p.ReadOnlyLink(fake), emit=lambda _: None)
        self.assertEqual(len(fake.sent), 2)

    def test_nonzero_torque_stops_without_cleanup_writes(self):
        for torque in (1, 2, 255):
            fake = FakeSerial(override={(1, 40): torque})
            with self.assertRaisesRegex(p.ProbeError, "WITHOUT disabling"):
                p.probe(p.ReadOnlyLink(fake), emit=lambda _: None)
            self.assertEqual(len(fake.sent), 7)
            self.assertEqual(fake.sent[-1][5], 40)

    def test_signed_magnitude_raw_position(self):
        fake = FakeSerial(override={(1, 56): 0x8064})
        self.assertEqual(p.ReadOnlyLink(fake).read_register(1, "present_position"), -100)

    def test_partial_transmission_no_retry(self):
        fake = FakeSerial(partial_write=True)
        with self.assertRaisesRegex(p.ProbeError, "Partial request"):
            p.ReadOnlyLink(fake).read_register(1, "model_number")
        self.assertEqual(len(fake.sent), 1)

    def test_timeout_and_truncation(self):
        for mutate in (lambda _: b"", lambda data: data[:-1]):
            fake = FakeSerial(mutate=mutate)
            with self.assertRaisesRegex(p.ProbeError, "Missing/truncated"):
                p.ReadOnlyLink(fake).read_register(1, "model_number")
            self.assertEqual(len(fake.sent), 1)

    def test_response_deadline(self):
        fake = FakeSerial()
        with patch.object(p.time, "monotonic", side_effect=[0, 1]):
            with self.assertRaisesRegex(p.ProbeError, "deadline"):
                p.ReadOnlyLink(fake).read_register(1, "model_number")

    def test_wrong_header_id_length(self):
        for mutate in (
            lambda x: b"\x00" + x[1:],
            lambda x: x[:2] + b"\x02" + x[3:],
            lambda x: x[:3] + b"\xff" + x[4:],
        ):
            with self.assertRaisesRegex(p.ProbeError, "header/ID/length"):
                p.ReadOnlyLink(FakeSerial(mutate=mutate)).read_register(1, "model_number")

    def test_checksum_error(self):
        with self.assertRaisesRegex(p.ProbeError, "checksum"):
            p.ReadOnlyLink(FakeSerial(mutate=lambda x: x[:-1] + bytes((x[-1] ^ 1,)))).read_register(1, "model_number")

    def test_alarm_status(self):
        with self.assertRaisesRegex(p.ProbeError, "status/alarm"):
            p.ReadOnlyLink(FakeSerial(mutate=lambda _: response(1, 777, 2, status=0x20))).read_register(1, "model_number")

    def test_execute_success_closes_fake(self):
        fake = FakeSerial()
        with patch.object(p, "open_verified_port", return_value=fake), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(p.main(["--execute", "--ack-power-risk"]), 0)
        self.assertTrue(fake.closed)
        self.assertEqual(len(fake.sent), 18)

    def test_execute_failure_closes_fake(self):
        fake = FakeSerial(override={(1, 3): 0})
        with patch.object(p, "open_verified_port", return_value=fake), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(p.main(["--execute", "--ack-power-risk"]), 1)
        self.assertTrue(fake.closed)
        self.assertEqual(len(fake.sent), 1)

    def test_wrong_usb_identity_never_constructs_serial(self):
        devices = [SimpleNamespace(device=p.PORT, vid=0x1234, pid=0x5678)]
        with patch("serial.tools.list_ports.comports", return_value=devices), patch("serial.Serial") as factory:
            with self.assertRaises(p.ProbeError):
                p.open_verified_port()
            factory.assert_not_called()

    def test_missing_port_never_constructs_serial(self):
        with patch("serial.tools.list_ports.comports", return_value=[]), patch("serial.Serial") as factory:
            with self.assertRaises(p.ProbeError):
                p.open_verified_port()
            factory.assert_not_called()

    def test_serial_setup_does_not_send_bytes(self):
        devices = [SimpleNamespace(device=p.PORT, vid=p.USB_ID[0], pid=p.USB_ID[1])]
        with patch("serial.tools.list_ports.comports", return_value=devices), patch("serial.Serial") as factory:
            port = p.open_verified_port()
            kwargs = factory.call_args.kwargs
            self.assertIsNone(kwargs["port"])
            self.assertEqual(kwargs["baudrate"], 1_000_000)
            self.assertTrue(kwargs["exclusive"])
            self.assertFalse(port.dtr)
            self.assertFalse(port.rts)
            self.assertEqual(port.port, p.PORT)
            port.open.assert_called_once_with()
            port.write.assert_not_called()

    def test_serial_open_failure_closes_without_writes(self):
        devices = [SimpleNamespace(device=p.PORT, vid=p.USB_ID[0], pid=p.USB_ID[1])]
        with patch("serial.tools.list_ports.comports", return_value=devices), patch("serial.Serial") as factory:
            port = factory.return_value
            port.open.side_effect = OSError("simulated failure")
            with self.assertRaises(OSError):
                p.open_verified_port()
            port.close.assert_called_once_with()
            port.write.assert_not_called()

    def test_keyboard_interrupt_closes_fake(self):
        fake = FakeSerial()
        with patch.object(p, "open_verified_port", return_value=fake), patch.object(p, "probe", side_effect=KeyboardInterrupt), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(p.main(["--execute", "--ack-power-risk"]), 130)
        self.assertTrue(fake.closed)
        self.assertEqual(fake.sent, [])


if __name__ == "__main__":
    unittest.main()
