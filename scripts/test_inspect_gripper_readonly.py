import contextlib
import io
import unittest
from unittest.mock import patch
import inspect_gripper_readonly as g


class FakeReadOnlyBus:
    def __init__(self,override=None,status=0):
        self.override=override or {}
        self.status=status
        self.requests=[]

    def read_register_with_error(self,motor_id,name,retries=0):
        self.requests.append((motor_id,name,retries))
        value={'model_number':777,'torque_enable':0,'present_position':2400,'id':motor_id,'operating_mode':0,'phase':12,'status':0}.get(name,0)
        return self.override.get((motor_id,name),value),self.status


class GripperReadTests(unittest.TestCase):
    def test_default_does_not_construct_bus(self):
        with patch.object(g.rustypot,'Bus',side_effect=AssertionError('no hardware')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(g.main([]),0)

    def test_reads_only_and_fixed_motor6_settings(self):
        raw=FakeReadOnlyBus()
        report=g.collect(raw)
        self.assertEqual(len(report['baseline']),6)
        self.assertTrue(all(i==6 for i,name,_ in raw.requests if name in g.REGISTERS))
        self.assertTrue(all(retries==0 for _,_,retries in raw.requests))

    def test_torque_or_model_or_alarm_stops(self):
        for raw in (FakeReadOnlyBus({(1,'torque_enable'):1}),FakeReadOnlyBus({(1,'model_number'):999}),FakeReadOnlyBus(status=1)):
            with self.assertRaises(g.base.ProbeError):
                g.collect(raw)

    def test_unknown_read_blocked(self):
        raw=FakeReadOnlyBus()
        with self.assertRaises(g.base.ProbeError):
            g.read(raw,6,'unknown')
        self.assertEqual(raw.requests,[])


if __name__=='__main__':
    unittest.main()
