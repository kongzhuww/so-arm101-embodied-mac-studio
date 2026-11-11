"""Fake-bus checks; no serial/camera/EEPROM accessed."""
import copy,json,tempfile,unittest
from pathlib import Path
from repair_gripper_zero import repair,NAMES


class Bus:
    def __init__(self,ignore_offset=False,voltage=123):
        self.cal={n:{'id':i,'drive_mode':0,'homing_offset':-288 if i==6 else 0,'range_min':2007 if i==6 else 0,'range_max':3438 if i==6 else 4095}for i,n in enumerate(NAMES,1)}
        self.r={i:{'model_number':777,'id':i,'torque_enable':1,'lock':1,'operating_mode':0,'phase':12,
                   'homing_offset':self.cal[n]['homing_offset'],'min_position_limit':self.cal[n]['range_min'],
                   'max_position_limit':self.cal[n]['range_max'],'goal_position':2099,'present_position':2000,'present_voltage':voltage}
                for i,n in enumerate(NAMES,1)}
        self.ignore_offset=ignore_offset;self.writes=[]
    def read_register_with_error(self,motor,name,retries):
        if motor==6 and name=='present_position':return 707-self.r[6]['homing_offset'],0
        return self.r[motor][name],0
    def write_register_with_error(self,motor,name,value,retries):
        self.writes.append((motor,name,value))
        if not(self.ignore_offset and name=='homing_offset'):self.r[motor][name]=value
        return 0


class RepairTests(unittest.TestCase):
    def test_only_six_offset_and_off_and_lock_written(self):
        bus=Bus();before=copy.deepcopy(bus.r)
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp)/'before.json';r=repair(bus,bus.cal,p,sign=-1)
            self.assertEqual(r['status'],'verified');self.assertTrue(p.exists())
        self.assertEqual(bus.writes,[(6,'torque_enable',0),(6,'lock',0),(6,'homing_offset',-1326),(6,'lock',1)])
        self.assertEqual(r['position_after']-r['position_before'],1038)
        for i in range(1,6):self.assertEqual(before[i],bus.r[i])
        self.assertEqual(bus.r[6]['goal_position'],2099)
    def test_offset_mismatch_never_retries_or_reenables(self):
        bus=Bus(ignore_offset=True)
        with tempfile.TemporaryDirectory()as temp:
            with self.assertRaisesRegex(RuntimeError,'readback mismatch'):repair(bus,bus.cal,Path(temp)/'before.json',sign=-1)
        self.assertEqual(len([w for w in bus.writes if w[1]=='homing_offset']),1)
        self.assertEqual(bus.r[6]['lock'],1);self.assertEqual(bus.r[6]['torque_enable'],0)
    def test_low_power_no_writes(self):
        bus=Bus(voltage=52)
        with tempfile.TemporaryDirectory()as temp:
            with self.assertRaisesRegex(RuntimeError,'12V'):repair(bus,bus.cal,Path(temp)/'before.json',sign=-1)
        self.assertEqual(bus.writes,[])

if __name__=='__main__':unittest.main()
