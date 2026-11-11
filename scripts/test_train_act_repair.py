"""Offline split provenance and validation leakage/padding regression tests."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from train_act_repair import prepare,evaluate
from test_studio_recording import rows


class RepairTests(unittest.TestCase):
    def test_split_uses_cropped_source_and_excludes_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=[root/f'original{g}' for g in (1,2,3)];cropped=root/'cropped'
            cropped.mkdir();content=json.dumps(rows()).encode()
            for directory in original:
                (directory/'metadata').mkdir(parents=True)
                for i in range(10):
                    (directory/f'episode_{i:04d}.json').write_bytes(content)
                    (directory/'metadata'/f'episode_{i:04d}.json').write_text(json.dumps({'version':{'name':'fixture'}}))
            for i in range(10):(cropped/f'episode_{i:04d}.json').write_bytes(content)
            for ident,manifest in [('act_20261106T174035Z',{'source':str(cropped)}),('act_20261107T070132Z',{'source':[str(p)for p in original]})]:
                target=root/'outputs'/ident;target.mkdir(parents=True);(target/'run-manifest.json').write_text(json.dumps(manifest))
            derived=root/'derived';manifest=prepare(root,derived)
            self.assertEqual(manifest['train_episodes'],23);self.assertEqual(manifest['validation_episodes'],6)
            records=manifest['records']
            self.assertTrue(all(Path(r['source']).parent==cropped for r in records if r['group']==1))
            train={(r['group'],r['episode'])for r in records if r['split']=='train'}
            val={(r['group'],r['episode'])for r in records if r['split']=='validation'}
            self.assertFalse(train&val);self.assertNotIn((1,0),train|val)
            self.assertEqual(val,{(g,e)for g in (1,2,3)for e in (8,9)})
            with self.assertRaises(ValueError):prepare(root,derived)

    def test_validation_excludes_padded_targets_and_future_actions_from_model(self):
        names=['pan','lift','elbow','wrist','roll','gripper'];cal={n:{'range_min':0,'range_max':4095}for n in names}
        class Policy:
            def eval(self):return self
            def predict_action_chunk(self,batch):
                assert set(batch)=={'observation.state','observation.images.external_front'}
                return torch.zeros((len(batch['observation.state']),50,6))
        samples=[]
        for g in (1,2,3):
            target=torch.full((50,6),10000.);target[0]=0
            pad=torch.ones(50,dtype=torch.bool);pad[0]=False
            samples.append((g,{'observation.state':torch.zeros(6),'observation.images.external_front':torch.zeros(3,8,8),
                'action':target,'action_is_pad':pad}))
        result=evaluate(Policy(),lambda x:x,lambda x:x,samples,cal,names)
        self.assertEqual(result['valid_future_targets'],3)
        self.assertEqual(result['selection_arm_mae'],0)
        self.assertEqual(result['joint_mae'],[0]*6)


if __name__=='__main__':unittest.main()
