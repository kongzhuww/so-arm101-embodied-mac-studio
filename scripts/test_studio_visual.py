"""Offline image transformation and label-integrity tests. No hardware."""
import unittest
import cv2,numpy as np,torch
from studio_visual import BackgroundView,translate_batch,BatchView


class VisualTests(unittest.TestCase):
    def image(self):return np.random.default_rng(42).integers(0,256,(240,320,3),dtype=np.uint8)
    def test_background_translation_without_labels(self):
        image=self.image();view=BackgroundView.from_rgb(image)
        shifted=cv2.warpAffine(image,np.array([[1,0,24],[0,1,10]],np.float32),(320,240),borderMode=cv2.BORDER_REPLICATE)
        result=view.estimate(shifted)
        self.assertEqual((result['dx'],result['dy']),(24,10))
        aligned=view.transform_rgb(shifted)
        np.testing.assert_array_equal(aligned[20:200,20:280],image[20:200,20:280])
        self.assertEqual(view.offset['dx'],24)
        with self.assertRaises(ValueError):view.transform_rgb(np.zeros_like(image),cache=False)
    def test_training_tensor_transform_matches_inference_cv(self):
        image=self.image();tensor=torch.from_numpy(image).permute(2,0,1).float().unsqueeze(0)/255
        out=translate_batch(tensor,[{'dx':24,'dy':10}])
        expected=cv2.warpAffine(image,np.array([[1,0,24],[0,1,10]],np.float32),(320,240),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
        np.testing.assert_allclose(out[0].permute(1,2,0).numpy(),expected/255,atol=5e-5)
    def test_augmentation_never_changes_states_actions(self):
        state=torch.ones(1,6);action=torch.ones(1,50,6)
        batch={'episode_index':torch.tensor([0]),'observation.state':state,'action':action,
               'observation.images.external_front':torch.rand(1,3,240,320)}
        out=BatchView([{'dx':0,'dy':0}])(batch,augment=True)
        self.assertIs(out['observation.state'],state);self.assertIs(out['action'],action)
        self.assertTrue(torch.all((out['observation.images.external_front']>=0)&(out['observation.images.external_front']<=1)))

if __name__=='__main__':unittest.main()
