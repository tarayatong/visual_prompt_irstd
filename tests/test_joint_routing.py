import unittest
import numpy as np
import torch
from torch.nn import functional as F

from scripts.run_joint_routing_study import evaluate
from visual_prompt.models.lightweight_heads import ObjectnessMLP,soft_iou_loss


class JointRoutingTests(unittest.TestCase):
    def test_segmentation_gradient_reaches_objectness_only_when_coupled(self):
        torch.manual_seed(111);model=ObjectnessMLP(7)
        f=torch.randn(2,7);z=torch.randn(2,8,8,requires_grad=True);target=torch.zeros(2,8,8);target[:,4,4]=1
        independent=soft_iou_loss(z,target)
        self.assertTrue(all(g is None for g in torch.autograd.grad(independent,list(model.parameters()),allow_unused=True)))
        obj=model(f);joint=soft_iou_loss(z+.25*obj[:,None,None],target)
        gradients=torch.autograd.grad(joint,list(model.parameters()))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0)
        # Additive conditioning preserves within-ROI logit differences.
        fused=z+.25*obj[:,None,None]
        torch.testing.assert_close(fused[:,1,1]-fused[:,4,4],z[:,1,1]-z[:,4,4])

    def test_rejecting_roi_still_counts_missed_target_and_full_background(self):
        gt=np.zeros((80,80),np.uint8);gt[40,40]=1
        z=np.full((1,64,64),-10.,np.float32);z[0,32,32]=2;z[0,32,33]=2
        rows=[dict(sample_id='a',y=40,x=40)];objects=np.array([-4.])
        result,_=evaluate(rows,z,objects,{'a':gt},dict(id='none',kind='bypass',nms=0,threshold=0,weight=0),'model')
        self.assertEqual(result['miou_percent'],50)
        self.assertAlmostEqual(result['fa_per_million'],1e6/6400)
        rejected,_=evaluate(rows,z,objects,{'a':gt},dict(id='hard',kind='hard',nms=0,threshold=.9,weight=0),'model')
        self.assertEqual(rejected['pd_percent'],0)
        self.assertEqual(rejected['miou_percent'],0)
        self.assertEqual(rejected['targets'],1)


if __name__=='__main__':unittest.main()
