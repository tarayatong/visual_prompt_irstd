import unittest

import torch
from torch.nn import functional as F

from visual_prompt.models.lightweight_heads import MaskVectorMLP, ObjectnessMLP
from visual_prompt.models.radial_augmented_heads import F4RadialObjectness, RadialMaskVectorMLP, RadialResidualObjectness
from visual_prompt.models.residual_objectness import ResidualObjectness


class AugmentedHeadsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_mask_initialization_keeps_original_descriptor_and_output(self):
        e0 = F.normalize(torch.randn(7), dim=0)
        torch.manual_seed(111); original = MaskVectorMLP(7, 128, e0, 'm2_residual')
        torch.manual_seed(111); new = RadialMaskVectorMLP(7, 11, e0)
        self.assertTrue(torch.equal(new.net[0].weight[:, :7], original.net[0].weight[:, :7]))
        self.assertTrue(torch.equal(new.net[0].weight[:, 18:25], original.net[0].weight[:, 7:14]))
        self.assertEqual(float(new.net[0].weight[:, 7:18].abs().sum()), 0)
        sam = F.normalize(torch.randn(2, 7, 8, 8),dim=1); radial = torch.randn(2,11,8,8)
        sim = torch.einsum('bchw,c->bhw',sam,e0); z = torch.zeros_like(sim)
        torch.testing.assert_close(new(sam,radial,e0), original(sam,e0,sim,z,z,.25,.1),atol=1e-6,rtol=1e-6)

    def test_radial_mask_columns_learn_without_changing_sam_or_e0(self):
        e0 = F.normalize(torch.randn(7),dim=0); before=e0.clone()
        model=RadialMaskVectorMLP(7,11,e0);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        sam=F.normalize(torch.randn(2,7,8,8),dim=1);radial=torch.randn(2,11,8,8)
        for _ in range(3):
            loss=F.binary_cross_entropy_with_logits(model(sam,radial,e0),torch.rand(2,8,8))
            optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertTrue(torch.equal(e0,before))
        self.assertGreater(float(model.net[0].weight[:,7:18].abs().sum()),0)
        self.assertGreater(float(model.net[-1].weight[7:18].abs().sum()),0)

    def test_obj_original_path_and_frozen_base(self):
        base=ObjectnessMLP(1440).state_dict()
        torch.manual_seed(111);old=ResidualObjectness(base)
        torch.manual_seed(111);new=RadialResidualObjectness(base,11)
        layers=[torch.randn(2,c,32,32) for c in (32,64,256)];raw=torch.rand(2,1,64,64);f4=torch.randn(2,1440)
        radial=torch.randn(2,11)
        torch.testing.assert_close(new(layers,raw,f4,radial),old(layers,raw,f4),atol=0,rtol=0)
        self.assertTrue(torch.equal(new.branch.mlp[0].weight[:,:96],old.branch.mlp[0].weight))
        optimizer=torch.optim.AdamW([p for p in new.parameters() if p.requires_grad],lr=.001)
        for _ in range(3):
            optimizer.zero_grad();F.binary_cross_entropy_with_logits(new(layers,raw,f4,radial),torch.tensor([1.,0.])).backward();optimizer.step()
        self.assertTrue(all(torch.equal(v,new.base.state_dict()[k]) for k,v in base.items()))
        self.assertGreater(float(new.branch.mlp[0].weight[:,96:].abs().sum()),0)

    def test_direct_obj_updates_both_feature_paths_and_has_no_residual_bound(self):
        model = F4RadialObjectness()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 217345)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        f4, radial = torch.randn(4, 1440), torch.randn(4, 256)
        before = model.net[0].weight.detach().clone()
        loss = F.binary_cross_entropy_with_logits(model(f4, radial), torch.tensor([0., 1., 0., 1.]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertGreater(float((model.net[0].weight[:, :1440] - before[:, :1440]).abs().sum()), 0)
        self.assertGreater(float((model.net[0].weight[:, 1440:] - before[:, 1440:]).abs().sum()), 0)
        with torch.no_grad():
            model.net[-1].weight.zero_(); model.net[-1].bias.fill_(10.)
        torch.testing.assert_close(model(f4, radial), torch.full((4,), 10.))


if __name__=='__main__':
    unittest.main()
