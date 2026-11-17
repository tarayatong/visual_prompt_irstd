import unittest
import torch
from torch.nn import functional as F
from visual_prompt.models.radial_adaptive_heads import AdaptiveRadialMask, AdaptiveRadialObjectness


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(111)

    def test_obj_projection_and_auxiliary_learn_and_zero_at_concat(self):
        model = AdaptiveRadialObjectness(11, dict(projection='nonlinear', auxiliary='radial_classification'), f4_channels=7)
        f4, ray, target = torch.randn(3, 7), torch.randn(3, 11), torch.tensor([0., 1., 0.])
        logits = model(f4, ray)
        loss = F.binary_cross_entropy_with_logits(logits, target)+model.auxiliary_loss(target)
        loss.backward()
        self.assertGreater(float(model.project[0].weight.grad.abs().sum()), 0)
        expected = model.main(torch.cat((f4, torch.zeros(3, 32)), 1))
        torch.testing.assert_close(model.forward_radial_zero(f4, ray), expected)

    def test_mask_auxiliaries_and_candidate_zero_intervention(self):
        e0 = F.normalize(torch.randn(7), dim=0)
        features = torch.randn(4, 18, 8, 8); target = torch.zeros(4, 8, 8); target[:2, 3:5, 3:5] = 1
        for kind in ('none', 'radial_mask', 'hard_negative', 'edge'):
            model = AdaptiveRadialMask(11, e0, dict(projection='nonlinear', condition='candidate', auxiliary=kind), sam_channels=7)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            for _ in range(3):
                logits = model.forward_features(features, e0)
                loss = F.binary_cross_entropy_with_logits(logits, target)+model.auxiliary_loss(logits, target, 2)
                self.assertTrue(torch.isfinite(loss))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            self.assertGreater(float(model.project[0].weight.grad.abs().sum()), 0)
            zero = model.forward_features(features, e0, zero_radial=True)
            expected = model.main(features[:, :7], torch.zeros(4, 32, 8, 8), e0)
            torch.testing.assert_close(zero, expected)

    def test_uniform_zero_radial_projection_has_no_added_bias(self):
        e0 = F.normalize(torch.randn(7), dim=0)
        for mode in ('linear', 'nonlinear'):
            model = AdaptiveRadialMask(11, e0, dict(projection=mode), sam_channels=7)
            torch.testing.assert_close(model.radial_representation(torch.zeros(2, 11, 8, 8)), torch.zeros(2, 32, 8, 8))

    def test_bounded_projection_preserves_gradients_and_final_zero_intervention(self):
        e0 = F.normalize(torch.randn(7), dim=0)
        model = AdaptiveRadialMask(11, e0, dict(projection='nonlinear',condition='candidate',output_norm=.25), sam_channels=7)
        features = torch.randn(2,18,8,8)
        representation = model.radial_representation(features[:,7:])
        torch.testing.assert_close(representation.norm(dim=1),torch.full((2,8,8),.25))
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
        for _ in range(3):
            logits = model.forward_features(features,e0)
            optimizer.zero_grad();logits.square().mean().backward();optimizer.step()
        self.assertGreater(float(model.project[0].weight.grad.abs().sum()),0)
        expected = model.main(features[:,:7],torch.zeros_like(representation),e0)
        torch.testing.assert_close(model.forward_features(features,e0,zero_radial=True),expected)


if __name__ == '__main__':
    unittest.main()
