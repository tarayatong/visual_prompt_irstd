"""Check the baseline-preserving initialization and sparse boundary supervision."""
import unittest
import torch
from torch.nn import functional as F

from visual_prompt.models.irstd_detail_refinement import DetailMask, band_bce, boundary_band


class DetailRefinementTests(unittest.TestCase):
    def test_residual_starts_as_exact_baseline_and_can_learn(self):
        torch.manual_seed(1)
        e0 = F.normalize(torch.randn(352), dim=0)
        torch.manual_seed(111)
        base = DetailMask(e0, False)
        torch.manual_seed(111)
        detail = DetailMask(e0, True)
        for key, value in base.base.state_dict().items():
            self.assertTrue(torch.equal(value, detail.base.state_dict()[key]))
        feature = torch.randn(2, 416, 16, 16)
        gray = torch.rand(2, 1, 16, 16)
        expected, actual = base(feature, gray, e0), detail(feature, gray, e0)
        self.assertTrue(torch.equal(expected, actual))
        actual.square().mean().backward()
        self.assertGreater(float(detail.output.weight.grad.abs().sum()), 0)

    def test_band_excludes_invalid_and_does_not_depend_on_prediction(self):
        gt = torch.zeros(1, 9, 9)
        gt[0, 4, 4] = 1
        valid = torch.ones_like(gt)
        band = boundary_band(gt, valid)
        self.assertEqual(int(band.sum()), 9)
        logits = torch.zeros_like(gt, requires_grad=True)
        loss = band_bce(logits, gt, valid)
        loss.backward()
        self.assertEqual(int((logits.grad != 0).sum()), 9)
        # If no valid band remains, retain a finite differentiable zero.
        zero_loss = band_bce(logits, gt, torch.zeros_like(valid))
        self.assertEqual(float(zero_loss), 0)
        self.assertTrue(torch.isfinite(zero_loss))

    def test_empty_target_does_not_create_edge_supervision(self):
        gt = torch.zeros(2, 9, 9)
        logits = torch.randn_like(gt, requires_grad=True)
        self.assertEqual(float(band_bce(logits, gt, torch.ones_like(gt))), 0)


if __name__ == '__main__':
    unittest.main()
