import unittest

import numpy as np
import torch
from torch.nn import functional as F

from visual_prompt.evaluation.radial_diagnostics import RadialDiagnostics, masked_channel_statistics
from visual_prompt.models.radial_augmented_heads import RadialMaskVectorMLP
from scripts.run_radial_independent_search import mask_descriptor, cached_mask_forward, transform_radial, component_indices
import scripts.run_radial_independent_search as search


class RadialDiagnosticsTests(unittest.TestCase):
    def test_component_slices_keep_type_direction_radius_order(self):
        channels = np.arange(256).reshape(4, 8, 8)
        np.testing.assert_equal(np.asarray(component_indices('B1')), channels[1].ravel())
        np.testing.assert_equal(np.asarray(component_indices('B2')), channels[1:3].ravel())
        np.testing.assert_equal(np.asarray(component_indices('B3')), channels[1:4].ravel())

    def test_opposite_pair_encoding_is_invertible_when_valid(self):
        original = search.GEOMETRY
        search.GEOMETRY = dict(original, opposite_pairs=True)
        try:
            x = torch.randn(3, 256); valid = torch.ones_like(x, dtype=torch.bool)
            result, mask = search.TrainOnlyRadialBank.encode_pairs(x, valid)
            values = result.reshape(3, 4, 8, 8)
            recovered = torch.cat((values[:, :, :4]+values[:, :, 4:], values[:, :, :4]-values[:, :, 4:]), 2)
            torch.testing.assert_close(recovered.reshape_as(x), x)
            self.assertTrue(mask.all())
            v2 = x[0, :, None, None].expand(-1, 5, 5)
            y2, _ = search.TrainOnlyRadialBank.encode_pairs(v2, torch.ones_like(v2, dtype=torch.bool))
            torch.testing.assert_close(y2[:, 2, 2], result[0])
        finally:
            search.GEOMETRY = original

    def test_image_balancing_and_paired_margin(self):
        d = RadialDiagnostics()
        d.add('a', 'target', np.full((50, 1), 2.), np.full(50, 4.), np.full(50, 1.))
        d.add('b', 'target', [[4.]], [6.], [3.])
        d.add('a', 'background', [[0.]], [-2.], [-1.])
        d.add('b', 'background', [[0.]], [-2.], [-1.])
        r = d.result()
        self.assertAlmostEqual(r['Fisher_J'], 9/(1+1e-6))
        self.assertAlmostEqual(r['margin_gain'], 4.)
        self.assertAlmostEqual(r['target_logit_change'], 3.)
        self.assertAlmostEqual(r['background_logit_change'], -1.)

    def test_statistics_exclude_invalid_and_normalization_keeps_them_zero(self):
        x = np.array([[2., 999.], [4., 999.]], np.float32)
        valid = np.array([[True, False], [True, False]])
        mu, std = masked_channel_statistics(x, valid)
        np.testing.assert_allclose(mu, [3., 0.]); np.testing.assert_allclose(std, [1., .01])
        r = transform_radial(torch.tensor([[3., 9.]]), torch.tensor([[True, False]]), 'A2', 2.)
        torch.testing.assert_close(r, torch.tensor([[.5, 0.]]))

    def test_cached_mask_matches_live_and_zero_recomputes_vector(self):
        torch.manual_seed(111); torch.set_num_threads(2)
        e0 = F.normalize(torch.randn(7), dim=0)
        model = RadialMaskVectorMLP(7, 11, e0)
        with torch.no_grad():
            model.net[-1].weight.normal_(0, .01)
            model.net[0].weight[:, 7:18].normal_(0, .01)
        sam = F.normalize(torch.randn(2, 7, 8, 8), dim=1); ray = torch.randn(2, 11, 8, 8)
        z = torch.cat((sam, ray), 1); desc = mask_descriptor(z, sam, e0)
        torch.testing.assert_close(cached_mask_forward(model, z, desc, e0), model(sam, ray, e0), atol=1e-6, rtol=1e-6)
        z0 = torch.cat((sam, torch.zeros_like(ray)), 1)
        desc0 = mask_descriptor(z0, sam, e0)
        torch.testing.assert_close(cached_mask_forward(model, z0, desc0, e0), model(sam, torch.zeros_like(ray), e0), atol=1e-6, rtol=1e-6)
        self.assertFalse(torch.equal(model.net(desc), model.net(desc0)))


if __name__ == '__main__':
    unittest.main()
