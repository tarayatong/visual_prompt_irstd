"""Analytic geometry, masking, and autograd checks for the reusable operator."""
from dataclasses import replace
import io
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.radial_profile import DirectionalRadialOperator, RadialProfileConfig


class RadialProfileTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_plane_derivatives_and_nonuniform_spacing(self):
        y, x = torch.meshgrid(torch.arange(21.), torch.arange(23.), indexing='ij')
        image = (2*x+3*y+7)[None, None]
        cfg = RadialProfileConfig(num_directions=4, radii=(1., 3., 5.), components=('first', 'second'), fusion='none')
        op = DirectionalRadialOperator(1, cfg)
        p = op.sample_profiles(image, torch.tensor([[[11., 10.]]]))
        expected = -(2*p.angles.cos()+3*p.angles.sin())
        torch.testing.assert_close(p.values[0, 0, 0, 0], expected[:, None].expand(-1, 3), atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(p.values[0, 0, 0, 1], torch.zeros(4, 3), atol=2e-5, rtol=0)
        self.assertTrue(p.valid[0, 0, 0].all())
        self.assertTrue(p.valid[0, 0, 1, :, :-1].all())
        self.assertFalse(p.valid[0, 0, 1, :, -1].any())

    def test_quadratic_curvature(self):
        y, x = torch.meshgrid(torch.arange(-8., 9.), torch.arange(-8., 9.), indexing='ij')
        image = (2*x*x+3*y*y)[None, None]
        cfg = RadialProfileConfig(num_directions=4, radii=(1., 2., 4.), components=('second',), fusion='none')
        p = DirectionalRadialOperator(1, cfg).sample_profiles(image, torch.tensor([[[8., 8.]]]))
        expected = torch.tensor([4., 6., 4., 6.])[:, None].expand(-1, 2)
        torch.testing.assert_close(p.values[0, 0, 0, 0, :, :2], expected, atol=3e-5, rtol=1e-5)

    def test_offset_invariance_of_contrast_and_derivatives(self):
        cfg = RadialProfileConfig(radii=(1., 2., 3.), fusion='none')
        op = DirectionalRadialOperator(2, cfg)
        image = torch.randn(1, 2, 15, 16)
        points = torch.tensor([[[7.2, 6.3]]])
        a, b = op(image, points), op(image+10, points)
        torch.testing.assert_close(a[:, :, :, 1:], b[:, :, :, 1:], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(b[:, :, :, 0]-a[:, :, :, 0], torch.full_like(a[:, :, :, 0], 10), atol=3e-6, rtol=1e-5)

    def test_boundaries_do_not_create_false_gradients(self):
        for padding in ('zeros', 'border', 'reflection'):
            cfg = RadialProfileConfig(radii=(1., 2., 4.), components=('first', 'second'), fusion='none', padding_mode=padding)
            p = DirectionalRadialOperator(1, cfg).sample_profiles(torch.ones(1, 1, 9, 10), torch.tensor([[[0., 0.], [4., 4.], [-4., -4.]]]))
            torch.testing.assert_close(p.values, torch.zeros_like(p.values), atol=1e-6, rtol=0)
            self.assertFalse(p.valid[:, 2].any())
            self.assertTrue(p.valid[:, 0].any())
            self.assertFalse(p.valid[:, 0].all())
            # k=2 is the downward axial ray: cos(pi/2) roundoff must not reject it.
            self.assertTrue(p.valid[0, 0, 0, 2].all())

    def test_dense_point_and_chunk_consistency(self):
        image = torch.randn(2, 2, 8, 9)
        points = torch.tensor([[[2., 3.], [5., 4.]], [[1., 6.], [7., 2.]]])
        for fusion in ('none', 'concat', 'mean', 'learned'):
            cfg = RadialProfileConfig(radii=(1., 2.), fusion=fusion, chunk_size=5)
            op = DirectionalRadialOperator(2, cfg)
            dense, selected = op(image), op(image, points)
            if fusion == 'none':
                sampled = torch.stack([dense[b, ..., points[b, :, 1].long(), points[b, :, 0].long()].movedim(-1, 0) for b in range(2)])
            else:
                sampled = torch.stack([dense[b, :, points[b, :, 1].long(), points[b, :, 0].long()].T for b in range(2)])
            torch.testing.assert_close(selected, sampled, atol=2e-6, rtol=2e-5)
            unchunked = DirectionalRadialOperator(2, replace(cfg, chunk_size=1000))
            unchunked.load_state_dict(op.state_dict())
            torch.testing.assert_close(dense, unchunked(image), atol=2e-6, rtol=2e-5)

    def test_elliptical_opposites_and_offcenter_asymmetry(self):
        y, x = torch.meshgrid(torch.arange(-12., 13.), torch.arange(-12., 13.), indexing='ij')
        image = torch.exp(-.5*((x/5)**2+(y/2)**2))[None, None]
        cfg = RadialProfileConfig(radii=(1., 2., 4., 6.), components=('intensity',), fusion='none')
        op = DirectionalRadialOperator(1, cfg)
        center = op(image, torch.tensor([[[12., 12.]]]))[0, 0, 0, 0]
        off = op(image, torch.tensor([[[14., 12.]]]))[0, 0, 0, 0]
        torch.testing.assert_close(center[:4], center[4:], atol=1e-6, rtol=1e-5)
        self.assertGreater(float((center[0]-center[2]).abs().mean()), .1)
        self.assertGreater(float((off[:4]-off[4:]).abs().mean()), .05)

    def test_input_and_position_gradcheck(self):
        cfg = RadialProfileConfig(num_directions=4, radii=(.8, 1.7), fusion='none')
        op = DirectionalRadialOperator(1, cfg).double()
        image = torch.randn(1, 1, 7, 8, dtype=torch.double, requires_grad=True)
        points = torch.tensor([[[3.21, 3.37]]], dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: op(a, b), (image, points), eps=1e-6, atol=1e-5, rtol=1e-3))

    def test_geometry_and_readout_are_trainable(self):
        cfg = RadialProfileConfig(radii=(.8, 1.7, 2.9, 4.), learnable_radii=True, learnable_rotation=True)
        op = DirectionalRadialOperator(2, cfg)
        image = torch.randn(1, 2, 15, 15, requires_grad=True)
        points = torch.tensor([[[7.23, 7.31], [6.41, 6.27]]], requires_grad=True)
        radii, _ = op.sampling_geometry()
        torch.testing.assert_close(radii, torch.tensor(cfg.radii), atol=1e-6, rtol=1e-6)
        optimizer = torch.optim.SGD(op.parameters(), lr=.05)
        before = op.projection.weight.detach().clone()
        loss = op(image, points).square().mean()
        loss.backward()
        for name, parameter in op.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(op.raw_gaps.grad.abs().sum()), 0)
        self.assertGreater(float(op.rotation.grad.abs().sum()), 0)
        self.assertGreater(float(points.grad.abs().sum()), 0)
        self.assertGreater(float(image.grad.abs().sum()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, op.projection.weight))
        radii, _ = op.sampling_geometry()
        self.assertTrue((torch.diff(torch.cat((radii.new_zeros(1), radii))) > 0).all())
        self.assertAlmostEqual(float(radii[-1].detach()), 4., places=5)
        angles = op.sampling_geometry()[1]
        torch.testing.assert_close(angles[4:]-angles[:4], torch.full((4,), math.pi))

    def test_learnable_geometry_gradcheck(self):
        cfg = RadialProfileConfig(num_directions=4, radii=(.7, 1.6, 2.7), fusion='none',
                                  learnable_radii=True, learnable_rotation=True)
        op = DirectionalRadialOperator(1, cfg).double()
        with torch.no_grad():
            op.rotation.fill_(.23)
        image = torch.randn(1, 1, 11, 12, dtype=torch.double)
        points = torch.tensor([[[5.21, 5.37]]], dtype=torch.double)
        def call(gaps, rotation):
            return torch.func.functional_call(op, {'raw_gaps': gaps, 'rotation': rotation}, (image, points))
        self.assertTrue(torch.autograd.gradcheck(call, (op.raw_gaps, op.rotation), eps=1e-6, atol=1e-5, rtol=1e-3))

    def test_mean_direction_pool_preserves_discrete_rotation_invariance(self):
        op = DirectionalRadialOperator(1, RadialProfileConfig(radii=(1., 2., 3.)))
        image = torch.randn(1, 1, 13, 13)
        center = torch.tensor([[[6., 6.]]])
        torch.testing.assert_close(op(image, center), op(torch.rot90(image, 1, (-2, -1)), center), atol=1e-6, rtol=1e-5)

    def test_geometry_only_modes_and_shape_controls(self):
        image = torch.randn(1, 3, 5, 6)
        raw = DirectionalRadialOperator(3, RadialProfileConfig(num_directions=6, radii=(1., 2., 3.), components=('intensity',), fusion='none'))
        self.assertEqual(raw(image).shape, (1, 3, 1, 6, 3, 5, 6))
        self.assertEqual(sum(p.numel() for p in raw.parameters()), 0)
        mean = DirectionalRadialOperator(3, RadialProfileConfig(fusion='mean'))
        self.assertEqual(mean(image).shape, (1, 12, 5, 6))
        self.assertEqual(sum(p.numel() for p in mean.parameters()), 0)
        flat = DirectionalRadialOperator(3, RadialProfileConfig(direction_pool='flatten', out_channels=7))
        self.assertEqual(flat(image).shape, (1, 7, 5, 6))
        odd = DirectionalRadialOperator(1, RadialProfileConfig(num_directions=5, opposite_pairs=False))
        self.assertEqual(odd(image[:, :1]).shape, (1, 16, 5, 6))

    def test_dense_mask_network_training_and_state_dict(self):
        op = DirectionalRadialOperator(1)
        head = nn.Conv2d(16, 1, 1)
        image = torch.randn(1, 1, 7, 8)
        target = torch.zeros(1, 1, 7, 8); target[:, :, 2:4, 3:5] = 1
        F.binary_cross_entropy_with_logits(head(op(image)), target).backward()
        self.assertGreater(float(op.radial[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(op.radius_logits.grad.abs().sum()), 0)
        buffer = io.BytesIO(); torch.save(op.state_dict(), buffer); buffer.seek(0)
        restored = DirectionalRadialOperator(1)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(restored(image), op(image))

    def test_concat_preserves_every_profile_value(self):
        image = torch.randn(1, 2, 9, 10)
        points = torch.tensor([[[4.2, 4.1], [2., 3.]]])
        cfg = RadialProfileConfig(radii=(1., 2.), fusion='none')
        raw = DirectionalRadialOperator(2, cfg)(image, points)
        stacked = DirectionalRadialOperator(2, replace(cfg, fusion='concat'))(image, points)
        self.assertEqual(stacked.shape, (1, 2, 2*4*8*2))
        torch.testing.assert_close(stacked, raw.flatten(2), atol=0, rtol=0)

    def test_constant_image_has_no_learned_boundary_response(self):
        op = DirectionalRadialOperator(1)
        for value in (0., .3, .7, 1.):
            out = op(torch.full((1, 1, 17, 19), value))
            torch.testing.assert_close(out, torch.zeros_like(out), atol=2e-7, rtol=0)

    def test_centered_readout_rejects_brightness_offset(self):
        op = DirectionalRadialOperator(2)
        image = torch.randn(1, 2, 11, 12)
        torch.testing.assert_close(op(image), op(image+5.), atol=2e-7, rtol=1e-4)

    def test_region_uses_original_context_and_matches_dense_crop(self):
        image = torch.randn(1, 1, 21, 23, requires_grad=True)
        for fusion in ('none', 'concat', 'mean', 'learned'):
            op = DirectionalRadialOperator(1, RadialProfileConfig(radii=(1., 2., 3.), fusion=fusion))
            region = op.forward_region(image, 6, 5, 8, 9)
            torch.testing.assert_close(region, op(image)[..., 6:14, 5:14], atol=2e-6, rtol=1e-4)
            region.square().mean().backward()
        # Context pixels outside the output ROI still influence its features.
        self.assertGreater(float(image.grad[..., 3:6, 5:14].abs().sum()), 0)
        with self.assertRaises(ValueError):
            op.forward_region(image, -1, 2, 8, 9)

    def test_invalid_configs_and_single_axis(self):
        for cfg in (RadialProfileConfig(radii=(1., 1.)), RadialProfileConfig(num_directions=3),
                    RadialProfileConfig(radii=(1.,), components=('second',)),
                    RadialProfileConfig(components=()), RadialProfileConfig(fusion='bad')):
            with self.assertRaises(ValueError):
                DirectionalRadialOperator(1, cfg)
        op = DirectionalRadialOperator(1, RadialProfileConfig(radii=(1., 2.), fusion='mean'))
        self.assertEqual(op(torch.ones(1, 1, 1, 8)).shape, (1, 4, 1, 8))
        with self.assertRaises(ValueError):
            op(torch.ones(1, 1, 4, 4), torch.ones(1, 2, 2, dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
