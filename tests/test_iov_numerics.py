"""Regress the author PDE zero-norm backward failure without changing forward."""
import unittest
import torch
from scripts.comparison_added_irstd_adapter import ZeroSafeSqrt

class IOVNumericsTests(unittest.TestCase):
    def test_forward_exact_and_zero_norm_gradient_finite(self):
        x=torch.tensor([0.,1e-16,.25,4.],requires_grad=True)
        y=ZeroSafeSqrt.apply(x)
        self.assertTrue(torch.equal(y,torch.sqrt(x)))
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(x.grad[0].item(),0.)
        torch.testing.assert_close(x.grad[1:],1/(2*x.detach()[1:].sqrt()))

if __name__=='__main__':unittest.main()
