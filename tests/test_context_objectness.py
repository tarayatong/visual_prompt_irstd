import unittest
import torch
from visual_prompt.models.lightweight_heads import ObjectnessMLP
from visual_prompt.models.context_objectness import ContextObjectness


class ContextObjectnessTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(111)

    def test_preserves_old_function_then_new_columns_learn(self):
        original=ObjectnessMLP(1440,128);model=ContextObjectness(original.state_dict(),32)
        f4=torch.randn(6,1440);extra=torch.randn(6,32)
        torch.testing.assert_close(model(f4,extra),original(f4),atol=1e-6,rtol=1e-5)
        model(f4,extra).square().mean().backward()
        self.assertGreater(float(model.main.net[0].weight.grad[:,1440:].abs().sum()),0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

if __name__=='__main__':unittest.main()
