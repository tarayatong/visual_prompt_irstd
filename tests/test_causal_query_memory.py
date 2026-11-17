import unittest
import torch
from torch.nn import functional as F

from visual_prompt.models.causal_query_memory import CausalQueryMask, sequence_forward, bce_dice


class CausalMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(111)
        self.e0 = F.normalize(torch.randn(352), dim=0)
        self.model = CausalQueryMask(self.e0, 8)
        # Nonzero reader exposes causal dependencies that zero initialization hides.
        torch.nn.init.normal_(self.model.readout[-1].weight, std=.1)
        self.support = torch.randn(1, 416, 8, 8)
        self.mask = torch.zeros(1, 8, 8); self.mask[:, 2:5, 2:5] = 1
        self.queries = torch.randn(1, 3, 416, 8, 8)
        self.valid = torch.ones(1, 3, 8, 8)

    def forward_sequence(self, queries, mode='propagation'):
        return sequence_forward(self.model, queries, self.e0, self.support, self.mask,
                                torch.ones_like(self.mask), self.valid, mode)

    def test_future_queries_cannot_change_past_outputs(self):
        original = self.forward_sequence(self.queries)
        changed = self.queries.clone(); changed[:, 2] *= -10
        result = self.forward_sequence(changed)
        self.assertTrue(torch.equal(original[:, :2], result[:, :2]))

    def test_writeback_changes_later_outputs_and_carries_gradient(self):
        queries = self.queries.clone().requires_grad_(True)
        result = self.forward_sequence(queries)
        support_only = self.forward_sequence(queries, 'support_only')
        self.assertTrue(torch.equal(result[:, 0], support_only[:, 0]))
        self.assertFalse(torch.equal(result[:, 1], support_only[:, 1]))
        result[:, -1].mean().backward()
        self.assertGreater(float(queries.grad[:, 0].abs().sum()), 0)

    def test_padding_is_excluded_from_loss(self):
        target = torch.zeros(1, 8, 8)
        valid = torch.ones_like(target); valid[:, 0] = 0
        logits = torch.zeros_like(target)
        expected = bce_dice(logits, target, valid)
        logits[:, 0] = 100
        self.assertTrue(torch.equal(expected, bce_dice(logits, target, valid)))

    def test_memory_is_reset_for_each_sequence(self):
        first = self.forward_sequence(self.queries)
        self.forward_sequence(self.queries * -2)
        self.assertTrue(torch.equal(first, self.forward_sequence(self.queries)))


if __name__ == '__main__':
    unittest.main()
