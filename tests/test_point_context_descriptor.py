import unittest
import torch
from visual_prompt.models.point_context_descriptor import point_context_descriptor


class PointContextTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_constant_fields_have_zero_center_surround_even_at_source_boundary(self):
        layers=[torch.ones(1,c,16,16) for c in (32,64,256)]
        points=torch.tensor([[32.,32.],[0.,0.]])
        output=point_context_descriptor(layers,points,64,source_size=64)
        offset=0
        for c in (32,64,256):
            torch.testing.assert_close(output[:,offset+c:offset+4*c],torch.zeros(2,3*c),atol=1e-6,rtol=0)
            offset+=4*c
        self.assertEqual(output.shape,(2,1408))

    def test_shared_and_batched_features_agree_and_exact_center_is_read(self):
        layers=[torch.randn(1,c,64,64) for c in (32,64,256)];points=torch.tensor([[32.,32.],[20.,20.]])
        shared=point_context_descriptor(layers,points,64,source_size=64)
        batched=point_context_descriptor([x.expand(2,-1,-1,-1).clone() for x in layers],points,64,source_size=64)
        torch.testing.assert_close(shared,batched)
        torch.testing.assert_close(shared[0,:32],torch.nn.functional.normalize(layers[0][0,:,32,32],dim=0),atol=1e-5,rtol=1e-5)


if __name__=='__main__':unittest.main()
