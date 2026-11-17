import unittest
import torch
from torch.nn import functional as F

from scripts.train_locked_radial_full50 import make_model
from scripts.run_radial_independent_search import cached_mask_forward, mask_descriptor


class FinalMetricsTests(unittest.TestCase):
    def test_final_factory_mask_forward_matches_normal_and_zero_interfaces(self):
        torch.set_num_threads(2);torch.manual_seed(111)
        e0=F.normalize(torch.randn(352),dim=0)
        for variant,spec in [('A0',None),('C1',None),('D2',dict(projection='nonlinear',condition='candidate',output_norm=.25))]:
            selected=dict(candidate=dict(variant=variant),architecture=spec,raw_channels=64)
            model=make_model('mask',selected,e0,torch.device('cpu'))
            # Give the residual readout nonzero learned values so radial intervention is meaningful.
            with torch.no_grad():
                net=model.main.net if spec is not None else model.net
                net[-1].weight.normal_(std=.01)
            sam=F.normalize(torch.randn(2,352,8,8),dim=1);ray=torch.randn(2,64,8,8)
            features=sam if variant=='A0' else torch.cat((sam,ray),1)
            desc=mask_descriptor(features,sam,e0)
            actual=cached_mask_forward(model,features,desc,e0)
            if spec is not None:expected=model.main(sam,model.radial_representation(ray),e0)
            elif variant=='A0':
                sim=torch.einsum('bchw,c->bhw',sam,e0);zeros=torch.zeros_like(sim)
                expected=model(sam,e0,sim,zeros,zeros,.25,.1)
            else:expected=model(sam,ray,e0)
            torch.testing.assert_close(actual,expected)
            f0=features.clone()
            if variant!='A0':f0[:,352:]=0
            zero=cached_mask_forward(model,f0,mask_descriptor(f0,sam,e0),e0,zero_radial=True)
            if spec is not None:torch.testing.assert_close(zero,model.main(sam,torch.zeros(2,32,8,8),e0))
            elif variant!='A0':torch.testing.assert_close(zero,model(sam,torch.zeros_like(ray),e0))


if __name__=='__main__':unittest.main()
