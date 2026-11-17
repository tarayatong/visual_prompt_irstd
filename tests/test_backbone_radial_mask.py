import torch
from visual_prompt.models.backbone_radial_mask import BackboneRadialMask
from visual_prompt.models.learned_radial_center_sam import LearnedRadialCenterSAM


def test_matches_retained_center_radial_head():
    torch.manual_seed(111)
    mean=torch.randn(5,64);std=torch.rand(5,64)+.1
    original=LearnedRadialCenterSAM(mean,std)
    generic=BackboneRadialMask(352,mean[0],std[0])
    generic.head.load_state_dict(original.base.head.state_dict())
    with torch.no_grad():generic.mapping.copy_(original.mapping)
    sam=torch.nn.functional.normalize(torch.randn(2,352,8,8),dim=1)
    raw=torch.randn(2,64,8,8);valid=torch.rand(2,64,8,8)>.2
    torch.testing.assert_close(generic(sam,raw,valid),original(sam,raw,valid),rtol=0,atol=0)


def test_mobile_channel_width_has_finite_gradients():
    model=BackboneRadialMask(992,torch.zeros(64),torch.ones(64))
    sam=torch.nn.functional.normalize(torch.randn(2,992,8,8),dim=1)
    raw=torch.randn(2,64,8,8);valid=torch.ones_like(raw,dtype=torch.bool)
    output=model(sam,raw,valid)
    assert output.shape==(2,8,8)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
