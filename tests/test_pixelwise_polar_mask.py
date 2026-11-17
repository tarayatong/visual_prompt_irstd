import unittest
import torch

from visual_prompt.models.pixelwise_polar_mask import PixelwisePolarMask, masked_batch_iou


def make_model(representation='polar', head='logit'):
    return PixelwisePolarMask(torch.zeros(64),torch.ones(64),
        dict(representation=representation,head=head,polar_hidden=4),torch.zeros(5,64),torch.ones(5,64))


def check_pixel_prediction_does_not_pool_crop(head):
    torch.manual_seed(11)
    model=make_model(head=head).eval()
    sam=torch.randn(2,352,5,5);raw=torch.randn(2,64,5,5);valid=torch.ones_like(raw,dtype=torch.bool)
    before=model(sam,raw,valid)
    # Change every other spatial location, including the crop center. The
    # top-left prediction must not depend on a crop-wide descriptor.
    sam2=sam+10;raw2=raw-7
    sam2[:,:,0,0]=sam[:,:,0,0];raw2[:,:,0,0]=raw[:,:,0,0]
    after=model(sam2,raw2,valid)
    torch.testing.assert_close(before[:,0,0],after[:,0,0],rtol=0,atol=0)
    assert not hasattr(model,'e0')
    assert model.net[0].in_channels==416


def test_removed_crop_head_cannot_be_selected():
    try:
        make_model(head='crop')
    except ValueError as exc:
        assert 'crop heads were removed' in str(exc)
    else:
        raise AssertionError('Removed crop head was accepted')


def test_ignored_pixels_have_zero_gradient():
    z=torch.zeros(2,4,4,requires_grad=True);target=torch.zeros_like(z);target[0,1,1]=1
    valid=torch.ones_like(z,dtype=torch.bool);valid[:,0,:]=False
    masked_batch_iou(z,target,valid).backward()
    assert torch.count_nonzero(z.grad[~valid])==0
    assert (z.grad[1][valid[1]]>0).all()


def test_invalid_radial_output_stays_zero():
    model=make_model()
    sam=torch.randn(1,352,4,4);raw=torch.randn(1,64,4,4);valid=torch.ones_like(raw,dtype=torch.bool)
    valid[:,::2,:,0]=False
    _,radial=model.encode(sam,raw,valid)
    assert torch.count_nonzero(radial[~valid])==0


class PixelwiseMaskTests(unittest.TestCase):
    def test_logit_has_no_crop_pooling(self):
        check_pixel_prediction_does_not_pool_crop('logit')

    def test_dynamic_has_no_crop_pooling(self):
        check_pixel_prediction_does_not_pool_crop('dynamic_normalized')

    def test_crop_head_rejected(self):
        test_removed_crop_head_cannot_be_selected()

    def test_ignored_gradient(self):
        test_ignored_pixels_have_zero_gradient()

    def test_invalid_radial_output(self):
        test_invalid_radial_output_stays_zero()


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
