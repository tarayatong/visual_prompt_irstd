"""Numerical regression checks for sparse segmentation objectives."""
import torch
from visual_prompt.training.losses import SegmentationCriterion
from visual_prompt.training.lora_model import segmentation_loss


def main():
    torch.manual_seed(42)
    baseline = SegmentationCriterion()
    robust = SegmentationCriterion(loss_mode='balanced_hard')
    x = torch.randn(2, 1, 16, 16)
    y = torch.zeros_like(x); y[0, 0, 7, 7] = 1
    assert torch.allclose(baseline(x, y)[0], segmentation_loss(x, y))
    # Empty, full, singleton foreground, and severely wrong saturated predictions.
    for target in [torch.zeros_like(y), torch.ones_like(y), y]:
        for value in [-1000., 0., 1000.]:
            z = torch.full_like(x, value, requires_grad=True)
            loss, _ = robust(z, target)
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    z = torch.zeros(1, 1, 32, 32, requires_grad=True)
    target = torch.zeros_like(z); target[0, 0, 16, 16] = 1
    robust(z, target)[0].backward()
    assert z.grad[0, 0, 16, 16] < 0  # gradient descent raises missed foreground
    assert z.grad[0, 0, 0, 0] > 0    # gradient descent suppresses false background
    # More background pixels cannot dilute the foreground classification term.
    foreground = []
    for size in [32, 64]:
        z = torch.zeros(1, 1, size, size)
        target = torch.zeros_like(z); target[0, 0, 16, 16] = 1
        foreground.append(robust(z, target)[1]['foreground'])
    assert torch.allclose(*foreground)
    # A confident false-positive island must increase loss.
    z = torch.full((1, 1, 32, 32), -6.)
    target = torch.zeros_like(z); target[0, 0, 16, 16] = 1
    z[0, 0, 16, 16] = 6
    clean = robust(z, target)[0]
    z[0, 0, :3, :3] = 6
    assert robust(z, target)[0] > clean
    # Formula and top-K gradient agree with a hand-computed small example.
    z = torch.tensor([[[[0., 1., 2., -1.]]]], requires_grad=True)
    target = torch.tensor([[[[1., 0., 0., 0.]]]])
    criterion = SegmentationCriterion(loss_mode='balanced_hard', dice_weight=0,
                                     hard_negative_min=1, hard_negative_ratio=0, hard_fn_fraction=1)
    loss, parts = criterion(z, target)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z, target, reduction='none')
    fp_hard = bce[0,0,0,2]
    fn_hard = bce[0,0,0,0]
    expected = bce[0,0,0,0] + .25 * bce[0,0,0,1:].mean() + .5 * .5 * (fp_hard + fn_hard)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.allclose(z.grad[0,0,0,2], z.sigmoid()[0,0,0,2] * (.25/3+.25))
    assert z.grad[0,0,0,0] < 0  # hard FN pushes a missed target upward
    print('PASS: baseline equivalence, empty/full/tiny masks, extreme logits, gradient signs, '
          'foreground balance, FP/FN penalties, exact loss formula, top-K gradients')


if __name__ == '__main__':
    main()
