import numpy as np
import torch
from torch.nn import functional as F
from visual_prompt.core.coarse_window_gate import WINDOWS
from visual_prompt.models.window_objectness import owner_windows, sample_window_rois, sample_window_points
from visual_prompt.models.full_image_objectness import _roi_grid, describe_full_image_points


def test_window_roi_coordinates_match_full_resolution_field():
    points = [(0, 0), (511, 511), (96, 96), (250, 251), (500, 32)]
    yy, xx = torch.meshgrid(torch.arange(512), torch.arange(512), indexing='ij')
    feature = torch.stack((torch.ones_like(xx), yy/512, xx/512)).float()[None]
    cache = {}
    for index in owner_windows(points):
        y,x = WINDOWS[int(index)]; cache[int(index)] = [feature[:,:,y:y+128,x:x+128]]
    actual = sample_window_rois(points, cache)
    grid = _roi_grid(points, (512,512), 64, 32, torch.device('cpu'))
    expected = F.normalize(F.grid_sample(feature.expand(len(points),-1,-1,-1), grid, align_corners=True),dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert owner_windows([(95,95),(96,96)]).tolist() == [0,8]


def test_injected_roi_descriptors_do_not_encode_and_preserve_statistics():
    class Predictor:
        device = torch.device('cpu')
        supports_external_sim_features = True
        sim_feature_layers = []
        calls = 0
        def set_image(self, image):
            self.calls += 1; self.original_size = image.shape[:2]
        def get_named_sim_features(self, align=False):
            return [('native', feature)]
    rng = np.random.default_rng(1)
    image = rng.integers(0,256,(512,512,3),dtype=np.uint8)
    score = rng.random((512,512),dtype=np.float32)
    feature = torch.from_numpy(rng.random((1,3,64,64),dtype=np.float32))
    points = [(32,33),(0,0)]
    p = Predictor()
    old = describe_full_image_points(p,image,score,points,64,7)
    grid = _roi_grid(points,(512,512),64,32,p.device)
    rois = F.normalize(F.grid_sample(feature.expand(2,-1,-1,-1),grid,align_corners=True),dim=1)
    new = describe_full_image_points(p,image,score,points,64,7,roi_features=rois)
    assert p.calls == 1
    assert all(np.array_equal(old[key],new[key]) for key in old)


def test_window_point_uses_exact_coordinate_across_native_resolutions():
    points = [(0,0),(511,511),(95,95),(96,96),(250,251),(500,32)]
    cache = {}
    for index in owner_windows(points):
        y0,x0 = WINDOWS[int(index)]
        layers = []
        for size in (32,64):
            yy,xx = torch.meshgrid(torch.linspace(y0,y0+127,size),torch.linspace(x0,x0+127,size),indexing='ij')
            layers.append(torch.stack((torch.ones_like(xx),yy/512,xx/512))[None])
        cache[int(index)] = layers
    actual = sample_window_points(points,cache)
    expected = torch.tensor([[1,y/512,x/512,1,y/512,x/512] for y,x in points])
    expected = F.normalize(expected,dim=1)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
