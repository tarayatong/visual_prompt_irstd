"""Packaging-only fixes for the author TVGTANet code; no model changes."""
import ast
import importlib
import importlib.util
import sys
import types
import cv2
import numpy as np
import torch


def load_native(root):
    source = root / "sources/TVGTANet"
    sys.path[:0] = [str(source.resolve()), str((source / "clip_cam").resolve())]
    sys.path.append(str((root / "runtime_packages").resolve()))
    # Author imports the same CAM package under two names, causing circular
    # __init__ imports. Give both names one namespace and load its real classes.
    package = types.ModuleType("clip_cam.pytorch_grad_cam")
    package.__path__ = [str(source / "clip_cam/pytorch_grad_cam")]
    sys.modules["clip_cam.pytorch_grad_cam"] = package
    sys.modules["pytorch_grad_cam"] = package
    package.GradCAM = importlib.import_module("clip_cam.pytorch_grad_cam.grad_cam").GradCAM
    # Missing CAM helper utilities are supplied by the other official author
    # repository which contains the same CAM implementation, without rewriting.
    spec = importlib.util.spec_from_file_location("comparison_pi_clip_cam_utils", root / "sources/PI-CLIP/utils.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    utils = importlib.import_module("utils")
    utils.parse_xml_to_dict = helpers.parse_xml_to_dict
    utils.scoremap2bbox = helpers.scoremap2bbox
    cam = importlib.import_module("clip_cam.lung_cam")
    # Only the absent, unused benchmark dataset factory is bypassed. The actual
    # feature maker, fitting loss, network and inference functions remain native.
    path = source / "core/runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and node.module == "data.dataset")]
    runner = types.ModuleType("comparison_tvgtanet_native_runner")
    runner.__file__ = str(path)
    exec(compile(tree, str(path), "exec"), runner.__dict__)
    return runner, cam, source


class NativeInfraredCAM:
    """Author attention-refined GradCAM with fixed infrared foreground text."""
    def __init__(self, native_cam, weight):
        self.native = native_cam
        native_cam.device = "cuda"
        self.model, _ = native_cam.clip.load(str(weight), device="cuda")
        self.model.eval()
        # Fixed before any infrared testing; same author template for both texts.
        self.bg = native_cam.zeroshot_classifier(["infrared background"], ["a clean origami {}."], self.model)
        self.fg = native_cam.zeroshot_classifier(["infrared small target"], ["a clean origami {}."], self.model)
        self.cam = native_cam.GradCAM(model=self.model,
                                     target_layers=[self.model.visual.transformer.resblocks[-1].ln_1],
                                     reshape_transform=native_cam.reshape_transform)

    def __call__(self, image_path):
        native = self.native
        # Preserve author1000px pre-resize (rounded to patch16 multiple1008).
        image = native.img_ms_and_flip(str(image_path), 1000, 1000, scales=[1.0])[0][None].cuda()
        h, w = image.shape[-2:]
        features, weights = self.model.encode_image(image, h, w)
        texts = torch.cat([self.fg, self.bg], dim=0)
        cam, _, last = self.cam(input_tensor=[features, texts, h, w],
                                targets=[native.ClipOutputTarget(0)], target_size=None)
        score = cam[0]
        weights.append(last)
        attention = torch.stack([a[:, 1:, 1:] for a in weights], dim=0)[-8:].mean(0)[0].cpu().detach().float()
        boxes, count = native.scoremap2bbox(scoremap=score, threshold=.4, multi_contour_eval=True)
        affinity = torch.zeros(score.shape)
        for index in range(count):
            x0, y0, x1, y1 = boxes[index]
            affinity[y0:y1, x0:x1] = 1
        transition = attention / attention.sum(dim=0, keepdim=True)
        transition = transition / transition.sum(dim=1, keepdim=True)
        for _ in range(2):
            transition = transition / transition.sum(dim=0, keepdim=True)
            transition = transition / transition.sum(dim=1, keepdim=True)
        transition = (transition + transition.T) / 2
        transition = transition @ transition
        transition = transition * affinity.view(1, -1)
        refined = (transition @ torch.as_tensor(score).float().view(-1, 1)).reshape(h // 16, w // 16).numpy()
        refined = native.scale_cam_image([refined], (400, 400))[0]
        # Native CAM writes uint8 PNG then dataset.read_mask thresholds at128.
        binary = ((refined * 255).astype(np.uint8) >= 128).astype(np.float32)
        if not np.isfinite(refined).all():
            raise ValueError("Native TVGTANet CAM produced nonfinite values")
        self.model.zero_grad(set_to_none=True)
        return torch.from_numpy(binary)
