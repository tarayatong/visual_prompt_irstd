"""Native MPA construction/loss and fixed-five-support augmentation adapter."""
import ast
import sys
import types
from PIL import Image
import torch


def load_native(root):
    source = root / "sources/MPA"
    sys.path.insert(0, str(source.resolve()))
    sys.path.append(str((root / "runtime_packages").resolve()))
    import model.resnet as resnet
    from model.MPA_matching import MPA_MatchingNet
    from data.lung_MPA import DatasetLungMPA
    original = resnet.resnet50
    def pretrained(pretrained=True):
        model = original(pretrained=False)
        state = torch.load(root / "weights/resnet50_deepstem_mpa.pth", map_location="cpu", weights_only=True)
        native = model.state_dict()
        assert all(k in state and state[k].shape == value.shape for k, value in native.items())
        model.load_state_dict({k: state[k] for k in native}, strict=True)
        return model
    resnet.resnet50 = pretrained
    try:
        model = MPA_MatchingNet("resnet50", refine=True, shot=5)
    finally:
        resnet.resnet50 = original
    for layer in (model.layer0, model.layer1):
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    # Reuse exactly the native loss function without importing its unused
    # benchmark factory or main entry point.
    path = source / "mpa.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "loss_calculator"]
    namespace = {}
    exec(compile(tree, str(path), "exec"), namespace)
    return model, DatasetLungMPA, namespace["loss_calculator"]


def fixed_support_dataset(native_class, manifest, transform, episodes):
    class FixedSupport(native_class):
        def sample_episode(self, index):
            return self.support_names[0], list(self.support_names), 0

        def load_frame(self, query_name, support_names):
            images = [Image.open(self.mask_to_image[name]).convert("RGB") for name in support_names]
            return images[0], self.read_mask(support_names[0]), images, support_names

    dataset = FixedSupport.__new__(FixedSupport)
    dataset.shot, dataset.num, dataset.transform = 5, episodes, transform
    dataset.support_names = [manifest["records"][sid]["mask"] for sid in manifest["support_ids"]]
    dataset.mask_to_image = {manifest["records"][sid]["mask"]: manifest["records"][sid]["image"] for sid in manifest["support_ids"]}
    import albumentations as A
    dataset.q_colorAug = A.Compose([A.RandomBrightnessContrast(p=.5),
                                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=1)])
    return dataset
