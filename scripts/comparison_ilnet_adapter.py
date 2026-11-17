"""Native ILNet-L outputs and transforms, constrained to frozen manifests."""
import numpy as np
import torch
from PIL import Image
from scripts.run_comparison_hdnet import ManifestDataset


class ILNetManifestDataset(ManifestDataset):
    def __init__(self, manifest, training, official_data_class):
        super().__init__(manifest, training, official_data_class)
        self.transformer.img_size = 512

    def __getitem__(self, index):
        sid = self.ids[index]
        record = self.manifest["records"][sid]
        image = Image.open(record["image"]).convert("RGB")
        mask = Image.open(record["mask"]).convert("L").resize(image.size, Image.Resampling.NEAREST)
        transform = self.transformer._sync_transform if self.training else self.transformer._testval_sync_transform
        image, mask = transform(image, mask)
        x = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255)
        stats = self.manifest["train50_rgb_stats"]
        x = (x - torch.tensor(stats["mean"])[:, None, None]) / torch.tensor(stats["std"])[:, None, None]
        return x, torch.from_numpy((np.array(mask) > 0).astype(np.float32))[None], sid


class ILNetNativeOutputs(torch.nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native

    def forward(self, x, warm_flag):
        outputs = self.native(x)
        return (list(outputs[1:]), outputs[0]) if isinstance(outputs, (list, tuple)) else ([], outputs)


def train50_stats(manifest):
    total, squares, count = np.zeros(3), np.zeros(3), 0
    for sid in manifest["train_ids"]:
        x = np.array(Image.open(manifest["records"][sid]["image"]).convert("RGB"), dtype=np.float64) / 255
        total += x.sum(axis=(0, 1))
        squares += np.square(x).sum(axis=(0, 1))
        count += x.shape[0] * x.shape[1]
    mean = total / count
    return dict(mean=mean.tolist(), std=np.maximum(np.sqrt(np.maximum(squares / count - mean**2, 0)), 1e-6).tolist(), source="fixed train50 only")
