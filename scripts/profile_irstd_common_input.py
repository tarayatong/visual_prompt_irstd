"""Count supported FLOPs at a shared input or at the IRSTD-1k evaluation sizes."""
import argparse
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch

from scripts import run_recent_a_comparison as base
from scripts.analyze_top8_comparison import read, record, save
from scripts.profile_tiny_ablation_flops import count

SIZE = 256
OUT = ROOT / 'outputs/top8_analysis_20261105/complexity_input256'
EVAL_OUT = OUT.with_name('complexity_eval_input')
EVAL_SIZES = {'ILNet': 512, 'IRPruneDet': 512, 'IRSAM': 512, 'SANSA': 640, 'Ours': 640}
CNN = ('ISNet', 'DNANet', 'MSHNet', 'SCTransNet', 'RPCANet', 'ILNet', 'HDNet', 'MSHNet+NS-FPN')
SAM = ('IRPruneDet', 'IRSAM', 'IR-SAM2', 'SAMamba')
METHODS = (*CNN, 'MLP-Net', 'MPCNet', 'SAM-SPL', *SAM, 'SANSA', 'Ours')


def image(m, sid, rgb, size):
    with Image.open(m['records'][sid]['image']) as im:
        arr = np.array(im.convert('RGB' if rgb else 'L').resize((size, size), Image.Resampling.BILINEAR),
                       dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1) if rgb else torch.from_numpy(arr)[None]


def profile(method, evaluation_input=False):
    torch.set_num_threads(2)
    base.seed_all()
    m = base.manifest_for('IRSTD-1k')
    size = EVAL_SIZES.get(method, SIZE) if evaluation_input else SIZE
    sid = m['test_ids'][0]
    device = torch.device('cpu')
    cp = None
    checkpoint_note = 'Retained IRSTD-1k task checkpoint'
    internal = None
    support_flops = None
    if method in CNN:
        from scripts.run_nuaa_coordinate_corrected import build_infrared
        if method == 'HDNet':
            # Its import creates fixed directional kernels on CUDA unconditionally.
            # Build the identical kernels directly on the CPU for this counter.
            with patch.object(torch.Tensor, 'cuda', lambda self, *args, **kwargs: self):
                ns = build_infrared(method)
        else:
            ns = build_infrared(method)
        model = ns['HDNet'](3)
        forward = lambda x: model(x, True)[1]
        rgb = method not in ('SCTransNet', 'RPCANet')
        rec = record('IRSTD-1k', method)
        cp = ROOT / rec['checkpoint'] if rec.get('checkpoint') else None
    elif method in ('MLP-Net', 'SAM-SPL'):
        from scripts.comparison_added_irstd_adapter import setup as setup_added
        model, *_ = setup_added(method, m, device, False)
        forward = (lambda x: model(x)[0]) if method == 'SAM-SPL' else model
        rgb = True
        rec = record('IRSTD-1k', method)
        cp = ROOT / rec['checkpoint']
    elif method == 'MPCNet':
        from scripts.run_mpcnet_comparison import setup, RUN
        model, *_ = setup(method, m, device)
        model.Train = False
        forward = model
        rgb = False
        cp = RUN / 'IRSTD-1k/MPCNet/checkpoints/epoch_0120.pt'
    elif method in SAM:
        from scripts.profile_public_irstd_models import build
        from scripts.train_sam_extension_train50 import RUN
        model, forward, identity = build(method, task_weights=False)
        internal = identity.get('encoder_internal_size')
        cp = RUN / 'IRSTD-1k' / method / 'checkpoints/epoch_0120.pt'
        rgb = True
    else:
        from visual_prompt.models.tiny_prototype_memory_replacement import install_prototype_memory
        from visual_prompt.models.tiny_prototype_feature_radial import install_prototype_feature_radial
        from visual_prompt.models.tiny_shared_memory import build_episode_keys, support_regions
        from scripts.run_tiny_ablation_combined import restore
        model, *_ = base.setup_sansa(m, device)
        if method == 'Ours':
            install_prototype_memory(model)
            install_prototype_feature_radial(model, active_stages=(1,))
            cp = ROOT / 'outputs/tiny_radial_detail_irstd_seed113/only_256/combined/checkpoints/epoch_0120.pt'
            restore(model, torch.load(cp, map_location='cpu', weights_only=False)['model'])
        else:
            cp = ROOT / record('IRSTD-1k', method)['checkpoint']
            base.load_saved(model, torch.load(cp, map_location='cpu', weights_only=False)['model'])
        model.eval()
        from util.promptable_utils import build_prompt_dict, rescale_prompt

        internal = model.sam.image_size
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        normalize = lambda x: (x / 255. - mean) / std
        ids = m['support_ids']
        pairs = [base.raw_pair(m, i, size, True) for i in ids]
        supports = torch.stack([normalize(torch.from_numpy(a).permute(2, 0, 1)) for a, _ in pairs])[None]
        masks = torch.stack([torch.from_numpy(y) for _, y in pairs])[None]

        def encode(x):
            processed, _, _, sizes = model._preprocess_visual_features(x, internal)
            return model._forward_backbone(processed, sizes), sizes

        def support():
            features, sizes = encode(supports)
            if method == 'Ours':
                regions = [support_regions(a.astype(np.uint8), y > 0) for a, y in pairs]
                return model.prototype_memory.prepare(build_episode_keys(model, features, range(5), regions))
            prompt = build_prompt_dict(masks, 'mask', 5, False, device)
            bank = {}
            for index in range(5):
                mask = rescale_prompt(prompt[0][index]['prompt'], 'mask', sizes[index], internal)
                decoded = model.sam._use_mask_as_output(features, mask, index)
                bank[index] = model._compute_memory_bank_dict(decoded, features, index)
            return bank

        def forward(x):
            features, _ = encode(normalize(x)[None])
            if method == 'SANSA':
                return model._compute_decoder_out_w_mem(features, 0, 5, cache).masks
            model.radial_branch.begin([(None, None)])
            result = model.prototype_memory.decode(model, features, 0, cache)
            model.radial_branch.finish()
            return result.masks

        rgb = True
        with torch.no_grad():
            cache, support_flops = count(support)

    if method not in ('SANSA', 'Ours'):
        if cp is not None and cp.is_file():
            model.load_state_dict(torch.load(cp, map_location='cpu', weights_only=False)['model'], strict=True)
        else:
            cp = None
            checkpoint_note = 'Architecture only; retained task checkpoint unavailable'
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.requires_grad_(False)
    # Grayscale adapters accept 1 channel; some legacy wrappers accept RGB and convert internally.
    if evaluation_input and method == 'ILNet':
        config = read(cp.parent.parent / 'config.json')
        assert config['input_size'] == size
        m['train50_rgb_stats'] = config['normalization']
        x, _, loaded_sid = ns['dataset_class'](m, False, ns['IRSTD_Dataset'])[0]
        assert loaded_sid == sid
        x = x[None]
    else:
        x = image(m, sid, rgb, size)[None]
        if method not in ('SANSA', 'Ours') and not (evaluation_input and method == 'IRSAM'):
            x = x / 255.
    with torch.no_grad():
        y, flops = count(lambda: forward(x))
    assert isinstance(y, torch.Tensor) and torch.isfinite(y).all() and flops > 0
    return dict(method=method, input_size=size, channels=x.shape[1], native_internal_encoder_size=internal,
                flops=flops, support_setup_flops=support_flops, parameters_total=total,
                parameters_trainable=trainable, query_id=sid, support_ids=m['support_ids'] if support_flops else None,
                prototype_tokens=model.prototype_memory.last_count if method == 'Ours' else None,
                output_shape=list(y.shape), checkpoint=str(cp.relative_to(ROOT)) if cp else None,
                checkpoint_sha256=base.sha(cp) if cp else None, checkpoint_note=checkpoint_note,
                torch_version=torch.__version__, device='CPU FP32')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-input', action='store_true',
                        help='Use each retained IRSTD-1k run’s external evaluation size.')
    parser.add_argument('methods', nargs='*')
    args = parser.parse_args()
    if len(args.methods) == 1:
        method = args.methods[0]
        assert method in METHODS
        row = profile(method, evaluation_input=args.evaluation_input)
        output = EVAL_OUT if args.evaluation_input else OUT
        save(output / f'{method}.json', row)
        print(method, row['flops'], row['parameters_trainable'], flush=True)
    else:
        failed = []
        options = ['--evaluation-input'] if args.evaluation_input else []
        for method in args.methods or METHODS:
            run = subprocess.run([sys.executable, __file__, *options, method], cwd=ROOT)
            if run.returncode:
                failed.append(method)
        if failed:
            raise SystemExit(f'Profiling failed: {failed}')


if __name__ == '__main__':
    main()
