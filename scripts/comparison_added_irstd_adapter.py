"""Official 2025 IRSTD architectures under the fixed train50 protocol.

Run in separate processes: author repositories reuse generic module names.
Changes to data/batch/schedule are disclosed in each immutable run config.
"""
import random
import sys
import types
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scripts.run_recent_a_comparison import RUN, DEFAULT_OUTPUT, MAX_EPOCHS, raw_pair, train_stats, module_from_file, commit, sha


class ZeroSafeSqrt(torch.autograd.Function):
    """Preserve sqrt forward exactly; select zero subgradient at zero norm."""
    @staticmethod
    def forward(ctx, value):
        root=value.sqrt();ctx.save_for_backward(root);return root
    @staticmethod
    def backward(ctx, gradient):
        root,=ctx.saved_tensors
        denominator=torch.where(root>0,2*root,torch.ones_like(root))
        return torch.where(root>0,gradient/denominator,torch.zeros_like(gradient))


def stabilize_iov_pde():
    from model.pde import pde
    class TorchProxy:
        sqrt=staticmethod(ZeroSafeSqrt.apply)
        def __getattr__(self,key):return getattr(torch,key)
    original=pde.forward
    namespace=dict(original.__globals__,torch=TorchProxy())
    pde.forward=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__)


def setup(method, m, device, smoke=False):
    size = m.get('working_size', 256)
    source = RUN/'sources'/method
    sys.path[:0] = [str(source), str(RUN/'runtime_packages'), str(DEFAULT_OUTPUT/'runtime_packages')]
    stats = train_stats(m)
    rgb = method in ('SAM-SPL','MLP-Net')
    identity = dict(source_commit=commit(source), adapter_sha256=sha(Path(__file__)),
        variant=method+'_train50', input_size=size, batch_size=2 if method=='SAM-SPL' else 4,
        normalization='RGB /255' if method=='MLP-Net' else 'train50-only grayscale mean/std (RGB repeated for SAM-SPL)',
        augmentation=f'resize{size}, independent horizontal/vertical flips and transpose; no extra labeled images',
        deviations=f'fixed train50, seed111, batch2/4, resize{size} with flips/transpose rather than native crop/copy-paste; all50 every epoch; fixed120 horizon; no validation/test model selection',
        train_only_statistics=stats, pretrained=None)
    if method=='SFDTNet':
        from model.ffnet import SFDTNet
        model = SFDTNet(in_channels=1,deep_supervision=True,mode='train')
        criterion = module_from_file('sfdt_native_loss',source/'loss.py').SoftIoULoss()
        lr=.05
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,MAX_EPOCHS,eta_min=1e-5)
        identity.update(loss='author sum of six SoftIoULoss heads',optimizer='Adam',lr=lr,
            inference_head='native test mode output head (not fused deep-supervision head)',
            source_issue='train.py default SFDTNet mismatches FFNet branch; instantiate official SFDTNet directly')
    elif method=='IOVarNet':
        from model.PIP import PIP
        stabilize_iov_pde()
        model=PIP(1,1,mode='train',deepsuper=True)
        def initialize(layer):
            if isinstance(layer,(torch.nn.Conv2d,torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(layer.weight,a=0,mode='fan_in')
            elif isinstance(layer,torch.nn.BatchNorm2d):
                torch.nn.init.normal_(layer.weight,1.,.02);torch.nn.init.constant_(layer.bias,0.)
        model.apply(initialize)
        criterion=torch.nn.BCELoss()
        lr=1e-4 if m['dataset']=='IRSTD-1k' else 1e-3
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,MAX_EPOCHS,eta_min=1e-5)
        identity.update(loss='author BCE on final sigmoid output',optimizer='Adam',lr=lr,gradient_clip=.5,
            numerical_fix='PDE sqrt forward unchanged; zero subgradient at zero directional norm prevents native SqrtBackward0 NaNs (anomaly-confirmed)')
    elif method=='MLP-Net':
        from model.MLPnet import MLPNet
        utils=module_from_file('mlp_native_utils',source/'utils.py')
        model=MLPNet(img_size=size,patch_size=4,in_chans=3,num_classes=1000,embed_dim=64,
            layers=[2,2,2,2],drop_rate=.5,norm_layer=torch.nn.BatchNorm2d,
            alpha=3,use_dropout=False,patch_norm=True)
        criterion=utils.WeightedDiceBCE(dice_weight=.5,BCE_weight=.5)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        scheduler=utils.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=5e-5)
        identity.update(loss='author WeightedDiceBCE (.5/.5)',optimizer='Adam',lr=.001,
            normalization='RGB /255',schedule='author 10-epoch cosine warm restarts, eta_min5e-5')
    elif method=='SAM-SPL':
        import yaml
        from sam_spl.base_model import make_adaptor
        cfg=yaml.safe_load((source/'sam_spl/configs/spl_t.yaml').read_text(encoding='utf-8'))
        cfg['sam_ckpt_path']=None
        model=make_adaptor(**cfg)
        checkpoint=RUN/'weights/sam2.1_hiera_tiny.pt'
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)['model']
        state=model.state_dict()
        # The author architecture intentionally truncates SAM after stage3.
        required={k for k in state if k.startswith('image_encoder.trunk.') and 'promote_genertor' not in k}
        loaded={k:weights[k] for k in state if k in weights and state[k].shape==weights[k].shape}
        if not required <= loaded.keys():raise ValueError('Missing frozen SAM backbone weights: '+str(required-loaded.keys()))
        model.load_state_dict(loaded,strict=False);model._freeze_encoder()
        Adan=module_from_file('spl_native_adan',source/'training/Adan.py').Adan
        optimizer=Adan(model.parameters(),lr=.01,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,MAX_EPOCHS,eta_min=1e-6)
        criterion=torch.nn.BCELoss()
        identity.update(loss='author sum BCE(sigmoid) over all deep-supervision logits',optimizer='author Adan',lr=.01,
            pretrained=str(checkpoint),pretrained_sha256=sha(checkpoint),loaded_pretrained_tensors=len(loaded),
            frozen_backbone_tensors=len(required),variant='SAM-SPL official spl_t (SAM2.1 tiny truncated stages) train50')
    else:raise ValueError(method)
    model=model.to(device)

    class FixedDataset(Dataset):
        def __len__(self):return len(m['train_ids'])
        def pair(self,sid,augment):
            x,y=raw_pair(m,sid,size,rgb=rgb)
            if augment:
                if random.random()<.5:x,y=x[::-1],y[::-1]
                if random.random()<.5:x,y=x[:,::-1],y[:,::-1]
                if random.random()<.5:x,y=x.transpose(1,0,2) if rgb else x.T,y.T
            if method=='MLP-Net':x=x/255.
            else:x=(x-stats['mean'])/stats['std']
            if not rgb:x=x[None]
            else:x=x.transpose(2,0,1)
            return torch.from_numpy(np.ascontiguousarray(x)),torch.from_numpy(np.ascontiguousarray(y))[None],sid
        def __getitem__(self,i):return self.pair(m['train_ids'][i],True)
    dataset=FixedDataset()
    loader=DataLoader(dataset,batch_size=identity['batch_size'],shuffle=True,num_workers=0,drop_last=False)
    def objective(batch,epoch):
        x,y,_=batch
        if method=='SFDTNet':model.mode='train'
        prediction=model(x.to(device));y=y.to(device)
        if method=='SAM-SPL':return sum(criterion(p.sigmoid(),y) for p in prediction)
        if method=='SFDTNet':return sum(criterion(p,y) for p in prediction)
        return criterion(prediction,y)
    def predict(sid):
        if method=='SFDTNet':model.mode='test'
        x,_,_=dataset.pair(sid,False)
        prediction=model(x[None].to(device))
        if method=='SAM-SPL':prediction=prediction[0].sigmoid()
        return prediction[0,0]
    return model,optimizer,scheduler,loader,objective,predict,identity
