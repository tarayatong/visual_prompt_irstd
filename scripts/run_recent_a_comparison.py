"""Isolated seven added baselines: fixed train50, report epoch120 only.

Every method is imported in its own process. Test labels are read only by the
final evaluation of both predeclared checkpoints after all training completes.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
import types
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.prepare_comparison import DEFAULT_OUTPUT, write_json
from visual_prompt.evaluation.comparison import image_metrics, summarize

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/comparison_recent_a_20261104'
from scripts.comparison_method_metadata import NEW_METHODS
METHODS = NEW_METHODS
DATASETS = ('IRSTD-1k', 'NUDT-SIRST', 'SIRST')
MAX_EPOCHS = 120
RETAIN = (120,)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def commit(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()

def seed_all():
    random.seed(111); np.random.seed(111); torch.manual_seed(111)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(111)
    torch.set_num_threads(2)

def manifest_for(dataset):
    m = json.loads((DEFAULT_OUTPUT/dataset/'protocol/manifest.json').read_text(encoding='utf-8'))
    assert len(m['train_ids']) == len(set(m['train_ids'])) == 50
    assert len(m['support_ids']) == 5 and set(m['support_ids']) <= set(m['train_ids'])
    assert not set(m['train_ids']) & set(m['test_ids'])
    assert len(m['test_ids']) == {'IRSTD-1k':201, 'NUDT-SIRST':664, 'SIRST':86}[dataset]
    return m

def raw_pair(m, sid, size, rgb=False):
    rec = m['records'][sid]
    with Image.open(rec['image']) as image:
        image = image.convert('RGB' if rgb else 'L')
        original_size = image.size
        x = np.array(image.resize((size,size), Image.Resampling.BILINEAR), dtype=np.float32)
    with Image.open(rec['mask']) as mask:
        # Canonicalize before author transforms (some upstream labels are 0/1).
        mask = mask.convert('L').resize(original_size, Image.Resampling.NEAREST)
        mask = Image.fromarray((np.asarray(mask)>0).astype(np.uint8)*255)
        y = np.array(mask.resize((size,size), Image.Resampling.NEAREST), dtype=np.float32)/255
    return x, y

def train_stats(m):
    total, square, n = 0., 0., 0
    for sid in m['train_ids']:
        a = np.asarray(Image.open(m['records'][sid]['image']).convert('L'), dtype=np.float64)
        total += a.sum(); square += np.square(a).sum(); n += a.size
    return dict(mean=float(total/n), std=float(max(square/n-(total/n)**2, 1e-12)**.5))

class RGBManifest(Dataset):
    def __init__(self, m, native, training):
        self.m, self.training = m, training
        self.ids = m['train_ids'] if training else m['test_ids']
        self.native = native.__new__(native)
        self.native.crop_size = self.native.base_size = 256
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        sid = self.ids[i]; rec = self.m['records'][sid]
        image = Image.open(rec['image']).convert('RGB')
        mask = Image.open(rec['mask']).convert('L').resize(image.size, Image.Resampling.NEAREST)
        mask = Image.fromarray((np.asarray(mask)>0).astype(np.uint8)*255)
        image, mask = (self.native._sync_transform if self.training else self.native._testval_sync_transform)(image, mask)
        x = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2,0,1)/255)
        x = (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        y = torch.from_numpy((np.array(mask.convert('L'))>0).astype(np.float32))[None]
        return x, y, sid

def replace_low_convs(module, pconv, prefix=''):
    changed = []
    for name, child in list(module.named_children()):
        path = f'{prefix}.{name}' if prefix else name
        if isinstance(child, torch.nn.Conv2d) and child.kernel_size == (3,3) and child.groups == 1:
            setattr(module, name, pconv(child.in_channels, child.out_channels, 3, child.stride[0]))
            changed.append(path)
        else:
            changed.extend(replace_low_convs(child, pconv, path))
    return changed

def setup_cnn(method, m, device, smoke=False):
    sys.path.append(str(DEFAULT_OUTPUT/'runtime_packages'))
    if method in ('PConv+SD', 'MSHNet+NS-FPN'):
        source = DEFAULT_OUTPUT/'sources'/('MSHNet' if method=='PConv+SD' else 'NS-FPN')
        sys.path.insert(0,str(source))
        if method=='PConv+SD':
            from model.MSHNet import MSHNet
            psource = RUN/'sources/PConv'
            PConv = module_from_file('recent_pconv', psource/'model/APConv.py').PConv
            loss = module_from_file('recent_sdm', psource/'model/loss.py').SLSIoULoss()
            model = MSHNet(3)
            # Explicit integration choice: both 3x3 convolutions of first two encoder levels.
            replacements = []
            for level in ('encoder_0','encoder_1'):
                replacements += replace_low_convs(getattr(model, level), PConv, level)
            identity = dict(source_commit=commit(psource), backbone_commit=commit(source),
                deviations='MSHNet integration: replace 3x3 convolutions in encoder_0/encoder_1 by author PConv; native SDM loss; placement is our declared choice, not an author end-to-end release',
                replaced_modules=replacements, variant='PConv+SD_MSHNet_train50')
        else:
            from scripts.comparison_nsfpn_adapter import enable_author_reference_backend
            enable_author_reference_backend()
            from model.MSHNet_NSFPN import MSHNet_NSFPN
            from model.loss import SLSIoULoss
            model, loss = MSHNet_NSFPN(3), SLSIoULoss()
            identity = dict(source_commit=commit(source),
                deviations='Author PyTorch deformable-attention reference backend; batch4; fixed500 horizon',
                variant='MSHNet+NS-FPN_train50')
        from utils.data import IRSTD_Dataset
        model = model.to(device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=.05)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1.)
        def objective(batch, epoch):
            x,y,_ = batch; x,y=x.to(device),y.to(device)
            auxiliary, logits = model(x,epoch>5)
            value = loss(logits,y,5,epoch)
            current = y
            for j, head in enumerate(auxiliary):
                if j: current = F.max_pool2d(current,2,2)
                value = value+loss(head,current,5,epoch)
            return value/(len(auxiliary)+1)
        def predict(sid):
            view = dict(m, test_ids=[sid])
            x,_,_ = RGBManifest(view,IRSTD_Dataset,False)[0]
            return model(x[None].to(device),True)[1].sigmoid()[0,0]
        loader = DataLoader(RGBManifest(m,IRSTD_Dataset,True),batch_size=4,shuffle=True,num_workers=0,drop_last=False)
        identity.update(optimizer='Adagrad',lr=.05,batch_size=4,input_size=256,normalization='ImageNet RGB')
        return model,optimizer,scheduler,loader,objective,predict,identity

    source=RUN/'sources/DGNet';sys.path.insert(0,str(source))
    sys.path.insert(0,str(RUN/'sources/CLIP'))
    from model import DGNetModel
    from dataset import InfraredSmallTargetDataset
    from utils import TrainingObjective
    from cda_loss import ConsensusKnowledgeDirectionalAlignmentLoss
    stats=train_stats(m)
    class BudgetDataset(InfraredSmallTargetDataset):
        def __init__(self):
            self.files=m['train_ids']; self.patch_size=self.base_size=256
            self.mode='train';self.use_aug=True;self.use_sync_transform=m['dataset']=='IRSTD-1k'
            self.img_norm_cfg=stats
        def _load_pair(self,sid):
            rec=m['records'][sid]
            image=Image.open(rec['image']).convert('I')
            mask=Image.open(rec['mask']).convert('L').resize(image.size,Image.Resampling.NEAREST)
            return image,Image.fromarray((np.asarray(mask)>0).astype(np.uint8)*255)
    model=DGNetModel().to(device)
    text_path=source/'weights/dgnet_text_features.pth'
    text=torch.load(text_path,map_location=device,weights_only=True)
    target=text['target'].float().reshape(1,512);background=text['background'].float().reshape(1,512)
    clip_path=RUN/'weights/ViT-B-32.pt'
    # Even CPU preflight uses the real frozen CLIP; never substitute a dummy CDA loss.
    cda=ConsensusKnowledgeDirectionalAlignmentLoss(device=str(device),model_name=str(clip_path),image_mean=stats['mean'],image_std=stats['std'])
    training_objective=TrainingObjective(model.loss,cda)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-4)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[300,450],gamma=.9)
    def objective(batch,epoch):
        image,augmented,y=batch
        x=torch.cat([image,augmented]).to(device);y=torch.cat([y,y]).to(device)
        out=model(x,target.expand(x.shape[0],-1),background.expand(x.shape[0],-1))
        return training_objective(epoch,out,y,x)
    def predict(sid):
        x,_=raw_pair(m,sid,256)
        x=torch.from_numpy((x-stats['mean'])/stats['std'])[None,None].to(device)
        return model(x,target,background)[0,0]
    loader=DataLoader(BudgetDataset(),batch_size=2,shuffle=True,num_workers=0,drop_last=False)
    identity=dict(source_commit=commit(source),variant='DGNet_train50',optimizer='Adam',lr=5e-4,
        batch_size=2,input_size=256,normalization=stats,clip_sha256=sha(clip_path),text_sha256=sha(text_path),
        deviations='batch2 doubled by native statistics augmentation vs author16; train50-only normalization; 500 rather than600 epochs; no published infrared checkpoint initialization')
    return model,optimizer,scheduler,loader,objective,predict,identity

def setup_sansa(m,device,smoke=False):
    sys.path.insert(0,str(RUN/'runtime_packages'))
    source=RUN/'sources/SANSA';sys.path.insert(0,str(source))
    from util import path_utils
    weight=RUN/'weights/sam2_hiera_tiny.pt'
    path_utils.SAM2_PATHS_CONFIG['tiny']=(str(weight),'../sam2/sam2_configs/sam2_hiera_t.yaml')
    from models.sansa.sansa import build_sansa
    from util.promptable_utils import build_prompt_dict
    from util.losses import loss_masks
    model=build_sansa('tiny',[2,3],.3,str(device)).to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,MAX_EPOCHS*50)
    mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
    def read(sid):
        x,y=raw_pair(m,sid,640,True)
        return (torch.from_numpy(x.transpose(2,0,1)/255)-mean)/std,torch.from_numpy(y)
    class Episodes(Dataset):
        def __len__(self):return 50
        def __getitem__(self,i):
            query=m['train_ids'][i]
            # Every allocated image is used once as query per epoch; no test data.
            supports=random.sample([sid for sid in m['train_ids'] if sid!=query],1)
            pairs=[read(sid) for sid in supports+[query]]
            return torch.stack([x for x,y in pairs]),torch.stack([y for x,y in pairs]),query
    loader=DataLoader(Episodes(),batch_size=1,shuffle=True,num_workers=0)
    def objective(batch,epoch):
        samples,masks,_=batch;samples=samples.to(device);masks=masks.to(device)
        prompt=build_prompt_dict(masks,'mask',1,True,device)
        out=model(samples,prompt)
        return sum(loss_masks(out['pred_masks'],masks,num_frames=1).values())
    def predict(sid):
        # Keep each test query independent. Support set remains the frozen five.
        xs,ys=zip(*(read(i) for i in m['support_ids']))
        # Do not read query GT: it is unnecessary to construct query prompts.
        rec=m['records'][sid]
        with Image.open(rec['image']) as image:
            a=np.array(image.convert('RGB').resize((640,640),Image.Resampling.BILINEAR),dtype=np.float32)
        q=(torch.from_numpy(a.transpose(2,0,1)/255)-mean)/std
        samples=torch.stack([*xs,q])[None].to(device)
        masks=torch.stack([*ys,torch.zeros_like(ys[0])])[None].to(device)
        prompt=build_prompt_dict(masks,'mask',5,False,device)
        return model(samples,prompt)['pred_masks'][-1].sigmoid()
    identity=dict(source_commit=commit(source),variant='SANSA_tiny_train50',optimizer='AdamW',lr=1e-4,
        batch_size=1,input_size=640,backbone_input_size=1024,support_shots=5,
        initialization_sha256=sha(weight),train_support_shots=1,
        deviations='author-supported tiny backbone rather than paper large; adapter stages2/3; fixed one-support training for10GB; inference uses frozen5; target-only adapter training instead of source benchmark training; native focal+Dice loss')
    return model,optimizer,scheduler,loader,objective,predict,identity

def setup_pahnet(m,device,smoke=False):
    source=RUN/'sources/PAHNet'
    sys.path[:0]=[str(source),str(RUN/'runtime_packages'),str(DEFAULT_OUTPUT/'runtime_packages')]
    # Import only the unmodified mmseg utilities actually used by PAHNet.
    # Importing the entire model registry unnecessarily loads unrelated CUDA ops.
    import mmseg
    base=Path(mmseg.__file__).parent/'models'
    namespace=types.ModuleType('mmseg.models');namespace.__path__=[str(base)]
    sys.modules['mmseg.models']=namespace
    utils=types.ModuleType('mmseg.models.utils');utils.__path__=[str(base/'utils')]
    sys.modules['mmseg.models.utils']=utils
    embed=module_from_file('recent_mmseg_embed',base/'utils/embed.py')
    shapes=module_from_file('recent_mmseg_shapes',base/'utils/shape_convert.py')
    utils.PatchEmbed=embed.PatchEmbed;utils.nchw_to_nlc=shapes.nchw_to_nlc;utils.nlc_to_nchw=shapes.nlc_to_nchw
    import HDMNet_PAHNet.model.HDMNetPlus as native
    import HDMNet_PAHNet.model.resnet as backbone
    import HDMNet_PAHNet.model.resnet_ssp as ssp_backbone
    checkpoint=RUN/'weights/pahnet_hdm_pascal_split0_5shot.pth'
    full=torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']
    full={k.removeprefix('module.'):v for k,v in full.items()}
    args=SimpleNamespace(data_set='pascal',low_fea='layer2',layers=50,ignore_label=255,
        shot=5,vgg=False,pre_weight='retained_full_PAHNet_PSP_component',split=0,kshot_trans_dim=2,
        weight_decay=.01,zoom_factor=8)
    # The released full checkpoint embeds both pretrained components. Recover
    # their exact tensors for the author's constructor, then strict-load all980 keys.
    original_load=torch.load;original_backbone=backbone.resnet50;original_ssp=ssp_backbone.resnet50
    backbone.resnet50=lambda pretrained=True,**kw:original_backbone(pretrained=False,**kw)
    ssp_backbone.resnet50=lambda pretrained=True:original_ssp(pretrained=False)
    psp=native.PSPNet(args)
    psp_weights={}
    for key in psp.state_dict():
        mapped=key
        if key.startswith('encoder.'):
            pieces=key.split('.');mapped='layer'+pieces[1]+'.'+'.'.join(pieces[2:])
        if mapped.startswith('cls.'):
            pieces=mapped.split('.');index=int(pieces[1])
            if index>=2:mapped='base_learnear.'+str(index-2)+'.'+'.'.join(pieces[2:])
        if mapped not in full:raise KeyError(f'Missing pretrained PSP tensor {key} -> {mapped}')
        psp_weights[key]=full[mapped]
    def component_load(path,*a,**kw):
        if str(path)==args.pre_weight:return {'state_dict':psp_weights}
        if 'initmodel/SSPNet/' in str(path).replace('\\','/'):
            return {k[4:]:v for k,v in full.items() if k.startswith('ssp.')}
        return original_load(path,*a,**kw)
    torch.load=component_load
    try:model=native.OneModel(args,cls_type='Novel')
    finally:
        torch.load=original_load;backbone.resnet50=original_backbone;ssp_backbone.resnet50=original_ssp
    model.load_state_dict(full,strict=True);model.freeze_modules(model);model=model.to(device)
    optimizer=model.get_optim(model,args,1e-4)
    optimized={id(p) for group in optimizer.param_groups for p in group['params']}
    for p in model.parameters():p.requires_grad_(id(p) in optimized)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:(1-min(e,MAX_EPOCHS)/MAX_EPOCHS)**.9)
    mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
    size=257
    def image_only(sid):
        with Image.open(m['records'][sid]['image']) as image:
            a=np.array(image.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)
        return (torch.from_numpy(a.transpose(2,0,1)/255)-mean)/std
    def label(sid):
        _,y=raw_pair(m,sid,size,True);return torch.from_numpy(y)
    class Episodes(Dataset):
        def __len__(self):return 50
        def __getitem__(self,i):
            query=m['train_ids'][i];supports=random.sample([s for s in m['train_ids'] if s!=query],5)
            return image_only(query),label(query),torch.stack([image_only(s) for s in supports]),torch.stack([label(s) for s in supports])
    loader=DataLoader(Episodes(),batch_size=1,shuffle=True,num_workers=0)
    def objective(batch,epoch):
        for layer in model.modules():
            if isinstance(layer,torch.nn.BatchNorm2d):layer.eval()
        x,y,sx,sy=[v.to(device) for v in batch]
        result=model(x=x,y_m=y,y_b=torch.zeros_like(y),s_x=sx,s_y=sy)
        return result[1]+result[2]+result[3]
    def predict(sid):
        sx=torch.stack([image_only(s) for s in m['support_ids']])[None].to(device)
        sy=torch.stack([label(s) for s in m['support_ids']])[None].to(device)
        return model(x=image_only(sid)[None].to(device),s_x=sx,s_y=sy)[0].softmax(1)[0,1]
    identity=dict(source_commit=commit(source),variant='PAHNet_HDM_Pascal_split0_train50',optimizer='native AdamW',lr=1e-4,
        batch_size=1,input_size=size,support_shots=5,train_support_shots=5,initialization_sha256=sha(checkpoint),
        pretrained_state_keys=len(full),initialization_strict=True,
        deviations='author full Pascal split0 5shot source pretraining; target50 novel-class adaptation, not native source benchmark training; batch1 and257 input vs4 and473; all target pixels map to background for frozen Pascal base learner; original prototype predictor/fusion/native three loss terms retained; no test-based adaptation')
    return model,optimizer,scheduler,loader,objective,predict,identity

def evaluate(model,predict,m,folder,config,epoch,checkpoint_sha):
    model.eval();pred_dir=folder/f'epoch_{epoch:04d}'/'predictions';pred_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sid in m['test_ids']:
        with torch.no_grad(): a=predict(sid).float().cpu().numpy()
        h,w=m['records'][sid]['size_hw'];a=cv2.resize(a,(w,h),interpolation=cv2.INTER_LINEAR)
        if not np.isfinite(a).all():raise ValueError(f'Nonfinite prediction {sid}')
        # Save compact binary masks, enough to recompute every reported metric.
        binary=(a>=.5).astype(np.uint8)*255
        Image.fromarray(binary).save(pred_dir/f'{sid}.png')
        with Image.open(m['records'][sid]['mask']) as im:
            gt=np.array(im.convert('L').resize((w,h),Image.Resampling.NEAREST))>0
        rows.append(image_metrics(binary>0,gt,sid))
    if {p.stem for p in pred_dir.glob('*.png')} != set(m['test_ids']):
        raise ValueError('Prediction coverage mismatch')
    out=pred_dir.parent/'metrics';out.mkdir(exist_ok=True)
    with (out/'per_image.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=summarize(rows)
    result.update(config,epochs=epoch,training_horizon=config['training_horizon'],
                  complete_test=config.get('evaluation_split','test') == 'test',checkpoint_sha256=checkpoint_sha,
                  prediction_coverage_verified=True)
    write_json(out/'summary.json',result)
    print(json.dumps(dict(event='evaluation_complete',method=config['method'],dataset=m['dataset'],epoch=epoch,IoU=result['IoU_percent'])),flush=True)

def saved_model(model):
    # SANSA frozen SAM weights are separately content-addressed in config.
    if hasattr(model,'sam'):
        return {k:v for k,v in model.state_dict().items() if 'adapter' in k}
    return model.state_dict()

def load_saved(model,weights):
    if hasattr(model,'sam'):
        expected={k for k in model.state_dict() if 'adapter' in k}
        if set(weights)!=expected:raise ValueError('Adapter checkpoint keys changed')
        model.load_state_dict(weights,strict=False)
    else:model.load_state_dict(weights,strict=True)

def worker(args):
    seed_all();m=manifest_for(args.dataset);device=torch.device(args.device)
    folder=RUN/args.dataset/args.method
    summary_path=folder/'epoch_0120/metrics/summary.json'
    if not args.smoke and summary_path.exists():
        from scripts.report_recent_a_comparison import verified_summary
        verified_summary(summary_path,m,120)
        print('Existing verified epoch120 result retained; no retraining',flush=True)
        return
    from scripts.comparison_added_irstd_adapter import setup as setup_added
    model,optimizer,scheduler,loader,objective,predict,identity=(setup_pahnet(m,device,args.smoke) if args.method=='PAHNet' else setup_sansa(m,device,args.smoke) if args.method=='SANSA'
        else setup_added(args.method,m,device,args.smoke) if args.method in ('SAM-SPL','SFDTNet','IOVarNet','MLP-Net')
        else setup_cnn(args.method,m,device,args.smoke))
    if args.smoke:
        model.train();batch=next(iter(loader));value=objective(batch,6)
        if not torch.isfinite(value):raise ValueError('Nonfinite smoke loss')
        value.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        optimizer.step();model.eval()
        # Training image only; no test labels are touched by preflight.
        with torch.no_grad():a=predict(m['train_ids'][0])
        assert a.ndim==2 and torch.isfinite(a).all()
        evidence=dict(event='smoke_pass',method=args.method,dataset=args.dataset,device=str(device),
                      loss=float(value.detach()),shape=list(a.shape),identity=identity)
        write_json(RUN/'preflight'/str(device)/f'{args.method}_{args.dataset}.json',evidence)
        print(json.dumps(evidence),flush=True)
        return
    folder.mkdir(parents=True,exist_ok=True)
    retained=folder/'checkpoints/epoch_0120.pt'
    if retained.exists():
        # Reuse the already saved checkpoint; never train it again after schedule change.
        state=torch.load(retained,map_location='cpu',weights_only=False)
        old=state['config']
        assert state['epoch']==120 and old['method']==args.method and old['dataset']==args.dataset
        assert old['manifest_sha256']==m['manifest_sha256'] and old['train_annotation_ids']==m['train_ids']
        assert old['actual_annotation_budget']==50 and old['seed']==111
        assert old['source_commit']==identity['source_commit']
        load_saved(model,state['model'])
        reused=dict(old,reused_existing_epoch120=True,
            reporting_protocol='only epoch120; existing checkpoint retained with its original learning-rate horizon; no further training')
        evaluate(model,predict,m,folder,reused,120,sha(retained))
        write_json(folder/'status.json',dict(status='complete',retained_epochs=[120],reused_existing_epoch120=True))
        return
    config=dict(method=args.method,dataset=args.dataset,seed=111,actual_annotation_budget=50,
        train_annotation_ids=m['train_ids'],manifest_sha256=m['manifest_sha256'],training_horizon=MAX_EPOCHS,
        runner_sha256=sha(Path(__file__)),
        retained_epochs=list(RETAIN),threshold=.5,test_selection='fixed120 only; evaluate after training only',
        epoch_definition='one pass over all50 training images, including final partial batch',**identity)
    path=folder/'config.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=config:raise ValueError('Run identity changed')
    write_json(path,config);cp=folder/'checkpoints/latest.pt';history=[];start=0
    if cp.exists():
        state=torch.load(cp,map_location='cpu',weights_only=False)
        if state['config']!=config:raise ValueError('Checkpoint configuration mismatch')
        load_saved(model,state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start=state['epoch'];history=state['history']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(start,MAX_EPOCHS):
        t0=time.monotonic();model.train();losses=[]
        for batch in loader:
            optimizer.zero_grad(set_to_none=True);value=objective(batch,epoch)
            if not torch.isfinite(value):raise ValueError(f'Nonfinite loss at epoch{epoch+1}')
            value.backward()
            gradients=[p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or any(not torch.isfinite(p.grad).all() for p in gradients):raise ValueError('Invalid gradients')
            if args.method=='SANSA':torch.nn.utils.clip_grad_norm_(gradients,.1)
            if args.method=='IOVarNet':torch.nn.utils.clip_grad_norm_(gradients,.5)
            optimizer.step();losses.append(float(value.detach()))
            if args.method=='SANSA':scheduler.step()
        if args.method!='SANSA':scheduler.step()
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.monotonic()-t0,
                            unique_supervised_images=50,optimizer_updates=(epoch+1)*len(loader)))
        write_json(folder/'training_history.json',history)
        write_json(folder/'status.json',dict(status='training',epoch=epoch+1,maximum_epochs=MAX_EPOCHS))
        if (epoch+1)%25==0 or epoch+1 in RETAIN:
            cp.parent.mkdir(exist_ok=True)
            state=dict(config=config,epoch=epoch+1,model=saved_model(model),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
            tmp=cp.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(cp)
            if epoch+1 in RETAIN:torch.save(state,cp.parent/f'epoch_{epoch+1:04d}.pt')
        print(f'{args.method} {args.dataset} epoch{epoch+1}/{MAX_EPOCHS} loss{history[-1]["loss"]:.6f} seconds{history[-1]["seconds"]:.2f}',flush=True)
    for epoch in RETAIN:
        checkpoint=cp.parent/f'epoch_{epoch:04d}.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['config']!=config or state['epoch']!=epoch:raise ValueError('Retained identity mismatch')
        load_saved(model,state['model']);evaluate(model,predict,m,folder,config,epoch,sha(checkpoint))
    write_json(folder/'status.json',dict(status='complete',retained_epochs=list(RETAIN)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--device',default='cuda');args=p.parse_args()
    if args.method=='MPA':
        command=[sys.executable,'-u','-m','scripts.run_fss50','--method','MPA','--dataset',args.dataset,
                 '--epochs','500','--run-root',str(RUN),'--retain-epochs','120','500']
        if args.smoke:command+=['--preflight' if args.device=='cpu' else '--gpu-smoke']
        subprocess.run(command,check=True)
        if args.smoke:
            write_json(RUN/'preflight'/args.device/f'MPA_{args.dataset}.json',
                       dict(status='passed',device=args.device,dataset=args.dataset,command=command,
                            scope='augmentation50 and native50support CPU forward' if args.device=='cpu' else 'native400 GPU training/backward and50support inference'))
    else:worker(args)

if __name__=='__main__':main()
