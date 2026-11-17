"""Validation-selected FSS input/representation adaptation; preserve native runs."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import types
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from scripts.prepare_comparison import DEFAULT_OUTPUT,write_json
from scripts.run_comparison_recovery_exploration import event,save_csv,process_alive
from scripts.run_comparison_task_adaptation import DEFAULT_RUN,load_manifest,load_split,seed_all,ensure_finite_gradients
from visual_prompt.evaluation.comparison import image_metrics,summarize
from visual_prompt.evaluation.spatial_alignment import restore_to_original


def supports(m,transform,crop=0):
    sx=[];sy=[]
    for sid in m['support_ids']:
        rec=m['records'][sid];im=Image.open(rec['image']).convert('RGB')
        mask=Image.open(rec['mask']).convert('L').resize(im.size,Image.Resampling.NEAREST)
        if crop:
            points=np.argwhere(np.asarray(mask)>0)
            if len(points):
                center=points.mean(0);cy,cx=points[np.square(points-center).sum(1).argmin()]
                left=int(np.clip(cx-crop/2,0,max(0,im.width-crop)))
                top=int(np.clip(cy-crop/2,0,max(0,im.height-crop)))
                box=(left,top,min(im.width,left+crop),min(im.height,top+crop));im=im.crop(box);mask=mask.crop(box)
        sx.append(transform(im)[None].cuda())
        size=tuple(reversed(sx[-1].shape[-2:]));a=np.asarray(mask.resize(size,Image.Resampling.NEAREST))>0
        sy.append(torch.from_numpy(a.astype(np.float32))[None].cuda())
    if not any(bool(mask.any()) for mask in sy):
        raise ValueError('All resized support masks are empty; refusing artificial foreground fallback')
    return sx,sy


def local_side_for(m):
    sizes=[min(m['records'][sid]['size_hw']) for sid in m['train_ids']]
    return max(64,min(256,int(np.median(sizes)//2)))


def windows(size,side=256):
    w,h=size
    def positions(n):
        return sorted(set(list(range(0,max(1,n-side+1),max(1,int(side*.75))))+[max(0,n-side)]))
    return [(x,y,min(w,x+side),min(h,y+side)) for y in positions(h) for x in positions(w)]


def score_predictions(predict,m,ids,out,label):
    folder=out/label;folder.mkdir(parents=True,exist_ok=True);rows=[]
    for index,sid in enumerate(ids):
        dest=folder/f'{sid}.png';rec=m['records'][sid]
        if not dest.exists():
            seed_all(111+index)
            # The predictor receives only the image path; no query label or GT geometry.
            pred=predict(rec['image'])
            assert pred.shape==tuple(rec['size_hw']) and np.isfinite(pred).all()
            Image.fromarray((pred>.5).astype(np.uint8)*255).save(dest)
            if hasattr(predict,'cache_audit') and predict.cache_audit.get('passed'):
                write_json(out/'support_cache_check.json',predict.cache_audit)
        pred=np.asarray(Image.open(dest).convert('L'))>0
        h,w=rec['size_hw'];gt=np.asarray(Image.open(rec['mask']).convert('L').resize((w,h),Image.Resampling.NEAREST))>0
        rows.append(image_metrics(pred,gt,sid))
        if (index+1)%10==0:event(out,'evaluation',group=label,images=index+1,total=len(ids))
    save_csv(out/'metrics'/f'{label}_per_image.csv',rows)
    return summarize(rows)


def configure_pool(model,mode):
    if mode=='original':return
    def pooling(self,feature,mask):
        mass=F.interpolate(mask.unsqueeze(1),size=feature.shape[-2:],mode='area')
        return (feature*mass).sum((2,3))/(mass.sum((2,3))+1e-5)
    model.masked_average_pooling=types.MethodType(pooling,model)


def make_episode(native,m,ids,transform,out,method):
    import albumentations as A
    folder=out/'canonical_train_masks';folder.mkdir(parents=True,exist_ok=True);lookup={}
    for sid in ids:
        rec=m['records'][sid];im=Image.open(rec['image'])
        mask=np.asarray(Image.open(rec['mask']).convert('L').resize(im.size,Image.Resampling.NEAREST))>0
        p=folder/f'{sid}.png';Image.fromarray(mask.astype(np.uint8)*255).save(p);lookup[str(p)]=rec['image']
    class Budget(native):
        def sample_episode(self,index):
            paths=[str(folder/f'{sid}.png') for sid in self.active_ids]
            return paths[0],paths,0
        def load_frame(self,query_name,support_names):
            images=[Image.open(lookup[p]).convert('RGB') for p in support_names]
            return images[0],self.read_mask(support_names[0]),images,support_names
    d=Budget.__new__(Budget);d.shot=5;d.num=math.ceil(len(ids)/5);d.transform=transform;d.active_ids=ids[:5]
    color=A.Compose([A.RandomBrightnessContrast(p=.5),A.HueSaturationValue(hue_shift_limit=20,sat_shift_limit=30,val_shift_limit=20,p=1)])
    if method=='MPA':d.q_colorAug=color
    else:
        d.q_transform=A.Compose([A.HorizontalFlip(p=.5),A.VerticalFlip(p=.5),A.RandomRotate90()]);d.q_transform2=color
    return d


def balanced_ce_loss(logits,target):
    loss=F.cross_entropy(logits,target,ignore_index=255,reduction='none')
    axes=tuple(range(1,target.ndim));pos=target==1;neg=target==0
    fg=(loss*pos).sum(axes)/pos.sum(axes).clamp_min(1)
    bg=(loss*neg).sum(axes)/neg.sum(axes).clamp_min(1)
    return torch.where(pos.sum(axes)>0,.5*(fg+bg),bg).mean()


def train_episode_model(args,pool,ids,out,updates,balanced_ce=False,decay_updates=None):
    from scripts.run_fss50 import native_model,ifa_loss
    seed_all();m=load_manifest(args.dataset);model,native,lossfn=native_model(DEFAULT_OUTPUT,args.method)
    configure_pool(model,pool);model.cuda()
    transform=transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    config=dict(method=args.method,dataset=args.dataset,pool=pool,training_ids=ids,seed=111,updates=updates,
                B=50,training_episode_shots=5,deployment_shots=5,loss='native output coefficients; balanced class-mean CE' if balanced_ce else 'native coefficients and CE',
                balanced_ce=balanced_ce,initialization='external pretraining only',input_size=400,
                version='A area pooling' if pool=='area' else 'T original pooling, bounded update budget',
                maximum_epochs=100,old_MPA_paused_checkpoint_read=False)
    decay_updates=decay_updates if decay_updates is not None else [updates//3,updates*2//3]
    config.update(optimizer='SGD',lr=.0005 if args.method=='IFA' else .001,momentum=.9,weight_decay=.0005,
                  learning_rate_decay_updates=decay_updates,MPA_phase=1 if args.method=='MPA' else None)
    if (out/'model.pt').exists():
        state=torch.load(out/'model.pt',map_location='cpu',weights_only=False)
        assert state['config']==config,'Completed training config differs; use a new output directory'
        model.load_state_dict(state['model']);return model,transform
    write_json(out/'training_config.json',config)
    d=make_episode(native,m,ids,transform,out,args.method)
    lr=.0005 if args.method=='IFA' else .001
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=lr,momentum=.9,weight_decay=.0005)
    ce=balanced_ce_loss if balanced_ce else torch.nn.CrossEntropyLoss(ignore_index=255)
    step=0;epoch=0;history=[];phase=1
    while step<updates:
        model.train();model.shot=5
        for module in model.modules():
            if isinstance(module,torch.nn.BatchNorm2d):module.eval()
        order=list(ids);random.Random(111+epoch).shuffle(order);losses=[];used=[]
        # Both allocated counts are divisible by5; each pass covers every training label.
        for offset in range(0,len(order),5):
            d.active_ids=order[offset:offset+5]
            sx,sy,qx,qy,*_=d[offset//5];sx=[x[None].cuda() for x in sx];sy=[y[None].cuda() for y in sy]
            opt.zero_grad(set_to_none=True)
            if args.method=='MPA':
                # Bounded phase1 diagnostic: same view/stage in original and area arms.
                qx=[q[None].cuda() for q in qx[:phase]];qy=[q[None].long().cuda() for q in qy[:phase]]
                output=model(sx,sy,qx,qy);parts=lossfn(*output,qy,torch.cat(sy).long(),True,ce)
                loss=parts[0]+.4*parts[1]+.1*parts[2]+parts[3]+.2*parts[4]
            else:
                qx=qx[None].cuda();qy=qy[None].long().cuda();output=model(sx,sy,qx,qy)
                loss=ifa_loss(output,qy,torch.cat(sy).long(),ce)
            if not torch.isfinite(loss):raise ValueError('nonfinite episode loss')
            loss.backward()
            ensure_finite_gradients(model)
            opt.step();step+=1;losses.append(float(loss.detach()));used+=d.active_ids
            if step in decay_updates:
                for group in opt.param_groups:group['lr']/=10
            del output,loss,sx,sy,qx,qy
            if args.method=='MPA':del parts
            if step>=updates:break
        epoch+=1;history.append(dict(epoch=epoch,updates=step,loss=float(np.mean(losses)),phase=phase))
        assert set(used)<=set(ids)
        save_csv(out/'training_curve.csv',history)
        if epoch%10==0:event(out,'training',**history[-1])
    torch.save(dict(model=model.state_dict(),config=config),out/'model.pt')
    return model,transform


def episode_predictor(model,method,m,transform,tiled=False):
    model.eval();model.shot=5;side=local_side_for(m)
    stages=[model.layer0,model.layer1,model.layer2,model.layer3]
    originals=[]
    for stage in stages:
        original=getattr(stage,'_support_cache_native_forward',stage.forward)
        stage.forward=original;originals.append(original)
    sx,sy=supports(m,transform,crop=side if tiled else 0)
    caches=[{} for _ in stages];control=dict(enabled=True);audit=dict(passed=False)
    with torch.no_grad():
        for tensor in sx:
            value=tensor
            for index,original in enumerate(originals):
                output=original(value);caches[index][value.data_ptr()]=(value,output);value=output
    for index,stage in enumerate(stages):
        stage._support_cache_native_forward=originals[index]
        def forward(self,value,_index=index):
            if control['enabled'] and value.data_ptr() in caches[_index]:
                return caches[_index][value.data_ptr()][1]
            return originals[_index](value)
        stage.forward=types.MethodType(forward,stage)
    def run(q):
        output=model(sx,sy,[q],[None]) if method=='MPA' else model(sx,sy,q,None)
        if not audit['passed']:
            control['enabled']=False
            reference=model(sx,sy,[q],[None]) if method=='MPA' else model(sx,sy,q,None)
            control['enabled']=True
            torch.testing.assert_close(output[0],reference[0],rtol=1e-5,atol=1e-5)
            audit.update(passed=True,supports=len(sx),max_abs_difference=float((output[0]-reference[0]).abs().max()),
                         policy='fixed support features only; query encoding recomputed')
        return output
    @torch.no_grad()
    def predict(path):
        image=Image.open(path).convert('RGB');w,h=image.size
        if not tiled:
            q=transform(image)[None].cuda()
            output=run(q)
            pred=output[0].argmax(1)[0].float().cpu().numpy()
            return restore_to_original(pred,(h,w),discrete=True)
        result=np.zeros((h,w),np.float32);count=np.zeros_like(result)
        for box in windows(image.size,side):
            q=transform(image.crop(box))[None].cuda()
            output=run(q)
            prob=output[0].softmax(1)[0,1].cpu().numpy()
            left,top,right,bottom=box;prob=cv2.resize(prob,(right-left,bottom-top),interpolation=cv2.INTER_LINEAR)
            result[top:bottom,left:right]+=prob;count[top:bottom,left:right]+=1
        return (result/count)>.5
    predict.cache_audit=audit
    return predict


def episodic_worker(args):
    from scripts.run_fss50 import native_model
    seed_all();m=load_manifest(args.dataset);split=load_split(args.dataset);root=args.output/args.method/args.dataset;root.mkdir(parents=True,exist_ok=True)
    if (root/'report.json').exists():return
    candidates=[];base_updates=800 if args.method=='IFA' else 200
    # Reuse completed40/10 IFA training only, independently restore/score its fixed100-epoch model.
    for pool in ('original','area'):
        folder=root/'validation'/pool;folder.mkdir(parents=True,exist_ok=True)
        old=Path('outputs/comparison_recovery_20261102/IFA')/args.dataset/f'IFA-2_{pool}_retry1/111/latest.pt'
        checkpoint=folder/'model.pt'
        if args.method=='IFA' and old.exists():
            state=torch.load(old,map_location='cpu',weights_only=False)
            assert state['config']['fit_ids']==split['fit_ids'] and state['epoch']==100
            model,_,_=native_model(DEFAULT_OUTPUT,args.method);configure_pool(model,pool);model.load_state_dict(state['model']);model.cuda()
            transform=transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
            write_json(folder/'reuse.json',dict(source=str(old),scope='train40 weights only; rescore val10 with consistent spatial restoration'))
            candidate_decay=[34*8,67*8]  # exact legacy schedule, not rounded thirds
        else:
            model,transform=train_episode_model(args,pool,split['fit_ids'],folder,base_updates)
            candidate_decay=[base_updates//3,base_updates*2//3]
        for tiled in (False,True) if pool=='area' else (False,):
            name=pool+(f'_tiles{local_side_for(m)}' if tiled else '')
            metric=score_predictions(episode_predictor(model,args.method,m,transform,tiled),m,split['validation_ids'],folder,name)
            candidates.append(dict(pool=pool,tiled=tiled,updates=base_updates,decay_updates=candidate_decay,validation=metric))
            write_json(root/'validation_results.json',candidates)
        del model;gc.collect();torch.cuda.empty_cache()
    if max(c['validation']['IoU_percent'] for c in candidates)<20:
        # One predeclared loss follow-up to the chosen representation/input; no test scores.
        base=max(candidates,key=lambda c:(c['validation']['IoU_percent'],-c['validation']['Fa_per_million']))
        folder=root/'validation'/'balanced_ce';folder.mkdir(parents=True,exist_ok=True)
        model,transform=train_episode_model(args,base['pool'],split['fit_ids'],folder,base_updates,True,base['decay_updates'])
        metric=score_predictions(episode_predictor(model,args.method,m,transform,base['tiled']),m,split['validation_ids'],folder,'predictions')
        candidates.append(dict(pool=base['pool'],tiled=base['tiled'],balanced_ce=True,updates=base_updates,
                               decay_updates=base['decay_updates'],validation=metric))
        write_json(root/'validation_results.json',candidates)
        del model;gc.collect();torch.cuda.empty_cache()
    selected=max(candidates,key=lambda c:(c['validation']['IoU_percent'],-c['validation']['Fa_per_million'],-c['updates']))
    write_json(root/'selection.json',dict(**selected,official_test_labels_read=False,version='IR-adapted/budget-adapted; no native result overwritten'))
    final=root/'full50';final.mkdir(exist_ok=True)
    model,transform=train_episode_model(args,selected['pool'],split['allocated_ids'],final,selected['updates'],
                                      selected.get('balanced_ce',False),selected['decay_updates'])
    score=score_predictions(episode_predictor(model,args.method,m,transform,selected['tiled']),m,m['test_ids'],final,'predictions')
    score.update(method=args.method,dataset=args.dataset,complete_test=True,annotation_budget=50,support_shots=5,
                 pool=selected['pool'],query_tile_side=local_side_for(m) if selected['tiled'] else None,
                 training_updates=selected['updates'],version='IR-adapted/budget-adapted')
    write_json(final/'metrics/summary.json',score);write_json(root/'report.json',dict(status='complete',method=args.method,dataset=args.dataset,selection=selected,test_metrics=score))


def affinity_worker(args):
    """ABC/TVG: isolate adaptation duration, then foreground-preserving pooling/text."""
    seed_all();m=load_manifest(args.dataset);split=load_split(args.dataset)
    root=args.output/args.method/args.dataset;root.mkdir(parents=True,exist_ok=True)
    if (root/'report.json').exists():return
    source=DEFAULT_OUTPUT/'sources'/args.method
    sys.path[:0]=[str(source.resolve()),str((DEFAULT_OUTPUT/'runtime_packages').resolve())]
    torch.hub.set_dir(str((DEFAULT_OUTPUT/'runtime_cache/torch_hub').resolve()))
    cam=None;native_cam=None
    if args.method=='TVGTANet':
        from scripts.comparison_tvgtanet_adapter import load_native,NativeInfraredCAM
        runner,native_cam,_=load_native(DEFAULT_OUTPUT)
        cam=NativeInfraredCAM(native_cam,DEFAULT_OUTPUT/'weights/ViT-B-16.pt')
    else:
        from core import runner
    from core.backbone import Backbone
    from utils import segutils
    backbone=Backbone('resnet50').cuda().eval();backbone.requires_grad_(False)
    native_downsample=segutils.downsample_mask
    transform=transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    sx,sy=supports(m,transform);sx=torch.cat(sx)[None];sy=torch.cat(sy)[None]
    original_epochs=25 if args.method=='ABCDFSS' else 20
    def predictor(spec,folder):
        side=local_side_for(m)
        if spec.get('local'):
            lsx,lsy=supports(m,transform,crop=side);lsx=torch.cat(lsx)[None];lsy=torch.cat(lsy)[None]
        else:lsx,lsy=sx,sy
        if spec['pool']=='area':
            def downsample(mask,h,w):
                x=mask.unsqueeze(1)
                if h<x.shape[-2] or w<x.shape[-1]:return F.interpolate(x,size=(h,w),mode='area').squeeze(1)
                return F.interpolate(x,size=(h,w),mode='bilinear',align_corners=False).squeeze(1)
            segutils.downsample_mask=downsample
        else:segutils.downsample_mask=native_downsample
        runner.set_args(SimpleNamespace(benchmark='infrared' if cam else 'lung',datapath='',nshot=5,
                        adapt_to='every-episode',postprocessing='off',logpath=str(folder/'logs'),verbosity=0))
        config=runner.makeConfig();config.fitting.num_epochs=spec['epochs'];config.featext.fit_every_episode=True
        maker=runner.makeFeatureMaker(SimpleNamespace(class_ids=[0]),config,device=torch.device('cuda'),feat_extr_method=backbone.extract_feats)
        def get_cam(path):
            sid=Path(path).stem;dest=root/'cams'/spec['template']/f'{sid}.npy';dest.parent.mkdir(parents=True,exist_ok=True)
            old=DEFAULT_OUTPUT/'full16'/args.dataset/'TVGTANet/clip_cams'/f'{sid}.npy'
            if not dest.exists():
                if spec['template']=='original' and old.exists():np.save(dest,np.load(old))
                else:np.save(dest,cam(path).numpy())
            return torch.from_numpy(np.load(dest)).float()
        if cam:
            template='a clean origami {}.' if spec['template']=='original' else 'a thermal image of {}.'
            cam.bg=native_cam.zeroshot_classifier(['infrared background'],[template],cam.model)
            cam.fg=native_cam.zeroshot_classifier(['infrared small target'],[template],cam.model)
            support_paths=[]
            for sid in m['support_ids']:
                rec=m['records'][sid]
                if not spec.get('local'):support_paths.append(rec['image']);continue
                im=Image.open(rec['image']).convert('RGB')
                mask=np.asarray(Image.open(rec['mask']).convert('L').resize(im.size,Image.Resampling.NEAREST))>0
                points=np.argwhere(mask);center=points.mean(0);cy,cx=points[np.square(points-center).sum(1).argmin()]
                left=int(np.clip(cx-side/2,0,max(0,im.width-side)));top=int(np.clip(cy-side/2,0,max(0,im.height-side)))
                dest=root/'local_cam_inputs'/f'{sid}__support_side{side}.png';dest.parent.mkdir(parents=True,exist_ok=True)
                im.crop((left,top,min(im.width,left+side),min(im.height,top+side))).save(dest);support_paths.append(str(dest))
            support_cams=torch.stack([get_cam(path) for path in support_paths])[None].cuda()
        def run(path):
            batch=dict(query_img=transform(Image.open(path).convert('RGB'))[None].cuda(),support_imgs=lsx,support_masks=lsy,class_id=torch.tensor([0],device='cuda'))
            if cam:batch.update(query_pascal_label=get_cam(path)[None].cuda(),support_pascal_labels=support_cams)
            task=runner.SingleSampleEval(batch,maker)
            _,pred=task.forward();array=pred[0].detach().float().cpu().numpy()
            w,h=Image.open(path).size;del task,batch,pred
            return restore_to_original(array,(h,w),discrete=True)
        def predict(path):
            if not spec.get('local'):return run(path)
            image=Image.open(path).convert('RGB');w,h=image.size
            result=np.zeros((h,w),np.float32);count=np.zeros_like(result)
            for left,top,right,bottom in windows(image.size,side):
                dest=root/'local_cam_inputs'/f'{Path(path).stem}__tile{left}_{top}_{right}_{bottom}.png'
                dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():image.crop((left,top,right,bottom)).save(dest)
                result[top:bottom,left:right]+=run(str(dest));count[top:bottom,left:right]+=1
            return result/count>.5
        return predict
    specs=[dict(epochs=original_epochs,pool='original',template='original'),dict(epochs=100,pool='original',template='original')]
    scored=[]
    for index,spec in enumerate(specs):
        folder=root/'validation'/f'duration_{spec["epochs"]}';folder.mkdir(parents=True,exist_ok=True)
        score=score_predictions(predictor(spec,folder),m,split['validation_ids'],folder,'predictions')
        scored.append(dict(settings=spec,validation=score));write_json(root/'validation_results.json',scored)
    best=max(scored,key=lambda r:(r['validation']['IoU_percent'],-r['validation']['Fa_per_million'],-r['settings']['epochs']))
    spec={**best['settings']}
    spec['pool']='area'
    folder=root/'validation'/'representation';folder.mkdir(parents=True,exist_ok=True)
    score=score_predictions(predictor(spec,folder),m,split['validation_ids'],folder,'predictions')
    scored.append(dict(settings=spec,validation=score));write_json(root/'validation_results.json',scored)
    if args.method=='TVGTANet':
        base=max(scored,key=lambda r:(r['validation']['IoU_percent'],-r['validation']['Fa_per_million'],-r['settings']['epochs']))
        spec={**base['settings'],'template':'thermal'}
        folder=root/'validation'/'thermal_text';folder.mkdir(parents=True,exist_ok=True)
        score=score_predictions(predictor(spec,folder),m,split['validation_ids'],folder,'predictions')
        scored.append(dict(settings=spec,validation=score));write_json(root/'validation_results.json',scored)
    if max(r['validation']['IoU_percent'] for r in scored)<20:
        base=max(scored,key=lambda r:(r['validation']['IoU_percent'],-r['validation']['Fa_per_million'],-r['settings']['epochs']))
        spec={**base['settings'],'local':True,'local_side':local_side_for(m)}
        folder=root/'validation'/'local_input';folder.mkdir(parents=True,exist_ok=True)
        score=score_predictions(predictor(spec,folder),m,split['validation_ids'],folder,'predictions')
        scored.append(dict(settings=spec,validation=score));write_json(root/'validation_results.json',scored)
    selected=max(scored,key=lambda r:(r['validation']['IoU_percent'],-r['validation']['Fa_per_million'],-r['settings']['epochs']))
    write_json(root/'selection.json',dict(**selected,official_test_labels_read=False,annotation_budget=50,adaptation_support_images=5,validation_images=10,
               note='allocated50 labels used to stratify split; budget50 validation-adapted, not a pure5-label reference'))
    folder=root/'full_test';folder.mkdir(exist_ok=True)
    score=score_predictions(predictor(selected['settings'],folder),m,m['test_ids'],folder,'predictions')
    score.update(method=args.method,dataset=args.dataset,complete_test=True,annotation_budget=50,support_shots=5,
                 settings=selected['settings'],version='budget/IR-adapted native episodic model')
    write_json(folder/'metrics/summary.json',score);write_json(root/'report.json',dict(status='complete',method=args.method,dataset=args.dataset,selection=selected,test_metrics=score))


def foundation_worker(args):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')
    seed_all();m=load_manifest(args.dataset);split=load_split(args.dataset)
    root=args.output/args.method/args.dataset;root.mkdir(parents=True,exist_ok=True)
    if (root/'report.json').exists():return
    sys.path.append(str((DEFAULT_OUTPUT/'runtime_packages').resolve()))
    torch.hub.set_dir(str((DEFAULT_OUTPUT/'runtime_cache/torch_hub').resolve()))
    if args.method=='HERA':
        source=DEFAULT_OUTPUT/'sources/HERA'
        sys.path[:0]=[str(source.resolve()),str((DEFAULT_OUTPUT/'sources/dinov3').resolve())]
        import main_hera as native
        hub=(DEFAULT_OUTPUT/'sources/dinov3_backbone_hub').resolve()
        weight=(DEFAULT_OUTPUT/'weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth').resolve()
        model=native.SSP_MatchingNet('DINOv3',dinov3_repo=str(hub),dinov3_ckpt_path=str(weight)).cuda().eval()
        model.requires_grad_(False);refine=native.RefineModule().cuda().eval()
        old=DEFAULT_OUTPUT/args.dataset/'HERA/checkpoints/support_adaptation.pt'
        state=torch.load(old,map_location='cpu',weights_only=False)
        assert state['manifest_sha256']==m['manifest_sha256']
        assert len(state['selected_feat_ids'])==1,'Fusion state needs a separate audited adapter'
        params=dict(model.named_parameters())
        with torch.no_grad():
            for name,value in state['adapted_parameters'].items():params[name].copy_(value)
        refine.load_state_dict(state['refine']);layer=state['selected_feat_ids'][0]
        write_json(root/'reuse.json',dict(source=str(old),labels='same fixed5 supports only; no train50 weights',
                   changes='PAC off; area prototype pooling; fixed23 layer diagnostic'))
        transform=transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        sx,sy=supports(m,transform);native_pool=model.masked_average_pooling
        specs=[dict(layer=layer,pool='original',refine=bool(state['apply_refine']),diagnostic=False),
               dict(layer=layer,pool='original',refine=False,diagnostic=True),
               dict(layer=layer,pool='area',refine=False,diagnostic=True),
               dict(layer=23,pool='area',refine=False,diagnostic=True,fixed_layer=True)]
        def predictor(spec):
            model.masked_average_pooling=native_pool;configure_pool(model,spec['pool']);model.feat_ids=[spec['layer']]
            def extractor(img,bb,ids,**kwargs):
                return model.extract_feat_dino_single(img,model.backbone,[spec['layer']],attn_strategy='dual_attn_gauss',**kwargs)
            model.extract_feats=extractor
            side=spec.get('local_side',256)
            msx,msy=supports(m,transform,crop=side) if spec.get('local') else (sx,sy)
            with torch.no_grad():fixed=model(msx,msy,None,None,tta_flag=True)
            @torch.no_grad()
            def predict(path):
                im=Image.open(path).convert('RGB');w,h=im.size
                boxes=windows(im.size,side) if spec.get('local') else [(0,0,w,h)]
                result=np.zeros((h,w),np.float32);count=np.zeros_like(result)
                for left,top,right,bottom in boxes:
                    q=transform(im.crop((left,top,right,bottom)))[None].cuda()
                    qfeat=model(None,None,q,None,q_flag=True)
                    logits=model.ssp_func(msx,msy,q,None,fixed,qfeat)[0]
                    if spec['refine']:
                        feat,attn=model.get_feat_and_attn(q,return_attn=True);logits=refine(feat,logits,q,attn)
                    if not spec.get('local'):return restore_to_original(logits.argmax(1)[0].float().cpu().numpy(),(h,w),discrete=True)
                    prob=logits.softmax(1)[0,1].cpu().numpy()
                    result[top:bottom,left:right]+=cv2.resize(prob,(right-left,bottom-top),interpolation=cv2.INTER_LINEAR)
                    count[top:bottom,left:right]+=1
                return result/count>.5
            return predict
    else:
        source=DEFAULT_OUTPUT/'sources/Matcher';sys.path.insert(0,str(source.resolve()))
        if 'int' not in np.__dict__:np.int=int
        from matcher.Matcher import build_matcher_oss
        settings=SimpleNamespace(device=torch.device('cuda'),dinov2_size='vit_large',sam_size='vit_h',
            dinov2_weights=(DEFAULT_OUTPUT/'weights/dinov2_vitl14_pretrain.pth').resolve().as_uri(),
            sam_weights=str((DEFAULT_OUTPUT/'weights/sam_vit_h_4b8939.pth').resolve()),
            points_per_side=64,pred_iou_thresh=.88,stability_score_thresh=.95,sel_stability_score_thresh=.90,
            iou_filter=.85,box_nms_thresh=.65,output_layer=3,dense_multimask_output=0,use_dense_mask=1,
            multimask_output=1,emd_filter=0.,purity_filter=.02,coverage_filter=0.,use_score_filter=True,
            deep_score_filter=.33,deep_score_norm_filter=.1,topk_scores_threshold=0.,num_centers=8,
            use_box=False,use_points_or_centers=True,sample_range=(1,6),max_sample_iterations=64,
            alpha=1.,beta=0.,exp=0.,num_merging_mask=5)
        model=build_matcher_oss(settings)
        transform=transforms.Compose([transforms.Resize((518,518)),transforms.ToTensor()])
        sx,sy=supports(m,transform);sx=torch.cat(sx)[None];sy=torch.cat(sy)[None]
        specs=[dict(merging=5,stability=.95),dict(merging=1,stability=.95),dict(merging=1,stability=.90)]
        def predictor(spec):
            model.num_merging_mask=spec['merging'];model.generator.stability_score_thresh=spec['stability']
            side=spec.get('local_side',256)
            if spec.get('local'):
                lsx,lsy=supports(m,transform,crop=side);lsx=torch.cat(lsx)[None];lsy=torch.cat(lsy)[None]
            else:lsx,lsy=sx,sy
            @torch.no_grad()
            def predict(path):
                im=Image.open(path).convert('RGB');w,h=im.size
                boxes=windows(im.size,side) if spec.get('local') else [(0,0,w,h)]
                result=np.zeros((h,w),np.float32);count=np.zeros_like(result)
                for left,top,right,bottom in boxes:
                    model.set_reference(lsx,lsy.clone());model.set_target(transform(im.crop((left,top,right,bottom)))[None].cuda())
                    pred=model.predict()[0].float().cpu().numpy();model.clear()
                    pred=restore_to_original(pred,(bottom-top,right-left),discrete=True)
                    result[top:bottom,left:right]+=pred;count[top:bottom,left:right]+=1
                return result/count>.5
            return predict
    candidates=[]
    for index,spec in enumerate(specs):
        folder=root/'validation'/f'candidate{index}';folder.mkdir(parents=True,exist_ok=True)
        score=score_predictions(predictor(spec),m,split['validation_ids'],folder,'predictions')
        candidates.append(dict(settings=spec,validation=score));write_json(root/'validation_results.json',candidates)
    area_tta_predict=None;area_tta_model=None
    if args.method=='HERA':
        # Correct the verified prototype loss *during* native HLS/PGR, not just after
        # historical adaptation that saw zero foreground prototypes.
        from common.logger import Logger
        from common.evaluation import Evaluator
        model.cpu();gc.collect();torch.cuda.empty_cache()
        area_tta_model=native.SSP_MatchingNet('DINOv3',dinov3_repo=str(hub),dinov3_ckpt_path=str(weight)).cuda().eval()
        area_tta_model.requires_grad_(False);configure_pool(area_tta_model,'area')
        fusion=native.FusionMLP().cuda();area_refine=native.RefineModule().cuda()
        batch=dict(support_imgs=torch.cat(sx)[None],support_masks=torch.cat(sy)[None],
                   query_img=sx[0],query_mask=torch.zeros(1,400,400),class_id=torch.tensor([0]))
        class Episode:
            dataset=SimpleNamespace(benchmark='lung',class_ids=[0])
            def __iter__(self):yield batch
            def __len__(self):return 1
        Logger.initialize(str(root/'area_native_tta/logs'),'support_adaptation.log');Evaluator.initialize()
        decision={};original_vote=native.should_apply_refine_by_vote
        def vote(**kwargs):
            apply,stats=original_vote(**kwargs);decision.update(apply=bool(apply),stats=stats);return apply,stats
        native.should_apply_refine_by_vote=vote
        try:
            native.test(area_tta_model,Episode(),5,torch.device('cuda'),fusion,area_refine,
                        SimpleNamespace(fusion='on',attn_strategy='dual_attn_gauss',refine='auto',benchmark='lung'))
        finally:native.should_apply_refine_by_vote=original_vote
        area_tta_model.eval()
        for parameter in area_tta_model.parameters():parameter.grad=None
        torch.cuda.empty_cache()
        @torch.no_grad()
        def area_tta_predict(path):
            im=Image.open(path).convert('RGB');w,h=im.size;q=transform(im)[None].cuda()
            logits=area_tta_model(sx,sy,q,None)[0]
            if decision.get('apply'):
                feat,attn=area_tta_model.get_feat_and_attn(q,return_attn=True)
                logits=area_refine(feat,logits,q,attn)
            return restore_to_original(logits.argmax(1)[0].float().cpu().numpy(),(h,w),discrete=True)
        folder=root/'validation'/'area_native_tta';folder.mkdir(parents=True,exist_ok=True)
        score=score_predictions(area_tta_predict,m,split['validation_ids'],folder,'predictions')
        spec=dict(pool='area',native_HLS_PGR_PAC_recomputed=True,selected_layers=list(area_tta_model.feat_ids),refine=decision.get('apply',False))
        torch.save(dict(adapted_parameters={k:p.detach().cpu() for k,p in area_tta_model.named_parameters() if p.requires_grad},
                        fusion=fusion.state_dict(),refine=area_refine.state_dict(),settings=spec),root/'area_native_tta/support_adaptation.pt')
        candidates.append(dict(settings=spec,validation=score));write_json(root/'validation_results.json',candidates)
        area_tta_model.cpu();model.cuda();gc.collect();torch.cuda.empty_cache()
    if max(r['validation']['IoU_percent'] for r in candidates)<10:
        # One bounded input-scale follow-up, predeclared here and selected on val10 only.
        eligible=[c for c in candidates if not c['settings'].get('native_HLS_PGR_PAC_recomputed')]
        best=max(eligible,key=lambda r:(r['validation']['IoU_percent'],-r['validation']['Fa_per_million']))
        spec={**best['settings'],'local':True,'local_side':local_side_for(m)}
        folder=root/'validation'/'local_input';folder.mkdir(parents=True,exist_ok=True)
        score=score_predictions(predictor(spec),m,split['validation_ids'],folder,'predictions')
        candidates.append(dict(settings=spec,validation=score));write_json(root/'validation_results.json',candidates)
    selected=max(candidates,key=lambda r:(r['validation']['IoU_percent'],-r['validation']['Fa_per_million']))
    write_json(root/'selection.json',dict(**selected,official_test_labels_read=False,annotation_budget=50,adaptation_support_images=5,
               note='validation-selected diagnostic/IR adaptation; ablated variants explicitly identified'))
    folder=root/'full_test';folder.mkdir(exist_ok=True)
    if selected['settings'].get('native_HLS_PGR_PAC_recomputed'):
        model.cpu();area_tta_model.cuda();final_predict=area_tta_predict
    else:final_predict=predictor(selected['settings'])
    score=score_predictions(final_predict,m,m['test_ids'],folder,'predictions')
    score.update(method=args.method,dataset=args.dataset,complete_test=True,annotation_budget=50,support_shots=5,
                 settings=selected['settings'],version='validation-selected diagnostic/IR-adapted version')
    write_json(folder/'metrics/summary.json',score);write_json(root/'report.json',dict(status='complete',method=args.method,dataset=args.dataset,selection=selected,test_metrics=score))


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--output',type=Path,default=DEFAULT_RUN);args=p.parse_args()
    dependency=args.output/'pre_fss_dependency.json'
    if dependency.exists():
        dependency_meta=json.loads(dependency.read_text(encoding='utf-8'));dependent_pid=dependency_meta['pid']
        while process_alive(dependent_pid):
            state=args.output/dependency_meta.get('state_path','followup_queue/status.json')
            if state.exists() and json.loads(state.read_text(encoding='utf-8')).get('stage','').startswith('complete'):break
            print(json.dumps(dict(stage='waiting_for_prior_recovery_followup',pid=dependent_pid)),flush=True);time.sleep(20)
    if args.method in ('IFA','MPA'):episodic_worker(args)
    elif args.method in ('ABCDFSS','TVGTANet'):affinity_worker(args)
    elif args.method in ('HERA','Matcher'):foundation_worker(args)
    else:raise ValueError(f'Backend not yet registered: {args.method}')


if __name__=='__main__':main()
