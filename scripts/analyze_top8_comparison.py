"""Audit retained top-eight results and profile their IRSTD inference only.

Never trains or modifies retained weights/results. Each model runs in a separate
process to preserve the original third-party import namespaces.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/top8_analysis_20261105'
SELECTED = ROOT / 'experiments/retention_comparison_20261105/selected'
METHODS = ('Ours', 'SANSA', 'ILNet', 'HDNet', 'DNANet', 'MSHNet', 'SCTransNet', 'MSHNet+NS-FPN')
DATASETS = ('SIRST', 'IRSTD-1k', 'NUDT-SIRST')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_write(p, rows, fields=None):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader(); w.writerows(rows)


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def manifest(dataset):
    return read(ROOT/'outputs/comparison_50shot_5support'/dataset/'protocol/manifest.json')


def record(dataset, method):
    return read(SELECTED/dataset/method/'record.json')


def load_rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return [{k: (v if k=='sample_id' else float(v) if k=='image_iou' else int(v))
                 for k,v in r.items()} for r in csv.DictReader(f)]


def matches(rows, expected, ids):
    if len(rows)!=len(ids) or len({r['sample_id'] for r in rows})!=len(ids) or {r['sample_id'] for r in rows}!=set(ids):
        return False
    keys = ('tp','fp','fn','pixels','detected','targets')
    tp,fp,fn,pixels,detected,targets = [sum(r[k] for r in rows) for k in keys]
    actual = dict(IoU=100*tp/(tp+fp+fn), nIoU=100*statistics.mean(r['image_iou'] for r in rows),
                  F1=200*tp/(2*tp+fp+fn), Pd=100*detected/targets, Fa=1e6*fp/pixels)
    return all(abs(actual[k]-expected[k])<1e-7 for k in actual)


def collect():
    audits=[]; summary=[]; missing=[]
    for dataset in DATASETS:
        m=manifest(dataset); ids=m['test_ids']; bymethod={}; allrows=[]
        for method in METHODS:
            if method=='Ours':
                rec=next(r for r in read(ROOT/'experiments/retention_comparison_20261105/ours_protected.json') if r['dataset']==dataset)
                source=ROOT/rec['evaluation_directory']
                rows=[read(source/f'{sid}.json')['rows']['learned'] for sid in ids]
                values=read(SELECTED/dataset/'Ours/summary.json')
                expected={k:values[v] for k,v in dict(IoU='IoU_percent',nIoU='nIoU_percent',F1='F1_percent',Pd='Pd_percent',Fa='Fa_per_million').items()}
            else:
                rec=record(dataset,method); source=ROOT/rec['source']; run=ROOT/rec['run_directory']; expected=rec['values']
                candidates=[OUT/'recovered'/dataset/method/'per_image.csv',source.parent/'per_image.csv',
                            source.parent/'per_image_recount.csv',run/'audit/per_image_recount.csv',run/'metrics/per_image.csv']
                rows=None
                for candidate in candidates:
                    if candidate.is_file():
                        trial=load_rows(candidate)
                        if matches(trial,expected,ids): rows=trial;source=candidate;break
                if rows is None:
                    missing.append(dict(dataset=dataset,method=method,reason='No existing per-image counts match all five selected table metrics'))
                    continue
            assert matches(rows,expected,ids),(dataset,method)
            bymethod[method]={r['sample_id']:r for r in rows}
            bad=[]
            for sid in ids:
                r=bymethod[method][sid];tp,fp,fn=[r[k] for k in ('tp','fp','fn')];union=tp+fp+fn
                iou=tp/union if union else 1.
                assert abs(iou-r['image_iou'])<1e-12
                prediction=''
                if method!='Ours':
                    dirs=[source.parent/'predictions',source.parent.parent/'predictions',run/'predictions',run/'full50/predictions']
                    prediction=next((str(p.resolve()) for folder in dirs for ext in ('.png','.npy') if (p:=folder/(sid+ext)).is_file()),'')
                item=dict(dataset=dataset,method=method,sample_id=sid,iou=iou,tp=tp,fp=fp,fn=fn,
                          image_path=m['records'][sid]['image'],mask_path=m['records'][sid]['mask'],prediction_path=prediction,source=str(source.relative_to(ROOT)))
                allrows.append(item)
                if union and 5*tp<union: bad.append(item)
            fields=['dataset','method','sample_id','iou','tp','fp','fn','image_path','mask_path','prediction_path','source']
            csv_write(OUT/'bad_samples'/dataset/f'{method}.csv',bad,fields)
            p=OUT/'bad_samples'/dataset/f'{method}_ids.txt'
            p.write_text(''.join(r['sample_id']+'\n' for r in bad),encoding='utf-8')
            summary.append(dict(dataset=dataset,method=method,total=len(rows),below_0_2=len(bad),fraction=len(bad)/len(rows)))
            audits.append(dict(dataset=dataset,method=method,source=str(source.relative_to(ROOT)),all_five_metrics_match=True,images=len(rows)))
        csv_write(OUT/'bad_samples'/dataset/'all_per_image.csv',allrows)
        matrix=[]
        for sid in ids:
            values={method:bymethod[method][sid]['image_iou'] for method in bymethod}
            failed=[method for method,v in values.items() if v<.2]
            matrix.append(dict(sample_id=sid,**{method:values.get(method,'') for method in METHODS},failed_methods=';'.join(failed),failed_method_count=len(failed)))
        csv_write(OUT/'bad_samples'/dataset/'iou_matrix.csv',matrix)
        union=[r for r in matrix if r['failed_method_count']]
        csv_write(OUT/'bad_samples'/dataset/'union_bad_samples.csv',union,list(matrix[0]))
        shortlist=sorted(union,key=lambda r:(-r['failed_method_count'],sum(float(r[k]) for k in bymethod)))[:20]
        csv_write(OUT/'bad_samples'/dataset/'hardest_shortlist.csv',shortlist,list(matrix[0]))
        advantage=[r for r in union if r.get('Ours','')!='' and r['Ours']>=.5 and any(r[k]!='' and r[k]<.2 for k in METHODS[1:])]
        advantage.sort(key=lambda r:(-r['failed_method_count'],-float(r['Ours'])))
        csv_write(OUT/'bad_samples'/dataset/'ours_advantage_shortlist.csv',advantage[:20],list(matrix[0]))
        (OUT/'bad_samples'/dataset/'union_ids.txt').write_text(''.join(r['sample_id']+'\n' for r in union),encoding='utf-8')
        intersection=[r['sample_id'] for r in matrix if r['failed_method_count']==len(METHODS)] if len(bymethod)==8 else None
        save(OUT/'bad_samples'/dataset/'sets.json',dict(complete=len(bymethod)==8,available_methods=list(bymethod),union=[r['sample_id'] for r in union],intersection=intersection))
        for r in union:
            sid=r['sample_id']
            for key,folder in [('image','images'),('mask','masks')]:
                src=Path(m['records'][sid][key]);dest=OUT/'bad_samples'/dataset/folder/(sid+src.suffix)
                dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():shutil.copy2(src,dest)
    csv_write(OUT/'bad_sample_summary.csv',summary)
    save(OUT/'bad_sample_audit.json',dict(threshold=.2,comparison='strictly less than; exact integer check 5*tp < tp+fp+fn',audits=audits,missing=missing))
    print(json.dumps(dict(summary=summary,missing=missing),ensure_ascii=False),flush=True)


def build_cnn(dataset,method,device):
    import torch
    from scripts.run_nuaa_coordinate_corrected import build_infrared
    rec=record(dataset,method);cfg=read(ROOT/rec['config']);m=manifest(dataset)
    ns=build_infrared(method);model=ns['HDNet'](3).to(device).eval()
    checkpoint=torch.load(ROOT/rec['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    if method=='ILNet':m['train50_rgb_stats']=cfg['normalization']
    if method=='SCTransNet':m['train50_gray_stats']=cfg['normalization']
    data=ns['dataset_class'](m,False,ns['IRSTD_Dataset'])
    return model,data,ns,rec


def recover(dataset,method,device,reuse=True):
    import torch
    import numpy as np
    from PIL import Image
    from visual_prompt.evaluation.comparison import image_metrics
    from visual_prompt.evaluation.spatial_alignment import restore_to_original
    torch.set_num_threads(2)
    model,data,ns,rec=build_cnn(dataset,method,device);m=manifest(dataset)
    out=OUT/'recovered'/dataset/method;rows=[];reused=0
    with torch.inference_mode():
        for i in range(len(data)):
            x,_,sid=data[i];dest=out/'predictions'/f'{sid}.png'
            h,w=m['records'][sid]['size_hw']
            if reuse and dest.exists():
                pred=np.asarray(Image.open(dest).convert('L'))>0
                reused+=1
            else:
                _,logits=model(x[None].to(device),True)
                a=(logits if ns['prediction_is_probability'] else logits.sigmoid())[0,0].float().cpu().numpy()
                pred=restore_to_original(a,(h,w),discrete=False)
            gt=np.asarray(Image.open(m['records'][sid]['mask']).convert('L').resize((w,h),Image.Resampling.NEAREST))>0
            rows.append(image_metrics(pred,gt,sid))
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not reuse or not dest.exists():Image.fromarray((pred>=.5).astype(np.uint8)*255).save(dest)
            if (i+1)%100==0:print(json.dumps(dict(method=method,images=i+1)),flush=True)
    csv_write(out/'per_image.csv',rows)
    save(out/'audit.json',dict(checkpoint=rec['checkpoint'],checkpoint_sha256=sha(ROOT/rec['checkpoint']),
                              images=len(rows),selected_five_metrics_match=matches(rows,rec['values'],m['test_ids'])))
    if not matches(rows,rec['values'],m['test_ids']) and reused and device.startswith('cuda'):
        save(out/'cached_prediction_mismatch.json',dict(reused=reused,action='rerun all images on GPU with retained checkpoint'))
        del model;torch.cuda.empty_cache()
        return recover(dataset,method,device,reuse=False)
    assert matches(rows,rec['values'],m['test_ids']),(dataset,method,'retained checkpoint does not reproduce table')


def gpu_state():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    name,util,used,total=[s.strip() for s in text.splitlines()[0].split(',')]
    return dict(name=name,utilization=int(util),used_MiB=int(used),total_MiB=int(total))


def build_ours(device):
    import torch
    from PIL import Image
    import numpy as np
    from visual_prompt.pipelines.p2p_framework import P2PModel,normalize_f4,DESCRIPTOR_NAMES,describe_full_image_points,local_sam,paste_local
    from visual_prompt.models.learned_radial_center_sam import LearnedRadialCenterSAM
    from visual_prompt.core.candidate_proposals import greedy_score_peaks
    from scripts.run_p2p_memory_polar import memory_scores_once,working
    u=P2PModel('IRSTD-1k','distance_obj',device=device,load_mask=False)
    rec=next(r for r in read(ROOT/'experiments/retention_comparison_20261105/ours_protected.json') if r['dataset']=='IRSTD-1k')
    ck=torch.load(ROOT/rec['checkpoint'],map_location='cpu',weights_only=False);state=ck['model']
    mask=LearnedRadialCenterSAM(state['base.feature_mean'],state['base.feature_std']).to(device).eval();mask.load_state_dict(state,strict=True)
    predictor=u.pred();model=predictor.model
    components=dict(encoder=model,reader=u.reader,objectness=u.obj,mask=mask)
    m=read(ROOT/'outputs/retained/p2p/IRSTD-1k/manifest.json')
    def prepare(sid):return working(m['records'][sid])[0]
    def forward(image):
        score,canonical=memory_scores_once(u,image)
        points=[(y,x) for y,x,_ in greedy_score_peaks(score,5,7)]
        desc=describe_full_image_points(u.pred(),image,canonical,points,64,u.args.local_contrast_sigma,32)
        raw=np.concatenate([desc[name] for name in DESCRIPTOR_NAMES],1)
        probs=u.obj(torch.from_numpy(normalize_f4(raw,u.ck['statistics'])).to(device)).sigmoid().cpu().tolist()
        canvas=np.zeros((512,512),np.uint8)
        for (y,x),p in zip(points,probs):
            if p<.1:continue
            s=local_sam(u.pred(),image,y,x).half().float();r,rv=u.radial.sample(image,y,x)
            pred=mask(s,r[None].to(device,dtype=torch.float32),rv[None].to(device))[0]>0
            paste_local(canvas,pred.cpu().numpy().astype(np.uint8),y,x)
        return canvas
    # Historical two-stage task optimization: reader + Obj + learned mask;
    # shared encoder remains frozen. Avoid counting obsolete stored Mask heads.
    return components,prepare,forward,m,rec


def profile(method,device,timing=False):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    torch.set_num_threads(2);torch.manual_seed(111)
    before=gpu_state() if device.startswith('cuda') else None
    m=manifest('IRSTD-1k');ids=m['test_ids'][:10]
    if method=='Ours':
        components,prepare,forward,m,rec=build_ours(device)
        counts={name:sum(p.numel() for p in module.parameters()) for name,module in components.items() if module is not None}
        total=sum(counts.values());trainable=sum(v for k,v in counts.items() if k!='encoder')
        inputs=[prepare(sid) for sid in ids]
        geometry='512x512 image, SAM2 B+ encoder512, sliding windows, Top5, Obj>=0.1, local64; full uncached query pipeline'
    elif method=='SANSA':
        from scripts.run_recent_a_comparison import setup_sansa,load_saved
        from PIL import Image
        import numpy as np
        model,optimizer,scheduler,loader,objective,predict,identity=setup_sansa(m,torch.device(device))
        rec=record('IRSTD-1k',method);ck=torch.load(ROOT/rec['checkpoint'],map_location='cpu',weights_only=False)
        load_saved(model,ck['model']);model.eval()
        from util.promptable_utils import build_prompt_dict
        mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
        def image(sid):
            a=np.asarray(Image.open(m['records'][sid]['image']).convert('RGB').resize((640,640),Image.Resampling.BILINEAR),dtype=np.float32)
            return (torch.from_numpy(a.transpose(2,0,1)/255)-mean)/std
        supports=[image(sid) for sid in m['support_ids']]
        labels=[]
        for sid in m['support_ids']:
            a=np.asarray(Image.open(m['records'][sid]['mask']).convert('L'))>0
            labels.append(torch.from_numpy(np.asarray(Image.fromarray(a.astype(np.uint8)).resize((640,640),Image.Resampling.NEAREST)).copy()).float())
        masks=torch.stack([*labels,torch.zeros_like(labels[0])])[None].to(device)
        prompt=build_prompt_dict(masks,'mask',5,False,torch.device(device))
        inputs=[torch.stack([*supports,image(sid)])[None].to(device) for sid in ids]
        def forward(samples):
            return model(samples,prompt)['pred_masks'][-1].sigmoid()
        total=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        counts=dict(model=total);geometry='5 support + 1 query, RGB640x640, native tiny backbone resizes to1024; supports reencoded per query'
    else:
        model,data,ns,rec=build_cnn('IRSTD-1k',method,device)
        inputs=[data[i][0][None].to(device) for i in range(10)]
        def forward(x):
            _,logits=model(x,True)
            return logits if ns['prediction_is_probability'] else logits.sigmoid()
        total=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        counts=dict(model=total);geometry=str(list(inputs[0].shape))
    out=OUT/'complexity'/f'{method}.json'
    result=read(out) if out.exists() else {}
    result.update(method=method,dataset='IRSTD-1k',parameters_total=total,parameters_trainable=trainable,
                  component_parameters=counts,input_geometry=geometry,checkpoint=rec['checkpoint'],checkpoint_sha256=sha(ROOT/rec['checkpoint']),
                  flop_convention='FMA=2 FLOPs; torch FlopCounterMode supported convolution/matmul/attention operators, common elementwise/interpolation/CPU operations not counted',
                  torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark)
    with torch.inference_mode():
        if not timing:
            flops=[]
            for i,x in enumerate(inputs[:3] if method=='Ours' else inputs[:1]):
                with FlopCounterMode(display=False) as counter: y=forward(x)
                flops.append(counter.get_total_flops())
                print(json.dumps(dict(method=method,sample=i,flops=flops[-1])),flush=True)
            result.update(flops=statistics.mean(flops),flops_samples=flops,flops_sample_ids=ids[:len(flops)],flops_device=device)
        else:
            assert device.startswith('cuda')
            validations=[]
            for i,x in enumerate(inputs[:3]):
                warm=forward(x)
                if method=='Ours':
                    import numpy as np
                    from PIL import Image
                    from visual_prompt.evaluation.comparison import image_metrics
                    gt=np.asarray(Image.open(m['records'][ids[i]]['mask']).convert('L'))>0
                    current=image_metrics(warm,gt,ids[i]);expected=read(ROOT/rec['evaluation_directory']/f'{ids[i]}.json')['rows']['learned']
                    same=all(current[k]==expected[k] for k in ('tp','fp','fn','detected','targets'))
                    validations.append(dict(sample_id=ids[i],counts_match_retained=same))
                    assert same,(ids[i],'Ours profiling path differs from retained inference')
            torch.cuda.synchronize();elapsed=[]
            for repeat in range(3):
                start=time.perf_counter()
                for x in inputs:forward(x)
                torch.cuda.synchronize();elapsed.append(time.perf_counter()-start)
            result.update(gpu=before['name'],gpu_before=before,gpu_after=gpu_state(),batch_size_queries=1,
                          fps=len(inputs)/statistics.median(elapsed),timing_repeat_seconds=elapsed,timing_query_ids=ids,
                          timing_scope='CNN/SANSA: GPU-resident input to probability (SANSA reencodes all5 supports per query); Ours: preloaded image, uncached full pipeline including intrinsic CPU stages',
                          concurrent_gpu_load=before['utilization']>10)
            if validations:result['prediction_validation']=validations
    save(out,result);print(json.dumps(result),flush=True)


def report():
    rows=[]
    # Complexity-only extension; failure-sample audit remains the original eight.
    for method in (*METHODS, 'SAM-SPL', 'IRPruneDet', 'IRSAM', 'IR-SAM2', 'SAMamba'):
        p=OUT/'complexity'/f'{method}.json'
        if p.exists():
            r=read(p);rows.append(dict(method=method,trainable_M=r['parameters_trainable']/1e6,total_M=r['parameters_total']/1e6,
                    GFLOPs=r['flops']/1e9 if r.get('flops') is not None else '',GPU_FPS=r.get('fps') or '',concurrent_gpu_load=r.get('concurrent_gpu_load',''),input_geometry=r['input_geometry'],
                    weights_status=r.get('weights_status','project retained trained weights'),backend=r.get('backend','native'),
                    source_commit=r.get('commit',r.get('source_commit','')),checkpoint=r.get('checkpoint',''),
                    profile_epoch=r.get('profile_epoch',''),input_size=r.get('input_size','')))
    if rows:csv_write(OUT/'complexity_IRSTD-1k.csv',rows)
    audit=read(OUT/'bad_sample_audit.json')
    summary=None
    with (OUT/'bad_sample_summary.csv').open(encoding='utf-8-sig',newline='') as f:summary=list(csv.DictReader(f))
    lookup={(r['dataset'],r['method']):r for r in summary}
    lines=['# IRSTD-1k 复杂度（前八与 SAM 扩展）、三数据集指标及低 IoU 样本','',
           '原前八方法包含 Ours；复杂度补充 SAM-SPL、IRPruneDet、IRSAM、IR-SAM2、SAMamba。新增四种方法按用户确认的三域train50/seed111/epoch120设置训练，逐组状态与五项测试指标见下文；训练后复杂度以IRSTD-1k第120轮任务权重及实际输入尺寸重新测量，表内披露权重与实现差异。这些是固定小样本适配结果，不声称复现论文原生全监督精度。低 IoU 样本仍只统计原前八，原有实验结果保留。','',
           '## IRSTD-1k 参数量、推理 FLOPs 与 GPU FPS','',
           '| 方法 | 可训练参数 M | 总参数 M | GFLOPs/查询 | GPU FPS | 实测设置 / 权重状态 |',
           '|---|---:|---:|---:|---:|---|']
    settings={
        'SAM-SPL':'RGB256；本项目 train50 保留权重',
        'IRPruneDet':'RGB512；作者发布剪枝任务权重（0.290M 版本）',
        'IRSAM':'RGB512→编码器1024；当前公开新版，仅 MobileSAM 预训练底座',
        'IR-SAM2':'RGB256；频域分支开启，仅 SAM2.1 预训练底座',
        'SAMamba':'RGB1024；PyTorch 参考 scan，非原生 CUDA 内核；权重详情见下文',
    }
    for r in rows:
        flop=f"{r['GFLOPs']:.3f}" if r['GFLOPs']!='' else '待完成'
        fps=f"{r['GPU_FPS']:.3f}" if r['GPU_FPS']!='' else '待完成'
        suffix='‡' if r['method']=='SAMamba' else ''
        setting=settings.get(r['method'],'原保留实现与任务权重，输入见下文')
        if r['profile_epoch']==120:
            setting=f"RGB{r['input_size']}；本项目 train50/epoch120 任务权重"+('；PyTorch参考scan，非原生CUDA' if r['method']=='SAMamba' else '')
            if r['method']=='IRSAM':setting+='；内部编码器1024'
        lines.append(f"| {r['method']+suffix} | {r['trainable_M']:.6f} | {r['total_M']:.6f} | {flop} | {fps} | {setting} |")
    lines+=['',
            '- 参数量：按实际构造模型统计。Ours 可训练参数为 Reader、Obj、当前学习径向 Mask 的总和，表示分阶段训练的任务参数；冻结 SAM2 和支持记忆／统计量另计。SANSA 仅 adapter 可训练。',
            '- 输入：HDNet、DNANet、MSHNet、MSHNet+NS-FPN 为 RGB256；SCTransNet 为灰度256；ILNet 为 RGB512；SANSA 为5支持+1查询、RGB640（原生编码器1024）；Ours 为512整图、SAM2 B+编码器512、滑窗、Top5、64局部裁剪。保留表中原生设置，未强行统一分辨率。',
            '- FLOPs：FMA=2；使用 PyTorch FlopCounterMode 统计执行到的卷积、矩阵乘、attention 等受支持算子。常规逐元素运算、插值、grid_sample、CPU后处理不计，是可复算的受支持算子计数。NS-FPN 保留作者 PyTorch deformable-attention 参考后端。',
            '- SANSA 每个查询重新编码5张支持图；Ours 包含整图滑窗检索、Obj及通过门限的局部编码和 Mask，未用缓存特征替代。Ours FLOPs 随候选通过数量变化，取测试清单前3张均值，原始值见 complexity/Ours.json。',
            '- GPU：NVIDIA GeForce RTX 3080，batch=1查询，FP32，3次预热，10张不同查询×3轮；每轮前后同步，以中位耗时换算FPS。不含模型加载和磁盘读图。CNN/SANSA与新增模型输入预置GPU；Ours包含原生流水线内CPU计算与传输。未启用compile或混合精度加速。SAMamba‡ 使用另行标注的参考后端，其他新增模型保留原前向分支。',
            '- 新增模型的可训练参数按作者代码的 requires_grad 设置统计；仅为 FLOPs/推理测量临时禁用梯度，不改变该列。IRPruneDet 发布版含未调用的融合模块，总参数仍计入这些参数，不把“非零权重数”当模型参数量。',
            '- SAMamba‡ 的 GFLOPs 为相同计数器支持的卷积/矩阵乘/attention计数，不含以逐元素递推实现的 selective scan；参考后端 FPS 不能用于评价作者原生 CUDA 内核的效率。IR-SAM2 的 FFT、排序等运算也不计入 GFLOPs，但完整执行并计入 FPS 耗时。',
            '- 每项权重路径与 SHA256、硬件状态、原始时间与查询 ID 位于 complexity/*.json。若 concurrent_gpu_load=true，该项有并行负载，需要重测。','',
            ]
    measured=OUT/'public_models_reproduction.md'
    if measured.exists():lines += [measured.read_text(encoding='utf-8').strip(), '']
    metrics=OUT/'sam_train50_metrics.md'
    if metrics.exists():lines += [metrics.read_text(encoding='utf-8').strip(), '']
    extension=OUT/'sam_based_extension.md'
    if extension.exists():lines += [extension.read_text(encoding='utf-8').strip(), '']
    lines += ['## 严格单图 IoU < 0.2 的样本数','',
            '| 方法 | SIRST /86 | IRSTD-1k /201 | NUDT-SIRST /664 |','|---|---:|---:|---:|']
    for method in METHODS:
        values=[lookup.get((d,method),{}).get('below_0_2','待补评测') for d in DATASETS]
        lines.append('| '+method+' | '+' | '.join(values)+' |')
    lines+=['','单图 IoU = TP/(TP+FP+FN)，空预测且空GT记1。采用严格小于0.2（整数判断5×TP < TP+FP+FN），不是小于等于。逐图计数重新聚合后，必须与选中表格的IoU、nIoU、F1、Pd、Fa全部一致，且测试ID无缺失、无重复。','',
            '## 样本文件','',
            '- [完整方法统计](bad_sample_summary.csv)、[24组来源核验](bad_sample_audit.json)、[复杂度CSV](complexity_IRSTD-1k.csv)。',
            '- 每个数据集下 METHOD_ids.txt 为逐方法ID，METHOD.csv 含IoU、TP/FP/FN及原图/GT路径、已有预测路径（缺失留空，不为筛样本重跑）；iou_matrix.csv 汇总8方法，union_ids.txt 为任一方法差的ID，sets.json还包含8方法共同失败交集。images/和masks/保存并集样本的原图与标注副本。',
            '- hardest_shortlist.csv：最多20个失败方法最多的困难样本；ours_advantage_shortlist.csv：Ours IoU≥0.5且至少1个对比方法IoU<0.2的最多20个可视化候选。完整ID/矩阵均保留，这两个文件仅便于初筛。','']
    for dataset in DATASETS:
        sets=read(OUT/'bad_samples'/dataset/'sets.json')
        lines.append(f"- {dataset}：并集 {len(sets['union'])} 张，共同失败 {len(sets['intersection']) if sets['intersection'] is not None else '待完成'} 张；[并集ID](bad_samples/{dataset}/union_ids.txt)、[8方法IoU矩阵](bad_samples/{dataset}/iou_matrix.csv)。")
    if audit['missing']:lines+=['','尚缺来源：'+json.dumps(audit['missing'],ensure_ascii=False)]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['collect','profile','recover','report']);p.add_argument('--method',choices=METHODS)
    p.add_argument('--dataset',choices=DATASETS,default='IRSTD-1k');p.add_argument('--device',default='cpu');p.add_argument('--timing',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.action=='collect':collect()
    elif a.action=='recover':recover(a.dataset,a.method,a.device)
    elif a.action=='profile':profile(a.method,a.device,a.timing)
    else:report()


if __name__=='__main__':main()
