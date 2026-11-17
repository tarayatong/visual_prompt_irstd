"""GT constructs/scorers synthetic pairs; inference receives only the resulting image."""
from pathlib import Path
import hashlib
import json
from functools import lru_cache
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/comparison_50shot_5support'
OUT = ROOT / 'outputs/background_replacement_baselines_20261109'
DOMAINS = ['NUDT-SIRST', 'IRSTD-1k', 'SIRST']
METHODS = ['DNANet', 'SCTransNet', 'RPCANet', 'ILNet']
RECIPE = dict(version=1, seed=111, protected_radius=4, feather_outer_radius=10,
              donor_erasure_radius=3, inpaint_radius=5, gain_limits=[.5, 2.],
              replicates=2, donor_pool='official test split; same-domain excludes receiver',
              matching='per-channel mean/std outside recipient and donor 10px halos',
              donor_resize='inpaint at donor resolution, then bilinear resize',
              label_alignment='binarize >0 first; nearest to actual image dimensions',
              evaluation='fixed model; original-resolution probability interpolation; threshold .5')

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

@lru_cache(maxsize=3)
def manifest(domain):
    return json.loads((BASE/domain/'protocol/manifest.json').read_text(encoding='utf-8'))

@lru_cache(maxsize=24)
def read(domain, sid):
    r = manifest(domain)['records'][sid]
    im = np.asarray(Image.open(r['image']).convert('RGB')).copy()
    gt = (np.asarray(Image.open(r['mask']).convert('L')) > 0).astype('uint8')
    if gt.shape != im.shape[:2]:
        gt = cv2.resize(gt, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
    distance = cv2.distanceTransform(1-gt, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return im, gt, distance

def donor_id(domain, sid, donor_domain, replicate):
    pool = sorted(s for s in manifest(donor_domain)['test_ids'] if domain != donor_domain or s != sid)
    key = f"111/{domain}/{sid}/{donor_domain}".encode()
    start = int(hashlib.sha256(key).hexdigest(), 16) % len(pool)
    return pool[(start + replicate) % len(pool)]

def compose(domain, sid, donor_domain, replicate, identity=False):
    im, gt, dist = read(domain, sid)
    dsid = sid if identity else donor_id(domain, sid, donor_domain, replicate)
    donor, dgt, ddist = read(donor_domain, dsid)
    clean = cv2.inpaint(donor, (ddist <= 3).astype('uint8')*255, 5, cv2.INPAINT_TELEA)
    donor_bg = clean[ddist > 10].astype('float32')
    ref_bg = im[dist > 10].astype('float32')
    if not len(donor_bg) or not len(ref_bg): raise ValueError('No background pixels')
    gain = np.clip(ref_bg.std(0)/np.maximum(donor_bg.std(0), 1e-6), .5, 2.)
    offset = ref_bg.mean(0)-gain*donor_bg.mean(0)
    bg = cv2.resize(clean, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR).astype('float32')
    bg = np.clip(bg*gain+offset, 0, 255)
    alpha = np.clip((10-dist)/6, 0, 1)[..., None]
    result = np.rint(alpha*im+(1-alpha)*bg).clip(0,255).astype('uint8')
    assert np.array_equal(result[dist <= 4], im[dist <= 4])
    meta = dict(donor_domain=donor_domain, donor_id=dsid, replicate=replicate,
                gain=gain.tolist(), offset=offset.tolist(),
                input_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
                changed_background_fraction=float(np.any(result != im, axis=2)[dist > 10].mean()))
    return result, meta

def checkpoint(method, domain):
    if method == 'RPCANet' and domain == 'NUDT-SIRST':
        folder = ROOT/'outputs/comparison_task_adaptation_20261104/RPCANet/NUDT-SIRST/full50'
        return folder/'model.pt', folder/'config.json', dict(updates=1000, adaptation='foreground-balanced BCE + SoftIoU; train-only validation selection; full50 refit')
    epoch = 125 if domain == 'NUDT-SIRST' and method in ('ILNet', 'SCTransNet') else 500
    folder = BASE/'full16'/domain/method
    return folder/f'checkpoints/epoch_{epoch:04d}.pt', folder/'config.json', dict(epoch=epoch, adaptation='native training recipe; 50 labeled images')

def preflight():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    locked = dict(recipe=RECIPE, checkpoints=[], manifests={})
    for d in DOMAINS:
        locked['manifests'][d] = digest(BASE/d/'protocol/manifest.json')
        for method in METHODS:
            p,c,info = checkpoint(method,d)
            locked['checkpoints'].append(dict(method=method, domain=d, path=str(p), sha256=digest(p), config=json.loads(c.read_text(encoding='utf-8')), **info))
    fig, axes = plt.subplots(3,4,figsize=(13,10))
    checks=[]
    for row,d in enumerate(DOMAINS):
        sid=manifest(d)['train_ids'][0]
        im,gt,_=read(d,sid)
        identity,_=compose(d,sid,d,0,identity=True)
        assert np.array_equal(identity,im), (d,sid,'identity reconstruction failed')
        images=[im]+[compose(d,sid,dd,0)[0] for dd in DOMAINS]
        for col,img in enumerate(images):
            axes[row,col].imshow(img);axes[row,col].contour(gt,levels=[.5],colors=['red'],linewidths=.4)
            axes[row,col].set_title(f'{d}: {sid}\n'+('Original' if col==0 else f'Background: {DOMAINS[col-1]}'),fontsize=9)
            axes[row,col].axis('off')
        checks.append(dict(domain=d,sample_id=sid,identity_exact=True,foreground_and_4px_halo_exact=True))
    fig.suptitle('Background intervention QA (training examples; red = GT for display only)')
    fig.tight_layout();fig.savefig(OUT/'construction_qa.png',dpi=150);plt.close(fig)
    locked['qa']=checks
    path=OUT/'protocol.json'
    if path.exists(): assert json.loads(path.read_text(encoding='utf-8'))==locked,'Protocol changed'
    else: dump(path,locked)
    print(json.dumps(dict(status='preflight_complete',output=str(OUT),checkpoints=len(locked['checkpoints']))),flush=True)

if __name__ == '__main__': preflight()
