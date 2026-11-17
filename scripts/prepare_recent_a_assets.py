"""Download only generic external pretraining, never infrared test-trained weights."""
import ast
import concurrent.futures
import json
import urllib.request
from scripts.run_recent_a_comparison import RUN, sha
from scripts.prepare_comparison import write_json

def download(item):
    name,url,expected=item
    folder=RUN/'weights';folder.mkdir(parents=True,exist_ok=True)
    dest=folder/name
    if not dest.exists():
        partial=dest.with_suffix('.part')
        with urllib.request.urlopen(url,timeout=120) as response,partial.open('wb') as out:
            expected_bytes=response.headers.get('Content-Length')
            while True:
                block=response.read(1024*1024)
                if not block:break
                out.write(block)
        if expected_bytes and partial.stat().st_size!=int(expected_bytes):raise ValueError('Incomplete download')
        partial.replace(dest)
    digest=sha(dest)
    if expected and digest!=expected:raise ValueError(f'Pretraining checksum mismatch: {name}')
    row=dict(name=name,url=url,bytes=dest.stat().st_size,sha256=digest)
    print(json.dumps(row),flush=True)
    return row

def main():
    module=ast.parse((RUN/'sources/CLIP/clip/clip.py').read_text(encoding='utf-8'))
    models=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='_MODELS' for t in n.targets))
    url=models['ViT-B/32']
    items=[('ViT-B-32.pt',url,url.split('/')[-2]),
        ('sam2_hiera_tiny.pt','https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt',None)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(download,items))
    pahnet=RUN/'weights/pahnet_hdm_pascal_split0_5shot.pth'
    if pahnet.exists():
        rows.append(dict(name=pahnet.name,url='https://drive.google.com/file/d/1-qVhq0_Xi47buc4nLjsiJqOezbqzfaHo/view',
                         author_folder='https://drive.google.com/drive/folders/1RJvMJlL088Idpfv5nGkAQgDe9cl9Ug27',
                         bytes=pahnet.stat().st_size,sha256=sha(pahnet),scope='author Pascal split0 five-shot source model; no infrared training'))
    write_json(RUN/'asset_provenance.json',rows)

if __name__=='__main__':main()
