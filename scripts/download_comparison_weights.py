"""Fetch official foundation weights needed by ready comparison adapters."""
import hashlib
import json
from pathlib import Path
import sys
from scripts.prepare_comparison import DEFAULT_OUTPUT, write_json
sys.path.append(str((DEFAULT_OUTPUT / "runtime_packages").resolve()))
import requests

WEIGHTS = {
    "dinov2_vitl14_pretrain.pth": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "resnet50-11ad3fa6.pth": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    "ViT-B-16.pt": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}


def main():
    folder = DEFAULT_OUTPUT / "weights"
    folder.mkdir(exist_ok=True)
    records = []
    for name, url in WEIGHTS.items():
        destination = folder / name
        partial = folder / f"{name}.part"
        if not destination.exists():
            offset = partial.stat().st_size if partial.exists() else 0
            with requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {}, stream=True, timeout=60) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if not append:
                    offset = 0
                expected = offset + int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an inconsistent resume offset")
                with partial.open("ab" if append else "wb") as stream:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                if expected is not None and partial.stat().st_size != expected:
                    raise ValueError(f"Incomplete weight file: {name}")
            partial.replace(destination)
        digest = hashlib.sha256()
        with destination.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        records.append(dict(name=name, url=url, bytes=destination.stat().st_size, sha256=digest.hexdigest()))
        write_json(folder / "matcher_weight_provenance.json", records)
        print(json.dumps(records[-1]), flush=True)


if __name__ == "__main__":
    main()
