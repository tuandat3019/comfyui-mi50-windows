"""Opt-in model downloads, pinned to the revisions used by the reference setup."""
from pathlib import Path
import argparse
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mi50.paths import REPO, runtime_path
from mi50.downloads import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['image', 'trellis'], required=True)
    parser.add_argument('--runtime')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    models = runtime_path(args.runtime) / 'ComfyUI/models'
    records = json.loads((REPO / 'config/models.json').read_text(encoding='utf-8'))[args.profile]
    if args.plan:
        print(json.dumps({'models_directory': str(models), 'sources': records, 'gpu_work': False}, indent=2))
        return
    from huggingface_hub import hf_hub_download, snapshot_download
    for record in records:
        if 'filename' in record:
            downloaded = Path(hf_hub_download(repo_id=record['repo'], revision=record['revision'], filename=record['filename']))
            verify(downloaded, record)
            target = models / record['destination']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded, target)
        else:
            destination = models / record['directory']
            if 'files' in record:
                for name in record['files']:
                    hf_hub_download(repo_id=record['repo'], revision=record['revision'], filename=name, local_dir=destination)
            else:
                snapshot_download(repo_id=record['repo'], revision=record['revision'], local_dir=destination,
                                  allow_patterns=record['allow_patterns'])
    print('[models] Download complete. No GPU inference has been run.')


if __name__ == '__main__':
    main()
