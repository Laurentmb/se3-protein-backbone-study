"""Fetch the recorded upstream commits and verify published checkpoint hashes.

This downloads model assets only. It does not submit amino-acid sequences.
Existing checkouts must match the requested commits; they are never reset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm='sha256'):
    """Hash large model files without loading a second copy into memory."""
    value = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def checkout(path, url, commit, check_only):
    """Create a shallow detached checkout, or verify an existing one."""
    if not path.exists():
        if check_only:
            raise RuntimeError('Missing source checkout: '+str(path))
        path.mkdir(parents=True)
        for arguments in [['init', str(path)], ['-C', str(path), 'remote', 'add', 'origin', url],
                          ['-C', str(path), 'fetch', '--depth', '1', 'origin', commit],
                          ['-C', str(path), 'checkout', '--detach', 'FETCH_HEAD']]:
            subprocess.run(['git']+arguments, check=True)
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != commit:
        raise RuntimeError('Checkout commit mismatch: '+str(path))
    print('Verified source:', path.name, commit, flush=True)


def download(url, path):
    """Write a temporary download and replace the destination only on completion."""
    temporary = path.with_suffix(path.suffix+'.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'SE3-backbone-study/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as stream:
        shutil.copyfileobj(response, stream, length=4*1024*1024)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'external')
    parser.add_argument('--check-only', action='store_true', help='Verify local assets without downloading')
    args = parser.parse_args()
    destination = args.output.resolve()
    sources = json.loads((ROOT/'configs/sources.json').read_text())
    for name in ['frameflow', 'framediff']:
        item = sources[name]
        checkout(destination/name, item['url'], item['commit'], args.check_only)

    # ProteinMPNN's checkpoint is already tracked in the pinned FrameDiff tree.
    weights = destination/'proteinmpnn_weights/v_48_020.pt'
    if not weights.exists() and not args.check_only:
        weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(destination/'framediff/ProteinMPNN/vanilla_model_weights/v_48_020.pt'), str(weights))

    ff = destination/'frameflow'
    checkpoint = ff/'weights/pdb/published.ckpt'
    config = ff/'weights/pdb/config.yaml'
    if (not checkpoint.exists() or not config.exists()) and not args.check_only:
        archive = destination/'frameflow_weights.tar.gz'
        expected = sources['frameflow']['archive_md5']
        if not archive.exists() or digest(archive, 'md5') != expected:
            download(sources['frameflow']['weights_url'], archive)
        if digest(archive, 'md5') != expected:
            raise RuntimeError('FrameFlow archive checksum mismatch')
        # Read only the two named regular files; never extract arbitrary archive paths.
        selected = {}
        with tarfile.open(str(archive), 'r:gz') as packed:
            for member in packed.getmembers():
                name = member.name
                while name.startswith('./'):
                    name = name[2:]
                if name in {'weights/pdb/published.ckpt', 'weights/pdb/config.yaml'}:
                    if not member.isfile() or name in selected:
                        raise RuntimeError('Unexpected model archive member')
                    target = ff/name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with packed.extractfile(member) as incoming, target.open('wb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
                    selected[name] = True
        if len(selected) != 2:
            raise RuntimeError('Required FrameFlow files missing from archive')
    checks = [(checkpoint, sources['frameflow']['checkpoint_sha256']),
        (config, sources['frameflow']['config_sha256']),
        (destination/'framediff/weights/paper_weights.pth', sources['framediff']['checkpoint_sha256']),
        (weights, sources['proteinmpnn']['checkpoint_sha256'])]
    for path, expected in checks:
        if not path.exists() or digest(path) != expected:
            raise RuntimeError('Checkpoint/config checksum mismatch: '+str(path))
        print('Verified asset:', path.name, flush=True)


if __name__ == '__main__':
    main()
