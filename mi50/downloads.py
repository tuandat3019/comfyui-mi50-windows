"""Checksum-verified downloads and bounded, traversal-safe ZIP extraction."""
from pathlib import Path, PurePosixPath
import hashlib
import os
import shutil
import urllib.request
import zipfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(path, record):
    path = Path(path)
    if record.get('bytes') is not None and path.stat().st_size != record['bytes']:
        raise RuntimeError(f'Size mismatch: {path.name}')
    if digest(path) != record['sha256']:
        raise RuntimeError(f'SHA-256 mismatch: {path.name}')


def fetch(record, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    name = record['name']
    if Path(name).name != name or ':' in name or '\\' in name:
        raise ValueError('Download filename must be a basename')
    target = cache / name
    if target.exists():
        verify(target, record)
        return target
    temporary = target.with_suffix(target.suffix + '.part')
    headers = {'User-Agent': 'comfyui-mi50-windows/0.1', **record.get('headers', {})}
    request = urllib.request.Request(record['url'], headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        verify(temporary, record)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def extract_zip(archive, destination, strip_root=False, headers_only=False):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        # Validate every member before writing any of them.
        prepared = []
        roots = set()
        for entry in entries:
            if '\\' in entry.orig_filename or ':' in entry.orig_filename:
                raise ValueError('Unsupported ZIP path')
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('ZIP path escapes destination')
            if not name.parts:
                continue
            roots.add(name.parts[0])
            relative = Path(*name.parts[1:] if strip_root else name.parts)
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('ZIP symlinks are unsupported')
            if headers_only and (not relative.parts or relative.parts[0] not in ('include', 'lib')):
                continue
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('ZIP path escapes destination')
            prepared.append((entry, target))
        if strip_root and len(roots) != 1:
            raise ValueError('Expected one archive root directory')
        for entry, target in prepared:
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
