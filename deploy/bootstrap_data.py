"""Install a pinned private dataset snapshot into ModelScope persistent storage."""
import hashlib
import json
import os
from pathlib import Path


def file_sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_manifest(folder, expected_hash):
    manifest = folder / 'data_manifest.json'
    if file_sha(manifest) != expected_hash:
        raise RuntimeError('Private data manifest differs from the configured release.')
    value = json.loads(manifest.read_text(encoding='utf-8'))
    for item in value['files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Invalid private manifest path.')
        path = folder / relative
        if not path.is_file() or path.stat().st_size != item['bytes'] or file_sha(path) != item['sha256']:
            raise RuntimeError('Private dataset verification failed: ' + item['path'])
    print('Private dataset verified:', value['file_count'], 'files; originals remain outside the public repository.', flush=True)


def ensure_data(folder):
    expected_hash = os.environ.get('ATLAS_DATA_MANIFEST_SHA256', '')
    marker = folder / '.verified-release'
    if (folder / 'data_manifest.json').is_file():
        # Local installations can use an explicitly mounted, previously verified tree.
        if not expected_hash:
            return
        if marker.is_file() and marker.read_text(encoding='utf-8').strip() == expected_hash and file_sha(folder / 'data_manifest.json') == expected_hash:
            return
        try:
            verify_manifest(folder, expected_hash)
        except (RuntimeError, FileNotFoundError):
            if not os.environ.get('ATLAS_DATA_REPO') or not os.environ.get('ATLAS_DATA_TOKEN'):
                raise
        else:
            marker.write_text(expected_hash)
            return
    repo = os.environ.get('ATLAS_DATA_REPO', '')
    token = os.environ.get('ATLAS_DATA_TOKEN', '')
    revision = os.environ.get('ATLAS_DATA_REVISION', 'master')
    if not repo or not token or len(expected_hash) != 64:
        raise RuntimeError('Configure the private data repository, read-only secret token and release manifest SHA-256 before starting.')
    from modelscope_hub import HubApi
    api = HubApi(endpoint='https://modelscope.cn', token=token)
    info = api.get_repo(repo, repo_type='dataset')
    if info.private is False or info.visibility not in (None, 1) or (info.private is not True and info.visibility != 1):
        raise RuntimeError('Refusing to install a dataset unless the API confirms private visibility.')
    folder.mkdir(parents=True, exist_ok=True)
    api.download_file(repo, repo_type='dataset', file_path='data_manifest.json',
                      revision=revision, local_dir=folder, expected_sha256=expected_hash)
    manifest = json.loads((folder / 'data_manifest.json').read_text(encoding='utf-8'))
    api.download_repo(repo, repo_type='dataset', revision=revision, local_dir=folder,
                      allow_patterns=[item['path'] for item in manifest['files']], max_workers=4)
    verify_manifest(folder, expected_hash)
    marker.write_text(expected_hash)


if __name__ == '__main__':
    ensure_data(Path(os.environ.get('ATLAS_DATA_DIR', '/mnt/workspace/wing-atlas-data')))
