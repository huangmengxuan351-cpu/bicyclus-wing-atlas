"""Build one independent Pages upload directory using verified, public previews."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=('zh', 'en'), required=True)
    parser.add_argument('--pack-dir', type=Path, help='Existing local public preview pack directory.')
    args = parser.parse_args()
    source = ROOT / 'public' / args.language
    target = ROOT / 'dist' / args.language
    target.mkdir(parents=True, exist_ok=True)
    # Refuse a stale output tree rather than accidentally publishing old files.
    if any(target.iterdir()):
        raise RuntimeError('Output directory must be empty: ' + str(target))
    for file in source.rglob('*'):
        if file.is_file() and file.name not in ('preview-packs.json', 'wrangler.jsonc'):
            destination = target / file.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
    manifest = json.loads((source / 'preview-packs.json').read_text(encoding='utf-8'))
    (target / 'packs').mkdir()

    def copy_pack(item):
        destination = target / 'packs' / item['name']
        if args.pack_dir:
            shutil.copy2(args.pack_dir / item['name'], destination)
        else:
            request = urllib.request.Request(manifest['base_url'] + item['name'], headers={'User-Agent': 'WingAtlas-CI/1'})
            with urllib.request.urlopen(request, timeout=120) as response, destination.open('wb') as output:
                shutil.copyfileobj(response, output)
        if destination.stat().st_size != item['bytes'] or digest(destination) != item['sha256']:
            raise RuntimeError('Public preview pack verification failed: ' + item['name'])

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(copy_pack, manifest['packs']))
    origin = os.environ.get('ATLAS_ORIGIN', '').rstrip('/')
    if origin:
        from urllib.parse import urlsplit
        parsed = urlsplit(origin)
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise RuntimeError('ATLAS_ORIGIN must be an HTTPS service URL without credentials.')
        origin += '/' + args.language
    config = {'$schema': './node_modules/wrangler/config-schema.json',
              'name': 'bicyclus-wing-atlas-' + args.language,
              'pages_build_output_dir': 'dist/' + args.language,
              'compatibility_date': '2026-10-20', 'vars': {'ATLAS_ORIGIN': origin}}
    (ROOT / 'wrangler.jsonc').write_text(json.dumps(config, indent=2))
    print('Built', args.language, 'with', len(manifest['packs']), 'verified preview packs. Cloud prediction configured:', bool(origin))


if __name__ == '__main__':
    main()
