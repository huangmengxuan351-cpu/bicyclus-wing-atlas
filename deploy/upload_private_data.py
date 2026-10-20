"""Upload the reviewed data folder only to an already-created PRIVATE dataset."""
import argparse
import os
import json
import time
from pathlib import Path
from bootstrap_data import verify_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--repo', required=True, help='Existing private ModelScope dataset owner/name.')
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    token = os.environ.get('MODELSCOPE_API_KEY', '')
    if not token:
        raise RuntimeError('Set MODELSCOPE_API_KEY locally; never paste it into chat or source code.')
    progress_path = args.data_dir.parent / 'private_upload_status.json'
    started = time.time()
    last_print = [0.0]
    def progress(value):
        safe = {key: value.get(key) for key in ('event', 'batch_index', 'num_batches', 'committed_files', 'committed_bytes', 'uploaded_bytes', 'uploaded_files', 'total_files', 'total_bytes', 'elapsed')}
        safe['updated_at_unix'] = time.time()
        progress_path.write_text(json.dumps(safe, indent=2), encoding='utf-8')
        if safe['event'] != 'upload_progress' or time.monotonic() - last_print[0] >= 30:
            print(json.dumps(safe), flush=True)
            last_print[0] = time.monotonic()
    progress({'event': 'verifying_local_release'})
    verify_manifest(args.data_dir, args.manifest_sha256)
    from modelscope_hub import HubApi
    api = HubApi(endpoint='https://modelscope.cn', token=token)
    info = api.get_repo(args.repo, repo_type='dataset')
    if info.private is False or info.visibility not in (None, 1) or (info.private is not True and info.visibility != 1):
        raise RuntimeError('The target is not confirmed PRIVATE. No files were uploaded.')
    value = json.loads((args.data_dir / 'data_manifest.json').read_text(encoding='utf-8'))
    release_bytes = value['total_bytes'] + (args.data_dir / 'data_manifest.json').stat().st_size
    progress({'event': 'uploading', 'total_files': value['file_count'] + 1, 'total_bytes': release_bytes})
    api.upload_folder(args.repo, repo_type='dataset', folder_path=args.data_dir,
                       allow_patterns=['data_manifest.json'] + [item['path'] for item in value['files']],
                       max_workers=8, tracker_path=args.data_dir.parent / 'private-upload-progress.json',
                       disable_tqdm=True, use_cache=True, progress_callback=progress,
                       commit_message='Reviewed 12-gene platform private data release')
    progress({'event': 'completed', 'committed_files': value['file_count'] + 1, 'committed_bytes': release_bytes,
              'total_files': value['file_count'] + 1, 'total_bytes': release_bytes, 'elapsed': time.time() - started})
    print('Private dataset upload completed. Keep dataset visibility private.', flush=True)


if __name__ == '__main__':
    main()
