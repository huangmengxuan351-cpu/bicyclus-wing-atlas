"""Update an existing free Docker Studio from a code-only GitHub checkout."""
import json
import base64
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hardware_items(value):
    if isinstance(value, dict):
        if 'name' in value and 'resource_type' in value:
            yield value
        for item in value.values():
            yield from hardware_items(item)
    elif isinstance(value, list):
        for item in value:
            yield from hardware_items(item)


def main():
    studio = os.environ.get('MODELSCOPE_STUDIO', '')
    token = os.environ.get('MODELSCOPE_API_KEY', '')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', studio) or not token:
        raise RuntimeError('Configure MODELSCOPE_STUDIO and the private MODELSCOPE_API_KEY GitHub secret.')
    from modelscope_hub import HubApi
    api = HubApi(endpoint='https://modelscope.cn', token=token)
    info = api.get_repo(studio, repo_type='studio')
    if info.sdk_type != 'docker':
        raise RuntimeError('The existing Studio must use Docker.')
    choices = list(hardware_items(api.openapi.list_studio_hardware(sdk_type='docker', studio=studio)))
    if not any(item['name'] == info.hardware and item['resource_type'] == 'free' and not item['name'].startswith('paid/') for item in choices):
        raise RuntimeError('The current account API did not confirm this Studio hardware is free. No code or resource changes were made.')
    # An allowlist prevents model/data/cache files from entering the Studio source.
    allowed = {'Dockerfile', 'requirements.txt', '.dockerignore', 'README.md'}
    paths = [p for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink()
             and ((p.relative_to(ROOT).parts[0] in ('code_python', 'output_results', 'deploy') and p.suffix in ('.py', '.html', '.css', '.js', '.json', '.png'))
                  or p.relative_to(ROOT).as_posix() in allowed)]
    if any(p.stat().st_size > 10 * 1024 * 1024 for p in paths):
        raise RuntimeError('Unexpected large file in the public code checkout.')
    with tempfile.TemporaryDirectory(prefix='atlas-studio-sync-') as directory:
        work = Path(directory)
        authorization = base64.b64encode(('oauth2:' + token).encode()).decode()
        env = dict(os.environ, GIT_TERMINAL_PROMPT='0',
                   GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='http.https://modelscope.cn/.extraheader',
                   GIT_CONFIG_VALUE_0='Authorization: Basic ' + authorization,
                   GIT_AUTHOR_NAME='Wing Atlas deployment', GIT_AUTHOR_EMAIL='deployment@users.noreply.github.com',
                   GIT_COMMITTER_NAME='Wing Atlas deployment', GIT_COMMITTER_EMAIL='deployment@users.noreply.github.com')
        checkout = work / 'studio'

        def git(*args):
            result = subprocess.run(['git', '-c', 'credential.helper=', *args], env=env, capture_output=True, text=True)
            if result.returncode:
                # Keep API credentials and private endpoint errors out of CI logs.
                raise RuntimeError('Studio Git synchronization failed; inspect authentication and branch settings in the account.')
            return result.stdout

        git('clone', '--branch', 'master', 'https://modelscope.cn/studios/' + studio + '.git', str(checkout))
        managed = []
        for path in paths:
            relative = path.relative_to(ROOT)
            destination = checkout / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            managed.append(relative.as_posix())
        git('-C', str(checkout), 'add', '--', *managed)
        changed = git('-C', str(checkout), 'diff', '--cached', '--name-only')
        if changed.strip():
            git('-C', str(checkout), 'commit', '-m', 'Deploy reviewed Wing Atlas code from GitHub')
            git('-C', str(checkout), 'push', 'origin', 'HEAD:master')
        # Recheck billing status immediately before deployment; never switch tiers.
        current = api.get_repo(studio, repo_type='studio')
        if current.hardware != info.hardware:
            raise RuntimeError('Hardware changed during synchronization. Deployment was not triggered.')
        api.deploy_repo(studio, repo_type='studio')
    print('Existing free Docker Studio deployment requested. Check its build and runtime logs; this is not a readiness confirmation.')


if __name__ == '__main__':
    main()
