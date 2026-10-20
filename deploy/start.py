"""Start the existing validated engines with private data and persistent runtime state."""
import os
import json
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('ATLAS_DATA_DIR', '/mnt/workspace/wing-atlas-data'))
STATE = Path(os.environ.get('ATLAS_STATE', '/mnt/workspace/wing-atlas-state'))
PRIVATE = ROOT / 'output_results/platform_local_secure_20261019'
PUBLIC = ROOT / 'output_results/platform_public_bilingual_20261019'
children = []


def mount_tree(source, target):
    if not target.exists() and not target.is_symlink():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=source.is_dir())
    elif source.is_dir() and target.is_dir() and not target.is_symlink():
        for item in source.iterdir():
            mount_tree(item, target / item.name)
    elif not target.is_symlink() or target.resolve() != source.resolve():
        raise RuntimeError('Private data collides with code: ' + str(target.relative_to(ROOT)))


def launch(script, *arguments):
    process = subprocess.Popen([sys.executable, '-u', str(script), *arguments], cwd=ROOT)
    children.append(process)


def wait_for(url):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in children):
            raise RuntimeError('A required service exited during startup.')
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(.5)
    raise TimeoutError('Service readiness timeout.')


def stop(*_):
    for process in reversed(children):
        if process.poll() is None:
            process.terminate()
    for process in children:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    def phase(name):
        (STATE / 'startup_progress.json').write_text(json.dumps({'phase': name}), encoding='utf-8')
    phase('initializing')
    launch(ROOT / 'deploy/startup_status.py')
    loading = children[-1]
    wait_for('http://127.0.0.1:' + os.environ.get('PORT', '7860') + '/healthz')
    if os.environ.get('ATLAS_SETUP_ONLY', '').lower() == 'true':
        phase('deployment_preflight')
        while loading.poll() is None:
            time.sleep(1)
        raise RuntimeError('Setup service exited.')
    phase('installing_private_data')
    from bootstrap_data import ensure_data
    ensure_data(DATA)
    if not (DATA / 'data_manifest.json').is_file():
        raise RuntimeError('Install the private data bundle before starting this service.')
    if DATA.resolve() == ROOT.resolve() or DATA.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError('Private data must be outside the code repository.')
    phase('starting_prediction_services')
    (STATE / 'preview_cache').mkdir(exist_ok=True)
    (STATE / 'prediction').mkdir(exist_ok=True)
    for item in DATA.iterdir():
        if item.name not in ('data_manifest.json', '.verified-release', '.cache', '.modelscope') and not item.name.startswith('.'):
            mount_tree(item, ROOT / item.name)
    os.environ['ATLAS_STATE'] = str(STATE)
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    sys.path.insert(0, str(PRIVATE))
    import access_control
    access_control.initialize()
    launch(PRIVATE / 'backend_runner.py')
    wait_for('http://127.0.0.1:8881/api/health')
    launch(PRIVATE / 'secure_gateway.py')
    wait_for('http://127.0.0.1:8880/healthz')
    launch(PUBLIC / 'public_gateway.py', '--language', 'zh', '--port', '8890')
    launch(PUBLIC / 'public_gateway.py', '--language', 'en', '--port', '8891')
    wait_for('http://127.0.0.1:8890/healthz')
    wait_for('http://127.0.0.1:8891/healthz')
    loading.terminate()
    loading.wait(timeout=5)
    children.remove(loading)
    launch(ROOT / 'deploy/dispatcher.py')
    print('Both language backends started; only port 7860 is public.', flush=True)
    while all(process.poll() is None for process in children):
        time.sleep(1)
    raise RuntimeError('A service exited; restart the container.')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        main()
    finally:
        stop()
