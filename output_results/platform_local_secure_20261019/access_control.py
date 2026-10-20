"""Portable credentials for private services and password-gated original exports."""
import base64
import copy
import json
import os
import secrets
import threading
import time
from pathlib import Path

STATE = Path(os.environ.get('ATLAS_STATE', '/mnt/workspace/wing-atlas-state'))
VAULT = STATE / 'access_roles.json'
LOCK = threading.RLock()
CACHED = None
CACHED_AT = 0.0


def atomic_save(value):
    STATE.mkdir(parents=True, exist_ok=True)
    temporary = VAULT.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.chmod(temporary, 0o600)
    temporary.replace(VAULT)


def initialize():
    with LOCK:
        if VAULT.exists():
            return
        atomic_save({'schema': 1, 'scope': 'private runtime storage; never commit', 'roles': {
            'viewer': {'username': 'atlas_internal', 'password': secrets.token_urlsafe(32),
                       'enabled': True, 'can_download': False},
            'exporter': {'username': 'atlas_download', 'password': secrets.token_urlsafe(32),
                         'enabled': False, 'can_download': True}},
            'maintenance': secrets.token_urlsafe(40)})


def load():
    global CACHED, CACHED_AT
    initialize()
    with LOCK:
        if CACHED is None or time.monotonic() - CACHED_AT > .5:
            CACHED = json.loads(VAULT.read_text(encoding='utf-8'))
            CACHED_AT = time.monotonic()
        value = copy.deepcopy(CACHED)
    # Configure this variable in the hosting platform, never in public code.
    password = os.environ.get('ATLAS_DOWNLOAD_PASSWORD', '')
    if password:
        value['roles']['exporter']['password'] = password
        value['roles']['exporter']['enabled'] = os.environ.get('ATLAS_DOWNLOAD_ENABLED', 'true').lower() == 'true'
    return value


def credentials(role='viewer'):
    item = load()['roles'][role]
    return item['username'], item['password']


def policy():
    return {key: {field: value for field, value in item.items() if field != 'password'}
            for key, item in load()['roles'].items()}


def authenticate(header):
    if not header.startswith('Basic '):
        return None
    try:
        username, password = base64.b64decode(header[6:], validate=True).decode().split(':', 1)
    except (ValueError, UnicodeError):
        return None
    for role, item in load()['roles'].items():
        if item['enabled'] and secrets.compare_digest(username.encode(), item['username'].encode()) and secrets.compare_digest(password.encode(), item['password'].encode()):
            return dict(item, role=role)
    return None


def set_download_enabled(enabled):
    global CACHED
    if not isinstance(enabled, bool):
        raise ValueError('enabled must be a boolean')
    with LOCK:
        initialize()
        value = json.loads(VAULT.read_text())
        value['roles']['exporter']['enabled'] = enabled
        atomic_save(value)
        CACHED = None
    return policy()
