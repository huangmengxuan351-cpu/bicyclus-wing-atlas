"""Keep the cloud port responsive while private data is installed or verified."""
import json
import os
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

STATE = Path(os.environ.get('ATLAS_STATE', '/mnt/workspace/wing-atlas-state'))
STATUS = STATE / 'startup_progress.json'


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/__bootstrap_wait' and self.command == 'GET':
            # Function Compute freezes background work between invocations.
            # A bounded provisioning request lets first-time initialization run.
            # It is called during deployment only, never as a permanent pinger.
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            deadline = time.monotonic() + 45
            try:
                while time.monotonic() < deadline:
                    value = {'ready': False, 'phase': 'initializing'}
                    for file in (STATUS, STATE / 'bootstrap_transfer.json'):
                        try:
                            value.update(json.loads(file.read_text(encoding='utf-8')))
                        except (OSError, ValueError):
                            pass
                    self.wfile.write((json.dumps(value) + '\n').encode())
                    self.wfile.flush()
                    time.sleep(5)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if path in ('/', '/healthz') or path in ('/zh/healthz', '/en/healthz'):
            code = 200
        else:
            code = 503
        value = {'ready': False, 'status': 'preparing',
                 'message': 'The prediction service is preparing. Please try again later.'}
        try:
            value['phase'] = json.loads(STATUS.read_text())['phase']
        except (OSError, ValueError, KeyError):
            pass
        body = json.dumps(value).encode()
        self.send_response(code)
        for key, value in {'Content-Type': 'application/json; charset=utf-8', 'Content-Length': str(len(body)),
                            'Cache-Control': 'no-store', 'Retry-After': '30'}.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    do_HEAD = do_GET


if __name__ == '__main__':
    STATE.mkdir(parents=True, exist_ok=True)
    data_parent = Path(os.environ.get('ATLAS_DATA_DIR', '/mnt/workspace/wing-atlas-data')).parent
    data_parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(data_parent)
    # Only size figures go to owner-accessible run logs. This is a filesystem
    # observation, not proof of an account-specific storage quota.
    virtual_capacity = usage.total >= 2 ** 60
    print(json.dumps({'event': 'cloud_filesystem_capacity',
                      'total_bytes': None if virtual_capacity else usage.total,
                      'free_bytes': None if virtual_capacity else usage.free,
                      'virtual_capacity_unbounded': virtual_capacity,
                      'account_quota_verified': False}), flush=True)
    server = ThreadingHTTPServer(('0.0.0.0', int(os.environ.get('PORT', '7860'))), Handler)
    server.serve_forever()
