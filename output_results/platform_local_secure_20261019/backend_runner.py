"""Private loopback backend accepting only the gateway's internal credential."""
import os
import secrets
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from access_control import initialize,load

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'code_python'))
from dapi_stage_atlas_portal_v2_server import App,DEFAULT_ATLAS,make_handler

def main():
    initialize()
    app=App(DEFAULT_ATLAS,Path(os.environ.get('ATLAS_STATE', '/mnt/workspace/wing-atlas-state'))/'prediction')
    base=make_handler(app)
    class PrivateHandler(base):
        def allowed(self):
            return secrets.compare_digest(self.headers.get('X-Atlas-Internal-Token',''),load()['maintenance'])
        def do_GET(self):
            if urlsplit(self.path).path == '/api/health':
                return self.send_json({'ready':True,'app':'private-platform-backend','predictor_loaded':app.engine is not None})
            if not self.allowed():
                return self.send_json({'error':'Private backend. Use the authorized display gateway.'},403)
            return super().do_GET()
        def do_POST(self):
            if not self.allowed():
                return self.send_json({'error':'Private backend. Use the authorized display gateway.'},403)
            return super().do_POST()
    server=ThreadingHTTPServer(('127.0.0.1',8881),PrivateHandler)
    server.daemon_threads=True
    print('Private backend: loopback 8881, internal credential required.',flush=True)
    server.serve_forever()

if __name__=='__main__':
    main()
