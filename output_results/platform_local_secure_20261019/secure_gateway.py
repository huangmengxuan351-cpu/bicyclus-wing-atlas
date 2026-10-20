"""Authenticated display previews with server-enforced original download rights."""
from __future__ import annotations
import os
import copy
import hashlib
import hmac
import io
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import unquote, urlsplit
from PIL import Image, PngImagePlugin
from access_control import authenticate, initialize, load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UI = ROOT/'code_python/dapi_stage_atlas_portal_v2'
ANATOMY = ROOT/'output_results/dapi_stage_atlas_portal_v2/vein_assets_v1'
BACKEND = 'http://127.0.0.1:8881'
MAX_EDGE = 960
CACHE = Path(os.environ.get('ATLAS_STATE', '/mnt/workspace/wing-atlas-state'))/'preview_cache'
CACHE.mkdir(exist_ok=True)
FRONTENDS=HERE/'frontends'
LOCKED = '/downloads/authorization-required'
PREVIEW_ASSETS = frozenset(('gray.png','mask.png','selected_aligned_gray.png','selected_aligned_mask.png','forewing_aligned_gray.png','forewing_aligned_mask.png','hindwing_aligned_gray.png','hindwing_aligned_mask.png','manual_aligned_gray.png','manual_aligned_mask.png'))
LOG = logging.getLogger('access')
LOG.setLevel(logging.INFO)
handler = RotatingFileHandler(Path(os.environ.get('ATLAS_STATE', '/mnt/workspace/wing-atlas-state'))/'secure_access.log',maxBytes=2_000_000,backupCount=3,encoding='utf-8')
handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
LOG.addHandler(handler)
STATE_LOCK = threading.RLock()
REGISTERED = {}
RUNS_FILE = Path(os.environ.get('ATLAS_STATE', '/mnt/workspace/wing-atlas-state'))/'viewer_prediction_runs.json'

def request_backend(route, method='GET', data=None, headers=None):
    internal_headers=dict(headers or {})
    internal_headers['X-Atlas-Internal-Token']=load()['maintenance']
    request = urllib.request.Request(BACKEND+route, data=data, method=method, headers=internal_headers)
    try:
        response = urllib.request.urlopen(request,timeout=180)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return response.status, dict(response.headers), response.read()

def local_asset(url):
    path = unquote(urlsplit(url).path)
    if '\\' in path or any(p in ('.','..') for p in path.split('/')):
        raise ValueError('Invalid asset path')
    if path.startswith('/atlas-assets/'):
        dataset, relative = path[len('/atlas-assets/'):].split('/',1)
        config = json.loads((UI/'atlas_sources.json').read_text(encoding='utf-8'))
        roots = {'main':ROOT/'output_results/gene_atlas_sop_run/seven_genes_20261007_complete_v2'}
        roots.update({i['key']:ROOT/i['path'] for i in config['additional_archives']})
        folder = roots[dataset].resolve()
    elif path.startswith('/anatomy-assets/'):
        folder, relative = ANATOMY.resolve(), path[len('/anatomy-assets/'):]
    elif path.startswith(('/predict/','/portal/','/atlas/')):
        folder, relative = UI.resolve(), path.lstrip('/')
    else:
        raise ValueError('Not a display asset')
    target = (folder/relative).resolve()
    if not target.is_relative_to(folder) or target.suffix.lower() != '.png' or not target.is_file():
        raise ValueError('Not a display image')
    return target

def register_preview(url):
    target = local_asset(url)
    stat = target.stat()
    identity = f'{url}|{stat.st_mtime_ns}|{stat.st_size}|{MAX_EDGE}|half-resolution-v1'
    token = hmac.new(load()['maintenance'].encode(),identity.encode(),hashlib.sha256).hexdigest()[:40]
    with STATE_LOCK:
        if len(REGISTERED) > 20000:
            REGISTERED.clear()
        REGISTERED[token] = target
    return '/preview/'+token+'.png'

def make_preview(target):
    stat = target.stat()
    digest = hashlib.sha256(f'{target}|{stat.st_mtime_ns}|{stat.st_size}|{MAX_EDGE}|v1'.encode()).hexdigest()
    cached = CACHE/(digest+'.png')
    if cached.is_file():
        return cached.read_bytes()
    with Image.open(target) as image:
        image.load()
        if image.mode not in ('RGB','RGBA','L','LA'):
            image = image.convert('RGBA' if 'A' in image.getbands() else 'RGB')
        factor = min(0.5,MAX_EDGE/max(image.size))
        size = tuple(max(1,round(x*factor)) for x in image.size)
        preview = image.resize(size,Image.Resampling.LANCZOS)
        info = PngImagePlugin.PngInfo()
        info.add_text('Access','Display preview only. Original downloads require separate authorization.')
        buffer = io.BytesIO()
        preview.save(buffer,format='PNG',pnginfo=info)
        data = buffer.getvalue()
    # A unique temp name avoids concurrent creation of the same preview.
    temporary = cached.with_suffix(f'.{threading.get_ident()}.tmp')
    temporary.write_bytes(data)
    temporary.replace(cached)
    return data

def viewer_catalog(value):
    data = copy.deepcopy(value)
    for key in ('report','experiments','manifest','coverage','display_comparison_url'):
        if key in data:
            data[key] = LOCKED
    for report in data.get('archive_reports',{}).values():
        for key in report:
            report[key] = LOCKED
    for group in data['groups']:
        for key in ('dapi','mask','comparison'):
            group[key] = register_preview(group[key])
        for variant in group['variants'].values():
            for key,url in list(variant['images'].items()):
                variant['images'][key] = LOCKED if key == 'intensity_16bit' else register_preview(url)
            variant['files'] = {key:LOCKED for key in variant['files']}
        for source in group['sources']:
            for key in ('aligned','intensity'):
                source[key] = register_preview(source[key])
    for case in data['anatomy_cases']:
        for key,url in list(case['assets'].items()):
            case['assets'][key] = register_preview(url) if key != 'manifest' else LOCKED
        case['rendering_comparison'] = register_preview(case['rendering_comparison'])
    data['access'] = {'role':'viewer','can_download':False,'preview_max_edge':MAX_EDGE,'preview_scale_at_most':0.5,'scientific_reference_grid':[2048,2048]}
    return data

def viewer_runs():
    with STATE_LOCK:
        return json.loads(RUNS_FILE.read_text(encoding='utf-8')) if RUNS_FILE.exists() else {}

def add_run(run_id, username):
    with STATE_LOCK:
        runs = viewer_runs()
        runs[run_id] = username
        temporary = RUNS_FILE.with_suffix('.tmp')
        temporary.write_text(json.dumps(runs,indent=2),encoding='utf-8')
        temporary.replace(RUNS_FILE)

def clean_paths(value):
    if isinstance(value,dict):
        return {k:clean_paths(v) for k,v in value.items()}
    if isinstance(value,list):
        return [clean_paths(v) for v in value]
    if isinstance(value,str) and re.match(r'^[A-Za-z]:[\\/]',value):
        return '[local file]'
    return value

def augment_html(data,language='en'):
    markup = data.decode('utf-8')
    return markup.replace('</body>',f'<script src="/{language}/access-display.js"></script></body>').encode('utf-8')

def frontend_target(path):
    matched=re.fullmatch(r'/(en|zh)(/.*)?',path)
    if not matched:
        return None
    language=matched[1]
    rest=matched[2] or '/'
    routes={'/':'portal/index.html','/atlas/':'atlas/index.html','/anatomy/':'atlas/anatomy.html','/predict/':'predict/index.html','/result/':'portal/result.html'}
    relative=routes.get(rest,rest.lstrip('/'))
    if relative!='access-display.js' and not relative.startswith(('atlas/','portal/','predict/')):
        return None
    folder=(FRONTENDS/language).resolve()
    target=(folder/relative).resolve()
    if not target.is_relative_to(folder) or not target.is_file():
        return None
    return language,target

class Handler(BaseHTTPRequestHandler):
    server_version = 'WingAtlas'

    def log_message(self,*_):
        pass

    def send(self,data,status=200,kind='application/json; charset=utf-8',extra=None):
        if not isinstance(data,bytes):
            data = json.dumps(data,ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('X-Frame-Options','SAMEORIGIN')
        self.send_header('Content-Security-Policy',"frame-ancestors 'self'")
        self.send_header('Referrer-Policy','same-origin')
        self.send_header('X-Robots-Tag','noindex, nofollow, noarchive')
        for key,value in (extra or {}).items():
            self.send_header(key,value)
        self.end_headers()
        if self.command != 'HEAD':
            try:
                self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):
                pass

    def deny(self,message='Original downloads require separate authorization.',status=403):
        LOG.info('DENY role=%s method=%s path=%s',getattr(self,'role',{}).get('role','anonymous'),self.command,urlsplit(self.path).path[:300])
        self.send({'error':message,'download_authorization_required':True},status)

    def proxy(self, route, method='GET', data=None, transform=None):
        headers = {key:value for key,value in self.headers.items() if key.lower() in ('content-type','accept')}
        status, response_headers, body = request_backend(route,method,data,headers)
        kind = response_headers.get('Content-Type','application/octet-stream')
        if transform and status == 200:
            body = json.dumps(transform(json.loads(body)),ensure_ascii=False).encode('utf-8')
        if 'text/html' in kind:
            body = augment_html(body)
        self.send(body,status,kind)

    def dispatch(self):
        raw = urlsplit(self.path).path
        path = unquote(raw)
        if path == '/healthz' and self.command in ('GET','HEAD'):
            return self.send({'ready':True,'service':'secure-display-gateway'})
        self.role = authenticate(self.headers.get('Authorization',''))
        if not self.role:
            return self.send({'error':'Sign in to view the platform.'},401,extra={'WWW-Authenticate':'Basic realm="Wing Atlas", charset="UTF-8"'})
        if '\\' in path or '%' in path or '//' in path or any(p in ('.','..') for p in path.split('/')):
            return self.deny('Invalid route.')
        if path == '/api/access-policy' and self.command in ('GET','HEAD'):
            return self.send({'role':self.role['role'],'can_download':self.role['can_download'],'preview_max_edge':MAX_EDGE,'local_management_url':'http://127.0.0.1:8882/'})
        if path == '/access-display.js' and self.command in ('GET','HEAD'):
            return self.send((HERE/'access-display.js').read_bytes(),kind='text/javascript; charset=utf-8')
        if path == '/favicon.ico' and self.command in ('GET','HEAD'):
            return self.send(b'',204,kind='image/x-icon')
        if self.command in ('GET','HEAD') and path == '/':
            return self.send(augment_html((FRONTENDS/'index.html').read_bytes()),kind='text/html; charset=utf-8')
        if self.command in ('GET','HEAD') and re.fullmatch(r'/(en|zh)(/(atlas|anatomy|predict|result))?',path):
            return self.send(b'',302,kind='text/plain',extra={'Location':self.path.replace(path,path+'/',1)})
        frontend=frontend_target(path)
        if self.command in ('GET','HEAD') and frontend:
            language,target=frontend
            suffix=target.suffix.lower()
            if suffix=='.html':
                return self.send(augment_html(target.read_bytes(),language),kind='text/html; charset=utf-8',extra={'X-Atlas-Language':language})
            if suffix in ('.css','.js'):
                return self.send(target.read_bytes(),kind='text/css; charset=utf-8' if suffix=='.css' else 'text/javascript; charset=utf-8',extra={'X-Atlas-Language':language})
            if suffix=='.png':
                return self.send(target.read_bytes() if self.role['can_download'] else make_preview(target),kind='image/png',extra={'X-Atlas-Asset':'original' if self.role['can_download'] else 'display-preview','X-Atlas-Language':language})
            return self.deny()
        if self.command in ('GET','HEAD') and re.fullmatch(r'/(atlas|anatomy|predict|result)/?',path):
            return self.send(b'',302,kind='text/plain',extra={'Location':'/en'+self.path})
        if self.command == 'POST':
            if path not in ('/api/predict','/api/re-align'):
                return self.deny('This entry is read-only except prediction.',405)
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                return self.deny('Cross-origin writes are not allowed.')
            length = int(self.headers.get('Content-Length',0))
            if not 0 < length <= 64*1024*1024:
                return self.deny('Upload limit: 64 MB.',413)
            body = self.rfile.read(length)
            if not self.role['can_download'] and path == '/api/re-align':
                run_id = str(json.loads(body).get('run_id',''))
                if viewer_runs().get(run_id) != self.role['username']:
                    return self.deny('Prediction record is not available to this account.')
            status,headers,response = request_backend(path,'POST',body,{'Content-Type':self.headers.get('Content-Type','application/json')})
            if status == 200:
                value = json.loads(response)
                if value.get('run_id'):
                    add_run(value['run_id'],self.role['username'])
                if not self.role['can_download']:
                    response = json.dumps(clean_paths(value),ensure_ascii=False).encode('utf-8')
            return self.send(response,status,headers.get('Content-Type','application/json'))
        if self.command not in ('GET','HEAD'):
            return self.deny('Method not allowed.',405)
        if self.role['can_download']:
            return self.proxy(self.path)
        if path == '/api/atlas/catalog':
            return self.proxy(path,transform=viewer_catalog)
        if path.startswith('/preview/'):
            match = re.fullmatch(r'/preview/([a-f0-9]{40})\.png',path)
            with STATE_LOCK:
                target = REGISTERED.get(match[1]) if match else None
            if not target:
                return self.deny('Preview expired. Reload the workspace.',404)
            return self.send(make_preview(target),kind='image/png',extra={'X-Atlas-Asset':'display-preview','Content-Disposition':'inline; filename="display-preview.png"'})
        if path.startswith('/api/file/'):
            match = re.fullmatch(r'/api/file/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)',path)
            if not match or match[2] not in PREVIEW_ASSETS or viewer_runs().get(match[1]) != self.role['username']:
                return self.deny('Prediction image is not available to this account.')
            target = Path(os.environ.get('ATLAS_STATE', '/mnt/workspace/wing-atlas-state'))/'prediction'/'uploads'/match[1]/'previews'/match[2]
            if not target.is_file():
                return self.deny('Preview not found.',404)
            return self.send(make_preview(target),kind='image/png',extra={'X-Atlas-Asset':'display-preview'})
        if path.startswith('/api/result/'):
            run_id = path[len('/api/result/'):]
            if viewer_runs().get(run_id) != self.role['username']:
                return self.deny('Prediction record is not available to this account.')
            return self.proxy(path,transform=clean_paths)
        if path == '/api/history':
            runs = viewer_runs()
            return self.proxy(path,transform=lambda d:{'items':[item for item in d['items'] if runs.get(item['run_id']) == self.role['username']]})
        if path in ('/api/health','/api/model-info'):
            return self.proxy(path,transform=clean_paths)
        pages = ('/','/atlas','/atlas/','/anatomy','/anatomy/','/predict','/predict/','/result','/result/')
        if path in pages:
            return self.proxy(self.path)
        if path.startswith(('/atlas/','/predict/','/portal/')) and not path.startswith('/atlas/data/'):
            suffix = Path(path).suffix.lower()
            if suffix in ('.css','.js','.html','.woff','.woff2','.ico'):
                return self.proxy(self.path)
            if suffix == '.png':
                return self.send(make_preview(local_asset(path)),kind='image/png',extra={'X-Atlas-Asset':'display-preview'})
        return self.deny()

    def handle_request(self):
        try:
            self.dispatch()
        except (BrokenPipeError,ConnectionResetError):
            pass
        except Exception as exc:
            LOG.error('Request failed path=%s type=%s',urlsplit(self.path).path[:300],type(exc).__name__)
            self.send({'error':'The request could not be completed.'},502)

    do_GET = handle_request
    do_HEAD = handle_request
    do_POST = handle_request
    do_PUT = handle_request
    do_DELETE = handle_request
    do_OPTIONS = handle_request

if __name__ == '__main__':
    initialize()
    server = ThreadingHTTPServer(('127.0.0.1',8880),Handler)
    server.daemon_threads = True
    print('Secure local display: http://127.0.0.1:8880/',flush=True)
    server.serve_forever()
