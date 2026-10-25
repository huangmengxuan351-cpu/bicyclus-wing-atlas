"""One anonymous, single-language public edition; password-gated original exports.

All scientific work is delegated to the already validated local secure platform.
No source data directory is mounted as a public static directory.
"""
from __future__ import annotations
import os
import argparse
import base64
import hashlib
import html
import json
import logging
import re
import secrets
import sqlite3
import sys
import urllib.error
import urllib.request
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1] / 'deploy'))
from frontend import resolve_frontend
PRIVATE = BASE.parent / 'platform_local_secure_20261019'
sys.path.insert(0, str(PRIVATE))
from access_control import authenticate, credentials

LOCAL = 'http://127.0.0.1:8880'
LANGUAGE = 'en'
INSTANCE = Path(os.environ.get('ATLAS_STATE', '/mnt/workspace/wing-atlas-state'))/'visitors/en'
MAX_UPLOAD = 64 * 1024 * 1024
SESSION_COOKIE = 'wing_atlas_visitor'
DOWNLOAD_EXTENSIONS = frozenset(('.png','.npy','.npz','.csv','.json','.txt','.md','.tif','.tiff','.zip'))
LOG = logging.getLogger('public')


def basic(role='viewer'):
    user, password = credentials(role)
    return 'Basic ' + base64.b64encode((user + ':' + password).encode()).decode()


def upstream(route, method='GET', data=None, headers=None, export=False):
    forwarded = {k:v for k,v in (headers or {}).items() if k.lower() in ('content-type','accept','range')}
    forwarded['Authorization'] = basic('exporter' if export else 'viewer')
    if method == 'POST':
        forwarded['Origin'] = LOCAL
    req = urllib.request.Request(LOCAL + route, method=method, data=data, headers=forwarded)
    try:
        response = urllib.request.urlopen(req, timeout=240)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, dict(response.headers), response.read()


def record_run(visitor, run_id):
    with sqlite3.connect(INSTANCE/'visitors.sqlite3', timeout=20) as db:
        db.execute('INSERT OR REPLACE INTO runs(run_id, visitor) VALUES (?,?)', (run_id, visitor))


def owns_run(visitor, run_id):
    with sqlite3.connect(INSTANCE/'visitors.sqlite3', timeout=20) as db:
        row = db.execute('SELECT visitor FROM runs WHERE run_id=?', (run_id,)).fetchone()
        return row is not None and secrets.compare_digest(row[0], visitor)


def labels():
    if LANGUAGE == 'zh':
        return {'download':'下载原始图像与数据（需要密码）', 'title':'原始数据下载',
                'intro':'展示和预测免费公开访问。此页的原图与原始数据需要下载账号及密码。',
                'back':'返回公开平台', 'gene':'基因', 'wing':'翅型', 'stage':'阶段',
                'files':'文件', 'note':'图像与数组保持原始文件内容；展示页使用降采样预览。',
                'blocked':'下载原始图像与数据需要下载密码。', 'disabled':'下载授权当前已关闭，请联系平台维护者。'}
    return {'download':'Download original images and data (password required)', 'title':'Original data downloads',
            'intro':'Browsing and prediction are public. Original images and data on this page require a download account and password.',
            'back':'Back to the public platform', 'gene':'Gene', 'wing':'Wing', 'stage':'Stage',
            'files':'Files', 'note':'Original image and array files are preserved. Display pages use reduced-resolution previews.',
            'blocked':'A download password is required for original images and data.',
            'disabled':'Download authorization is currently disabled. Contact the platform owner.'}


def download_paths(catalog):
    """Construct an exact allowlist from the current catalog, never arbitrary paths."""
    found = set()
    def visit(value):
        if isinstance(value,dict):
            for item in value.values(): visit(item)
        elif isinstance(value,list):
            for item in value: visit(item)
        elif isinstance(value,str) and value.startswith(('/atlas-assets/','/anatomy-assets/')):
            path = urlsplit(value).path
            if Path(path).suffix.lower() in DOWNLOAD_EXTENSIONS:
                found.add(path)
    visit(catalog)
    return found


def download_index(catalog):
    lab = labels()
    esc = html.escape
    def link(path, name):
        return f'<a href="/downloads{esc(path,quote=True)}" download>{esc(name)}</a>'
    rows = []
    for group in catalog['groups']:
        links = [link(group[k], k.upper()) for k in ('dapi','mask','comparison')]
        for method, variant in group['variants'].items():
            links.extend(link(value, method+' / '+name) for name,value in variant['images'].items())
            links.extend(link(value, method+' / '+name) for name,value in variant['files'].items())
        for i, source in enumerate(group['sources'],1):
            links.extend(link(source[k], f'{i} / {k}') for k in ('aligned','intensity'))
        wing = group.get('wing',group.get('wing_type',''))
        if LANGUAGE == 'zh': wing = {'forewing':'前翅','hindwing':'后翅'}.get(wing,wing)
        rows.append('<tr><td>'+esc(str(group.get('gene','')))+'<br>'+esc(str(wing))+' / '+esc(str(group.get('stage','')))
                    +'</td><td>'+' '.join(links)+'</td></tr>')
    anatomy=[]
    for case in catalog.get('anatomy_cases',[]):
        anatomy.extend(link(value, str(case.get('id','anatomy'))+' / '+name) for name,value in case['assets'].items()
                       if urlsplit(value).path in download_paths(catalog))
    title=esc(lab['title'])
    return f'''<!doctype html><html lang="{LANGUAGE}"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>{title}</title><style>body{{font:16px/1.65 system-ui;margin:35px auto;padding:0 24px;max-width:1200px;color:#183d36;background:#f6f5ee}}a{{color:#087569;margin-right:14px;display:inline-block}}td,th{{padding:15px;text-align:left;vertical-align:top;border-bottom:1px solid #cdd8d1}}td:first-child{{min-width:140px}}table{{width:100%}}</style>
    <a href="/">{esc(lab['back'])}</a><h1>{title}</h1><p>{esc(lab['intro'])}</p><p>{esc(lab['note'])}</p>
    <table><thead><tr><th>{esc(lab['gene'])} / {esc(lab['wing'])} / {esc(lab['stage'])}</th><th>{esc(lab['files'])}</th></tr></thead><tbody>{''.join(rows)}</tbody></table><p>{' '.join(anatomy)}</p></html>'''.encode()


class Handler(BaseHTTPRequestHandler):
    server_version = 'WingAtlasPublic'
    def log_message(self,*args): pass

    def send(self, data, status=200, kind='application/json; charset=utf-8', extra=None):
        if not isinstance(data,bytes): data=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(status)
        headers = {'Content-Type':kind, 'Content-Length':str(len(data)), 'Cache-Control':'no-store',
                   'X-Content-Type-Options':'nosniff', 'Referrer-Policy':'same-origin',
                   'X-Frame-Options':'SAMEORIGIN', 'Content-Security-Policy':"frame-ancestors 'self'",
                   'X-Atlas-Language':LANGUAGE, 'X-Atlas-Deployment':'independent-public-'+LANGUAGE}
        headers.update(extra or {})
        if getattr(self,'new_visitor',False):
            # Secure is correct for browser requests through HTTPS tunnels, and also
            # accepted by modern browsers for the loopback address used in QA.
            headers['Set-Cookie']=f'{SESSION_COOKIE}={self.visitor}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=2592000'
        for key,value in headers.items(): self.send_header(key,value)
        self.end_headers()
        if self.command!='HEAD':
            try: self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError): pass

    def deny_download(self):
        LOG.info('DOWNLOAD_AUTH_REQUIRED %s %s',self.command,urlsplit(self.path).path[:250])
        return self.send({'error':labels()['blocked'],'download_password_required':True},401,
                         extra={'WWW-Authenticate':f'Basic realm="Wing Atlas {LANGUAGE.upper()} downloads", charset="UTF-8"'})

    def visitor_session(self):
        cookie=SimpleCookie()
        try: cookie.load(self.headers.get('Cookie',''))
        except Exception: pass
        value=cookie.get(SESSION_COOKIE)
        self.visitor=value.value if value and re.fullmatch(r'[a-f0-9]{48}',value.value) else secrets.token_hex(24)
        self.new_visitor=not value or value.value!=self.visitor

    def proxy(self, route, method='GET', data=None, export=False):
        status, headers, body=upstream(route,method,data,self.headers,export)
        if status==200 and route=='/api/history':
            result=json.loads(body)
            result['items']=[item for item in result['items'] if owns_run(self.visitor,item['run_id'])]
            body=json.dumps(result,ensure_ascii=False).encode()
        if status==200 and method=='POST' and route=='/api/predict':
            result=json.loads(body)
            if result.get('run_id'): record_run(self.visitor,result['run_id'])
        extra={k:v for k,v in headers.items() if k.lower() in ('content-disposition','x-atlas-asset')}
        if export:
            extra['Content-Disposition']='attachment; filename="'+re.sub(r'[^A-Za-z0-9_.-]','_',Path(urlsplit(route).path).name)+'"'
            extra['X-Atlas-Asset']='authorized-original'
        return self.send(body,status,headers.get('Content-Type','application/octet-stream'),extra)

    def protected_download(self,path):
        role=authenticate(self.headers.get('Authorization',''))
        if not role or role['role']!='exporter': return self.deny_download()
        status,_,body=upstream('/api/atlas/catalog',export=True)
        if status!=200: return self.send({'error':labels()['disabled']},503)
        catalog=json.loads(body)
        if path in ('/downloads','/downloads/'):
            return self.send(download_index(catalog),kind='text/html; charset=utf-8')
        original=path[len('/downloads'):] if path.startswith('/downloads/') else path
        if original not in download_paths(catalog):
            return self.send({'error':'File is not in the published download catalog.'},403)
        return self.proxy(original,export=True)

    def dispatch(self):
        self.visitor_session()
        path=unquote(urlsplit(self.path).path)
        if '\\' in path or '%' in path or '//' in path or any(p in ('.','..') for p in path.split('/')):
            return self.send({'error':'Invalid route.'},403)
        if path=='/healthz' and self.command in ('GET','HEAD'):
            return self.send({'ready':True,'language':LANGUAGE,'access':'public','downloads':'password-required'})
        if self.command in ('GET','HEAD') and path.startswith(('/downloads','/atlas-assets/','/anatomy-assets/')):
            return self.protected_download(path)
        if self.command not in ('GET','HEAD','POST'):
            return self.send({'error':'Method not allowed.'},405)
        if self.command=='POST':
            if path not in ('/api/predict','/api/re-align'):
                return self.send({'error':'This route does not accept public writes.'},405)
            origin=self.headers.get('Origin')
            if origin and urlsplit(origin).netloc!=self.headers.get('Host'):
                return self.send({'error':'Cross-origin writes are not allowed.'},403)
            try: length=int(self.headers.get('Content-Length','0'))
            except ValueError: length=0
            if not 0<length<=MAX_UPLOAD: return self.send({'error':'Upload limit: 64 MB.'},413)
            body=self.rfile.read(length)
            if path=='/api/re-align':
                try: run_id=str(json.loads(body).get('run_id',''))
                except ValueError: return self.send({'error':'Invalid JSON.'},400)
                if not owns_run(self.visitor,run_id): return self.send({'error':'Prediction record unavailable.'},403)
            return self.proxy(path,'POST',body)
        if path=='/api/access-policy':
            return self.send({'role':'public','can_download':False,'download_password_required':True,
                              'download_entry':'/downloads/','preview_max_edge':960})
        if path in ('/api/atlas/catalog','/api/health','/api/model-info','/api/history') or re.fullmatch(r'/preview/[a-f0-9]{40}\.png',path):
            return self.proxy(self.path)
        if path.startswith(('/api/file/','/api/result/')):
            match=re.fullmatch(r'/api/(?:file|result)/([A-Za-z0-9_.-]+)(?:/([A-Za-z0-9_.-]+))?',path)
            if not match or not owns_run(self.visitor,match[1]):
                return self.send({'error':'Prediction record unavailable.'},403)
            return self.proxy(self.path)
        if path.startswith('/api/') or path.startswith(('/admin','/legacy-atlas','/runtime','/code_python','/output_results')):
            return self.send({'error':'Not a public route.'},403)
        if path=='/favicon.ico': return self.send(b'',204,'image/x-icon')
        if path in ('/atlas', '/anatomy', '/predict', '/result'):
            return self.send(b'',302,'text/plain',{'Location':'/'+LANGUAGE+path+'/'})
        resolved = resolve_frontend(LANGUAGE, path)
        if resolved is None:
            return self.send({'error':'Page not found in this language edition.'},404)
        target, kind = resolved
        return self.send(target.read_bytes(), kind=kind)

    def handle_request(self):
        try: self.dispatch()
        except (BrokenPipeError,ConnectionResetError): pass
        except Exception as exc:
            LOG.error('REQUEST_FAILED %s %s %s',self.command,urlsplit(self.path).path[:250],type(exc).__name__)
            self.send({'error':'Service temporarily unavailable.'},502)
    do_GET=do_HEAD=do_POST=do_PUT=do_DELETE=do_OPTIONS=handle_request


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--language',choices=('zh','en'),required=True)
    parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args()
    LANGUAGE=args.language
    INSTANCE=Path(os.environ.get('ATLAS_STATE', '/mnt/workspace/wing-atlas-state'))/'visitors'/LANGUAGE
    INSTANCE.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(INSTANCE/'visitors.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,visitor TEXT NOT NULL)')
    handler=RotatingFileHandler(INSTANCE/'access.log',maxBytes=2_000_000,backupCount=2,encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    LOG.addHandler(handler);LOG.setLevel(logging.INFO)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.daemon_threads=True
    print(f'Independent public {LANGUAGE} edition: http://127.0.0.1:{args.port}/',flush=True)
    server.serve_forever()
