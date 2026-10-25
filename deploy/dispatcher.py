"""Single-port browser and API adapter using the canonical public frontend."""
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from http.cookies import SimpleCookie


class Handler(BaseHTTPRequestHandler):
    server_version = 'WingAtlasCloud'

    def log_message(self, *_):
        pass

    def json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def dispatch(self):
        path = urlsplit(self.path).path
        if path == '/healthz' and self.command in ('GET', 'HEAD'):
            return self.json({'ready': True, 'service': 'cloud-prediction-origin',
                              'languages': ['zh', 'en'], 'downloads': 'password-required'})
        if path == '/' and self.command in ('GET', 'HEAD'):
            self.send_response(302)
            self.send_header('Location', '/en/')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        language = next((key for key in ('zh', 'en') if path.startswith('/' + key + '/')), None)
        if language is None:
            # Browser APIs use root-relative URLs. Same-host Referer determines
            # the edition; the cookie is a fallback for direct API clients.
            referer = urlsplit(self.headers.get('Referer', ''))
            language = next((key for key in ('zh', 'en')
                             if referer.netloc == self.headers.get('Host')
                             and referer.path.startswith('/' + key + '/')), None)
            if language is None:
                cookie = SimpleCookie()
                try:
                    cookie.load(self.headers.get('Cookie', ''))
                except Exception:
                    pass
                stored = cookie.get('atlas_language')
                language = stored.value if stored and stored.value in ('zh', 'en') else 'en'
        if self.command not in ('GET', 'HEAD', 'POST'):
            return self.json({'error': 'Method not allowed.'}, 405)
        if path.startswith('/' + language + '/_'):
            return self.json({'error': 'Unknown route.'}, 404)
        relative = self.path[len('/' + language):] if path.startswith('/' + language + '/') else self.path
        port = 8890 if language == 'zh' else 8891
        origin = 'http://127.0.0.1:' + str(port)
        headers = {key: value for key, value in self.headers.items()
                   if key.lower() in ('content-type', 'accept', 'cookie', 'range')}
        # ModelScope reserves Authorization. Pages forwards the user's Basic header
        # as X-Atlas-Download-Authorization; the validated private gateway checks it.
        authorization = self.headers.get('X-Atlas-Download-Authorization', '')
        direct_authorization = self.headers.get('Authorization', '')
        if not authorization and direct_authorization.startswith('Basic '):
            # Direct local Docker browsers send Basic authentication themselves;
            # the ModelScope service's Bearer credential must never be forwarded.
            authorization = direct_authorization
        if authorization:
            headers['Authorization'] = authorization
        request_origin = self.headers.get('Origin')
        if request_origin and urlsplit(request_origin).netloc != self.headers.get('Host'):
            return self.json({'error': 'Cross-origin writes are not allowed.'}, 403)
        if request_origin:
            headers['Origin'] = origin
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            return self.json({'error': 'Invalid request length.'}, 400)
        if length < 0 or length > 64 * 1024 * 1024:
            return self.json({'error': 'Upload limit: 64 MB.'}, 413)
        data = self.rfile.read(length) if self.command == 'POST' else None
        request = urllib.request.Request(origin + relative, method=self.command, data=data, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=240)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            body = response.read()
            self.send_response(response.status)
            if 'text/html' in response.headers.get('Content-Type', ''):
                self.send_header('Set-Cookie', 'atlas_language=' + language + '; Path=/; SameSite=Lax')
            for key, value in response.headers.items():
                if key.lower() not in ('transfer-encoding', 'connection', 'server', 'date', 'content-length'):
                    self.send_header(key, value)
            self.send_header('Content-Length', response.headers.get('Content-Length', '0')
                             if self.command == 'HEAD' else str(len(body)))
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)

    def handle_request(self):
        try:
            self.dispatch()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            print('Origin request failed: ' + type(error).__name__, flush=True)
            self.json({'error': 'Prediction service temporarily unavailable.'}, 503)

    do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = do_OPTIONS = handle_request


if __name__ == '__main__':
    server = ThreadingHTTPServer(('0.0.0.0', int(os.environ.get('PORT', '7860'))), Handler)
    server.daemon_threads = True
    server.serve_forever()
