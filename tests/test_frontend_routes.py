"""Integration checks for the shared public UI and browser/API dispatcher."""
from pathlib import Path
import http.client
import importlib.util
import io
import json
import sys
import threading
import unittest
from email.message import Message
from http.server import ThreadingHTTPServer
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))
from frontend import resolve_frontend
import dispatcher

spec = importlib.util.spec_from_file_location('atlas_public_gateway_test', ROOT /
    'output_results/platform_public_bilingual_20261019/public_gateway.py')
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


class HTTPTest(unittest.TestCase):
    def serve(self, handler):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def request(self, port, path, method='GET', headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def test_public_gateway_reads_current_pages_frontend_for_both_languages(self):
        port = self.serve(gateway.Handler)
        for language in ('en', 'zh'):
            with patch.object(gateway, 'LANGUAGE', language):
                for path in ('/', '/' + language + '/atlas/', '/' + language + '/anatomy/',
                             '/' + language + '/predict/', '/' + language + '/result/',
                             '/' + language + '/atlas/workspaces.js', '/manual/'):
                    status, _, data = self.request(port, path)
                    target, _ = resolve_frontend(language, path)
                    self.assertEqual(status, 200, path)
                    self.assertEqual(data, target.read_bytes(), path)

    def test_originals_and_private_paths_stay_protected(self):
        port = self.serve(gateway.Handler)
        with patch.object(gateway, 'authenticate', return_value=None):
            for path in ('/downloads/', '/atlas-assets/probe.png', '/anatomy-assets/probe.png'):
                self.assertEqual(self.request(port, path)[0], 401)
        for path in ('/_worker.js', '/en/_worker.js', '/output_results/file',
                     '/en/../../README.md', '/en/data/catalog.json'):
            self.assertGreaterEqual(self.request(port, path)[0], 400, path)

    def test_manual_is_anonymous_and_english_has_no_chinese_text(self):
        manual = resolve_frontend('en', '/manual/')[0].read_text(encoding='utf-8')
        self.assertNotRegex(manual, r'[\u4e00-\u9fff]')
        self.assertIn('Educational approximation', manual)
        self.assertIn('visual guidance only', manual)
        self.assertIn('中文', resolve_frontend('zh', '/manual/')[0].read_text(encoding='utf-8'))
        self.assertIsNone(resolve_frontend('en', '/zh/atlas/'))

    def test_dispatcher_browsing_root_and_edition_forwarding(self):
        calls = []
        def response(request, timeout):
            calls.append(request)
            value = io.BytesIO(b'<html>canonical frontend</html>')
            value.status = 200
            value.headers = Message()
            value.headers['Content-Type'] = 'text/html; charset=utf-8'
            return value
        port = self.serve(dispatcher.Handler)
        status, headers, _ = self.request(port, '/')
        self.assertEqual((status, headers['Location']), (302, '/en/'))
        with patch.object(dispatcher.urllib.request, 'urlopen', side_effect=response):
            for language, upstream_port in [('en', 8891), ('zh', 8890)]:
                status, headers, _ = self.request(port, '/' + language + '/atlas/')
                self.assertEqual(status, 200)
                self.assertEqual(calls[-1].full_url, f'http://127.0.0.1:{upstream_port}/atlas/')
                self.assertIn('atlas_language=' + language, str(headers))

    def test_dispatcher_root_api_uses_same_host_edition_referer(self):
        calls=[]
        def response(request, timeout):
            calls.append(request)
            value=io.BytesIO(b'{}'); value.status=200; value.headers=Message()
            value.headers['Content-Type']='application/json'
            return value
        port = self.serve(dispatcher.Handler)
        with patch.object(dispatcher.urllib.request, 'urlopen', side_effect=response):
            self.request(port, '/api/history', headers={'Referer':f'http://127.0.0.1:{port}/zh/predict/'})
            self.assertEqual(calls[-1].full_url, 'http://127.0.0.1:8890/api/history')
            self.request(port, '/api/history', headers={'Cookie':'atlas_language=en'})
            self.assertEqual(calls[-1].full_url, 'http://127.0.0.1:8891/api/history')
            self.request(port, '/en/downloads/', headers={'Authorization':'Basic test-local-only'})
            self.assertEqual(calls[-1].get_header('Authorization'), 'Basic test-local-only')
            self.request(port, '/en/downloads/', headers={'Authorization':'Bearer service-test-only'})
            self.assertIsNone(calls[-1].get_header('Authorization'))
        self.assertEqual(self.request(port, '/api/predict', 'POST', {'Origin':'https://unrelated.example'})[0], 403)


if __name__ == '__main__':
    unittest.main()
