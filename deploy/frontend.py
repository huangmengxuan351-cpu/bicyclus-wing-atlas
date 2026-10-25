"""Resolve the one editable frontend tree used by Pages and local gateways.

Only public display assets are eligible; private scientific files and Worker
source are outside this resolver's allowlist.
"""
from pathlib import Path
import mimetypes

ROOT = Path(__file__).resolve().parents[1]


def resolve_frontend(language, path):
    if language not in ('en', 'zh'):
        return None
    if '\\' in path or '%' in path or any(p in ('.', '..') for p in path.split('/')):
        return None
    prefix = '/' + language
    if path.startswith(('/en/', '/zh/')):
        if not path.startswith(prefix + '/'):
            return None
        path = path[len(prefix):]
    routes = {'/': 'portal/index.html', '/atlas/': 'atlas/index.html',
              '/anatomy/': 'atlas/anatomy.html', '/predict/': 'predict/index.html',
              '/result/': 'portal/result.html'}
    folder = (ROOT / 'public' / language).resolve()
    if path in ('/manual', '/manual/', '/docs/manual.html'):
        target = folder / 'docs/manual.html'
    else:
        relative = routes.get(path, path.lstrip('/'))
        target = folder / language / relative
    target = target.resolve()
    if not target.is_relative_to(folder) or not target.is_file():
        return None
    if target.suffix.lower() not in ('.html', '.js', '.css', '.png', '.svg', '.ico'):
        return None
    kind = {'.js': 'text/javascript', '.html': 'text/html', '.css': 'text/css'}.get(
        target.suffix.lower(), mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
    if target.suffix.lower() in ('.html', '.js', '.css', '.svg'):
        kind += '; charset=utf-8'
    return target, kind
