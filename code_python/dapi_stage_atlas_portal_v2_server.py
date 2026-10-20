"""Isolated atlas workbench; original portal code and atlas data remain read-only."""
from __future__ import annotations

import argparse
import csv
import importlib
import hashlib
import json
import mimetypes
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT/'code_python/dapi_stage_atlas_portal_v2'
LEGACY = ROOT/'code_python/stage_atlas_viewer_app'
DEFAULT_ATLAS = ROOT/'output_results/gene_atlas_sop_run/seven_genes_20261007_complete_v2'
OUTPUT = ROOT/'output_results/dapi_stage_atlas_portal_v2'
ANATOMY = OUTPUT/'vein_assets_v1'
VARIANTS = ('baseline', 'early_soft_q20', 'bilateral_soft_q20')


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=256)
def frame_hash(path, mtime_ns, size):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def asset(path, dataset='main'):
    return '/atlas-assets/'+dataset+'/'+quote(str(path).replace('\\', '/'), safe='/')


def catalog(root, dataset='main'):
    url = lambda path: asset(path, dataset)
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    validation = json.loads((root/'validation.json').read_text(encoding='utf-8'))
    if not validation.get('passed'):
        raise ValueError('The atlas archive has not passed integrity validation.')
    metrics = {(r['group'], r['variant']): r for r in rows(root/'metrics.csv')}
    members = rows(root/'accepted_manifest.csv')
    intensity = {r['id']: r for r in rows(root/'intensity_source_archive.csv')}
    groups = []
    for info in rows(root/'group_provenance.csv'):
        group = info['group']
        wing, stage = group.split('__')[1:]
        standard = root/group/'standard_dapi.png'
        stat = standard.stat()
        frame = frame_hash(str(standard), stat.st_mtime_ns, stat.st_size)
        standard_mask = root/group/'standard_mask.png'
        stat = standard_mask.stat()
        frame += frame_hash(str(standard_mask), stat.st_mtime_ns, stat.st_size)
        item = dict(id=group, gene=info['gene'], wing=wing,
            stage=stage.replace('stage_', '').replace('_', '.'), n=int(info['n']),
            origin=info['origin'], status=info['registration_status'], protocol=info['protocol'], coordinate_frame=frame,
            dapi=url(f'{group}/standard_dapi.png'), mask=url(f'{group}/standard_mask.png'),
            comparison=url(f'{group}/comparison.png'), variants={}, sources=[])
        for v in VARIANTS:
            m = metrics[group, v]
            images = ('heatmap', 'heatmap_dapi', 'heatmap_unclipped', 'expression_layer',
                'points_layer', 'points_dapi', 'intensity_gray', 'intensity_16bit')
            arrays = ('expression_float.npy', 'between_sample_sd.npy', 'coverage_count.npy',
                'missing_support.npy', 'outside_reference_conflict.npy', 'points.npz')
            item['variants'][v] = dict(
                images={k: url(f'{group}/{v}/{k}.png') for k in images},
                files={k: url(f'{group}/{v}/{k}') for k in arrays},
                metrics={k: float(m[k]) for k in ('strong_retention', 'weak_retention',
                    'outside_reference_fraction', 'target_coverage_fraction')})
        for r in [r for r in members if r['group'] == group]:
            src = intensity[r['id']]
            item['sources'].append(dict(id=r['id'], label=r['filename'],
                aligned=url(f'{group}/inputs/{r["id"]}.png'),
                intensity=url(src['archived']), stage=src['processing_stage'],
                tre=float(r['mean_tre_px_2048']) if r.get('mean_tre_px_2048') else None,
                method=r.get('actual_method') or r.get('method', 'Historical B/C')))
        groups.append(item)
    return dict(version='stage_atlas_portal_v2', genes=summary['genes'], groups=groups,
        samples=summary['samples'], group_count=len(groups), variant_count=summary['variants'],
        variants=list(VARIANTS), report=url('REPORT_ZH.md'), experiments=url('EXPERIMENTS_ZH.md'),
        manifest=url('accepted_manifest.csv'), coverage=url('gene_group_coverage.csv'),
        validated=True, units='relative fluorescence, not molecular counts')


class App:
    def __init__(self, atlas_root, output_dir):
        self.atlas = atlas_root.resolve()
        self.output = output_dir.resolve()
        self.atlases = {'main': self.atlas}
        self.data = self.refresh_catalog()
        self.engine = None
        self.engine_lock = threading.Lock()
        self.inference_lock = threading.Lock()

    def refresh_catalog(self):
        config = json.loads((UI/'atlas_sources.json').read_text(encoding='utf-8'))
        roots = {'main': self.atlas}
        for item in config.get('additional_archives', []):
            key = item['key']
            if not key.isalnum() or key in roots:
                raise ValueError('Archive keys must be unique alphanumeric identifiers.')
            path = Path(item['path'])
            roots[key] = (ROOT/path if not path.is_absolute() else path).resolve()
        loaded = [catalog(path, key) for key, path in roots.items()]
        hidden = set(config.get('hidden_genes', []))
        self.hidden_genes = hidden
        # Later explicit archive entries supersede whole groups, never mix samples/fields.
        groups = {g['id']: dict(g, dataset=key) for key, payload in zip(roots, loaded)
                  for g in payload['groups'] if g['gene'] not in hidden}
        registry = json.loads((ROOT/'output_results/gene_full_quality_audit_v1/aligned_color_manifest.json').read_text(encoding='utf-8'))
        names = sorted(({r['gene'] for r in registry if isinstance(r, dict) and r.get('gene')}
                        | {g['gene'] for g in groups.values()}) - hidden)
        inventory = []
        for gene in names:
            for wing in ('forewing', 'hindwing'):
                for stage in ('0.5', '0.75', '1.0'):
                    group = f'{gene}__{wing}__stage_{stage.replace(".", "_")}'
                    item = groups.get(group)
                    inventory.append(dict(gene=gene, wing=wing, stage=stage, available=item is not None,
                        n=item['n'] if item else 0, id=group))
        payload = dict(loaded[0], genes=names, processed_genes=sorted({g['gene'] for g in groups.values()}),
            groups=list(groups.values()), inventory=inventory, samples=sum(g['n'] for g in groups.values()),
            group_count=len(groups), variant_count=len(groups)*3,
            archive_reports={key: dict(report=p['report'], experiments=p['experiments'], manifest=p['manifest']) for key,p in zip(roots,loaded)})
        anatomy = []
        if (ANATOMY/'validation.json').is_file():
            if not json.loads((ANATOMY/'validation.json').read_text(encoding='utf-8')).get('passed'):
                raise ValueError('Final vein import has not passed validation.')
            structure = json.loads((ANATOMY/'catalog.json').read_text(encoding='utf-8'))
            for case in structure['cases']:
                anatomy.append(dict(case, assets={key: '/anatomy-assets/'+quote(value, safe='/')
                    for key, value in case['assets'].items()},
                    rendering_comparison='/anatomy-assets/'+case['path']+'/rendering_comparison.png'))
        payload['anatomy_cases'] = anatomy
        payload['anatomy_validated'] = len(anatomy) == 6
        payload['display_comparison_url'] = '/anatomy-assets/display_comparison.html'
        self.atlases = roots
        self.data = payload
        return payload

    def predictor(self):
        with self.engine_lock:
            if self.engine is None:
                module = importlib.import_module('dapi_stage_prediction_server')
                module.APP_DIR = UI/'predict'
                module.PORTAL_DIR = UI/'portal'
                state = module.StagePredictionState(self.output)
                self.engine = module.make_handler(state)
            return self.engine


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send_json(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def file(self, root, relative):
            root = root.resolve()
            path = (root/relative).resolve()
            if not path.is_relative_to(root):
                self.send_error(403)
                return
            if not path.is_file():
                self.send_error(404)
                return
            self.send_response(200)
            kind = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
            if kind in ('text/html', 'text/css', 'application/javascript', 'text/javascript'):
                kind += '; charset=utf-8'
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(path.stat().st_size))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-cache' if path.suffix in ('.html', '.js', '.css') else 'private, max-age=3600')
            self.end_headers()
            try:
                with path.open('rb') as f:
                    shutil.copyfileobj(f, self.wfile, 65536)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def serve_static_file(self, root, relative):
            self.file(root, relative.lstrip('/'))

        def legacy_file(self, relative):
            # Filter only the served metadata; original legacy data stay untouched.
            if relative.startswith('data/') and relative.endswith('/manifest.json'):
                path = (LEGACY/relative).resolve()
                if not path.is_relative_to(LEGACY.resolve()):
                    self.send_error(403)
                    return
                if not path.is_file():
                    self.send_error(404)
                    return
                data = json.loads(path.read_text(encoding='utf-8'))
                layers = data.get('layers', {})
                layers['genes'] = {key: value for key, value in layers.get('genes', {}).items()
                                   if key not in app.hidden_genes}
                data['has_genes'] = bool(layers['genes'])
                self.send_json(data)
            else:
                self.file(LEGACY, relative)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == '/api/health':
                self.send_json(dict(app='stage_atlas_portal_v2', ready=True,
                    predictor_loaded=app.engine is not None, original_port=8820))
            elif path == '/api/atlas/catalog':
                try:
                    self.send_json(app.refresh_catalog())
                except Exception as exc:
                    self.send_json(dict(error=str(exc)), 500)
            elif path.startswith('/atlas-assets/'):
                parts = path.removeprefix('/atlas-assets/').split('/', 1)
                if len(parts) != 2 or parts[0] not in app.atlases:
                    self.send_error(404)
                else:
                    self.file(app.atlases[parts[0]], parts[1])
            elif path in ('/', '/atlas', '/atlas/'):
                self.file(UI/'atlas' if path != '/' else UI/'portal', 'index.html')
            elif path in ('/anatomy', '/anatomy/'):
                self.file(UI/'atlas', 'anatomy.html')
            elif path.startswith('/anatomy-assets/'):
                self.file(ANATOMY, path.removeprefix('/anatomy-assets/'))
            elif path.startswith('/atlas/data/'):
                self.legacy_file(path.removeprefix('/atlas/'))
            elif path.startswith('/atlas/'):
                self.file(UI/'atlas', path.removeprefix('/atlas/'))
            elif path in ('/legacy-atlas', '/legacy-atlas/'):
                self.file(LEGACY, 'index.html')
            elif path.startswith('/legacy-atlas/'):
                self.legacy_file(path.removeprefix('/legacy-atlas/'))
            elif path.startswith('/portal/'):
                self.file(UI/'portal', path.removeprefix('/portal/'))
            elif path in ('/predict', '/predict/'):
                self.file(UI/'predict', 'index.html')
            elif path.startswith('/predict/'):
                self.file(UI/'predict', path.removeprefix('/predict/'))
            elif path in ('/result', '/result/'):
                self.file(UI/'portal', 'result.html')
            elif path.startswith('/api/'):
                try:
                    app.predictor().do_GET(self)
                except Exception as exc:
                    self.send_json(dict(error=str(exc)), 500)
            else:
                self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path
            if path not in ('/api/predict', '/api/re-align'):
                self.send_json(dict(error='This atlas interface is read-only.'), 405)
                return
            if int(self.headers.get('Content-Length', 0)) > 64*1024*1024:
                self.send_json(dict(error='Upload limit: 64 MB.'), 413)
                return
            if not app.inference_lock.acquire(blocking=False):
                self.send_json(dict(error='Another prediction is running. Please wait.'), 409)
                return
            try:
                app.predictor().do_POST(self)
            except Exception as exc:
                self.send_json(dict(error=str(exc)), 500)
            finally:
                app.inference_lock.release()
    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8870)
    p.add_argument('--atlas-dir', type=Path, default=DEFAULT_ATLAS)
    p.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = p.parse_args()
    app = App(args.atlas_dir, args.output_dir)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(app))
    server.daemon_threads = True
    print(f'Independent atlas portal: http://127.0.0.1:{args.port}/', flush=True)
    print('Atlas browsing does not load classification models.', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
