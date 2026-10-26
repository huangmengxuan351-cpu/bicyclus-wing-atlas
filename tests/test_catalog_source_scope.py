"""A source reused in two historical stage groups keeps the correct group route."""
from pathlib import Path
import csv
import importlib.util
import json
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('catalog_scope_test',ROOT/'code_python/dapi_stage_atlas_portal_v2_server.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class CatalogSourceScope(unittest.TestCase):
    def test_shared_source_is_resolved_within_its_own_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def table(name,values):
                with (root/name).open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
            (root/'summary.json').write_text(json.dumps(dict(genes=['hh'],samples=2,variants=6)))
            (root/'validation.json').write_text(json.dumps(dict(passed=True)))
            groups=['hh__forewing__stage_0_75','hh__forewing__stage_1_0']
            for group in groups:
                folder=root/group;folder.mkdir()
                # The catalog hashes these opaque reference bytes; no image decoder is involved.
                (folder/'standard_dapi.png').write_bytes(b'coordinate-reference')
                (folder/'standard_mask.png').write_bytes(b'coordinate-mask')
            table('group_provenance.csv',[dict(group=g,gene='hh',n=1,origin='legacy',registration_status='legacy',protocol='frozen') for g in groups])
            table('accepted_manifest.csv',[dict(group=g,id='shared-source',filename='same.png',method='B_robust_affine') for g in groups])
            table('intensity_source_archive.csv',[dict(group=g,id='shared-source',archived=g+'/intensity_source_colors/shared.png',processing_stage='source-export') for g in groups])
            table('metrics.csv',[dict(group=g,variant=v,strong_retention=1,weak_retention=1,outside_reference_fraction=0,target_coverage_fraction=1) for g in groups for v in module.VARIANTS])
            payload=module.catalog(root)
            for group in payload['groups']:
                self.assertEqual(group['sources'][0]['intensity'],'/atlas-assets/main/'+group['id']+'/intensity_source_colors/shared.png')

if __name__=='__main__':unittest.main()
