import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';

for (const language of ['en','zh']) {
  const base=new URL(`../public/${language}/`,import.meta.url);
  const catalog=JSON.parse(await readFile(new URL('data/catalog.json',base),'utf8'));
  test(`${language}: final HH membership, withdrawal and totals agree`,()=>{
    assert.deepEqual([catalog.samples,catalog.group_count,catalog.variant_count],[284,71,213]);
    const group=stage=>catalog.groups.find(g=>g.id===`hh__forewing__stage_${stage}`);
    assert.deepEqual(group('0_5').sources.map(s=>s.id).sort(),['7f4ca7b961b905233d62','99d2441a4ae7213dc9a6']);
    assert.deepEqual(group('0_75').sources.map(s=>s.id).sort(),['3204f827b7da06dd2e3c','b4798cb44433ef034bf7']);
    assert.equal(group('1_0'),undefined);
    assert.equal(catalog.inventory.find(g=>g.id==='hh__forewing__stage_1_0').available,false);
    assert.equal(catalog.withdrawn_groups.hh__forewing__stage_1_0.reason,'stage_assignment_conflict');
    assert.equal(group('0_75').point_display.sampling,'area_uniform');
    assert.deepEqual(group('0_75').point_display.threshold_percentiles,[40,50]);
    assert.equal(group('0_5').point_display.sampling,'intensity_proportional');
  });
  test(`${language}: updated previews are complete, content-addressed and served without the private origin`,async()=>{
    const index=JSON.parse(await readFile(new URL('data/preview-index.json',base),'utf8'));
    const source=await readFile(new URL('_worker.js',base),'utf8');
    const worker=(await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'))).default;
    for(const [route,entry] of Object.entries(index).filter(([,v])=>!Array.isArray(v))) {
      const bytes=await readFile(new URL(entry.asset.slice(1),base));
      assert.equal(createHash('sha256').update(bytes).digest('hex'),entry.sha256);
      assert.ok(bytes.readUInt32BE(16)<=960 && bytes.readUInt32BE(20)<=960);
      const env={ASSETS:{fetch:async req=>new URL(req.url).pathname==='/data/preview-index.json'?Response.json(index):new Response(bytes,{headers:{'Content-Type':'image/png'}})}};
      const response=await worker.fetch(new Request('https://example.test'+route),env);
      assert.equal(response.status,200);assert.equal(response.headers.get('X-Atlas-Asset'),'display-preview');
      assert.deepEqual(Buffer.from(await response.arrayBuffer()),bytes);
    }
  });
}
