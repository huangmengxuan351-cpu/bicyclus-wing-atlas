import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';

for (const language of ['en','zh']) {
  const base=new URL(`../public/${language}/`,import.meta.url);
  const catalog=JSON.parse(await readFile(new URL('data/catalog.json',base),'utf8'));
  test(`${language}: final HH membership, withdrawal and totals agree`,()=>{
    assert.deepEqual([catalog.samples,catalog.group_count,catalog.variant_count],[285,72,216]);
    const group=stage=>catalog.groups.find(g=>g.id===`hh__forewing__stage_${stage}`);
    assert.deepEqual(group('0_5').sources.map(s=>s.id).sort(),['7f4ca7b961b905233d62','99d2441a4ae7213dc9a6']);
    assert.deepEqual(group('0_75').sources.map(s=>s.id).sort(),['3204f827b7da06dd2e3c','b4798cb44433ef034bf7']);
    assert.equal(group('1_0').n,4);
    assert.equal(group('1_0').selection_status,'legacy_retained_by_user');
    assert.equal(catalog.inventory.find(g=>g.id==='hh__forewing__stage_1_0').available,true);
    assert.equal(catalog.withdrawn_groups.hh__forewing__stage_1_0,undefined);
    assert.equal(group('0_75').point_display.sampling,'area_uniform');
    assert.deepEqual(group('0_75').point_display.threshold_percentiles,[40,50]);
    assert.equal(group('0_5').point_display.sampling,'intensity_proportional');
    const hind=stage=>catalog.groups.find(g=>g.id===`hh__hindwing__stage_${stage}`);
    assert.deepEqual(hind('0_5').sources.map(s=>s.id).sort(),['a00934e3c2aabeef0ef1','e96bc4d93e5bd4df65c8','3a156cd46a22910734cc'].sort());
    assert.deepEqual(hind('0_75').sources.map(s=>s.id).sort(),['e97bb5c210c8a9842335','5e3c6940da0d810e8246'].sort());
    for(const stage of ['0_5','0_75']) {
      assert.equal(hind(stage).selection_status,'user_selected_final_composite');
      assert.equal(hind(stage).point_display.sampling,'intensity_proportional');
      assert.deepEqual(hind(stage).point_display.threshold_percentiles,[50,85]);
      assert.equal(hind(stage).n,hind(stage).sources.length);
    }
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
