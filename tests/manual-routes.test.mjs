import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

for(const language of ['en','zh']) {
  const source=await readFile(new URL(`../public/${language}/_worker.js`,import.meta.url),'utf8');
  const worker=(await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'))).default;
  test(`${language}: anonymous manual routes use local static assets`,async()=>{
    const calls=[];
    const env={ASSETS:{fetch:async req=>{calls.push(new URL(req.url).pathname);return new Response('local manual');}}};
    for(const path of ['/manual/','/docs/manual.html']) {
      const response=await worker.fetch(new Request('https://example.pages.dev'+path),env);
      assert.equal(response.status,200);
      assert.equal(await response.text(),'local manual');
    }
    assert.deepEqual(calls,['/docs/manual.html','/docs/manual.html']);
  });
}
const source=await readFile(new URL('../public/landing/_worker.js',import.meta.url),'utf8');
const entry=(await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'))).default;
test('project entry preserves edition routes and query strings',async()=>{
  const result=await entry.fetch(new Request('https://bicycluswingatlas.pages.dev/en/atlas/?gene=hh&stage=0.5'),{});
  assert.equal(result.status,302);
  assert.equal(result.headers.get('Location'),'https://bicyclus-wing-atlas-en.pages.dev/en/atlas/?gene=hh&stage=0.5');
  assert.equal((await entry.fetch(new Request('https://bicycluswingatlas.pages.dev/downloads/'),{})).status,401);
});
