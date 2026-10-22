import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

for (const language of ['zh','en']) {
  const source = await readFile(new URL(`../public/${language}/_worker.js`,import.meta.url),'utf8');
  const worker = (await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'))).default;
  const site = `https://bicyclus-wing-atlas-${language}.pages.dev`;
  const origin = `https://example.api-inference.modelscope.net/${language}`;
  const env = {ATLAS_ORIGIN:origin, MODELSCOPE_STUDIO_TOKEN:'test-only-service-credential'};

  test(`${language}: accept own browser origin and authenticate the service hop`, async t => {
    const calls=[];
    t.mock.method(globalThis,'fetch',async (url,options) => {
      calls.push({url:String(url),options,body:await new Response(options.body).text()});
      return Response.json({run_id:'test-run',summary:{decision:'needs_review'}});
    });
    const request = new Request(site+'/api/predict',{method:'POST',body:'public-test-data',
      headers:{Origin:site,'Content-Type':'multipart/form-data; boundary=test',Cookie:'wing_atlas_visitor=test',
        'X-ModelScope-Forged':'ignore','X-Atlas-Download-Authorization':'ignore'}});
    const result = await worker.fetch(request,env);
    assert.equal(result.status,200);
    assert.equal(calls.length,1);
    assert.equal(calls[0].url,origin+'/api/predict');
    assert.equal(calls[0].options.headers.has('Origin'),false);
    assert.equal(calls[0].options.headers.get('Authorization'),'Bearer test-only-service-credential');
    assert.equal(calls[0].options.headers.get('Cookie'),'wing_atlas_visitor=test');
    assert.equal(calls[0].options.headers.has('X-ModelScope-Forged'),false);
    assert.equal(calls[0].options.headers.has('X-Atlas-Download-Authorization'),false);
    assert.equal(calls[0].body,'public-test-data');
    assert.doesNotMatch(await result.text(),/test-only-service-credential/);
  });

  test(`${language}: reject foreign and opaque browser origins before forwarding`, async t => {
    let called=false;
    t.mock.method(globalThis,'fetch',async () => {called=true;return Response.json({});});
    for(const supplied of ['https://unrelated.example','null',`https://bicyclus-wing-atlas-${language==='zh'?'en':'zh'}.pages.dev`]) {
      const result = await worker.fetch(new Request(site+'/api/predict',{method:'POST',body:'test',headers:{Origin:supplied}}),env);
      assert.equal(result.status,403);
    }
    assert.equal(called,false);
  });

  test(`${language}: give the hosting proxy a fixed-length upload stream`,async t => {
    const tasks=[];
    const sent=Uint8Array.from([0,255,216,10,13,128,0,255,217]);
    const previous=Object.getOwnPropertyDescriptor(globalThis,'FixedLengthStream');
    Object.defineProperty(globalThis,'FixedLengthStream',{configurable:true,value:class {
      constructor(length) {
        assert.equal(length,sent.length);
        const stream=new TransformStream();
        this.readable=stream.readable;this.writable=stream.writable;
      }
    }});
    t.after(()=>{if(previous)Object.defineProperty(globalThis,'FixedLengthStream',previous);else delete globalThis.FixedLengthStream;});
    t.mock.method(globalThis,'fetch',async (_,options) => {
      assert.deepEqual(new Uint8Array(await new Response(options.body).arrayBuffer()),sent);
      return Response.json({run_id:'binary-upload'});
    });
    const result=await worker.fetch(new Request(site+'/api/predict',{method:'POST',body:sent,
      headers:{Origin:site,'Content-Length':String(sent.length)}}),env,{waitUntil:p=>tasks.push(p)});
    await Promise.all(tasks);
    assert.equal(result.status,200);
    assert.equal(result.headers.get('X-Atlas-Upload-Framing'),'fixed-length');
  });

  test(`${language}: private origin credential is required`,async t => {
    t.mock.method(globalThis,'fetch',async () => {throw Error('Must not call upstream');});
    const result=await worker.fetch(new Request(site+'/api/predict',{method:'POST',body:'test',headers:{Origin:site}}),{ATLAS_ORIGIN:origin});
    assert.equal(result.status,503);
  });

  test(`${language}: anonymous originals remain protected`,async t => {
    let called=false;
    t.mock.method(globalThis,'fetch',async () => {called=true;return Response.json({});});
    for(const path of ['/downloads/','/atlas-assets/main/test.png','/anatomy-assets/test.png']) {
      const result=await worker.fetch(new Request(site+path),env);
      assert.equal(result.status,401);
    }
    assert.equal(called,false);
  });

  test(`${language}: forward download authorization separately from service credential`,async t => {
    t.mock.method(globalThis,'fetch',async (_,options) => {
      assert.equal(options.headers.get('Authorization'),'Bearer test-only-service-credential');
      assert.equal(options.headers.get('X-Atlas-Download-Authorization'),'Basic invalid-test-credential');
      return Response.json({error:'Download password required'}, {status:401});
    });
    const result=await worker.fetch(new Request(site+'/downloads/',{headers:{Authorization:'Basic invalid-test-credential'}}),env);
    assert.equal(result.status,401);
  });

  test(`${language}: failed prediction submissions are not automatically repeated`,async t => {
    let calls=0;
    t.mock.method(globalThis,'fetch',async () => {calls++;throw Error('upstream unavailable');});
    const result=await worker.fetch(new Request(site+'/api/predict',{method:'POST',body:'test',headers:{Origin:site}}),env);
    assert.equal(result.status,503);
    assert.equal(calls,1);
  });

  const frontend = await readFile(new URL(`../public/${language}/${language}/predict/app.js`,import.meta.url),'utf8');
  const start=frontend.indexOf('async function predictionResponse(');
  const end=frontend.indexOf('\napplyManual.addEventListener',start);
  assert.ok(start>=0 && end>start);
  const parse=vm.runInNewContext(frontend.slice(start,end)+'\npredictionResponse;');

  test(`${language}: preserve successful prediction payload`,async () => {
    const payload={run_id:'test',summary:{decision:'needs_review'}};
    assert.deepEqual(await parse(Response.json(payload)),payload);
  });

  test(`${language}: explain non-JSON gateway errors without a JSON parse exception`,async () => {
    await assert.rejects(parse(new Response('<html>Gateway timeout</html>',{status:524})),error => {
      assert.match(error.message,/HTTP 524/);
      assert.doesNotMatch(error.message,/Unexpected token|<html>/);
      return true;
    });
  });

  test(`${language}: explain a busy backend`,async () => {
    await assert.rejects(parse(Response.json({error:'Another prediction is running. Please wait.'},{status:409})),error => {
      assert.match(error.message,/HTTP 409/);
      assert.match(error.message,language==='zh'?/后台正在处理/:/Another prediction/);
      return true;
    });
  });
}
