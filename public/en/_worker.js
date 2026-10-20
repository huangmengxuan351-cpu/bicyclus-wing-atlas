// Cloud-resident display resources. Only dynamic prediction/downloads use an origin.
const LANGUAGE = 'en';
const FALLBACK_ORIGIN = ''; // Set ATLAS_ORIGIN to the ModelScope origin plus /en.
let previewIndexPromise;
async function previewEntry(path,request,env) {
  if(!/^\/preview\/[a-f0-9]{40}\.png$/.test(path))return undefined;
  if(!previewIndexPromise)previewIndexPromise=env.ASSETS.fetch(new Request(new URL('/data/preview-index.json',request.url))).then(r=>{if(!r.ok)throw Error('Preview index unavailable.');return r.json();}).catch(error=>{previewIndexPromise=undefined;throw error;});
  return (await previewIndexPromise)[path];
}
const STATIC_ROUTES = {"/en/access-display.js":"/en/access-display.js","/en/atlas/anatomy.html":"/en/atlas/anatomy.html","/en/atlas/index.html":"/en/atlas/index.html","/en/atlas/styles.css":"/en/atlas/styles.css","/en/atlas/workspaces.js":"/en/atlas/workspaces.js","/en/portal/home-workspaces.css":"/en/portal/home-workspaces.css","/en/portal/home.js":"/en/portal/home.js","/en/portal/index.html":"/en/portal/index.html","/en/portal/result.html":"/en/portal/result.html","/en/portal/result.js":"/en/portal/result.js","/en/portal/styles.css":"/en/portal/styles.css","/en/predict/app.js":"/en/predict/app.js","/en/predict/index.html":"/en/predict/index.html","/en/predict/styles.css":"/en/predict/styles.css","/":"/en/portal/index.html","/en/":"/en/portal/index.html","/en/atlas/":"/en/atlas/index.html","/en/anatomy/":"/en/atlas/anatomy.html","/en/predict/":"/en/predict/index.html","/en/result/":"/en/portal/result.html","/en/predict/manual_anatomical_rotation_cross_overlay.png":"/en/predict/manual_anatomical_rotation_cross_overlay.png"};
const API_FILES = {'/api/atlas/catalog':'/data/catalog.json','/api/access-policy':'/data/access-policy.json'};

function reply(value,status=200,extra={}) {
  return Response.json(value,{status,headers:{'Cache-Control':'no-store','X-Atlas-Language':LANGUAGE,...extra}});
}
function unavailable() {
  return reply({error:LANGUAGE==='zh'?'预测或原始下载服务暂未连接；图谱浏览仍可使用。':'Prediction or original downloads are currently unavailable. Atlas browsing remains available.'},503);
}
async function asset(request,env,path) {
  const url=new URL(request.url);url.pathname=path;url.search='';
  let response=await env.ASSETS.fetch(new Request(url,{method:request.method,headers:request.headers}));
  // Pages canonicalizes /index.html and *.html. Resolve those asset redirects
  // internally so the visitor stays on this edition's public route.
  for(let i=0;i<4 && [301,302,307,308].includes(response.status);i++) {
    const location=response.headers.get('Location');
    if(!location)break;
    const canonical=new URL(location,url);
    if(canonical.origin!==url.origin)return reply({error:'Invalid asset redirect.'},502);
    url.pathname=canonical.pathname;url.search=canonical.search;
    response=await env.ASSETS.fetch(new Request(url,{method:request.method,headers:request.headers}));
  }
  const headers=new Headers(response.headers);
  headers.set('X-Atlas-Language',LANGUAGE);headers.set('X-Atlas-Hosting','cloud-static');
  headers.set('X-Content-Type-Options','nosniff');
  return new Response(request.method==='HEAD'?null:response.body,{status:response.status,headers});
}
async function dynamic(request,env) {
  const incoming=new URL(request.url);
  const origin=env.ATLAS_ORIGIN||FALLBACK_ORIGIN;
  let upstream;
  try {upstream=new URL(origin);if(upstream.protocol!=='https:'||upstream.username||upstream.password)throw Error();}
  catch {return unavailable();}
  if (request.method==='POST') {
    if (request.headers.get('Origin') && request.headers.get('Origin')!==incoming.origin) return reply({error:'Cross-origin writes are not allowed.'},403);
    if (Number(request.headers.get('Content-Length')||0)>64*1024*1024) return reply({error:'Upload limit: 64 MB.'},413);
  }
  const headers=new Headers(request.headers);
  for(const name of ['host','oai-sites-authorization','oai-authenticated-user-id','oai-authenticated-user-email','oai-authenticated-user-full-name'])headers.delete(name);
  const downloadAuthorization=headers.get('Authorization');
  for(const name of [...headers.keys()])if(name.toLowerCase()==='authorization'||name.toLowerCase()==='x-atlas-download-authorization'||/^x-(modelscope|studio)-/i.test(name))headers.delete(name);
  if(downloadAuthorization)headers.set('X-Atlas-Download-Authorization',downloadAuthorization);
  if(upstream.hostname.endsWith('.api-inference.modelscope.net')) {
    if(!env.MODELSCOPE_STUDIO_TOKEN)return unavailable();
    headers.set('Authorization','Bearer '+env.MODELSCOPE_STUDIO_TOKEN);
  }
  if(headers.has('Origin'))headers.set('Origin',upstream.origin);
  const prefix=upstream.pathname.replace(/\/$/,'');
  const target=new URL(prefix+incoming.pathname+incoming.search,upstream.origin);
  try {
    let response,attempts=0;
    const retryable=['GET','HEAD'].includes(request.method);
    for(let attempt=0;attempt<2;attempt++) {
      attempts++;
      try {
        response=await fetch(target,{method:request.method,headers,body:retryable?undefined:request.body,redirect:'manual'});
        if(!retryable||attempt||![502,503,504,520,521,522,523,524,525,526].includes(response.status))break;
        await response.body?.cancel();
      } catch(error) {if(!retryable||attempt)throw error;}
      await new Promise(resolve=>setTimeout(resolve,750));
    }
    const resultHeaders=new Headers(response.headers);resultHeaders.set('Cache-Control','no-store');resultHeaders.set('X-Atlas-Language',LANGUAGE);
    resultHeaders.set('X-Atlas-Origin-Attempts',String(attempts));
    const location=resultHeaders.get('Location');
    if(location?.startsWith(upstream.origin+prefix+'/'))resultHeaders.set('Location',incoming.origin+location.slice((upstream.origin+prefix).length));
    return new Response(response.body,{status:response.status,headers:resultHeaders});
  } catch {return unavailable();}
}

export default {
  async fetch(request,env) {
    const url=new URL(request.url),path=url.pathname;
    let decoded=path;
    try {for(let i=0;i<3;i++)decoded=decodeURIComponent(decoded);}catch{return reply({error:'Invalid URL.'},400);}
    if(decoded.includes('\\')||decoded.split('/').some(p=>p==='.'||p==='..'))return reply({error:'Invalid path.'},403);
    if(path==='/healthz'||path==='/__deployment_health')return reply({ready:true,display_ready:true,hosting:'cloud-static',language:LANGUAGE,genes:12,samples:291,groups:72,prediction_hosting:'external-origin'});
    const original=path.startsWith('/downloads')||path.startsWith('/atlas-assets/')||path.startsWith('/anatomy-assets/');
    if(original) {
      if(!request.headers.has('Authorization'))return reply({error:'Original downloads require a separate password.'},401,{'WWW-Authenticate':'Basic realm="Atlas original downloads", charset="UTF-8"'});
      if(!['GET','HEAD'].includes(request.method))return reply({error:'Method not allowed.'},405);
      return dynamic(request,env);
    }
    if(path in API_FILES && ['GET','HEAD'].includes(request.method))return asset(request,env,API_FILES[path]);
    const preview=await previewEntry(path,request,env);
    if(preview && ['GET','HEAD'].includes(request.method)) {
      const [pack,start,length]=preview;
      const target=new URL('/packs/'+pack,url.origin);
      const response=await env.ASSETS.fetch(new Request(target,{headers:{Range:`bytes=${start}-${start+length-1}`}}));
      if(!response.ok)return reply({error:'Preview not found.'},404);
      const buffer=await response.arrayBuffer();
      const bytes=response.status===206?buffer:buffer.slice(start,start+length);
      return new Response(request.method==='HEAD'?null:bytes,{headers:{'Content-Type':'image/png','Content-Length':String(length),'Cache-Control':'public, max-age=86400','X-Atlas-Asset':'display-preview','X-Atlas-Language':LANGUAGE,'X-Atlas-Hosting':'cloud-static','X-Content-Type-Options':'nosniff'}});
    }
    // These packs contain reduced display previews only. CI reuses them with SHA-256 checks.
    if(/^\/public-packs\/\d{4}\.bin$/.test(path)&&['GET','HEAD'].includes(request.method))return asset(request,env,path.replace('/public-packs/','/packs/'));
    if(/^\/preview\/[a-f0-9]{40}\.png$/.test(path)&&['GET','HEAD'].includes(request.method))return dynamic(request,env);
    if(STATIC_ROUTES[path] && ['GET','HEAD'].includes(request.method))return asset(request,env,STATIC_ROUTES[path]);
    if(/^\/api\/(?:model-info|history|health|result\/[A-Za-z0-9_.-]+|file\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)$/.test(path)&&['GET','HEAD'].includes(request.method))return dynamic(request,env);
    if(['/api/predict','/api/re-align'].includes(path)&&request.method==='POST')return dynamic(request,env);
    if(path==='/favicon.ico')return new Response(null,{status:204});
    // Explicit static allowlist; no raw data, admin, uploads or alternate-language tree.
    return reply({error:'Page not found.'},404);
  }
};
