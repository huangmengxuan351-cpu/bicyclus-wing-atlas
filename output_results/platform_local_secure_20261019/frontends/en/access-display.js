"use strict";
(async()=>{
  const response=await fetch('/api/access-policy',{cache:'no-store'});
  if(!response.ok)return;
  const policy=await response.json();
  if(policy.can_download)return;
  const note=document.createElement('div');
  note.id='display-access-note';
  note.textContent='Display access · Reduced-resolution previews · Original images and data exports require separate authorization';
  note.style.cssText='position:fixed;bottom:10px;left:50%;transform:translateX(-50%);z-index:9999;padding:8px 14px;border-radius:6px;background:#193c35ed;color:white;font:12px/1.5 Arial,sans-serif;max-width:90vw;text-align:center;pointer-events:none';
  document.body.append(note);
  function apply(){
    document.querySelectorAll('a[download],.report-links,#fullComparison').forEach(el=>{el.hidden=true;el.style.display='none'});
    document.querySelectorAll('details').forEach(el=>{
      const title=el.querySelector(':scope > summary');
      if(!title||!title.textContent.trim().startsWith('Download'))return;
      title.textContent='Original downloads require authorization';
      el.open=false;
    });
    document.querySelectorAll('a[href^="/legacy-atlas"]').forEach(el=>{el.hidden=true;el.style.display='none'});
  }
  apply();
  const observer=new MutationObserver(()=>{observer.disconnect();apply();observer.observe(document.body,{childList:true,subtree:true})});
  observer.observe(document.body,{childList:true,subtree:true});
})().catch(()=>{});
