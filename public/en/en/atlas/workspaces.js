"use strict";
const $ = id => document.getElementById(id);
const anatomical = document.body.dataset.workspace === "anatomy";
const query = new URLSearchParams(location.search);
const names = {baseline:"Baseline (control)",early_soft_q20:"Soft gating (default)",bilateral_soft_q20:"Bilateral + soft-q20"};
const tips = {
  baseline:"No additional soft-q20 or filtering. Historical gain or background handling may already be present; this is not untouched microscope data.",
  early_soft_q20:"Soft weights are estimated before replaying frozen registration. Low-SNR signal is attenuated; real weak expression can also be reduced.",
  bilateral_soft_q20:"Edge-preserving bilateral filtering precedes soft-q20. Registration, gain and display scale are fixed across processing versions."
};
const palette = {dll:"#56d4f0",dpp:"#ef9a45",fz9:"#ea82b6",hh:"#b5ce50",hth:"#f5d66a",mad6:"#938de6",omb:"#57c8a0",optix:"#e57368",sal:"#b896dd",tkv:"#83cd63",vg:"#83d5d0",wg:"#edca86",wnt10:"#cc92b5"};
const presets = {
  structure:{dapi:.52,expression:.60,vein:.66,blend:"source-over",label:"Structure first"},
  signal:{dapi:.20,expression:.86,vein:.40,blend:"screen",label:"Signal first"},
  balanced:{dapi:.40,expression:.72,vein:.54,blend:"source-over",label:"Balanced layers"}
};
const state = {gene:query.get("gene")||"dll",wing:(query.get("wing")||"hindwing").replaceAll("_",""),
  stage:(query.get("stage")||"0.5").replace("stage_","").replaceAll("_","."),
  variant:query.get("variant")||"early_soft_q20",representation:query.get("representation")||"heat",
  mode:query.get("mode")||"single",zoom:1,x:0,y:0};
let data, generation=0, tiles=[], sourcePage=0;
let selectedGenes=(query.get("genes")||"").split(",").filter(Boolean);
const geneColors={},geneOpacity={},imageCache=new Map(),tintCache=new Map();
const esc = text=>String(text).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = x=>`${Math.round(Number(x)*100)}%`;
const labelWing = w=>w==="forewing"?"Forewing":"Hindwing";
const colorFor = g=>geneColors[g]||palette[g]||"#a4c4d9";
const findGroup = (w=state.wing,s=state.stage,g=state.gene)=>data.groups.find(i=>i.gene===g&&i.wing===w&&i.stage===s);
const findAnatomy = (w=state.wing,s=state.stage)=>data.anatomy_cases.find(i=>i.wing===w&&i.stage===s);
const notice = text=>{ $("notice").textContent=text;$("notice").hidden=!text; };
if(query.get("embedded")==="1")document.body.classList.add("embedded");

function boundedCache(cache,key,value,limit){cache.delete(key);cache.set(key,value);while(cache.size>limit)cache.delete(cache.keys().next().value);return value;}
function loadImage(url){
  if(imageCache.has(url))return boundedCache(imageCache,url,imageCache.get(url),12);
  return boundedCache(imageCache,url,new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error(`Image unavailable: ${url}`));im.src=url;}),12);
}
function tintImage(url,color){
  const key=url+color;
  if(tintCache.has(key))return boundedCache(tintCache,key,tintCache.get(key),8);
  return boundedCache(tintCache,key,loadImage(url).then(image=>{
    const canvas=document.createElement("canvas");canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
    const c=canvas.getContext("2d");c.drawImage(image,0,0);c.globalCompositeOperation="source-in";
    c.fillStyle=color;c.fillRect(0,0,canvas.width,canvas.height);return canvas;
  }),8);
}
function syncUrl(){
  const p=new URLSearchParams();
  for(const k of ["gene","wing","stage","variant","representation","mode"])p.set(k,state[k]);
  if(document.body.classList.contains("embedded"))p.set("embedded","1");
  if(anatomical){
    p.set("genes",selectedGenes.join(","));
    for(const id of ["showDapi","showVein","showGenes"])p.set(id,$(id).checked?"1":"0");
    for(const id of ["displayPreset","dapiSource","dapiBrightness","veinView","veinColor","veinOpacity","opacity","blendMode","layerOrder"])p.set(id,$(id).value);
    p.set("colors",selectedGenes.map(g=>`${g}:${colorFor(g)}`).join(","));
    p.set("alphas",selectedGenes.map(g=>`${g}:${geneOpacity[g]??.9}`).join(","));
  }
  history.replaceState(null,"",`${anatomical?"/en/anatomy/":"/en/atlas/"}?${p}`);
  const destination=new URLSearchParams({gene:state.gene,wing:state.wing,stage:state.stage,variant:state.variant,representation:state.representation});
  if($("processingComparison"))$("processingComparison").href=`/en/atlas/?${destination}&mode=methods`;
  if(!anatomical){destination.set("genes",state.gene);destination.set("showGenes","1");}
  $("workspaceSwitch").href=`${anatomical?"/en/atlas/":"/en/anatomy/"}?${destination}`;
}
function displayOptions(tile){return tile.preset?presets[tile.preset]:{
  dapi:Number($("dapiBrightness").value),expression:Number($("opacity").value),
  vein:Number($("veinOpacity").value),blend:$("blendMode").value};}
function draw(tile){
  if(!tile.loaded||!tile.canvas.isConnected)return;
  const r=tile.viewport.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2);
  const w=Math.max(1,Math.round(r.width*ratio)),h=Math.max(1,Math.round(r.height*ratio));
  if(tile.canvas.width!==w||tile.canvas.height!==h){tile.canvas.width=w;tile.canvas.height=h;}
  const c=tile.canvas.getContext("2d");c.globalCompositeOperation="source-over";c.globalAlpha=1;c.fillStyle="#080d0c";c.fillRect(0,0,w,h);
  const unit=Math.min(w,h),size=unit*state.zoom,x=(w-size)/2+state.x*unit,y=(h-size)/2+state.y*unit;
  if(!anatomical){c.globalAlpha=Number($("opacity").value);c.drawImage(tile.signal,x,y,size,size);c.globalAlpha=1;return;}
  const opt=displayOptions(tile);
  if($("showDapi").checked){c.globalAlpha=opt.dapi;c.drawImage(tile.base,x,y,size,size);}
  const drawVein=()=>{if($("showVein").checked){c.globalCompositeOperation="source-over";c.globalAlpha=opt.vein;c.drawImage(tile.vein,x,y,size,size);}};
  const drawGenes=()=>{if($("showGenes").checked){c.globalCompositeOperation=opt.blend;for(const layer of tile.layers){c.globalAlpha=opt.expression*(geneOpacity[layer.gene]??.9);c.drawImage(layer.image,x,y,size,size);}}};
  if($("layerOrder").value==="gene-top"){drawVein();drawGenes();}else{drawGenes();drawVein();}
  c.globalCompositeOperation="source-over";c.globalAlpha=1;
}
function redraw(){tiles.forEach(draw);$("zoomValue").value=`${Math.round(state.zoom*100)}%`;}
function setZoom(z){state.zoom=Math.max(.5,Math.min(6,z));redraw();}
function bindViewport(tile){
  let drag=null;
  tile.viewport.onpointerdown=e=>{if(e.button!==0)return;tile.viewport.setPointerCapture(e.pointerId);tile.viewport.classList.add("dragging");drag={id:e.pointerId,x:e.clientX,y:e.clientY,sx:state.x,sy:state.y};};
  tile.viewport.onpointermove=e=>{if(!drag||drag.id!==e.pointerId)return;const r=tile.viewport.getBoundingClientRect(),unit=Math.min(r.width,r.height);state.x=drag.sx+(e.clientX-drag.x)/unit;state.y=drag.sy+(e.clientY-drag.y)/unit;redraw();};
  const release=()=>{drag=null;tile.viewport.classList.remove("dragging");};
  tile.viewport.onpointerup=release;tile.viewport.onpointercancel=release;
  tile.viewport.addEventListener("wheel",e=>{e.preventDefault();setZoom(state.zoom*Math.exp(-e.deltaY*.001));},{passive:false});
  tile.viewport.ondblclick=()=>{state.zoom=1;state.x=state.y=0;redraw();};
}
function tileElement(title,note,count=""){
  const el=document.createElement("article");el.className="tile";
  el.innerHTML=`<header class="tile-header"><strong>${esc(title)}</strong><small>${esc(count)}</small></header><div class="viewport"><canvas aria-label="${esc(title)}"></canvas><span class="loading-label">Loading layers...</span></div><div class="tile-note">${esc(note)}</div>`;
  $("tiles").append(el);return el;
}
function makeExpressionTile(group,variant,title,token){
  const el=tileElement(title,group?`${labelWing(group.wing)} / ${group.stage} / ${names[variant]}`:"Missing data is not inferred.",group?`n = ${group.n}`:"No data");
  if(!group){el.querySelector(".loading-label").textContent="No accepted samples for this combination";return;}
  const tile={canvas:el.querySelector("canvas"),viewport:el.querySelector(".viewport"),loaded:false};tiles.push(tile);bindViewport(tile);
  const url=group.variants[variant].images[state.representation==="points"?"points_layer":"heatmap"];
  loadImage(url).then(signal=>{if(token!==generation)return;tile.signal=signal;tile.loaded=true;el.querySelector(".loading-label").remove();draw(tile);}).catch(error=>{if(token===generation){el.querySelector(".loading-label").textContent="Layer unavailable";notice(error.message);}});
}
async function makeAnatomyTile(wing,stage,title,token,preset=null){
  const anatomy=findAnatomy(wing,stage);
  const candidates=$("showGenes").checked?selectedGenes.map(g=>findGroup(wing,stage,g)).filter(Boolean):[];
  const el=tileElement(title,`${labelWing(wing)} / ${stage} / ${candidates.map(g=>`${g.gene}: n=${g.n}`).join(" / ")||"DAPI and vein only"}`,anatomy?`${anatomy.branch_count} branches`:"No anatomy");
  if(!anatomy){el.querySelector(".loading-label").textContent="No validated anatomy data";return;}
  if(candidates.some(g=>g.coordinate_frame!==anatomy.coordinate_frame)){
    el.querySelector(".loading-label").textContent="Overlay blocked: incompatible coordinate frames";
    notice("The gene and vein frames differ. No automatic warping or unverified overlay is allowed.");return;
  }
  const tile={canvas:el.querySelector("canvas"),viewport:el.querySelector(".viewport"),loaded:false,preset,layers:[]};tiles.push(tile);bindViewport(tile);
  try{
    const baseUrl=anatomy.assets[$("dapiSource").value==="original"?"dapi_original":"dapi_display"];
    const veinUrl=anatomy.assets[$("veinView").value==="outline"?"vein_outline":"vein"];
    const loaded=await Promise.all([loadImage(baseUrl),tintImage(veinUrl,$("veinColor").value),...candidates.map(g=>tintImage(g.variants[state.variant].images[state.representation==="points"?"points_layer":"expression_layer"],colorFor(g.gene)))]);
    if(token!==generation)return;
    tile.base=loaded[0];tile.vein=loaded[1];tile.layers=candidates.map((g,i)=>({gene:g.gene,image:loaded[i+2]}));
    tile.loaded=true;el.querySelector(".loading-label").remove();draw(tile);
  }catch(error){if(token===generation){el.querySelector(".loading-label").textContent="Layer unavailable";notice(error.message);}}
}
function geneEvidence(group){
  $("methodHelp").textContent=tips[state.variant];$("metrics").replaceChildren();$("downloads").replaceChildren();
  if(!group){$("facts").innerHTML=`<dt>Gene</dt><dd>${esc(state.gene)}</dd><dt>Extracted expression</dt><dd>Not available yet</dd>`;$("reviewStatus").textContent="No extracted data for this combination. No surrogate expression is displayed.";$("statusBadge").textContent="Awaiting extracted data";$("fullComparison").hidden=true;const withdrawn=data.withdrawn_groups?.[`${state.gene}__${state.wing}__stage_${state.stage.replace(".","_")}`];if(withdrawn){$("reviewStatus").textContent=withdrawn.reason_en;$("statusBadge").textContent="Withdrawn: stage conflict";}return;}
  const facts={"Inspected gene":group.gene+(anatomical&&!$("showGenes").checked?" / not displayed":""),"Gene source images":`${group.n}${group.n===1?" / single-sample map":""}`,"Wing / stage":`${labelWing(group.wing)} / ${group.stage}`,"Layer size":"2048 x 2048","Coordinate protocol":group.protocol};
  $("facts").innerHTML=Object.entries(facts).map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  $("reviewStatus").textContent=group.origin==="legacy"?"Historical sample quality was accepted; registration remains experimental.":"Sample usability was accepted. The selected safe registration candidate remains pending method-level human confirmation.";
  $("statusBadge").textContent="Registration needs confirmation";
  if(group.selection_status==="user_selected_final_composite"){$("reviewStatus").textContent="Final composite selected by the user; independent landmark validation remains a separate evidence item.";$("statusBadge").textContent="Final composite selected";}
  const v=group.variants[state.variant],m=v.metrics;
  $("metrics").innerHTML=Object.entries({"Strong retained":pct(m.strong_retention),"Weak retained":pct(m.weak_retention),"Outside reference":pct(m.outside_reference_fraction),"Standard coverage":pct(m.target_coverage_fraction)}).map(([k,value])=>`<dt>${k}</dt><dd>${value}</dd>`).join("");
  const links={"Expression heatmap (PNG)":v.images.heatmap,"Transparent expression (PNG)":v.images.expression_layer,"Intensity particles (PNG)":v.images.points_layer,"Intensity (16-bit PNG)":v.images.intensity_16bit,"Float intensity (NPY)":v.files["expression_float.npy"],"Coverage (NPY)":v.files["coverage_count.npy"],"Uncertainty (NPY)":v.files["between_sample_sd.npy"],"Missing support (NPY)":v.files["missing_support.npy"],"Outside conflict (NPY)":v.files["outside_reference_conflict.npy"],"Particles (NPZ)":v.files["points.npz"]};
  $("downloads").innerHTML=Object.entries(links).map(([label,url])=>`<a download href="${url}">${label}</a>`).join("");
  const archive=data.archive_reports[group.dataset];$("reportLink").href=archive.report;$("experimentLink").href=archive.experiments;$("manifestLink").href=archive.manifest;
  if(!anatomical){$("fullComparison").hidden=false;$("fullComparison").href=group.comparison;}
}
function anatomyEvidence(){
  const a=findAnatomy();if(!a)return;
  const extra={"Standard DAPI":"Verified original + normalized display","Confirmed vein":`${a.branch_count} branches / ${a.candidate}`,"Frame audit":"Exact DAPI pixel match / no new registration","Overlay genes":$("showGenes").checked?selectedGenes.map(g=>{const item=findGroup(state.wing,state.stage,g);return `${g}: ${item?`n=${item.n}`:"no data"}`;}).join(" / ")||"None selected":"Hidden / anatomy only"};
  $("facts").insertAdjacentHTML("beforeend",Object.entries(extra).map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join(""));
  $("statusBadge").textContent="Frozen human-selected vein";
  $("reviewStatus").textContent="Vein geometry is the confirmed representative layer, unchanged. All published gene reference DAPI images matched this frame exactly. This validates coordinate compatibility, not biological registration accuracy."+ ($("showGenes").checked?" Gene registration approval remains experimental or pending.":"");
  $("anatomyDownloads").innerHTML=Object.entries({"Standard DAPI / original":a.assets.dapi_original,"Standard DAPI / normalized display":a.assets.dapi_display,"Confirmed vein / transparent PNG":a.assets.vein,"Confirmed vein / grayscale PNG":a.assets.vein_gray,"Approved vein + DAPI":a.assets.approved_overlay,"Frozen selection manifest":a.assets.manifest}).map(([label,url])=>`<a href="${url}" download>${label}</a>`).join("");
  $("fullComparison").hidden=false;$("fullComparison").href=a.assets.approved_overlay;
  $("anatomyLegend").innerHTML=($("showVein").checked?`<span><i style="background:${$("veinColor").value}"></i>Vein</span>`:"")+ ($("showGenes").checked?selectedGenes.filter(g=>findGroup(state.wing,state.stage,g)).map(g=>`<span><i style="background:${colorFor(g)}"></i>${esc(g)} / n=${findGroup(state.wing,state.stage,g).n}</span>`).join(""):"");
}
function geneControls(){
  $("geneLayers").replaceChildren();
  for(const gene of data.genes){
    const group=findGroup(state.wing,state.stage,gene),a=findAnatomy(),compatible=!!group&&a?.coordinate_frame===group.coordinate_frame;
    const row=document.createElement("div");row.className="gene-control";
    row.innerHTML=`<label class="check"><input type="checkbox" ${selectedGenes.includes(gene)?"checked":""} ${compatible?"":"disabled"}><span>${esc(gene)}</span><small>${group?`n=${group.n}${compatible?"":" / frame differs"}`:"No data"}</small></label><div><input aria-label="${esc(gene)} color" type="color" value="${colorFor(gene)}" ${compatible?"":"disabled"}><input aria-label="${esc(gene)} opacity" type="range" min="0" max="1" step="0.05" value="${geneOpacity[gene]??.9}" ${compatible?"":"disabled"}></div>`;
    row.querySelector("input[type=checkbox]").onchange=e=>{
      if(e.target.checked&&selectedGenes.length>=4){e.target.checked=false;notice("Choose up to four genes for a readable overlay.");return;}
      selectedGenes=e.target.checked?[...selectedGenes,gene]:selectedGenes.filter(g=>g!==gene);if(e.target.checked)$("showGenes").checked=true;render();
    };
    row.querySelector("input[type=color]").onchange=e=>{geneColors[gene]=e.target.value;render();};
    row.querySelector("input[type=range]").oninput=e=>{geneOpacity[gene]=Number(e.target.value);redraw();syncUrl();};
    $("geneLayers").append(row);
  }
  $("addGene").disabled=!findGroup()||selectedGenes.includes(state.gene);
}
function sources(group){
  const start=sourcePage*12;
  for(const s of group.sources.slice(start,start+12)){
    const el=document.createElement("article");el.className="tile source-card";
    el.innerHTML=`<h3>${esc(s.label)}</h3><a href="${s.aligned}" target="_blank" rel="noopener"><img loading="lazy" src="${s.aligned}" alt="Historical aligned color input"></a><div class="source-meta">${esc(s.method)}${s.tre!==null?`<br>Selection TRE: ${s.tre.toFixed(1)} px / not independent validation`:""}<br>${esc(s.stage.replaceAll("_"," "))}</div><div class="source-actions"><a href="${s.aligned}" target="_blank" rel="noopener">Aligned color</a><a href="${s.intensity}" target="_blank" rel="noopener">Intensity source color</a></div>`;$("tiles").append(el);
  }
  if(group.sources.length>12){const el=document.createElement("div");el.className="source-actions";el.innerHTML=`<button id="prevPage" ${sourcePage===0?"disabled":""}>Previous</button><span>${start+1}-${Math.min(start+12,group.n)} of ${group.n}</span><button id="nextPage" ${start+12>=group.n?"disabled":""}>Next</button>`;$("tiles").append(el);$("prevPage").onclick=()=>{sourcePage--;render();};$("nextPage").onclick=()=>{sourcePage++;render();};}
}
function render(){
  if(!data)return;
  const token=++generation;tiles=[];$("tiles").replaceChildren();$("tiles").className=`tiles ${state.mode==="styles"?"methods":state.mode}`;
  $("gene").value=state.gene;$("variant").value=state.variant;$("representation").value=state.representation;$("variant").disabled=state.mode==="methods";
  $("processingLabel").textContent=state.mode==="methods"?"Three-method comparison":names[state.variant];
  document.querySelectorAll("[data-wing],[data-stage],[data-mode]").forEach(b=>{const key=b.dataset.wing?"wing":b.dataset.stage?"stage":"mode";b.classList.toggle("active",b.dataset[key]===state[key]);});
  const group=findGroup();geneEvidence(group);syncUrl();
  $("viewTitle").textContent=`${anatomical?"Standard anatomy":state.gene} / ${labelWing(state.wing)}`;
  $("viewSubtitle").textContent=`Stage ${state.stage} / ${anatomical?"DAPI + confirmed vein + optional genes":names[state.variant]}`;
  if(anatomical){
    geneControls();anatomyEvidence();
    for(const id of ["dapiBrightness","veinOpacity","opacity","blendMode"])$(id).disabled=state.mode==="styles";
    notice($("showGenes").checked?"Qualitative overlays of independent samples. Gene hue identifies the layer; opacity carries relative signal. Color mixing does not prove cellular co-expression.":"Anatomy-only view. Turn on gene layers to add expression; approved vein geometry is never re-fitted.");
    $("modeNote").textContent=state.mode==="styles"?"Three fixed display presets. Layer visibility, colors and zoom are shared.":"DAPI, vein and gene layers share verified coordinates. Display controls do not alter scientific arrays.";
    if(state.mode==="styles")Object.entries(presets).forEach(([key,p])=>makeAnatomyTile(state.wing,state.stage,p.label,token,key));
    else if(state.mode==="stages")["0.5","0.75","1.0"].forEach(s=>makeAnatomyTile(state.wing,s,`Stage ${s}`,token));
    else if(state.mode==="wings")["forewing","hindwing"].forEach(w=>makeAnatomyTile(w,state.stage,labelWing(w),token));
    else makeAnatomyTile(state.wing,state.stage,"Standard DAPI / vein / expression",token);
  }else{
    $("legend").hidden=state.mode==="sources";$("legend").querySelector("i").style.background=state.representation==="points"?"linear-gradient(90deg,#47bdff,#ffae36,#fff187)":"";
    $("legend").querySelector("span").textContent=state.representation==="points"?(group?.point_display?.sampling==="area_uniform"?"Area-uniform samples / P40, P50 / blue-orange-yellow":"Baseline P50 / P85 tiers"):"Relative expression / 0-1";
    $("modeNote").textContent="Pure expression, no DAPI or vein background. Linked zoom and pan.";
    notice(!group?"No extracted data for this combination. No expression is invented.":group.n===1?"Single-sample representative map, not a population consensus.":state.mode==="stages"||state.mode==="wings"?"Compare spatial patterns, not absolute mRNA abundance. ROI gains and batches differ.":"");
    if(state.mode==="sources"&&group)sources(group);
    else if(state.mode==="methods")data.variants.forEach(v=>makeExpressionTile(group,v,names[v],token));
    else if(state.mode==="stages")["0.5","0.75","1.0"].forEach(s=>makeExpressionTile(findGroup(state.wing,s),state.variant,`Stage ${s}`,token));
    else if(state.mode==="wings")["forewing","hindwing"].forEach(w=>makeExpressionTile(findGroup(w),state.variant,labelWing(w),token));
    else makeExpressionTile(group,state.variant,names[state.variant],token);
  }
}
function applyPreset(key){const p=presets[key];if(!p)return;$("displayPreset").value=key;$("dapiBrightness").value=p.dapi;$("opacity").value=p.expression;$("veinOpacity").value=p.vein;$("blendMode").value=p.blend;updateOutputs();}
function updateOutputs(){for(const [id,label] of [["opacity","opacityValue"],["dapiBrightness","dapiValue"],["veinOpacity","veinValue"]])if($(id)&&$(label))$(label).value=pct($(id).value);}
$("gene").onchange=e=>{state.gene=e.target.value;sourcePage=0;render();};
$("variant").onchange=e=>{state.variant=e.target.value;render();};$("representation").onchange=e=>{state.representation=e.target.value;render();};
document.querySelectorAll("[data-wing],[data-stage],[data-mode]").forEach(b=>b.onclick=()=>{const key=b.dataset.wing?"wing":b.dataset.stage?"stage":"mode";state[key]=b.dataset[key];sourcePage=0;render();});
$("opacity").oninput=()=>{if(anatomical)$("displayPreset").value="custom";updateOutputs();redraw();syncUrl();};
if(anatomical){
  $("displayPreset").onchange=e=>{applyPreset(e.target.value);state.mode="single";render();};
  for(const id of ["dapiBrightness","veinOpacity"])$(id).oninput=()=>{$("displayPreset").value="custom";updateOutputs();redraw();syncUrl();};
  for(const id of ["showDapi","showVein","showGenes","dapiSource","veinView","veinColor","blendMode","layerOrder"])$(id).onchange=()=>{render();};
  $("addGene").onclick=()=>{if(!selectedGenes.includes(state.gene)&&findGroup()){if(selectedGenes.length>=4){notice("Choose up to four genes.");return;}selectedGenes.push(state.gene);$("showGenes").checked=true;render();}};
}
$("zoomIn").onclick=()=>setZoom(state.zoom*1.2);$("zoomOut").onclick=()=>setZoom(state.zoom/1.2);$("fit").onclick=()=>{state.zoom=1;state.x=state.y=0;redraw();};
$("fullscreen").onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else $("tiles").requestFullscreen().catch(()=>notice("Fullscreen is unavailable."));};
$("copyLink").onclick=async()=>{try{await navigator.clipboard.writeText(location.href);$("copyLink").textContent="Link copied";setTimeout(()=>$("copyLink").textContent="Copy view link",1500);}catch{notice("Copy the browser address to share this view.");}};
new ResizeObserver(redraw).observe($("tiles"));window.addEventListener("resize",redraw);
async function init(){
  if(!anatomical&&state.mode==="multi"){const p=new URLSearchParams(query);p.set("mode","single");p.set("showGenes","1");location.replace('/en/anatomy/?'+p);return;}
  const response=await fetch('/api/atlas/catalog',{cache:"no-store"});if(!response.ok)throw new Error("Atlas archive unavailable");data=await response.json();data.anatomy_cases??=[];
  if(!data.genes.includes(state.gene))state.gene=data.genes[0];if(!["forewing","hindwing"].includes(state.wing))state.wing="hindwing";
  if(!["0.5","0.75","1.0"].includes(state.stage))state.stage="0.5";if(!data.variants.includes(state.variant))state.variant="early_soft_q20";
  if(!["heat","points"].includes(state.representation))state.representation="heat";
  if(!(anatomical?["single","styles","stages","wings"]:["single","methods","stages","wings","sources"]).includes(state.mode))state.mode="single";
  selectedGenes=[...new Set(selectedGenes)].filter(g=>data.genes.includes(g)).slice(0,4);
  if(state.variant!=="early_soft_q20"||state.mode==="methods")$("processingSettings").open=true;
  $("gene").innerHTML=data.genes.map(g=>`<option value="${esc(g)}">${esc(g)}${data.processed_genes.includes(g)?"":" / awaiting data"}</option>`).join("");$("gene").disabled=false;
  if(anatomical){
    applyPreset(query.get("displayPreset")||"structure");$("showGenes").checked=selectedGenes.length>0;
    for(const id of ["showDapi","showVein","showGenes"])if(query.has(id))$(id).checked=query.get(id)==="1";
    if($("showGenes").checked)$("anatomySettings").open=false;
    for(const id of ["dapiSource","veinView","blendMode","layerOrder","displayPreset"]){const value=query.get(id);if(value&&[...$(id).options].some(o=>o.value===value))$(id).value=value;}
    for(const id of ["dapiBrightness","veinOpacity","opacity"]){const value=Number(query.get(id));if(query.has(id)&&Number.isFinite(value))$(id).value=Math.max(0,Math.min(1,value));}
    if(/^#[a-f0-9]{6}$/i.test(query.get("veinColor")||""))$("veinColor").value=query.get("veinColor");
    for(const pair of (query.get("colors")||"").split(",")){const [gene,color]=pair.split(":");if(data.genes.includes(gene)&&/^#[a-f0-9]{6}$/i.test(color||""))geneColors[gene]=color;}
    for(const pair of (query.get("alphas")||"").split(",")){const [gene,value]=pair.split(":");if(data.genes.includes(gene)&&value!==undefined&&Number.isFinite(Number(value)))geneOpacity[gene]=Math.max(0,Math.min(1,Number(value)));}
  }
  $("datasetCount").textContent=`${data.genes.length} registered / ${data.processed_genes.length} extracted / ${data.samples} images${anatomical?" / 6 confirmed veins":""}`;
  $("reportLink").href=data.report;$("experimentLink").href=data.experiments;$("manifestLink").href=data.manifest;updateOutputs();render();
}
init().catch(error=>{notice(error.message);$("viewTitle").textContent="Archive unavailable";});
