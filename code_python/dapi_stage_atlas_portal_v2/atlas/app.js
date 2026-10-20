"use strict";
const $ = id => document.getElementById(id);
const query = new URLSearchParams(location.search);
const names = {baseline:"Baseline", early_soft_q20:"Early soft-q20", bilateral_soft_q20:"Bilateral + soft-q20"};
const tips = {
  baseline:"No additional soft-q20 or filtering. Fixed historical background/gain may already be present. This is a processing control, not untouched microscope data.",
  early_soft_q20:"Soft weights are estimated per source image before replaying frozen registration. Low-SNR signal is attenuated; real weak expression can also be reduced.",
  bilateral_soft_q20:"Edge-preserving bilateral filtering precedes the same frozen soft-q20 rule. Registration, gain and the displayed scale remain fixed across processing versions."
};
let data, generation = 0, tiles = [], sourcePage = 0;
const imageCache = new Map();
const palette={dll:"#56d4f0",dpp:"#ef9a45",fz9:"#ea82b6",hh:"#b5ce50",hth:"#f5d66a",mad6:"#938de6",omb:"#57c8a0",optix:"#e57368",sal:"#b896dd",tkv:"#83cd63",vg:"#83d5d0",wg:"#edca86",wnt10:"#cc92b5"};
let selectedGenes = (query.get("genes") || "").split(",").filter(Boolean);
const geneOpacity = {}, geneColors = {};
const state = {gene:query.get("gene") || "dll", wing:(query.get("wing") || "hindwing").replaceAll("_", ""),
  stage:(query.get("stage") || "0.5").replace("stage_", "").replaceAll("_", "."),
  variant:query.get("variant") || "early_soft_q20", representation:query.get("representation") || "heat",
  mode:query.get("mode") || "single", zoom:1, x:0, y:0};
if (query.get("embedded") === "1") document.body.classList.add("embedded");
const esc = text => String(text).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = x => `${(Number(x)*100).toFixed(1)}%`;
const labelWing = wing => wing === "forewing" ? "Forewing" : "Hindwing";
const findGroup = (wing=state.wing, stage=state.stage) => data.groups.find(g => g.gene===state.gene && g.wing===wing && g.stage===stage);

function notice(message) { $("notice").textContent = message; $("notice").hidden = !message; }
function syncUrl() {
  const p = new URLSearchParams();
  for (const k of ["gene","wing","stage","variant","representation","mode"]) p.set(k, state[k]);
  if (document.body.classList.contains("embedded")) p.set("embedded","1");
  if (state.mode==="multi") p.set("genes",selectedGenes.join(","));
  history.replaceState(null,"",`/atlas/?${p}`);
}
function loadImage(url) {
  if (imageCache.has(url)) {
    const value = imageCache.get(url); imageCache.delete(url); imageCache.set(url, value); return value;
  }
  const promise = new Promise((resolve,reject) => {
    const im = new Image(); im.onload=()=>resolve(im); im.onerror=()=>reject(new Error(`Image unavailable: ${url}`)); im.src=url;
  });
  imageCache.set(url,promise);
  while (imageCache.size>8) imageCache.delete(imageCache.keys().next().value);
  return promise;
}
function draw(tile) {
  if (!tile.loaded || !tile.canvas.isConnected) return;
  const r=tile.viewport.getBoundingClientRect(), ratio=Math.min(devicePixelRatio||1,2);
  const w=Math.max(1,Math.round(r.width*ratio)), h=Math.max(1,Math.round(r.height*ratio));
  if (tile.canvas.width!==w || tile.canvas.height!==h) {tile.canvas.width=w;tile.canvas.height=h;}
  const c=tile.canvas.getContext("2d"); c.fillStyle="#080d0c";c.fillRect(0,0,w,h);
  const size=Math.min(w,h)*state.zoom, unit=Math.min(w,h);
  const x=(w-size)/2+state.x*unit, y=(h-size)/2+state.y*unit;
  if (tile.base && $("showDapi").checked) {
    c.globalAlpha=Number($("dapiBrightness").value);c.drawImage(tile.base,x,y,size,size);
  }
  if(tile.layers){
    c.globalCompositeOperation=$("blendMode").value;
    for(const layer of tile.layers){c.globalAlpha=(geneOpacity[layer.gene]??.8)*Number($("opacity").value);c.drawImage(layer.image,x,y,size,size);}
    c.globalCompositeOperation="source-over";
  }else{
    c.globalAlpha=tile.opaque ? 1 : Number($("opacity").value);c.drawImage(tile.signal,x,y,size,size);
  }
  c.globalAlpha=1;
}
function redraw() {tiles.forEach(draw);$("zoomValue").value=`${Math.round(state.zoom*100)}%`;}
function setZoom(zoom) {state.zoom=Math.max(.5,Math.min(6,zoom));redraw();}
function bindViewport(tile) {
  let drag=null;
  tile.viewport.addEventListener("pointerdown",e=>{
    if (e.button!==0) return;
    tile.viewport.setPointerCapture(e.pointerId);tile.viewport.classList.add("dragging");
    drag={id:e.pointerId,x:e.clientX,y:e.clientY,sx:state.x,sy:state.y};
  });
  tile.viewport.addEventListener("pointermove",e=>{
    if (!drag || drag.id!==e.pointerId) return;
    const r=tile.viewport.getBoundingClientRect(), unit=Math.min(r.width,r.height);
    state.x=drag.sx+(e.clientX-drag.x)/unit;state.y=drag.sy+(e.clientY-drag.y)/unit;redraw();
  });
  const release=()=>{drag=null;tile.viewport.classList.remove("dragging");};
  tile.viewport.addEventListener("pointerup",release);tile.viewport.addEventListener("pointercancel",release);
  tile.viewport.addEventListener("wheel",e=>{e.preventDefault();setZoom(state.zoom*Math.exp(-e.deltaY*.001));},{passive:false});
  tile.viewport.addEventListener("dblclick",()=>{state.zoom=1;state.x=state.y=0;redraw();});
}
function makeTile(group, variant, title, token) {
  const el=document.createElement("article");el.className="tile";
  el.innerHTML=`<header class="tile-header"><strong>${esc(title)}</strong><small>${group?`n = ${group.n}`:"No data"}</small></header><div class="viewport"><canvas aria-label="${esc(title)}"></canvas><span class="loading-label">${group?"Loading layer...":"No accepted samples for this combination"}</span></div><div class="tile-note">${group?esc(`${labelWing(group.wing)} / Stage ${group.stage} / ${names[variant]}`):"Missing data is not inferred."}</div>`;
  $("tiles").append(el);
  if (!group) return;
  const tile={canvas:el.querySelector("canvas"),viewport:el.querySelector(".viewport"),loaded:false};tiles.push(tile);bindViewport(tile);
  const images=group.variants[variant].images;
  let url=images.expression_layer;
  const base=group.dapi;
  if (state.representation==="points") url=images.points_layer;
  tile.opaque=false;
  Promise.all([loadImage(url),base?loadImage(base):Promise.resolve(null)]).then(([signal,dapi])=>{
    if (token!==generation || !el.isConnected) return;
    tile.signal=signal;tile.base=dapi;tile.loaded=true;el.querySelector(".loading-label").remove();draw(tile);
  }).catch(error=>{if(token===generation){el.querySelector(".loading-label").textContent="Could not load this layer";notice(error.message);}});
}
function evidence(group) {
  if(!group){
    $("facts").innerHTML=`<dt>Selected gene</dt><dd>${esc(state.gene)}</dd><dt>Wing / stage</dt><dd>${labelWing(state.wing)} / ${state.stage}</dd><dt>Extracted atlas</dt><dd>Not available yet</dd>`;
    $("reviewStatus").textContent="This combination has no published extracted data. No surrogate or generated expression is displayed.";
    $("statusBadge").textContent="Awaiting extracted data";$("metrics").replaceChildren();$("downloads").replaceChildren();$("methodHelp").textContent=tips[state.variant];$("fullComparison").hidden=true;
    return;
  }
  $("fullComparison").hidden=false;
  const m=group.variants[state.variant].metrics;
  const facts={"Selected gene":group.gene,"Input images":`${group.n}${group.n===1?" / single-sample map":""}`,"Wing / stage":`${labelWing(group.wing)} / ${group.stage}`,"Layer size":"2048 x 2048","Processing origin":group.origin==="legacy"?"Historical 111-image cohort":"8866 reviewed cohort","Coordinate protocol":group.protocol};
  $("facts").innerHTML=Object.entries(facts).map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  $("reviewStatus").textContent=group.origin==="legacy"?"Historical sample quality was accepted. Frozen registration is experimental; method-level acceptance has not been established by this archive.":"Sample usability was accepted. The selected safe registration candidate remains pending method-level human confirmation.";
  $("statusBadge").textContent="Registration needs confirmation";
  $("methodHelp").textContent=tips[state.variant];
  $("metrics").innerHTML=Object.entries({"Strong signal retained":pct(m.strong_retention),"Weak signal retained":pct(m.weak_retention),"Signal outside reference":pct(m.outside_reference_fraction),"Standard wing coverage":pct(m.target_coverage_fraction)}).map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join("");
  const v=group.variants[state.variant];
  const downloads={"Expression layer (PNG)":v.images.expression_layer,"Particle layer (PNG)":v.images.points_layer,"Heatmap + DAPI (PNG)":v.images.heatmap_dapi,"Particles + DAPI (PNG)":v.images.points_dapi,"Unclipped heatmap (PNG)":v.images.heatmap_unclipped,"Intensity (16-bit PNG)":v.images.intensity_16bit,"Float intensity (NPY)":v.files["expression_float.npy"],"Coverage (NPY)":v.files["coverage_count.npy"],"Uncertainty (NPY)":v.files["between_sample_sd.npy"],"Missing support (NPY)":v.files["missing_support.npy"],"Outside conflict (NPY)":v.files["outside_reference_conflict.npy"],"Particle coordinates (NPZ)":v.files["points.npz"]};
  $("downloads").innerHTML=Object.entries(downloads).map(([label,url])=>`<a download href="${url}">${label}</a>`).join("");
  $("downloads").parentElement.querySelector("summary").textContent=`Download selected gene / ${group.gene}`;
  $("fullComparison").href=group.comparison;
  const archive=data.archive_reports[group.dataset];
  if(archive){$("reportLink").href=archive.report;$("experimentLink").href=archive.experiments;$("manifestLink").href=archive.manifest;}
}
function geneControls(){
  $("geneLayers").replaceChildren();
  for(const gene of data.genes){
    const group=data.groups.find(g=>g.gene===gene&&g.wing===state.wing&&g.stage===state.stage);
    const color=geneColors[gene]||palette[gene]||"#a4c4d9";
    const row=document.createElement("div");row.className="gene-control";
    row.innerHTML=`<label class="check"><input type="checkbox" ${selectedGenes.includes(gene)?"checked":""} ${group?"":"disabled"}><span>${esc(gene)}</span><small>${group?`n=${group.n}`:"No data"}</small></label><div><input aria-label="${esc(gene)} color" type="color" value="${color}" ${group?"":"disabled"}><input aria-label="${esc(gene)} opacity" type="range" min="0" max="1" step="0.05" value="${geneOpacity[gene]??.8}" ${group?"":"disabled"}></div>`;
    row.querySelector("input[type=checkbox]").onchange=e=>{
      if(e.target.checked&&selectedGenes.length>=4){e.target.checked=false;notice("Use up to four gene layers for a readable overlay.");return;}
      selectedGenes=e.target.checked?[...selectedGenes,gene]:selectedGenes.filter(g=>g!==gene);render();
    };
    row.querySelector("input[type=color]").onchange=e=>{geneColors[gene]=e.target.value;render();};
    row.querySelector("input[type=range]").oninput=e=>{geneOpacity[gene]=Number(e.target.value);redraw();};
    $("geneLayers").append(row);
  }
}
async function multiTile(token){
  const groups=selectedGenes.map(gene=>data.groups.find(g=>g.gene===gene&&g.wing===state.wing&&g.stage===state.stage)).filter(Boolean);
  if(!groups.length){makeTile(null,state.variant,"Select gene layers at the left",token);return;}
  if(new Set(groups.map(g=>g.coordinate_frame)).size!==1){
    notice("These archives use different standard DAPI / mask frames. Overlay is blocked; use separate views until coordinate compatibility is verified.");return;
  }
  const el=document.createElement("article");el.className="tile";
  el.innerHTML=`<header class="tile-header"><strong>Color-coded expression / ${groups.length} genes</strong><small>${labelWing(state.wing)} / ${state.stage}</small></header><div class="viewport"><canvas aria-label="Multi-gene expression overlay"></canvas><span class="loading-label">Loading independent gene layers...</span></div><div class="tile-note gene-legend">${groups.map(g=>`<span><i style="background:${geneColors[g.gene]||palette[g.gene]||"#a4c4d9"}"></i>${esc(g.gene)} / n=${g.n}</span>`).join("")}</div>`;
  $("tiles").append(el);
  const tile={viewport:el.querySelector(".viewport"),canvas:el.querySelector("canvas"),loaded:false,layers:[]};tiles.push(tile);bindViewport(tile);
  try{
    tile.base=await loadImage(groups[0].dapi);
    for(const g of groups){
      const key=state.representation==="points"?"points_layer":"expression_layer";
      const original=await loadImage(g.variants[state.variant].images[key]);
      if(token!==generation)return;
      const tinted=document.createElement("canvas");tinted.width=2048;tinted.height=2048;
      const c=tinted.getContext("2d");c.drawImage(original,0,0);c.globalCompositeOperation="source-in";
      c.fillStyle=geneColors[g.gene]||palette[g.gene]||"#a4c4d9";c.fillRect(0,0,2048,2048);
      tile.layers.push({gene:g.gene,image:tinted});
    }
    if(token!==generation)return;
    tile.loaded=true;el.querySelector(".loading-label").remove();draw(tile);
  }catch(error){if(token===generation)notice(error.message);}
}
function sources(group) {
  const perPage=12, start=sourcePage*perPage;
  for (const s of group.sources.slice(start,start+perPage)) {
    const el=document.createElement("article");el.className="tile source-card";
    el.innerHTML=`<h3>${esc(s.label)}</h3><a href="${s.aligned}" target="_blank" rel="noopener"><img loading="lazy" src="${s.aligned}" alt="Historical aligned color input"></a><div class="source-meta">${esc(s.method)}${s.tre!==null?`<br>Selection TRE: ${s.tre.toFixed(1)} px (not independent validation)`:""}<br>${esc(s.stage.replaceAll("_"," "))}</div><div class="source-actions"><a href="${s.aligned}" target="_blank" rel="noopener">Aligned color</a><a href="${s.intensity}" target="_blank" rel="noopener">Intensity source color</a></div>`;
    $("tiles").append(el);
  }
  if(group.sources.length>perPage){
    const el=document.createElement("div");el.className="source-actions";
    el.innerHTML=`<button id="prevPage" ${sourcePage===0?"disabled":""}>Previous</button><span>${start+1}-${Math.min(start+perPage,group.n)} of ${group.n}</span><button id="nextPage" ${start+perPage>=group.n?"disabled":""}>Next</button>`;
    $("tiles").append(el);$("prevPage").onclick=()=>{sourcePage--;render();};$("nextPage").onclick=()=>{sourcePage++;render();};
  }
}
function render() {
  if(!data) return;
  const token=++generation;tiles=[];$("tiles").replaceChildren();$("tiles").className=`tiles ${state.mode==="multi"?"single":state.mode}`;
  $("gene").value=state.gene;$("variant").value=state.variant;$("representation").value=state.representation;
  document.querySelectorAll("[data-wing]").forEach(b=>b.classList.toggle("active",b.dataset.wing===state.wing));
  document.querySelectorAll("[data-stage]").forEach(b=>b.classList.toggle("active",b.dataset.stage===state.stage));
  document.querySelectorAll("[data-mode]").forEach(b=>b.classList.toggle("active",b.dataset.mode===state.mode));
  $("variant").disabled=state.mode==="methods";
  const group=findGroup();
  evidence(group);syncUrl();
  $("multiSettings").hidden=state.mode!=="multi";
  if(state.mode==="multi"){
    selectedGenes=selectedGenes.filter(g=>data.groups.some(item=>item.gene===g&&item.wing===state.wing&&item.stage===state.stage));
    geneControls();
  }
  $("viewTitle").textContent=`${state.gene} / ${labelWing(state.wing)}`;
  $("viewSubtitle").textContent=`Stage ${state.stage} / ${names[state.variant]} / ${group?`${group.n} input image${group.n===1?"":"s"}`:"No extracted atlas yet"}`;
  $("showDapi").disabled=state.mode==="sources";
  $("legend").hidden=state.mode==="sources"||state.mode==="multi";
  $("legend").querySelector("i").style.background=state.representation==="points"?"linear-gradient(90deg,#47bdff,#ffae36,#fff187)":"";
  $("legend").querySelector("span").textContent=state.representation==="points"?"Baseline P50 / P85 color tiers":"Relative expression";
  $("modeNote").textContent=state.mode==="sources"?"Historical aligned color and actual intensity-source exports are both preserved.":"Shared zoom and pan. Display controls do not alter scientific arrays.";
  notice(state.mode==="multi"?"Qualitative spatial overlay only. Colors mix at overlapping signals; this does not prove co-expression or compare absolute mRNA abundance. Each gene retains its own observations and sample count.":state.mode==="stages"||state.mode==="wings"?"Compare spatial patterns, not absolute expression levels: ROI gains and batches differ. A shared 0-1 display scale does not provide cross-sample calibration.":group?.n===1?"Single-sample representative map. This is not a population consensus.":"");
  if(state.mode==="multi"){
    $("viewTitle").textContent=`Multi-gene / ${labelWing(state.wing)}`;
    $("viewSubtitle").textContent=`Stage ${state.stage} / ${names[state.variant]} / ${selectedGenes.join(" + ")||"Select available genes"}`;
    const available=selectedGenes.map(gene=>data.groups.find(g=>g.gene===gene&&g.wing===state.wing&&g.stage===state.stage)).filter(Boolean);
    const extra=document.createElement("dt");extra.textContent="Overlay layers / per-gene inputs";
    const detail=document.createElement("dd");detail.textContent=available.map(g=>`${g.gene}: n=${g.n}`).join(" / ");
    $("facts").append(extra,detail);
    multiTile(token);
  }
  else if(state.mode==="sources"&&group) sources(group);
  else if(state.mode==="methods") data.variants.forEach(v=>makeTile(group,v,names[v],token));
  else if(state.mode==="stages") ["0.5","0.75","1.0"].forEach(s=>makeTile(findGroup(state.wing,s),state.variant,`Stage ${s}`,token));
  else if(state.mode==="wings") ["forewing","hindwing"].forEach(w=>makeTile(findGroup(w),state.variant,labelWing(w),token));
  else makeTile(group,state.variant,names[state.variant],token);
  if(!group&&state.mode!=="multi")notice("No extracted data for this combination yet. The page is ready for a future validated archive; no expression is invented.");
}
$("gene").onchange=e=>{state.gene=e.target.value;sourcePage=0;render();};
$("variant").onchange=e=>{state.variant=e.target.value;render();};
$("representation").onchange=e=>{state.representation=e.target.value;render();};
document.querySelectorAll("[data-wing],[data-stage],[data-mode]").forEach(b=>b.onclick=()=>{const key=b.dataset.wing?"wing":b.dataset.stage?"stage":"mode";state[key]=b.dataset[key];if(state.mode==="multi"&&!selectedGenes.length&&findGroup())selectedGenes=[state.gene];sourcePage=0;render();});
$("blendMode").onchange=redraw;
$("showDapi").onchange=redraw;
$("opacity").oninput=()=>{$("opacityValue").value=pct($("opacity").value);redraw();};
$("dapiBrightness").oninput=()=>{$("dapiValue").value=pct($("dapiBrightness").value);redraw();};
$("zoomIn").onclick=()=>setZoom(state.zoom*1.2);$("zoomOut").onclick=()=>setZoom(state.zoom/1.2);
$("fit").onclick=()=>{state.zoom=1;state.x=state.y=0;redraw();};
$("fullscreen").onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else $("tiles").requestFullscreen().catch(()=>notice("Fullscreen is not available in this browser."));};
$("copyLink").onclick=async()=>{try{await navigator.clipboard.writeText(location.href);$("copyLink").textContent="Link copied";setTimeout(()=>$("copyLink").textContent="Copy view link",1500);}catch{notice("Copy the address from the browser to share this view.");}};
new ResizeObserver(redraw).observe($("tiles"));window.addEventListener("resize",redraw);
fetch("/api/atlas/catalog").then(r=>{if(!r.ok)throw new Error("Archive unavailable");return r.json();}).then(payload=>{
  data=payload;$("gene").innerHTML=data.genes.map(g=>`<option value="${esc(g)}">${esc(g)}${data.processed_genes.includes(g)?"":" / awaiting data"}</option>`).join("");$("gene").disabled=false;
  if(!data.genes.includes(state.gene))state.gene=data.genes[0];
  if(!["forewing","hindwing"].includes(state.wing))state.wing="hindwing";
  if(!["0.5","0.75","1.0"].includes(state.stage))state.stage="0.5";
  if(!data.variants.includes(state.variant))state.variant="early_soft_q20";
  if(!["heat","points"].includes(state.representation))state.representation="heat";
  if(!["single","methods","stages","wings","multi","sources"].includes(state.mode))state.mode="single";
  selectedGenes=selectedGenes.filter(g=>data.genes.includes(g)).slice(0,4);
  if(!selectedGenes.length&&findGroup())selectedGenes=[state.gene];
  $("datasetCount").textContent=`${data.genes.length} registered / ${data.processed_genes.length} extracted / ${data.samples} images`;
  $("reportLink").href=data.report;$("experimentLink").href=data.experiments;$("manifestLink").href=data.manifest;render();
}).catch(error=>{notice(error.message);$("viewTitle").textContent="Archive could not be loaded";});
