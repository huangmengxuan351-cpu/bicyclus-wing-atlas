
function wingCN(value){return ({forewing:"前翅",hindwing:"后翅"})[value]||value||"所选翅型"}
function statusCN(value){return ({auto_candidate:"自动候选",needs_review:"需要复核",ready:"就绪",stable:"稳定",idle:"等待输入",loading:"加载中",error:"错误",manual:"人工",automatic:"自动"})[value]||value}
function diagnosticCN(value){let text=String(value||"");for(const [key,label] of Object.entries({alignment_confidence_below_threshold:"配准置信度低于阈值","wing_type_probability_below_0.62":"翅型概率低于0.62","wing_type_margin_below_0.06":"翅型前两类概率差低于0.06","stage_probability_below_0.48":"阶段概率低于0.48","stage_margin_below_0.04":"阶段前两类概率差低于0.04",ordinal_conflict:"序数证据冲突",stage_bundle_decision:"阶段模型判定"})){text=text.replaceAll(key,label)}return text}
function sourceStageCN(value){return value==="pre_historical_alignment_export"?"历史配准导出前的强度来源":value.replaceAll("_"," ")}
"use strict";
const $ = id => document.getElementById(id);
const anatomical = document.body.dataset.workspace === "anatomy";
const query = new URLSearchParams(location.search);
const names = {baseline:"基线处理",early_soft_q20:"早期软门控q20",bilateral_soft_q20:"双边滤波与软门控q20"};
const tips = {
  baseline:"不新增软门控或滤波；历史输入可能已经过增益或背景处理，并非未经处理的显微图像。",
  early_soft_q20:"在回放冻结配准之前估计软权重；低信噪比信号会被抑制，也可能降低真实弱表达。",
  bilateral_soft_q20:"先进行保边双边滤波，再应用软门控q20；不同分支共用冻结配准、增益与显示范围。"
};
const palette = {dll:"#56d4f0",dpp:"#ef9a45",fz9:"#ea82b6",hh:"#b5ce50",hth:"#f5d66a",mad6:"#938de6",omb:"#57c8a0",optix:"#e57368",sal:"#b896dd",tkv:"#83cd63",vg:"#83d5d0",wg:"#edca86",wnt10:"#cc92b5"};
const presets = {
  structure:{dapi:.52,expression:.60,vein:.66,blend:"source-over",label:"结构优先"},
  signal:{dapi:.20,expression:.86,vein:.40,blend:"screen",label:"信号优先"},
  balanced:{dapi:.40,expression:.72,vein:.54,blend:"source-over",label:"平衡显示"}
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
const labelWing = w=>w==="forewing"?"前翅":"后翅";
const colorFor = g=>geneColors[g]||palette[g]||"#a4c4d9";
const findGroup = (w=state.wing,s=state.stage,g=state.gene)=>data.groups.find(i=>i.gene===g&&i.wing===w&&i.stage===s);
const findAnatomy = (w=state.wing,s=state.stage)=>data.anatomy_cases.find(i=>i.wing===w&&i.stage===s);
const notice = text=>{ $("notice").textContent=text;$("notice").hidden=!text; };
if(query.get("embedded")==="1")document.body.classList.add("embedded");

function boundedCache(cache,key,value,limit){cache.delete(key);cache.set(key,value);while(cache.size>limit)cache.delete(cache.keys().next().value);return value;}
function loadImage(url){
  if(imageCache.has(url))return boundedCache(imageCache,url,imageCache.get(url),12);
  return boundedCache(imageCache,url,new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error(`图像暂不可用： ${url}`));im.src=url;}),12);
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
  history.replaceState(null,"",`${anatomical?"/zh/anatomy/":"/zh/atlas/"}?${p}`);
  const destination=new URLSearchParams({gene:state.gene,wing:state.wing,stage:state.stage,variant:state.variant,representation:state.representation});
  if(!anatomical){destination.set("genes",state.gene);destination.set("showGenes","1");}
  $("workspaceSwitch").href=`${anatomical?"/zh/atlas/":"/zh/anatomy/"}?${destination}`;
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
  const el=tileElement(title,group?`${labelWing(group.wing)} / ${group.stage} / ${names[variant]}`:"缺测数据不进行推断补齐。",group?`n = ${group.n}`:"暂无数据");
  if(!group){el.querySelector(".loading-label").textContent="该组合没有已接受的来源图像";return;}
  const tile={canvas:el.querySelector("canvas"),viewport:el.querySelector(".viewport"),loaded:false};tiles.push(tile);bindViewport(tile);
  const url=group.variants[variant].images[state.representation==="points"?"points_layer":"heatmap"];
  loadImage(url).then(signal=>{if(token!==generation)return;tile.signal=signal;tile.loaded=true;el.querySelector(".loading-label").remove();draw(tile);}).catch(error=>{if(token===generation){el.querySelector(".loading-label").textContent="图层暂不可用";notice(error.message);}});
}
async function makeAnatomyTile(wing,stage,title,token,preset=null){
  const anatomy=findAnatomy(wing,stage);
  const candidates=$("showGenes").checked?selectedGenes.map(g=>findGroup(wing,stage,g)).filter(Boolean):[];
  const el=tileElement(title,`${labelWing(wing)} / ${stage} / ${candidates.map(g=>`${g.gene}: n=${g.n}`).join(" / ")||"仅DAPI与翅脉"}`,anatomy?`${anatomy.branch_count} 条分支`:"暂无结构参照");
  if(!anatomy){el.querySelector(".loading-label").textContent="没有通过完整性验证的结构数据";return;}
  if(candidates.some(g=>g.coordinate_frame!==anatomy.coordinate_frame)){
    el.querySelector(".loading-label").textContent="已阻止叠加：参考坐标不一致";
    notice("基因与翅脉参考坐标不同，已阻止自动变形和未经核验的叠加。");return;
  }
  const tile={canvas:el.querySelector("canvas"),viewport:el.querySelector(".viewport"),loaded:false,preset,layers:[]};tiles.push(tile);bindViewport(tile);
  try{
    const baseUrl=anatomy.assets[$("dapiSource").value==="original"?"dapi_original":"dapi_display"];
    const veinUrl=anatomy.assets[$("veinView").value==="outline"?"vein_outline":"vein"];
    const loaded=await Promise.all([loadImage(baseUrl),tintImage(veinUrl,$("veinColor").value),...candidates.map(g=>tintImage(g.variants[state.variant].images[state.representation==="points"?"points_layer":"expression_layer"],colorFor(g.gene)))]);
    if(token!==generation)return;
    tile.base=loaded[0];tile.vein=loaded[1];tile.layers=candidates.map((g,i)=>({gene:g.gene,image:loaded[i+2]}));
    tile.loaded=true;el.querySelector(".loading-label").remove();draw(tile);
  }catch(error){if(token===generation){el.querySelector(".loading-label").textContent="图层暂不可用";notice(error.message);}}
}
function geneEvidence(group){
  $("methodHelp").textContent=tips[state.variant];$("metrics").replaceChildren();$("downloads").replaceChildren();
  if(!group){$("facts").innerHTML=`<dt>基因</dt><dd>${esc(state.gene)}</dd><dt>已提取表达</dt><dd>尚无可用结果</dd>`;$("reviewStatus").textContent="该组合尚无表达结果，不使用替代数据。";$("statusBadge").textContent="等待表达结果";$("fullComparison").hidden=true;return;}
  const facts={"当前查看基因":group.gene+(anatomical&&!$("showGenes").checked?" / 当前未显示":""),"来源图像数量":`${group.n}${group.n===1?" / 单图代表性结果":""}`,"翅型 / 阶段":`${labelWing(group.wing)} / ${group.stage}`,"科研参考网格":"2048 x 2048","当前基因处理协议":group.protocol};
  $("facts").innerHTML=Object.entries(facts).map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  $("reviewStatus").textContent=group.origin==="legacy"?"历史来源图像质量已接受，配准仍保留实验性状态。":"来源图像可用性已接受；当前安全配准候选仍需方法层面的人工确认。";
  $("statusBadge").textContent="配准仍需确认";
  const v=group.variants[state.variant],m=v.metrics;
  $("metrics").innerHTML=Object.entries({"强信号保留率":pct(m.strong_retention),"弱信号保留率":pct(m.weak_retention),"参考区外信号比例":pct(m.outside_reference_fraction),"标准翅区覆盖率":pct(m.target_coverage_fraction)}).map(([k,value])=>`<dt>${k}</dt><dd>${value}</dd>`).join("");
  const links={"表达热图（PNG）":v.images.heatmap,"透明表达图层（PNG）":v.images.expression_layer,"强度点位图（PNG）":v.images.points_layer,"16位强度图（PNG）":v.images.intensity_16bit,"浮点表达（NPY）":v.files["expression_float.npy"],"覆盖数（NPY）":v.files["coverage_count.npy"],"样本间标准差（NPY）":v.files["between_sample_sd.npy"],"缺测标记（NPY）":v.files["missing_support.npy"],"参考区外冲突（NPY）":v.files["outside_reference_conflict.npy"],"点位坐标（NPZ）":v.files["points.npz"]};
  $("downloads").innerHTML=Object.entries(links).map(([label,url])=>`<a download href="${url}">${label}</a>`).join("");
  const archive=data.archive_reports[group.dataset];$("reportLink").href=archive.report;$("experimentLink").href=archive.experiments;$("manifestLink").href=archive.manifest;
  if(!anatomical){$("fullComparison").hidden=false;$("fullComparison").href=group.comparison;}
}
function anatomyEvidence(){
  const a=findAnatomy();if(!a)return;
  const extra={"标准DAPI":"已核验原始参照与亮度标准化显示","最终选定翅脉":`${a.branch_count} 条分支 / ${a.candidate}`,"参考坐标核验":"DAPI参考像素一致 / 未重新配准","已叠加基因":$("showGenes").checked?selectedGenes.map(g=>{const item=findGroup(state.wing,state.stage,g);return `${g}: ${item?`n=${item.n}`:"暂无数据"}`;}).join(" / ")||"未选择基因":"已隐藏 / 仅显示结构"};
  $("facts").insertAdjacentHTML("beforeend",Object.entries(extra).map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join(""));
  $("statusBadge").textContent="已冻结人工选定翅脉";
  $("reviewStatus").textContent="当前为最终选定的代表性翅脉，几何未改动。各基因的DAPI参照与该坐标一致；这说明坐标兼容，不代表已独立验证生物学配准准确度。"+ ($("showGenes").checked?" 基因配准仍保留实验性或待确认状态。":"");
  $("anatomyDownloads").innerHTML=Object.entries({"标准DAPI / 原图":a.assets.dapi_original,"标准DAPI / 亮度标准化显示":a.assets.dapi_display,"最终翅脉 / 透明PNG":a.assets.vein,"最终翅脉 / 灰度PNG":a.assets.vein_gray,"最终翅脉与DAPI叠加":a.assets.approved_overlay,"冻结选择记录":a.assets.manifest}).map(([label,url])=>`<a href="${url}" download>${label}</a>`).join("");
  $("fullComparison").hidden=false;$("fullComparison").href=a.assets.approved_overlay;
  $("anatomyLegend").innerHTML=($("showVein").checked?`<span><i style="background:${$("veinColor").value}"></i>翅脉</span>`:"")+ ($("showGenes").checked?selectedGenes.filter(g=>findGroup(state.wing,state.stage,g)).map(g=>`<span><i style="background:${colorFor(g)}"></i>${esc(g)} / n=${findGroup(state.wing,state.stage,g).n}</span>`).join(""):"");
}
function geneControls(){
  $("geneLayers").replaceChildren();
  for(const gene of data.genes){
    const group=findGroup(state.wing,state.stage,gene),a=findAnatomy(),compatible=!!group&&a?.coordinate_frame===group.coordinate_frame;
    const row=document.createElement("div");row.className="gene-control";
    row.innerHTML=`<label class="check"><input type="checkbox" ${selectedGenes.includes(gene)?"checked":""} ${compatible?"":"disabled"}><span>${esc(gene)}</span><small>${group?`n=${group.n}${compatible?"":" / 坐标不同"}`:"暂无数据"}</small></label><div><input aria-label="${esc(gene)} 颜色" type="color" value="${colorFor(gene)}" ${compatible?"":"disabled"}><input aria-label="${esc(gene)} 透明度" type="range" min="0" max="1" step="0.05" value="${geneOpacity[gene]??.9}" ${compatible?"":"disabled"}></div>`;
    row.querySelector("input[type=checkbox]").onchange=e=>{
      if(e.target.checked&&selectedGenes.length>=4){e.target.checked=false;notice("为保持可读性，最多叠加四个基因。");return;}
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
    el.innerHTML=`<h3>${esc(s.label)}</h3><a href="${s.aligned}" target="_blank" rel="noopener"><img loading="lazy" src="${s.aligned}" alt="Historical aligned color input"></a><div class="source-meta">${esc(s.method)}${s.tre!==null?`<br>选择过程TRE： ${s.tre.toFixed(1)} 像素 / 不作为独立验证`:""}<br>${esc(sourceStageCN(s.stage))}</div><div class="source-actions"><a href="${s.aligned}" target="_blank" rel="noopener">对齐彩图预览</a><a href="${s.intensity}" target="_blank" rel="noopener">强度来源彩图预览</a></div>`;$("tiles").append(el);
  }
  if(group.sources.length>12){const el=document.createElement("div");el.className="source-actions";el.innerHTML=`<button id="prevPage" ${sourcePage===0?"disabled":""}>上一页</button><span>${start+1}-${Math.min(start+12,group.n)} 共 ${group.n}</span><button id="nextPage" ${start+12>=group.n?"disabled":""}>下一页</button>`;$("tiles").append(el);$("prevPage").onclick=()=>{sourcePage--;render();};$("nextPage").onclick=()=>{sourcePage++;render();};}
}
function render(){
  if(!data)return;
  const token=++generation;tiles=[];$("tiles").replaceChildren();$("tiles").className=`tiles ${state.mode==="styles"?"methods":state.mode}`;
  $("gene").value=state.gene;$("variant").value=state.variant;$("representation").value=state.representation;$("variant").disabled=state.mode==="methods";
  document.querySelectorAll("[data-wing],[data-stage],[data-mode]").forEach(b=>{const key=b.dataset.wing?"wing":b.dataset.stage?"stage":"mode";b.classList.toggle("active",b.dataset[key]===state[key]);});
  const group=findGroup();geneEvidence(group);syncUrl();
  $("viewTitle").textContent=`${anatomical?"标准解剖结构":state.gene} / ${labelWing(state.wing)}`;
  $("viewSubtitle").textContent=`阶段 ${state.stage} / ${anatomical?"DAPI、最终翅脉与可选基因":names[state.variant]}`;
  if(anatomical){
    geneControls();anatomyEvidence();
    for(const id of ["dapiBrightness","veinOpacity","opacity","blendMode"])$(id).disabled=state.mode==="styles";
    notice($("showGenes").checked?"各基因来自独立样本集合。颜色区分基因，透明度表达相对信号；混色不能证明细胞共表达。":"当前仅显示解剖结构。开启基因层可叠加表达；最终翅脉不会重新拟合。");
    $("modeNote").textContent=state.mode==="styles"?"比较三种固定显示预设；图层可见性、颜色和缩放保持同步。":"DAPI、翅脉与基因层使用已核验的参考坐标；显示控件不改写科研数组。";
    if(state.mode==="styles")Object.entries(presets).forEach(([key,p])=>makeAnatomyTile(state.wing,state.stage,p.label,token,key));
    else if(state.mode==="stages")["0.5","0.75","1.0"].forEach(s=>makeAnatomyTile(state.wing,s,`阶段 ${s}`,token));
    else if(state.mode==="wings")["forewing","hindwing"].forEach(w=>makeAnatomyTile(w,state.stage,labelWing(w),token));
    else makeAnatomyTile(state.wing,state.stage,"标准DAPI / 翅脉 / 基因表达",token);
  }else{
    $("legend").hidden=state.mode==="sources";$("legend").querySelector("i").style.background=state.representation==="points"?"linear-gradient(90deg,#47bdff,#ffae36,#fff187)":"";
    $("legend").querySelector("span").textContent=state.representation==="points"?"基线P50 / P85点位等级":"相对表达 / 0–1";
    $("modeNote").textContent="仅显示表达，不叠加DAPI或翅脉；缩放与平移同步。";
    notice(!group?"该组合尚无表达结果，不生成替代信号。":group.n===1?"当前为单图代表性结果，不代表群体共识。":state.mode==="stages"||state.mode==="wings"?"用于比较空间分布，不用于比较绝对mRNA数量；不同ROI增益和批次存在差异。":"");
    if(state.mode==="sources"&&group)sources(group);
    else if(state.mode==="methods")data.variants.forEach(v=>makeExpressionTile(group,v,names[v],token));
    else if(state.mode==="stages")["0.5","0.75","1.0"].forEach(s=>makeExpressionTile(findGroup(state.wing,s),state.variant,`阶段 ${s}`,token));
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
  $("addGene").onclick=()=>{if(!selectedGenes.includes(state.gene)&&findGroup()){if(selectedGenes.length>=4){notice("最多选择四个基因。");return;}selectedGenes.push(state.gene);$("showGenes").checked=true;render();}};
}
$("zoomIn").onclick=()=>setZoom(state.zoom*1.2);$("zoomOut").onclick=()=>setZoom(state.zoom/1.2);$("fit").onclick=()=>{state.zoom=1;state.x=state.y=0;redraw();};
$("fullscreen").onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else $("tiles").requestFullscreen().catch(()=>notice("当前无法进入全屏。"));};
$("copyLink").onclick=async()=>{try{await navigator.clipboard.writeText(location.href);$("copyLink").textContent="链接已复制";setTimeout(()=>$("copyLink").textContent="复制当前视图链接",1500);}catch{notice("可复制浏览器地址分享当前视图。");}};
new ResizeObserver(redraw).observe($("tiles"));window.addEventListener("resize",redraw);
async function init(){
  if(!anatomical&&state.mode==="multi"){const p=new URLSearchParams(query);p.set("mode","single");p.set("showGenes","1");location.replace('/zh/anatomy/?'+p);return;}
  const response=await fetch('/api/atlas/catalog',{cache:"no-store"});if(!response.ok)throw new Error("图谱归档暂不可用");data=await response.json();data.anatomy_cases??=[];
  if(!data.genes.includes(state.gene))state.gene=data.genes[0];if(!["forewing","hindwing"].includes(state.wing))state.wing="hindwing";
  if(!["0.5","0.75","1.0"].includes(state.stage))state.stage="0.5";if(!data.variants.includes(state.variant))state.variant="early_soft_q20";
  if(!["heat","points"].includes(state.representation))state.representation="heat";
  if(!(anatomical?["single","styles","stages","wings"]:["single","methods","stages","wings","sources"]).includes(state.mode))state.mode="single";
  selectedGenes=[...new Set(selectedGenes)].filter(g=>data.genes.includes(g)).slice(0,4);
  $("gene").innerHTML=data.genes.map(g=>`<option value="${esc(g)}">${esc(g)}${data.processed_genes.includes(g)?"":" / 等待数据"}</option>`).join("");$("gene").disabled=false;
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
  $("datasetCount").textContent=`${data.genes.length} 个已登记 / ${data.processed_genes.length} 个已提取 / ${data.samples} 个图像单元${anatomical?" / 六组最终翅脉":""}`;
  $("reportLink").href=data.report;$("experimentLink").href=data.experiments;$("manifestLink").href=data.manifest;updateOutputs();render();
}
init().catch(error=>{notice(error.message);$("viewTitle").textContent="归档暂不可用";});
