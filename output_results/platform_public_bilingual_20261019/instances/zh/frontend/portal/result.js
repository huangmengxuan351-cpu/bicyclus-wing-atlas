
function wingCN(value){return ({forewing:"前翅",hindwing:"后翅"})[value]||value||"所选翅型"}
function statusCN(value){return ({auto_candidate:"自动候选",needs_review:"需要复核",ready:"就绪",stable:"稳定",idle:"等待输入",loading:"加载中",error:"错误",manual:"人工",automatic:"自动"})[value]||value}
function diagnosticCN(value){let text=String(value||"");for(const [key,label] of Object.entries({alignment_confidence_below_threshold:"配准置信度低于阈值","wing_type_probability_below_0.62":"翅型概率低于0.62","wing_type_margin_below_0.06":"翅型前两类概率差低于0.06","stage_probability_below_0.48":"阶段概率低于0.48","stage_margin_below_0.04":"阶段前两类概率差低于0.04",ordinal_conflict:"序数证据冲突",stage_bundle_decision:"阶段模型判定"})){text=text.replaceAll(key,label)}return text}
function sourceStageCN(value){return value==="pre_historical_alignment_export"?"历史配准导出前的强度来源":value.replaceAll("_"," ")}
const WINGS = ["forewing", "hindwing"];
const STAGES = ["0.5", "0.75", "1.0"];
const WING_PATH = { forewing: "fore_wing", hindwing: "hind_wing" };
const STAGE_PATH = { "0.5": "stage_0_5", "0.75": "stage_0_75", "1.0": "stage_1_0" };

const params = new URLSearchParams(window.location.search);
const runId = params.get("run_id") || "";
const requestedWing = normalizeWing(params.get("wing"));
const requestedStage = normalizeStage(params.get("stage"));
const selectionSource = params.get("source") || "model";

let prediction = null;
let selectedWing = requestedWing;
let selectedStage = requestedStage;

function normalizeWing(value) {
  const key = String(value || "").toLowerCase().replaceAll("_", "");
  if (key === "forewing") return "forewing";
  if (key === "hindwing") return "hindwing";
  return "";
}

function normalizeStage(value) {
  const key = String(value || "").replace("stage_", "").replaceAll("_", ".");
  return STAGES.includes(key) ? key : "";
}

function percent(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "-";
}

function bestEntry(values) {
  return Object.entries(values || {}).sort((a, b) => Number(b[1]) - Number(a[1]))[0] || ["", 0];
}

function assetUrl(asset) {
  return `/api/file/${encodeURIComponent(runId)}/${encodeURIComponent(asset)}?v=${Date.now()}`;
}

function renderBars(targetId, values, labels) {
  const target = document.getElementById(targetId);
  target.innerHTML = "";
  for (const label of labels) {
    const value = Math.max(0, Math.min(1, Number(values?.[label] || 0)));
    const row = document.createElement("div");
    row.className = "probability-row";
    row.innerHTML = `<span>${wingCN(label)}</span><i><b style="width:${value * 100}%"></b></i><strong>${percent(value)}</strong>`;
    target.appendChild(row);
  }
}

function atlasUrl(wing, stage, embedded = false) {
  const query = new URLSearchParams({ wing: WING_PATH[wing], stage: STAGE_PATH[stage] });
  if (embedded) query.set("embedded", "1");
  return `/zh/atlas/?${query}`;
}

function resultUrl(wing, stage, source = "user") {
  const query = new URLSearchParams({ run_id: runId, wing, stage, source });
  return `/zh/result/?${query}`;
}

function isAutomaticCandidate(payload) {
  return payload?.summary?.decision !== "needs_review";
}

function showError(message) {
  document.getElementById("loadingState").hidden = true;
  document.getElementById("errorState").hidden = false;
  document.getElementById("errorMessage").textContent = message;
}

function renderCandidates() {
  const container = document.getElementById("jointCandidates");
  const candidates = [];
  for (const wing of WINGS) {
    for (const stage of STAGES) {
      candidates.push({ wing, stage, probability: Number(prediction.joint_probabilities?.[wing]?.[stage] || 0) });
    }
  }
  candidates.sort((a, b) => b.probability - a.probability);
  container.innerHTML = "";
  for (const candidate of candidates) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "candidate-button";
    button.innerHTML = `<span>${candidate.wing === "forewing" ? "前翅" : "后翅"} · 阶段 ${candidate.stage}</span><strong>${percent(candidate.probability)}</strong>`;
    button.addEventListener("click", () => window.location.assign(resultUrl(candidate.wing, candidate.stage)));
    container.appendChild(button);
  }
}

async function loadAtlas(wing, stage) {
  const response = await fetch("/api/atlas/catalog", { cache: "no-store" });
  if (!response.ok) throw new Error(`图谱暂不可用（${response.status})`);
  const catalog = await response.json();
  const group = catalog.groups.find(g => g.wing === wing && g.stage === stage);
  if (!group) throw new Error("该翅型与阶段尚无表达图谱。");
  document.getElementById("standardAtlasImage").src = group.dapi;
  document.getElementById("standardAtlasLabel").textContent = `${wing === "forewing" ? "前翅" : "后翅"} · 阶段 ${stage}`;
  const frame = document.getElementById("atlasFrame");
  frame.src = atlasUrl(wing, stage, true);
  frame.hidden = false;
  document.getElementById("atlasPlaceholder").hidden = true;
  document.getElementById("fullAtlasLink").href = atlasUrl(wing, stage);
  document.getElementById("expandAtlasLink").href = atlasUrl(wing, stage);
}

function setAlignmentView(mode) {
  const image = document.getElementById("alignedImage");
  const mask = document.getElementById("alignedMask");
  image.hidden = mode === "mask";
  mask.hidden = mode === "image";
  mask.classList.toggle("mask-only", mode === "mask");
  document.querySelectorAll("[data-alignment-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.alignmentView === mode);
  });
}

function renderPrediction() {
  const summary = prediction.summary || {};
  const prep = prediction.preprocessing || {};
  const [modelWing] = bestEntry(prediction.family_probabilities);
  const modelStageMap = prediction.conditional_stage_probabilities?.[modelWing] || {};
  const [modelStage] = bestEntry(modelStageMap);
  const automatic = isAutomaticCandidate(prediction);

  if (!selectedWing || !selectedStage) {
    if (automatic) {
      selectedWing = normalizeWing(summary.stage_family || modelWing);
      selectedStage = normalizeStage(summary.stage_pred || bestEntry(prediction.conditional_stage_probabilities?.[selectedWing] || {})[0]);
    }
  }

  document.getElementById("loadingState").hidden = true;
  document.getElementById("resultContent").hidden = false;
  document.getElementById("backToAlignment").href = `/zh/predict/?run_id=${encodeURIComponent(runId)}&stay=1`;
  document.getElementById("decisionPill").textContent = automatic ? "模型选择图谱" : "需要复核";
  document.getElementById("decisionPill").className = `decision-pill ${automatic ? "auto" : "review"}`;
  document.getElementById("resultTitle").textContent = selectedWing && selectedStage
    ? `${selectedWing === "forewing" ? "前翅" : "后翅"} · 阶段 ${selectedStage}`
    : "选择对应图谱";
  document.getElementById("resultSubtitle").textContent = !selectedWing || !selectedStage
    ? "模型未达到自动选择图谱的阈值，请在下方确认组合。"
    : selectionSource === "user"
      ? "当前图谱由用户选择，原模型结果保持不变。"
      : "当前图谱对应模型置信度最高的结果。";

  renderBars("familyBars", prediction.family_probabilities || {}, WINGS);
  const stageFamily = selectedWing || modelWing;
  document.getElementById("stageProbabilityTitle").textContent = `阶段概率： ${wingCN(stageFamily) || "当前翅型"}`;
  renderBars("stageBars", prediction.conditional_stage_probabilities?.[stageFamily] || {}, STAGES);

  document.getElementById("modelPrediction").textContent = `${wingCN(modelWing) || "-"} · 阶段 ${modelStage || "-"}`;
  document.getElementById("displayedAtlas").textContent = selectedWing && selectedStage ? `${wingCN(selectedWing)} · 阶段 ${selectedStage}` : "等待选择";
  document.getElementById("alignmentStatus").textContent = `${percent(summary.alignment_confidence)} · ${prep.manual_alignment ? "人工修订" : "自动"}`;
  document.getElementById("wingMargin").textContent = percent(summary.wing_type_margin);
  document.getElementById("stageMargin").textContent = percent(summary.stage_margin);
  document.getElementById("alignmentConfidence").textContent = percent(summary.alignment_confidence);
  document.getElementById("technicalReviewReason").textContent = diagnosticCN(summary.review_reason) || "无";

  const grayAsset = prep.gray_asset || "gray.png";
  const alignedGrayAsset = prep.selected_aligned_gray_asset || "selected_aligned_gray.png";
  const alignedMaskAsset = prep.selected_aligned_mask_asset || "selected_aligned_mask.png";
  document.getElementById("uploadedImage").src = assetUrl(grayAsset);
  document.getElementById("alignedImage").src = assetUrl(alignedGrayAsset);
  document.getElementById("alignedMask").src = assetUrl(alignedMaskAsset);
  document.getElementById("alignmentAssetLabel").textContent = prep.manual_alignment ? "用户校准后的配准依据" : "自动配准依据";

  const chooser = document.getElementById("reviewChooser");
  if (!automatic && (!requestedWing || !requestedStage)) {
    chooser.hidden = false;
    document.getElementById("reviewReason").textContent = diagnosticCN(summary.review_reason) || "模型未达到自动选择图谱的阈值。";
    renderCandidates();
  } else {
    chooser.hidden = true;
  }

  if (selectedWing && selectedStage) {
    loadAtlas(selectedWing, selectedStage).catch((error) => showError(error.message));
  }
}

async function init() {
  if (!runId) {
    showError("结果链接未包含预测记录编号。");
    return;
  }
  try {
    const response = await fetch(`/api/result/${encodeURIComponent(runId)}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`未找到预测结果（${response.status}).`);
    prediction = await response.json();
    renderPrediction();
  } catch (error) {
    showError(error.message);
  }
}

document.querySelectorAll("[data-alignment-view]").forEach((button) => {
  button.addEventListener("click", () => setAlignmentView(button.dataset.alignmentView));
});

init();
