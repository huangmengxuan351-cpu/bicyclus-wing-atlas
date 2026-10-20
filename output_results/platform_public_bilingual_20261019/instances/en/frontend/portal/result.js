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
    row.innerHTML = `<span>${label}</span><i><b style="width:${value * 100}%"></b></i><strong>${percent(value)}</strong>`;
    target.appendChild(row);
  }
}

function atlasUrl(wing, stage, embedded = false) {
  const query = new URLSearchParams({ wing: WING_PATH[wing], stage: STAGE_PATH[stage] });
  if (embedded) query.set("embedded", "1");
  return `/en/atlas/?${query}`;
}

function resultUrl(wing, stage, source = "user") {
  const query = new URLSearchParams({ run_id: runId, wing, stage, source });
  return `/en/result/?${query}`;
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
    button.innerHTML = `<span>${candidate.wing === "forewing" ? "Fore wing" : "Hind wing"} · Stage ${candidate.stage}</span><strong>${percent(candidate.probability)}</strong>`;
    button.addEventListener("click", () => window.location.assign(resultUrl(candidate.wing, candidate.stage)));
    container.appendChild(button);
  }
}

async function loadAtlas(wing, stage) {
  const response = await fetch("/api/atlas/catalog", { cache: "no-store" });
  if (!response.ok) throw new Error(`Atlas unavailable (${response.status})`);
  const catalog = await response.json();
  const group = catalog.groups.find(g => g.wing === wing && g.stage === stage);
  if (!group) throw new Error("No extracted atlas for this wing and stage.");
  document.getElementById("standardAtlasImage").src = group.dapi;
  document.getElementById("standardAtlasLabel").textContent = `${wing === "forewing" ? "Fore wing" : "Hind wing"} · Stage ${stage}`;
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
  document.getElementById("backToAlignment").href = `/en/predict/?run_id=${encodeURIComponent(runId)}&stay=1`;
  document.getElementById("decisionPill").textContent = automatic ? "model-selected atlas" : "review required";
  document.getElementById("decisionPill").className = `decision-pill ${automatic ? "auto" : "review"}`;
  document.getElementById("resultTitle").textContent = selectedWing && selectedStage
    ? `${selectedWing === "forewing" ? "Fore wing" : "Hind wing"} · Stage ${selectedStage}`
    : "Choose the matching atlas";
  document.getElementById("resultSubtitle").textContent = !selectedWing || !selectedStage
    ? "The model did not meet automatic atlas-selection thresholds. Confirm a combination below."
    : selectionSource === "user"
      ? "The displayed atlas was selected by the user; the original model result remains unchanged."
      : "The displayed atlas follows the highest-confidence model result.";

  renderBars("familyBars", prediction.family_probabilities || {}, WINGS);
  const stageFamily = selectedWing || modelWing;
  document.getElementById("stageProbabilityTitle").textContent = `Stage probability for ${stageFamily || "selected wing"}`;
  renderBars("stageBars", prediction.conditional_stage_probabilities?.[stageFamily] || {}, STAGES);

  document.getElementById("modelPrediction").textContent = `${modelWing || "-"} · Stage ${modelStage || "-"}`;
  document.getElementById("displayedAtlas").textContent = selectedWing && selectedStage ? `${selectedWing} · Stage ${selectedStage}` : "Awaiting selection";
  document.getElementById("alignmentStatus").textContent = `${percent(summary.alignment_confidence)} · ${prep.manual_alignment ? "manual revision" : "automatic"}`;
  document.getElementById("wingMargin").textContent = percent(summary.wing_type_margin);
  document.getElementById("stageMargin").textContent = percent(summary.stage_margin);
  document.getElementById("alignmentConfidence").textContent = percent(summary.alignment_confidence);
  document.getElementById("technicalReviewReason").textContent = summary.review_reason || "None";

  const grayAsset = prep.gray_asset || "gray.png";
  const alignedGrayAsset = prep.selected_aligned_gray_asset || "selected_aligned_gray.png";
  const alignedMaskAsset = prep.selected_aligned_mask_asset || "selected_aligned_mask.png";
  document.getElementById("uploadedImage").src = assetUrl(grayAsset);
  document.getElementById("alignedImage").src = assetUrl(alignedGrayAsset);
  document.getElementById("alignedMask").src = assetUrl(alignedMaskAsset);
  document.getElementById("alignmentAssetLabel").textContent = prep.manual_alignment ? "User-adjusted aligned evidence" : "Automatically aligned evidence";

  const chooser = document.getElementById("reviewChooser");
  if (!automatic && (!requestedWing || !requestedStage)) {
    chooser.hidden = false;
    document.getElementById("reviewReason").textContent = summary.review_reason || "The model did not meet automatic atlas-selection thresholds.";
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
    showError("The result URL does not include a run_id.");
    return;
  }
  try {
    const response = await fetch(`/api/result/${encodeURIComponent(runId)}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Prediction result not found (${response.status}).`);
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
