const form = document.getElementById("predictForm");
const imageInput = document.getElementById("imageInput");
const maskInput = document.getElementById("maskInput");
const imageName = document.getElementById("imageName");
const maskName = document.getElementById("maskName");
const pageParams = new URLSearchParams(window.location.search);
const embeddedAtlasMode = pageParams.get("embedded") === "1";
const stayOnPredictor = pageParams.get("stay") === "1";
const restoredRunId = pageParams.get("run_id") || "";
const atlasTargetOrigin = pageParams.get("atlas_origin") || "http://127.0.0.1:8817";

if (embeddedAtlasMode) {
  document.body.classList.add("embedded");
}
const modelStatus = document.getElementById("modelStatus");
const modelInfo = document.getElementById("modelInfo");
const summaryTitle = document.getElementById("summaryTitle");
const decisionBadge = document.getElementById("decisionBadge");
const jsonResult = document.getElementById("jsonResult");
const historyBody = document.getElementById("historyBody");
const refreshHistory = document.getElementById("refreshHistory");
const alignmentBadge = document.getElementById("alignmentBadge");
const alignmentFill = document.getElementById("alignmentFill");
const alignmentConfidence = document.getElementById("alignmentConfidence");
const alignmentTitle = document.getElementById("alignmentTitle");
const alignmentNote = document.getElementById("alignmentNote");
const toggleManual = document.getElementById("toggleManual");
const manualPanel = document.getElementById("manualPanel");
const manualFamily = document.getElementById("manualFamily");
const manualGrayPreview = document.getElementById("manualGrayPreview");
const manualMaskPreview = document.getElementById("manualMaskPreview");
const referenceContourPreview = document.getElementById("referenceContourPreview");
const rotationControl = document.getElementById("rotationControl");
const scaleControl = document.getElementById("scaleControl");
const translateXControl = document.getElementById("translateXControl");
const translateYControl = document.getElementById("translateYControl");
const flipControl = document.getElementById("flipControl");
const applyManual = document.getElementById("applyManual");
const resetManual = document.getElementById("resetManual");
const manualActionStatus = document.getElementById("manualActionStatus");
const atlasDecisionPanel = document.getElementById("atlasDecisionPanel");
const atlasDecisionTitle = document.getElementById("atlasDecisionTitle");
const atlasDecisionNote = document.getElementById("atlasDecisionNote");
const atlasCandidateGrid = document.getElementById("atlasCandidateGrid");

let currentPayload = null;
let manualBaseGeometry = {
  width: 2048,
  height: 2048,
  centroid: [1024, 1024],
};

const previewIds = {
  gray: "grayPreview",
  mask: "maskPreview",
  alignedGray: "alignedGrayPreview",
  alignedMask: "alignedMaskPreview",
};

function percent(value) {
  const number = Number(value || 0);
  return `${(number * 100).toFixed(1)}%`;
}

function renderBars(targetId, values, order) {
  const target = document.getElementById(targetId);
  target.innerHTML = "";
  order.forEach((label) => {
    const value = Number(values?.[label] || 0);
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span>${label}</span>
      <div class="track"><div class="fill" style="width:${Math.max(0, Math.min(100, value * 100))}%"></div></div>
      <strong>${percent(value)}</strong>
    `;
    target.appendChild(row);
  });
}

function setDecision(decision, reason) {
  decisionBadge.textContent = decision || "idle";
  decisionBadge.className = "decision";
  if (decision === "auto_candidate") {
    decisionBadge.classList.add("auto");
  } else if (decision) {
    decisionBadge.classList.add("review");
    if (reason) {
      decisionBadge.title = reason;
    }
  }
}

function bestEntry(values) {
  const entries = Object.entries(values || {}).map(([label, value]) => [label, Number(value || 0)]);
  entries.sort((a, b) => b[1] - a[1]);
  return entries[0] || ["", 0];
}

function imageUrl(runId, asset) {
  return `/api/file/${encodeURIComponent(runId)}/${encodeURIComponent(asset)}?t=${Date.now()}`;
}

function evidenceText(values) {
  const [label, value] = bestEntry(values || {});
  return label ? `${label} ${percent(value)}` : "-";
}

function resultUrl(runId, family, stage, source) {
  const query = new URLSearchParams({ run_id: runId, wing: family, stage, source });
  return `/result/?${query}`;
}

function renderAtlasCandidates(payload) {
  const candidates = [];
  for (const family of ["forewing", "hindwing"]) {
    for (const stage of ["0.5", "0.75", "1.0"]) {
      candidates.push({
        family,
        stage,
        probability: Number(payload.joint_probabilities?.[family]?.[stage] || 0),
      });
    }
  }
  candidates.sort((a, b) => b.probability - a.probability);
  atlasCandidateGrid.innerHTML = "";
  for (const candidate of candidates) {
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = `<span>${candidate.family === "forewing" ? "Fore wing" : "Hind wing"} · Stage ${candidate.stage}</span><strong>${percent(candidate.probability)}</strong>`;
    button.addEventListener("click", () => {
      window.location.assign(resultUrl(payload.run_id, candidate.family, candidate.stage, "user"));
    });
    atlasCandidateGrid.appendChild(button);
  }
}

function updateAtlasDestination(payload) {
  if (embeddedAtlasMode) return;
  const summary = payload.summary || {};
  const [bestFamily] = bestEntry(payload.family_probabilities || {});
  const selectedFamily = payload.preprocessing?.selected_family_for_stage || summary.stage_family || bestFamily;
  const [selectedStage] = bestEntry(payload.conditional_stage_probabilities?.[selectedFamily] || {});
  const automatic = summary.decision !== "needs_review";

  if (automatic && selectedFamily && selectedStage && !stayOnPredictor) {
    atlasDecisionPanel.hidden = false;
    atlasDecisionTitle.textContent = "Prediction complete. Opening the matching atlas...";
    atlasDecisionNote.textContent = `${selectedFamily === "forewing" ? "Fore wing" : "Hind wing"} · Stage ${selectedStage}`;
    atlasCandidateGrid.innerHTML = "";
    window.setTimeout(() => {
      window.location.assign(resultUrl(payload.run_id, selectedFamily, selectedStage, "model"));
    }, 650);
    return;
  }

  atlasDecisionPanel.hidden = false;
  atlasDecisionTitle.textContent = automatic
    ? "Prediction restored. Choose an atlas when ready."
    : "Review required before opening the atlas";
  atlasDecisionNote.textContent = automatic
    ? "The automatic result is retained while you inspect or refine alignment."
    : (summary.review_reason || "Select one of the six model-ranked wing and stage combinations.");
  renderAtlasCandidates(payload);
}

function notifyAtlas(payload) {
  if (!embeddedAtlasMode || window.parent === window) return;
  const summary = payload.summary || {};
  const [bestFamily, bestFamilyProb] = bestEntry(payload.family_probabilities || {});
  const selectedFamily = payload.preprocessing?.selected_family_for_stage || summary.stage_family || bestFamily;
  const selectedFamilyProb = Number(payload.family_probabilities?.[selectedFamily] || bestFamilyProb || 0);
  const selectedStageMap = payload.conditional_stage_probabilities?.[selectedFamily] || {};
  const [selectedStage, selectedStageProb] = bestEntry(selectedStageMap);
  if (!selectedFamily || !selectedStage) return;

  window.parent.postMessage({
    type: "dapi-stage-prediction-result",
    source: "dapi-stage-predictor",
    wing_type: selectedFamily,
    wing_confidence: selectedFamilyProb,
    stage: selectedStage,
    stage_confidence: Number(selectedStageProb || 0),
    decision: summary.decision || "",
    review_reason: summary.review_reason || "",
    run_id: payload.run_id || "",
  }, atlasTargetOrigin);
}

function updateManualTransform() {
  const rotation = Number(rotationControl.value || 0);
  const scale = Number(scaleControl.value || 1);
  const tx = Number(translateXControl.value || 0);
  const ty = Number(translateYControl.value || 0);
  const flip = flipControl.checked ? -1 : 1;
  document.getElementById("rotationValue").textContent = `${rotation.toFixed(0)}°`;
  document.getElementById("scaleValue").textContent = `${scale.toFixed(2)}×`;
  document.getElementById("translateXValue").textContent = `${tx.toFixed(0)} px`;
  document.getElementById("translateYValue").textContent = `${ty.toFixed(0)} px`;
  const shell = manualGrayPreview.parentElement;
  const naturalWidth = Number(manualGrayPreview.naturalWidth || manualBaseGeometry.width || 2048);
  const naturalHeight = Number(manualGrayPreview.naturalHeight || manualBaseGeometry.height || 2048);
  const shellWidth = Number(shell?.clientWidth || naturalWidth);
  const shellHeight = Number(shell?.clientHeight || naturalHeight);
  const previewFitScale = Math.min(shellWidth / naturalWidth, shellHeight / naturalHeight);
  const contentWidth = naturalWidth * previewFitScale;
  const contentHeight = naturalHeight * previewFitScale;
  const offsetX = (shellWidth - contentWidth) / 2;
  const offsetY = (shellHeight - contentHeight) / 2;
  const centroid = manualBaseGeometry.centroid || [naturalWidth / 2, naturalHeight / 2];
  const fullWidth = Number(manualBaseGeometry.width || naturalWidth);
  const fullHeight = Number(manualBaseGeometry.height || naturalHeight);
  const fullToPreviewX = contentWidth / fullWidth;
  const fullToPreviewY = contentHeight / fullHeight;
  const originX = offsetX + Number(centroid[0]) * fullToPreviewX;
  const originY = offsetY + Number(centroid[1]) * fullToPreviewY;
  const transformOrigin = `${originX}px ${originY}px`;
  const transform = `translate(${tx * fullToPreviewX}px, ${ty * fullToPreviewY}px) rotate(${rotation}deg) scale(${scale * flip}, ${scale})`;
  manualGrayPreview.style.transformOrigin = transformOrigin;
  manualMaskPreview.style.transformOrigin = transformOrigin;
  manualGrayPreview.style.transform = transform;
  manualMaskPreview.style.transform = transform;
}

function resetManualControls() {
  rotationControl.value = "0";
  scaleControl.value = "1";
  translateXControl.value = "0";
  translateYControl.value = "0";
  flipControl.checked = false;
  manualActionStatus.textContent = "";
  manualActionStatus.className = "manual-action-status";
  updateManualTransform();
}

function setManualPreview(payload, family) {
  const runId = payload?.run_id;
  if (!runId) return;
  const prep = payload.preprocessing || {};
  const familyAlignment = prep.family_alignments?.[family] || {};
  const grayAsset = familyAlignment.aligned_gray_asset;
  const maskAsset = familyAlignment.aligned_mask_asset;
  manualBaseGeometry = {
    width: Number(familyAlignment.canvas_width || 2048),
    height: Number(familyAlignment.canvas_height || 2048),
    centroid: familyAlignment.mask_centroid || [1024, 1024],
  };
  manualGrayPreview.src = imageUrl(runId, grayAsset || `${family}_aligned_gray.png`);
  manualMaskPreview.src = imageUrl(runId, maskAsset || `${family}_aligned_mask.png`);
  referenceContourPreview.src = "/predict/manual_anatomical_rotation_cross_overlay.png";
  manualGrayPreview.onload = updateManualTransform;
  manualMaskPreview.onload = updateManualTransform;
}

function renderAlignment(payload, selectedFamily) {
  const prep = payload.preprocessing || {};
  const alignment = prep.selected_alignment || prep.family_alignments?.[selectedFamily]?.diagnostics || {};
  const confidence = Number(alignment.confidence || 0);
  const low = Boolean(alignment.needs_manual_alignment);
  alignmentTitle.textContent = low
    ? "Low automatic-alignment confidence; manual alignment recommended"
    : "Automatic alignment is stable";
  alignmentBadge.textContent = low ? "review alignment" : "stable";
  alignmentBadge.className = `alignment-pill ${low ? "low" : "stable"}`;
  alignmentFill.style.width = `${Math.max(0, Math.min(100, confidence * 100))}%`;
  alignmentConfidence.textContent = percent(confidence);
  const retained = alignment.transform_retained_fraction;
  alignmentNote.textContent = `Reference overlap ${percent(alignment.reference_overlap)} · mask sanity ${percent(alignment.mask_sanity)}${retained == null ? "" : ` · retained ${percent(retained)}`} · threshold ${percent(alignment.threshold || 0.62)}`;
  manualFamily.value = selectedFamily || "forewing";
  setManualPreview(payload, manualFamily.value);
  if (low && !prep.manual_alignment) {
    manualPanel.hidden = false;
    toggleManual.textContent = "Hide alignment controls";
  } else if (!prep.manual_alignment) {
    manualPanel.hidden = true;
    toggleManual.textContent = "Manual alignment";
  }
}

function renderEvidence(payload, selectedFamily) {
  const stage = payload.stage_payloads?.[selectedFamily] || {};
  document.getElementById("evidenceTitle").textContent = `${selectedFamily || "selected wing"} · fused ${evidenceText(stage.probabilities)}`;
  document.getElementById("modelEvidence").textContent = evidenceText(stage.p_model_probabilities);
  document.getElementById("ordinalEvidence").textContent = stage.ordinal_stage_pred
    ? `${stage.ordinal_stage_pred} ${percent(stage.ordinal_stage_conf)}`
    : "-";
  document.getElementById("templateEvidence").textContent = evidenceText(stage.template_stage_probabilities);
  document.getElementById("referenceEvidence").textContent = `${evidenceText(stage.reference_stage_probabilities)} · d=${Number(stage.reference_best_distance || 0).toFixed(3)}`;
}

function renderResult(payload) {
  currentPayload = payload;
  const summary = payload.summary || {};
  const runId = payload.run_id;
  const [bestFamily, bestFamilyProb] = bestEntry(payload.family_probabilities || {});
  const selectedFamily = payload.preprocessing?.selected_family_for_stage || summary.stage_family || bestFamily;
  const selectedFamilyProb = Number(payload.family_probabilities?.[selectedFamily] || bestFamilyProb || 0);
  const selectedStageMap = payload.conditional_stage_probabilities?.[selectedFamily] || {};
  const [selectedStage, selectedStageProb] = bestEntry(selectedStageMap);
  summaryTitle.textContent = `${selectedFamily || summary.wing_type_pred || "-"} ${percent(selectedFamilyProb || summary.wing_type_conf)} · stage ${selectedStage || summary.stage_pred || "-"} ${percent(selectedStageProb || summary.stage_conf)}`;
  setDecision(summary.decision, summary.review_reason);

  renderBars("familyBars", payload.family_probabilities || {}, ["forewing", "hindwing"]);
  const title = document.getElementById("selectedStageTitle");
  title.textContent = selectedFamily ? `Stage for ${selectedFamily}` : "Stage for Selected Wing";
  renderBars("selectedStageBars", selectedStageMap, ["0.5", "0.75", "1.0"]);

  const prep = payload.preprocessing || {};
  document.getElementById(previewIds.gray).src = imageUrl(runId, prep.gray_asset || "gray.png");
  document.getElementById(previewIds.mask).src = imageUrl(runId, prep.mask_asset || "mask.png");
  document.getElementById(previewIds.alignedGray).src = imageUrl(runId, prep.selected_aligned_gray_asset || "selected_aligned_gray.png");
  document.getElementById(previewIds.alignedMask).src = imageUrl(runId, prep.selected_aligned_mask_asset || "selected_aligned_mask.png");

  renderAlignment(payload, selectedFamily);
  renderEvidence(payload, selectedFamily);

  jsonResult.textContent = JSON.stringify(payload, null, 2);
  notifyAtlas(payload);
  updateAtlasDestination(payload);
}

async function loadModelInfo() {
  const response = await fetch("/api/model-info");
  const payload = await response.json();
  const models = payload.models || {};
  const allLoaded = Object.values(models).every((item) => item.loaded);
  modelStatus.textContent = allLoaded ? "ready" : "check models";
  modelStatus.className = `status-pill ${allLoaded ? "ready" : ""}`;
  modelInfo.innerHTML = `
    <div><strong>Profile:</strong> ${payload.stage_profile || ""}</div>
    <div><strong>Type:</strong> ${models.wing_type?.bundle_kind || ""}</div>
    <div><strong>Hind:</strong> ${models.hindwing_stage?.best_candidate || models.hindwing_stage?.bundle_kind || ""}</div>
    <div><strong>Fore:</strong> ${models.forewing_stage?.best_candidate || models.forewing_stage?.bundle_kind || ""}</div>
  `;
}

async function loadHistory() {
  const response = await fetch("/api/history");
  const payload = await response.json();
  historyBody.innerHTML = "";
  (payload.items || []).forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.timestamp || ""}</td>
      <td>${row.original_filename || ""}</td>
      <td>${row.wing_type_pred || ""} ${percent(row.wing_type_conf)}</td>
      <td>${row.stage_pred || ""} ${percent(row.stage_conf)}</td>
      <td>${row.decision || ""}</td>
    `;
    historyBody.appendChild(tr);
  });
}

async function loadRestoredResult(runId) {
  if (!runId) return;
  const response = await fetch(`/api/result/${encodeURIComponent(runId)}`, { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Saved prediction could not be restored");
  renderResult(payload);
  summaryTitle.scrollIntoView({ behavior: "smooth", block: "start" });
}

imageInput.addEventListener("change", () => {
  imageName.textContent = imageInput.files?.[0]?.name || "PNG / JPG / TIFF";
});

maskInput.addEventListener("change", () => {
  maskName.textContent = maskInput.files?.[0]?.name || "No file selected";
});

refreshHistory.addEventListener("click", loadHistory);

toggleManual.addEventListener("click", () => {
  manualPanel.hidden = !manualPanel.hidden;
  toggleManual.textContent = manualPanel.hidden ? "Manual alignment" : "Hide alignment controls";
});

manualFamily.addEventListener("change", () => {
  if (currentPayload) setManualPreview(currentPayload, manualFamily.value);
  resetManualControls();
});

[rotationControl, scaleControl, translateXControl, translateYControl, flipControl].forEach((control) => {
  control.addEventListener("input", updateManualTransform);
  control.addEventListener("change", updateManualTransform);
});

resetManual.addEventListener("click", resetManualControls);

applyManual.addEventListener("click", async () => {
  if (!currentPayload?.run_id) return;
  applyManual.disabled = true;
  applyManual.textContent = "Re-predicting...";
  manualActionStatus.textContent = "Submitting alignment and recomputing wing family and stage...";
  manualActionStatus.className = "manual-action-status working";
  try {
    const response = await fetch("/api/re-align", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        run_id: currentPayload.run_id,
        family: manualFamily.value,
        rotation_deg: Number(rotationControl.value),
        scale: Number(scaleControl.value),
        translate_x: Number(translateXControl.value),
        translate_y: Number(translateYControl.value),
        flip_x: flipControl.checked,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Manual alignment failed");
    renderResult(payload);
    updateManualTransform();
    manualPanel.hidden = false;
    toggleManual.textContent = "Hide alignment controls";
    manualActionStatus.textContent = "Alignment applied. Predictions have been updated.";
    manualActionStatus.className = "manual-action-status success";
    await loadHistory();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    manualActionStatus.textContent = `Alignment not applied: ${message}`;
    manualActionStatus.className = "manual-action-status error";
  } finally {
    applyManual.disabled = false;
    applyManual.textContent = "Apply and re-predict";
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = form.querySelector("button");
  button.disabled = true;
  button.textContent = "Running...";
  try {
    const data = new FormData();
    data.append("image", imageInput.files[0]);
    if (maskInput.files?.[0]) {
      data.append("mask", maskInput.files[0]);
    }
    const response = await fetch("/api/predict", { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Prediction failed");
    }
    renderResult(payload);
    await loadHistory();
  } catch (error) {
    summaryTitle.textContent = "Prediction failed";
    setDecision("error", String(error));
    jsonResult.textContent = String(error);
  } finally {
    button.disabled = false;
    button.innerHTML = "<span>Run Prediction</span>";
  }
});

loadModelInfo().catch((error) => {
  modelStatus.textContent = "model error";
  modelInfo.textContent = String(error);
});

window.addEventListener("resize", updateManualTransform);
loadHistory().catch(() => {});
loadRestoredResult(restoredRunId).catch((error) => {
  summaryTitle.textContent = "Saved prediction unavailable";
  setDecision("error", String(error));
});
