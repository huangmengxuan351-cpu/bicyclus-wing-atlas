async function loadModelState() {
  const output = document.getElementById("modelState");
  try {
    const response = await fetch("/api/atlas/catalog", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    output.textContent = `${payload.genes.length} registered genes / ${payload.processed_genes.length} with extracted data / ${payload.samples} inputs. Models load only on the predictor page.`;
    output.classList.add("ready");
  } catch (error) {
    output.textContent = `Model status unavailable: ${error.message}`;
  }
}

loadModelState();
