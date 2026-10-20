async function loadModelState() {
  const output = document.getElementById("modelState");
  try {
    const response = await fetch("/api/atlas/catalog", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    output.textContent = `${payload.genes.length} 个已登记基因 / ${payload.processed_genes.length} 个已有表达结果 / ${payload.samples} 个图像单元。分类模型仅在预测页面加载。`;
    output.classList.add("ready");
  } catch (error) {
    output.textContent = `状态暂不可用： ${error.message}`;
  }
}

loadModelState();
