const predictionForm = document.querySelector("#prediction-form");
const batchForm = document.querySelector("#batch-form");
const riskPill = document.querySelector("#risk-pill");
const defectLabel = document.querySelector("#defect-label");
const probability = document.querySelector("#probability");
const recommendations = document.querySelector("#recommendations");

function formToPayload(form) {
  const data = new FormData(form);
  return Object.fromEntries([...data.entries()].map(([key, value]) => [key, Number(value)]));
}

function renderPrediction(result) {
  riskPill.textContent = `${result.risk_level} Risk`;
  riskPill.className = `risk-pill ${result.risk_level}`;
  defectLabel.textContent = result.defect_label;
  probability.textContent = `Defect probability: ${(result.defect_probability * 100).toFixed(1)}%`;
  recommendations.innerHTML = "";

  result.recommendations.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    recommendations.appendChild(li);
  });
}

function renderError(message) {
  riskPill.textContent = "Error";
  riskPill.className = "risk-pill High";
  defectLabel.textContent = "Prediction failed";
  probability.textContent = message;
}

predictionForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(formToPayload(predictionForm)),
  });

  const result = await response.json();
  if (!response.ok) {
    renderError(result.detail || "Unable to run prediction.");
    return;
  }

  renderPrediction(result);
});

batchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const response = await fetch("/predict/batch", {
    method: "POST",
    body: new FormData(batchForm),
  });

  if (!response.ok) {
    const result = await response.json();
    renderError(result.detail || "Batch prediction failed.");
    return;
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "quality_predictions.csv";
  link.click();
  URL.revokeObjectURL(url);
});
