const predictionForm = document.querySelector("#prediction-form");
const batchForm = document.querySelector("#batch-form");
const riskPill = document.querySelector("#risk-pill");
const defectLabel = document.querySelector("#defect-label");
const probability = document.querySelector("#probability");
const recommendations = document.querySelector("#recommendations");

// 🔹 Change this if backend URL changes
const BASE_URL = "http://127.0.0.1:5000";

// Convert form data → JSON
function formToPayload(form) {
  const data = new FormData(form);
  return Object.fromEntries(
    [...data.entries()].map(([key, value]) => [key, Number(value)])
  );
}

// 🔹 API function (clean version of your second code)
async function sendData(endpoint, data, isFormData = false) {
  const options = {
    method: "POST",
  };

  if (isFormData) {
    options.body = data;
  } else {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(data);
  }

  const response = await fetch(`${BASE_URL}${endpoint}`, options);

  let result;
  try {
    result = await response.json();
  } catch {
    result = { detail: "Invalid server response" };
  }

  if (!response.ok) {
    throw new Error(result.detail || "Request failed");
  }

  return result;
}

// Render success
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

// Render error
function renderError(message) {
  riskPill.textContent = "Error";
  riskPill.className = "risk-pill High";
  defectLabel.textContent = "Prediction failed";
  probability.textContent = message;
}

// 🔹 Single prediction
predictionForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  try {
    const payload = formToPayload(predictionForm);
    const result = await sendData("/predict", payload);
    renderPrediction(result);
  } catch (error) {
    renderError(error.message);
  }
});

// 🔹 Batch prediction
batchForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  try {
    const response = await fetch(`${BASE_URL}/predict/batch`, {
      method: "POST",
      body: new FormData(batchForm),
    });

    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.detail || "Batch prediction failed.");
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = "quality_predictions.csv";
    link.click();

    URL.revokeObjectURL(url);
  } catch (error) {
    renderError(error.message);
  }
});
