const predictionForm = document.querySelector("#prediction-form");
const batchForm = document.querySelector("#batch-form");
const riskPill = document.querySelector("#risk-pill");
const defectLabel = document.querySelector("#defect-label");
const probability = document.querySelector("#probability");
const recommendations = document.querySelector("#recommendations");

// Convert form data → JSON payload
function formToPayload(form) {
  const data = new FormData(form);
  return Object.fromEntries(
    [...data.entries()].map(([key, value]) => [key, Number(value)])
  );
}

// 🔹 Centralized API call (your second code merged properly)
async function sendData(data) {
  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Prediction failed");
    }

    return result;
  } catch (error) {
    throw error;
  }
}

// Render success result
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

// 🔹 Single prediction form
predictionForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = formToPayload(predictionForm);

  try {
    const result = await sendData(payload); // using merged function
    renderPrediction(result);
  } catch (error) {
    renderError(error.message);
  }
});

// 🔹 Batch prediction (file upload)
batchForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  try {
    const response = await fetch("http://127.0.0.1:5000/predict/batch", {
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
});
