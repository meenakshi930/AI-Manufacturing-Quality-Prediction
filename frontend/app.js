const API_BASE = "https://ai-manufacturing-quality-prediction.onrender.com";

document.getElementById("prediction-form").addEventListener("submit", async function(e) {
  e.preventDefault();

  const data = {};
  new FormData(this).forEach((v, k) => data[k] = Number(v));

  document.getElementById("prediction-text").innerText = "Processing...";

  try {
    const res = await fetch(API_BASE + "/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data)
    });

    const result = await res.json();

    const prediction = result.label || (result.prediction === 1 ? "Defect" : "Pass");
    const confidence = result.confidence;

    // 🎯 Risk logic
    let risk = "";
    let riskClass = "";

    if (confidence < 0.6) {
      risk = "Low Risk";
      riskClass = "low";
    } else if (confidence < 0.8) {
      risk = "Medium Risk";
      riskClass = "medium";
    } else {
      risk = "High Risk";
      riskClass = "high";
    }

    // UI update
    document.getElementById("prediction-text").innerText =
      `Prediction: ${prediction} (${(confidence * 100).toFixed(1)}%)`;

    const badge = document.getElementById("risk-badge");
    badge.innerText = risk;
    badge.className = riskClass;

    document.getElementById("confidence-bar").style.width =
      `${confidence * 100}%`;

    // 💡 Recommendations
    let rec = "";

    if (prediction === "Defect") {
      rec = "⚠️ High defect probability detected. Check machine calibration, reduce vibration & inspect pressure levels.";
    } else {
      rec = "✅ System stable. Maintain current operating conditions and monitor periodically.";
    }

    document.getElementById("recommendations").innerText = rec;

  } catch (err) {
    document.getElementById("prediction-text").innerText = "Error connecting to API";
  }
});
