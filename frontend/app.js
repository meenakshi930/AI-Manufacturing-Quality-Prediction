const API_BASE = "https://ai-manufacturing-quality-prediction.onrender.com";

// 🔹 Single Prediction
document.getElementById("prediction-form").addEventListener("submit", async function(e) {
  e.preventDefault();

  const data = {};
  new FormData(this).forEach((v, k) => data[k] = Number(v));

  document.getElementById("result").innerText = "Processing...";

  try {
    const res = await fetch(API_BASE + "/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(data)
    });

    const result = await res.json();

    document.getElementById("result").innerText =
      `Prediction: ${result.label} | Confidence: ${result.confidence}`;

  } catch (err) {
    document.getElementById("result").innerText = "Error connecting to API";
    console.error(err);
  }
});

// 🔹 Batch Prediction
document.getElementById("batch-form").addEventListener("submit", async function(e) {
  e.preventDefault();

  const formData = new FormData(this);

  try {
    const res = await fetch(API_BASE + "/predict-batch", {
      method: "POST",
      body: formData
    });

    const result = await res.json();

    alert("Batch prediction done!");
    console.log(result);

  } catch (err) {
    alert("Batch upload failed");
  }
});
