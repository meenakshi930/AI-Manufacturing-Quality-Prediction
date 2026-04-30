const API_BASE = "https://ai-manufacturing-quality-prediction.onrender.com";

let featureChart, historyChart;
let historyData = [], historyLabels = [];

function updateFeatureGraph(data) {
  const values = [
    data.temperature,
    data.pressure,
    data.humidity,
    data.vibration_level
  ];

  if (featureChart) featureChart.destroy();

  featureChart = new Chart(document.getElementById("featureChart"), {
    type: "bar",
    data: {
      labels: ["Temp", "Pressure", "Humidity", "Vibration"],
      datasets: [{ data: values }]
    }
  });
}

document.getElementById("prediction-form").addEventListener("submit", async function(e) {
  e.preventDefault();

  const data = {};
  new FormData(this).forEach((v,k)=> data[k]=Number(v));

  document.getElementById("loader").style.display = "block";

  try {
    const res = await fetch(API_BASE + "/predict", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(data)
    });

    const result = await res.json();
    console.log(result);

    const prediction = result.label || (result.prediction===1?"Defect":"Pass");
    const confidence = result.confidence ?? 0.5;

    document.getElementById("prediction-text").innerText =
      `${prediction} (${(confidence*100).toFixed(1)}%)`;

    // Risk logic (UI only)
    let risk="Low Risk", cls="low";
    if(prediction==="Defect" && confidence>0.8){risk="High Risk";cls="high";}
    else if(prediction==="Defect"){risk="Medium Risk";cls="medium";}

    const badge = document.getElementById("risk-badge");
    badge.innerText = risk;
    badge.className = cls;

    document.getElementById("confidence-bar").style.width =
      `${confidence*100}%`;

    // Insight
    document.getElementById("insight-box").innerText =
      prediction==="Defect"
      ? "⚠️ Possible machine failure detected. Check vibration & temperature."
      : "✅ System stable and operating normally.";

    document.getElementById("timestamp").innerText =
      "Updated: "+new Date().toLocaleTimeString();

    updateFeatureGraph(data);

    // History graph
    historyData.push(confidence);
    historyLabels.push("Run "+historyData.length);

    if(historyData.length>5){
      historyData.shift();
      historyLabels.shift();
    }

    if(historyChart) historyChart.destroy();

    historyChart = new Chart(document.getElementById("historyChart"), {
      type:"line",
      data:{labels:historyLabels,datasets:[{data:historyData}]}
    });

  } catch (err) {
    document.getElementById("prediction-text").innerText =
      "⚠️ Backend not responding";
  }

  document.getElementById("loader").style.display = "none";
});

// Load default graph
window.onload=()=>{
  updateFeatureGraph({
    temperature:55,
    pressure:210,
    humidity:0.6,
    vibration_level:105
  });
};
