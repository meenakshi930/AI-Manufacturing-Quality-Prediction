const API_BASE = "https://ai-manufacturing-quality-prediction.onrender.com";

let featureChart, historyChart;
let historyData = [], historyLabels = [];
let totalRuns = 0, defectCount = 0, confSum = 0;

// ── Chart defaults ────────────────────────────────────────────────────────────
const CHART_DEFAULTS = {
  color: "#94a3b8",
  font: { family: "'Space Mono', monospace", size: 11 }
};

Chart.defaults.color = CHART_DEFAULTS.color;
Chart.defaults.font  = CHART_DEFAULTS.font;

const ACCENT   = "#00d2c8";
const ACCENT2  = "#00a8e8";
const DEFECT_C = "#f43f5e";

// ── Feature chart ─────────────────────────────────────────────────────────────
function updateFeatureGraph(data) {
  if (featureChart) featureChart.destroy();

  const labels = [
    "Prod. Vol", "Prod. Cost", "Supplier Q", "Del. Delay",
    "Defect R", "Quality S", "Maint. H", "Downtime",
    "Inv. Turn", "Stockout", "Worker P", "Safety",
    "Energy C", "Energy E", "Add. Time", "Add. Cost"
  ];

  const values = [
    data.ProductionVolume, data.ProductionCost, data.SupplierQuality, data.DeliveryDelay,
    data.DefectRate, data.QualityScore, data.MaintenanceHours, data.DowntimePercentage,
    data.InventoryTurnover, data.StockoutRate, data.WorkerProductivity, data.SafetyIncidents,
    data.EnergyConsumption, data.EnergyEfficiency, data.AdditiveProcessTime, data.AdditiveMaterialCost
  ];

  featureChart = new Chart(document.getElementById("featureChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Value",
        data: values,
        backgroundColor: values.map((_, i) =>
          i % 2 === 0 ? "rgba(0,210,200,0.5)" : "rgba(0,168,232,0.4)"
        ),
        borderColor: ACCENT,
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0f1626",
          borderColor: "rgba(0,210,200,0.3)",
          borderWidth: 1,
          titleFont: { family: "'Space Mono', monospace", size: 10 },
          bodyFont:  { family: "'Space Mono', monospace", size: 11 },
        }
      },
      scales: {
        x: { grid: { color: "rgba(0,210,200,0.05)" }, ticks: { maxRotation: 45 } },
        y: { grid: { color: "rgba(0,210,200,0.05)" } }
      }
    }
  });
}

// ── Metrics update ────────────────────────────────────────────────────────────
function updateMetrics(prediction, confidence) {
  totalRuns++;
  confSum += confidence;
  if (prediction === "Defect") defectCount++;

  document.getElementById("total-runs").textContent    = totalRuns;
  document.getElementById("defect-count").textContent  = defectCount;
  document.getElementById("avg-confidence").textContent = (confSum / totalRuns * 100).toFixed(0) + "%";
  document.getElementById("last-result").textContent   = prediction;
}

// ── Form submit ───────────────────────────────────────────────────────────────
document.getElementById("prediction-form").addEventListener("submit", async function(e) {
  e.preventDefault();

  const data = {};
  new FormData(this).forEach((v, k) => data[k] = Number(v));

  // Loading state
  const btn = document.getElementById("predict-btn");
  document.getElementById("btn-text").style.display   = "none";
  document.getElementById("btn-loader").style.display = "inline";
  btn.disabled = true;

  try {
    // ── 1. Prediction ──────────────────────────────────────────────────────
    const res = await fetch(API_BASE + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showError("API Error " + res.status + ": " + (err.error || "Unknown error"));
      return;
    }

    const result  = await res.json();
    const prediction = result.label || (result.prediction === 1 ? "Defect" : "Pass");
    const confidence = result.confidence ?? 0.5;

    // ── 2. Update result panel ─────────────────────────────────────────────
    document.getElementById("result-idle").style.display   = "none";
    document.getElementById("result-active").style.display = "block";

    const verdictEl = document.getElementById("verdict-text");
    verdictEl.textContent = prediction;
    verdictEl.className   = "verdict " + (prediction === "Defect" ? "defect" : "pass");

    let riskText = "LOW RISK", riskCls = "low";
    if (prediction === "Defect" && confidence > 0.8) { riskText = "HIGH RISK";   riskCls = "high"; }
    else if (prediction === "Defect")                { riskText = "MEDIUM RISK"; riskCls = "medium"; }

    const chipEl = document.getElementById("risk-chip");
    chipEl.textContent = riskText;
    chipEl.className   = "risk-chip " + riskCls;

    document.getElementById("conf-value").textContent      = (confidence * 100).toFixed(1) + "%";
    document.getElementById("conf-fill").style.width       = (confidence * 100) + "%";
    document.getElementById("conf-fill").style.background  =
      prediction === "Defect"
        ? "linear-gradient(90deg, #f43f5e, #fb7185)"
        : "linear-gradient(90deg, #00d2c8, #00a8e8)";

    document.getElementById("timestamp").textContent =
      "Last updated " + new Date().toLocaleTimeString();

    // ── 3. Insight box ─────────────────────────────────────────────────────
    const insightEl = document.getElementById("insight-box");
    insightEl.style.display = "block";
    insightEl.className     = "insight-box" + (prediction === "Defect" ? " defect" : "");
    insightEl.textContent   = prediction === "Defect"
      ? "⚠  Defect detected — review recommendations below before continuing production."
      : "✓  No defect predicted — system operating within acceptable parameters.";

    // ── 4. Update counters and charts ──────────────────────────────────────
    updateMetrics(prediction, confidence);

    updateFeatureGraph(data);

    historyData.push(parseFloat((confidence * 100).toFixed(1)));
    historyLabels.push("#" + totalRuns);

    if (historyChart) historyChart.destroy();
    historyChart = new Chart(document.getElementById("historyChart"), {
      type: "line",
      data: {
        labels: historyLabels,
        datasets: [{
          label: "Confidence %",
          data: historyData,
          borderColor: ACCENT,
          backgroundColor: "rgba(0,210,200,0.08)",
          fill: true,
          tension: 0.4,
          pointRadius: 5,
          pointBackgroundColor: historyData.map((_, i) =>
            historyData[i] > 80 ? DEFECT_C : ACCENT
          ),
          pointBorderColor: "#0f1626",
          pointBorderWidth: 2
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#0f1626",
            borderColor: "rgba(0,210,200,0.3)",
            borderWidth: 1,
            titleFont: { family: "'Space Mono', monospace", size: 10 },
            bodyFont:  { family: "'Space Mono', monospace", size: 11 },
          }
        },
        scales: {
          x: { grid: { color: "rgba(0,210,200,0.05)" } },
          y: { min: 0, max: 100, grid: { color: "rgba(0,210,200,0.05)" } }
        }
      }
    });

    // ── 5. Recommendations ────────────────────────────────────────────────
    try {
      const recRes = await fetch(API_BASE + "/recommendations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });

      if (recRes.ok) {
        const recData = await recRes.json();
        const recs = recData.recommendations || [];
        const recsPanel = document.getElementById("recs-panel");
        const recsList  = document.getElementById("recs-list");

        if (recs.length > 0) {
          recsList.innerHTML = recs.map(r => `<li>${r}</li>`).join("");
          recsPanel.style.display = "block";
        }
      }
    } catch {
      // recommendations non-critical — skip silently
    }

  } catch (err) {
    showError("Connection failed — is the backend running?");
    console.error(err);
  } finally {
    document.getElementById("btn-text").style.display   = "inline";
    document.getElementById("btn-loader").style.display = "none";
    btn.disabled = false;
  }
});

function showError(msg) {
  document.getElementById("result-idle").style.display   = "none";
  document.getElementById("result-active").style.display = "block";
  document.getElementById("verdict-text").textContent    = "Error";
  document.getElementById("verdict-text").className      = "verdict defect";
  document.getElementById("risk-chip").textContent       = "";
  document.getElementById("insight-box").style.display   = "block";
  document.getElementById("insight-box").className       = "insight-box defect";
  document.getElementById("insight-box").textContent     = msg;
}

// ── Dark / light toggle ───────────────────────────────────────────────────────
function toggleDark() {
  document.documentElement.classList.toggle("light");
}

// ── Init chart on load ────────────────────────────────────────────────────────
window.onload = () => {
  updateFeatureGraph({
    ProductionVolume: 920, ProductionCost: 18400,
    SupplierQuality: 84,  DeliveryDelay: 4,
    DefectRate: 4.3,      QualityScore: 69,
    MaintenanceHours: 3,  DowntimePercentage: 4.1,
    InventoryTurnover: 3.4, StockoutRate: 8.2,
    WorkerProductivity: 83, SafetyIncidents: 5,
    EnergyConsumption: 4300, EnergyEfficiency: 0.18,
    AdditiveProcessTime: 8.1, AdditiveMaterialCost: 420
  });
};
