/**
 * frontend/app.js
 *
 * BASE_URL is resolved at runtime from the environment.
 * Set it in your .env file — never hardcode a deployed URL here.
 *
 * Vite:              VITE_API_BASE_URL=https://your-api.example.com
 * Create React App:  REACT_APP_API_BASE_URL=https://your-api.example.com
 * Plain HTML/JS:     set window.API_BASE_URL before this script loads
 */

const BASE_URL =
  (typeof window !== "undefined" && window.API_BASE_URL) ||
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE_URL) ||
  (typeof process !== "undefined" && process.env?.REACT_APP_API_BASE_URL) ||
  "http://127.0.0.1:5000";   // local dev fallback only


// ── API helpers ───────────────────────────────────────────────────────────────

/**
 * POST /predict
 * @param {{ temperature: number, pressure: number, humidity: number, vibration_level: number }} features
 * @returns {Promise<{ prediction: number, confidence: number, label: string }>}
 */
export async function getPrediction(features) {
  const response = await fetch(`${BASE_URL}/predict`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(features),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `API error ${response.status}`);
  }

  return response.json();
}

/**
 * GET /metrics
 * @returns {Promise<{ accuracy: number, f1: number, precision: number, recall: number }>}
 */
export async function getMetrics() {
  const response = await fetch(`${BASE_URL}/metrics`);

  if (!response.ok) {
    throw new Error(`Failed to load metrics: ${response.status}`);
  }

  return response.json();
}

/**
 * GET /recommendations  (POST with features)
 * @param {object} features
 * @returns {Promise<{ recommendations: string[] }>}
 */
export async function getRecommendations(features) {
  const response = await fetch(`${BASE_URL}/recommendations`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(features),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `API error ${response.status}`);
  }

  return response.json();
}

/**
 * GET /health
 * @returns {Promise<boolean>}
 */
export async function healthCheck() {
  try {
    const response = await fetch(`${BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
