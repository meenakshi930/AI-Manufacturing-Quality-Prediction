/**
 * app.js — Frontend API client
 *
 * BASE_URL is read from the environment at build time (Vite / CRA style)
 * or falls back to localhost for local development.
 *
 * Set in your .env file:
 *   VITE_API_BASE_URL=https://your-production-api.example.com
 * or for Create React App:
 *   REACT_APP_API_BASE_URL=https://your-production-api.example.com
 */

const BASE_URL =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE_URL) ||
  process.env.REACT_APP_API_BASE_URL ||
  "http://127.0.0.1:5000";  // local dev fallback only


/**
 * Send sensor readings to the prediction endpoint.
 * @param {{ temperature: number, pressure: number, humidity: number, vibration_level: number }} features
 * @returns {Promise<{ prediction: number, confidence: number, label: string }>}
 */
export async function getPrediction(features) {
  const response = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `API error: ${response.status}`);
  }

  return response.json();
}


/**
 * Fetch the model's evaluation metrics from the backend.
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
 * Health-check the API.
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
