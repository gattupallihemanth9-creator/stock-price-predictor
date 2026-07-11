/**
 * main.js – StockVision shared utilities
 * Loaded on every page. Page-specific logic lives inline in each template.
 */

"use strict";

// ── Number / currency helpers ────────────────────────────────────────────────

/**
 * Format a dollar amount with 2 decimal places.
 * @param {number} value
 * @returns {string}  e.g. "$1,234.56"
 */
function formatPrice(value) {
  if (value == null || isNaN(value)) return "–";
  return "$" + parseFloat(value).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

/**
 * Format a large number into a compact human-readable string.
 * @param {number} value
 * @returns {string}  e.g. "$1.23T", "$456.78B", "$12.34M"
 */
function formatLargeNumber(value) {
  if (value == null || isNaN(value)) return "–";
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9)  return `$${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6)  return `$${(value / 1e6).toFixed(2)}M`;
  return `$${value.toLocaleString()}`;
}

/**
 * Return a CSS class name based on whether a numeric change is positive or negative.
 * @param {number} change
 * @returns {"text-green"|"text-red"|""}
 */
function changeClass(change) {
  if (change > 0) return "text-green";
  if (change < 0) return "text-red";
  return "";
}

/**
 * Format a percentage change with sign and % symbol.
 * @param {number} pct
 * @returns {string}  e.g. "+3.45%" or "-1.20%"
 */
function formatPct(pct) {
  if (pct == null || isNaN(pct)) return "–";
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${parseFloat(pct).toFixed(2)}%`;
}

// ── DOM helpers ──────────────────────────────────────────────────────────────

/**
 * Show a loading spinner inside a container element.
 * @param {HTMLElement} el
 */
function showSpinner(el) {
  el.innerHTML = `
    <div class="loader-wrap">
      <div class="spinner"></div>
      <p>Loading…</p>
    </div>`;
}

/**
 * Show an error message inside a container element.
 * @param {HTMLElement} el
 * @param {string} message
 */
function showError(el, message) {
  el.innerHTML = `<div class="error-box"><i class="fas fa-exclamation-circle"></i> ${message}</div>`;
}

// ── Toast notifications ──────────────────────────────────────────────────────

let _toastTimeout = null;

/**
 * Show a brief toast message at the bottom of the screen.
 * @param {string} message
 * @param {"success"|"error"|"info"} [type="info"]
 */
function showToast(message, type = "info") {
  // Remove existing toast
  const existing = document.getElementById("sv-toast");
  if (existing) existing.remove();
  if (_toastTimeout) clearTimeout(_toastTimeout);

  const colors = { success: "#4ade80", error: "#f87171", info: "#38bdf8" };
  const icons  = { success: "fa-check-circle", error: "fa-times-circle", info: "fa-info-circle" };

  const toast = document.createElement("div");
  toast.id = "sv-toast";
  toast.style.cssText = `
    position: fixed;
    bottom: 1.5rem;
    left: 50%;
    transform: translateX(-50%) translateY(20px);
    background: #1e293b;
    border: 1px solid ${colors[type]};
    color: ${colors[type]};
    padding: 0.7rem 1.4rem;
    border-radius: 50px;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    z-index: 9999;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    opacity: 0;
    transition: opacity 0.3s ease, transform 0.3s ease;
  `;
  toast.innerHTML = `<i class="fas ${icons[type]}"></i> ${message}`;
  document.body.appendChild(toast);

  // Animate in
  requestAnimationFrame(() => {
    toast.style.opacity = "1";
    toast.style.transform = "translateX(-50%) translateY(0)";
  });

  // Auto-dismiss after 3 seconds
  _toastTimeout = setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(-50%) translateY(20px)";
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// ── Fetch wrapper ────────────────────────────────────────────────────────────

/**
 * Wrapper around fetch that throws a descriptive error on non-2xx responses.
 * @param {string} url
 * @param {RequestInit} [options]
 * @returns {Promise<any>}  Parsed JSON response
 */
async function apiFetch(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  if (data.error) {
    throw new Error(data.error);
  }
  return data;
}

// ── Watchlist helpers (shared across pages) ──────────────────────────────────

/**
 * Add a ticker to the watchlist and show a toast.
 * @param {string} ticker
 */
async function addToWatchlist(ticker) {
  try {
    const data = await apiFetch("/api/watchlist/add", {
      method: "POST",
      body: JSON.stringify({ ticker: ticker.toUpperCase() }),
    });
    showToast(data.message || `${ticker} added to watchlist`, "success");
    return true;
  } catch (err) {
    showToast(err.message, "error");
    return false;
  }
}

/**
 * Remove a ticker from the watchlist and show a toast.
 * @param {string} ticker
 */
async function removeFromWatchlist(ticker) {
  try {
    const data = await apiFetch("/api/watchlist/remove", {
      method: "DELETE",
      body: JSON.stringify({ ticker: ticker.toUpperCase() }),
    });
    showToast(data.message || `${ticker} removed`, "info");
    return true;
  } catch (err) {
    showToast(err.message, "error");
    return false;
  }
}

// ── Date helpers ─────────────────────────────────────────────────────────────

/**
 * Return the last N items from an array.
 * @param {Array} arr
 * @param {number} n  Pass 0 for all items.
 * @returns {Array}
 */
function lastN(arr, n) {
  if (!n || n <= 0) return arr;
  return arr.slice(-n);
}

// ── Export to global scope (no module bundler used) ──────────────────────────
window.SV = {
  formatPrice,
  formatLargeNumber,
  changeClass,
  formatPct,
  showSpinner,
  showError,
  showToast,
  apiFetch,
  addToWatchlist,
  removeFromWatchlist,
  lastN,
};
