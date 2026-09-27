// Small helpers shared by every module: API access, formatting, DOM building, preferences.

// ---- api ----

/** A request reached the server and it answered with an error status. */
export class ApiError extends Error {
    constructor(status, detail) {
        super(detail);
        this.status = status;
    }
}

/** The server could not be reached at all. */
export class OfflineError extends Error {}

function buildQuery(params) {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
        if (value !== null && value !== undefined) query.set(key, value);
    }
    return query.toString();
}

async function request(endpoint, params) {
    let response;
    try {
        response = await fetch(`${endpoint}?${buildQuery(params)}`, { cache: 'no-store' });
    } catch (error) {
        throw new OfflineError(error.message);
    }
    if (!response.ok) {
        let detail = response.statusText;
        try {
            detail = (await response.json()).detail ?? detail;
        } catch {
            // body was not JSON, keep the status text
        }
        throw new ApiError(response.status, detail);
    }
    return response;
}

export async function fetchJson(endpoint, params = {}) {
    return (await request(endpoint, params)).json();
}

export async function fetchRunFileText(run, path) {
    return (await request('/api/file', { run, path })).text();
}

export function runFileUrl(run, path) {
    return `/api/file?${buildQuery({ run, path })}`;
}

// ---- formatting ----

/** Format a metric value with 4 significant digits, exponent form for tiny or huge values. */
export function formatValue(value) {
    if (value === null || value === undefined || !Number.isFinite(value)) return '—';
    const magnitude = Math.abs(value);
    if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e5)) return value.toExponential(2);
    return String(+value.toPrecision(4));
}

/** Format an iteration count compactly: 950, 13.4k, 2.1M. */
export function formatIteration(value) {
    if (value === null || value === undefined) return '';
    const magnitude = Math.abs(value);
    if (magnitude >= 1e6) return `${+(value / 1e6).toFixed(1)}M`;
    if (magnitude >= 1e3) return `${+(value / 1e3).toFixed(1)}k`;
    return String(+value.toFixed(2));
}

export function formatAge(seconds) {
    if (seconds < 60) return `${Math.round(seconds)} s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
    if (seconds < 86400) return `${Math.round(seconds / 3600)} h`;
    return `${Math.round(seconds / 86400)} d`;
}

export function shortName(path) {
    return path.slice(path.lastIndexOf('/') + 1);
}

// ---- dom ----

/** Create an element with a class and optional text content (never parsed as HTML). */
export function el(tag, className = '', text = null) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== null) element.textContent = text;
    return element;
}

export function swatch(color, dashed, className = 'sw') {
    const element = el('span', dashed ? `${className} dash` : className);
    element.style.setProperty('--c', color);
    return element;
}

export function isTyping() {
    const active = document.activeElement;
    if (!active) return false;
    if (active.tagName === 'TEXTAREA' || active.tagName === 'SELECT') return true;
    return active.tagName === 'INPUT' && active.type !== 'range' && active.type !== 'checkbox';
}

// ---- arrays ----

/** Index of the first element ≥ `target` in an ascending array. */
export function lowerBound(values, target) {
    let low = 0;
    let high = values.length;
    while (low < high) {
        const middle = (low + high) >> 1;
        if (values[middle] < target) low = middle + 1;
        else high = middle;
    }
    return low;
}

/** Index of the element closest to `target` in an ascending array, -1 when empty. */
export function nearestIndex(values, target) {
    if (!values.length) return -1;
    const index = lowerBound(values, target);
    if (index === 0) return 0;
    if (index === values.length) return values.length - 1;
    return target - values[index - 1] <= values[index] - target ? index - 1 : index;
}

/** Debiased exponential moving average (TensorBoard), over a dense array without nulls. */
export function smoothEma(values, weight) {
    if (weight <= 0) return values;
    const smoothed = new Array(values.length);
    let last = 0;
    for (let ii = 0; ii < values.length; ii++) {
        last = last * weight + (1 - weight) * values[ii];
        smoothed[ii] = last / (1 - Math.pow(weight, ii + 1));
    }
    return smoothed;
}

// ---- timing ----

export function debounce(callback, delay_ms) {
    let timer = null;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => callback(...args), delay_ms);
    };
}

// ---- preferences (localStorage, may be unavailable) ----

const PREFS_KEY = 'ldtrain-viewer';

export function loadPref(key, fallback) {
    try {
        const prefs = JSON.parse(localStorage.getItem(PREFS_KEY) ?? '{}');
        return key in prefs ? prefs[key] : fallback;
    } catch {
        return fallback;
    }
}

export function savePref(key, value) {
    try {
        const prefs = JSON.parse(localStorage.getItem(PREFS_KEY) ?? '{}');
        prefs[key] = value;
        localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
    } catch {
        // storage blocked: preferences just do not persist
    }
}
