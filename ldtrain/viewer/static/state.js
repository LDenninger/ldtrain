// Global view state, the run index built from /api/tree, the selection model, run colors and URL state.

import { debounce, loadPref } from './util.js';

const LIVE_WINDOW_S = 600;
export const PALETTE_SIZE = 12;

export const state = {
    //--- run tree ---
    root: '',
    tree: null,
    runs: new Map(),        // path -> run node
    folders: new Map(),     // path -> folder node
    treeLoaded: false,

    //--- selection ---
    selected: new Map(),    // path -> palette slot
    focused: null,
    hidden: new Set(),
    lastClicked: null,

    //--- view ---
    metricFilter: '',
    collapsed: new Set(),   // metric groups and section ids
    smoothing: 0.6,
    logy: new Set(),
    xRange: null,           // [min, max] synced zoom, null = full range
    syncZoom: true,
    mediaStep: null,        // null = follow the latest step
    treeFilter: '',
    collapsedFolders: new Set(loadPref('collapsedFolders', [])),
    theme: loadPref('theme', 'auto'),
};

// ---- events ----

const listeners = new Map();

/** Subscribe to a change kind: tree, selection, visibility, smoothing, filter, zoom, media, theme. */
export function on(kind, callback) {
    if (!listeners.has(kind)) listeners.set(kind, []);
    listeners.get(kind).push(callback);
}

export function emit(kind, detail) {
    for (const callback of listeners.get(kind) ?? []) callback(detail);
    if (kind !== 'tree' && kind !== 'theme') writeUrlDebounced();
}

// ---- run index ----

export function indexTree(root) {
    state.tree = root;
    state.runs.clear();
    state.folders.clear();
    const visit = (node, parentPath, depth) => {
        node.parentPath = parentPath;
        node.depth = depth;
        if (node.is_run) {
            state.runs.set(node.path, node);
            return;
        }
        state.folders.set(node.path, node);
        for (const child of node.children) visit(child, node.path, depth + 1);
    };
    visit(root, null, -1);
    state.treeLoaded = true;
}

export function findNode(path) {
    return state.runs.get(path) ?? state.folders.get(path) ?? null;
}

export function runsBelow(node) {
    return node.is_run ? [node] : node.children.flatMap(runsBelow);
}

export function isLive(run) {
    return run.mtime !== null && Date.now() / 1000 - run.mtime < LIVE_WINDOW_S;
}

// ---- selection ----

/** Selected run paths, ordered by palette slot. */
export function selectedPaths() {
    return [...state.selected.entries()].sort((a, b) => a[1] - b[1]).map(([path]) => path);
}

/** Selected run paths in draw order: slot order with the focused run last (drawn on top). */
export function drawOrder() {
    const paths = selectedPaths().filter((path) => path !== state.focused);
    if (state.focused) paths.push(state.focused);
    return paths;
}

function lowestFreeSlot() {
    const used = new Set(state.selected.values());
    let slot = 0;
    while (used.has(slot)) slot++;
    return slot;
}

export function selectRun(path) {
    if (state.selected.has(path)) return;
    state.selected.set(path, lowestFreeSlot());
    if (!state.focused) state.focused = path;
}

export function deselectRun(path) {
    if (!state.selected.delete(path)) return;
    state.hidden.delete(path);
    if (state.focused === path) state.focused = selectedPaths()[0] ?? null;
}

export function toggleRun(path) {
    if (state.selected.has(path)) deselectRun(path);
    else selectRun(path);
}

export function focusRun(path) {
    selectRun(path);
    state.focused = path;
}

export function soloRun(path) {
    state.selected.clear();
    state.hidden.clear();
    focusRun(path);
}

export function clearSelection() {
    state.selected.clear();
    state.hidden.clear();
    state.focused = null;
}

/** Select every run below a folder, or deselect them all when all are selected already. */
export function toggleFolder(folder) {
    const runs = runsBelow(folder);
    const allSelected = runs.every((run) => state.selected.has(run.path));
    for (const run of runs) {
        if (allSelected) deselectRun(run.path);
        else selectRun(run.path);
    }
}

export function toggleHidden(path) {
    if (state.hidden.has(path)) state.hidden.delete(path);
    else state.hidden.add(path);
}

/** Drop selected paths the tree does not know (deleted runs, stale links). */
export function pruneSelection() {
    for (const path of [...state.selected.keys()]) {
        if (!state.runs.has(path)) deselectRun(path);
    }
    for (const path of [...state.hidden]) {
        if (!state.selected.has(path)) state.hidden.delete(path);
    }
}

// ---- run colors ----

let paletteColors = [];

/** Re-read the run palette from the CSS variables, after load and on every theme change. */
export function readPalette() {
    const style = getComputedStyle(document.documentElement);
    paletteColors = Array.from({ length: PALETTE_SIZE }, (_, ii) => style.getPropertyValue(`--run-${ii}`).trim());
}

export function slotColor(slot) {
    return paletteColors[slot % PALETTE_SIZE];
}

export function slotDash(slot) {
    return slot % PALETTE_SIZE >= 8 || slot >= PALETTE_SIZE ? [6, 3] : null;
}

export function runColor(path) {
    return slotColor(state.selected.get(path) ?? 0);
}

export function runDash(path) {
    return slotDash(state.selected.get(path) ?? 0);
}

export function withAlpha(hexColor, alpha) {
    const value = parseInt(hexColor.slice(1), 16);
    return `rgba(${value >> 16},${(value >> 8) & 255},${value & 255},${alpha})`;
}

// ---- url state ----

function splitList(value) {
    return value ? value.split(',').filter(Boolean) : [];
}

export function readUrl() {
    const params = new URLSearchParams(location.search);
    for (const entry of splitList(params.get('runs'))) {
        const colon = entry.lastIndexOf(':');
        const slot = colon > 0 ? Number(entry.slice(colon + 1)) : NaN;
        if (Number.isInteger(slot) && slot >= 0) state.selected.set(entry.slice(0, colon), slot);
        else selectRun(entry);
    }
    const focus = params.get('focus');
    state.focused = focus && state.selected.has(focus) ? focus : (selectedPaths()[0] ?? null);
    state.hidden = new Set(splitList(params.get('hide')).filter((path) => state.selected.has(path)));
    state.metricFilter = params.get('m') ?? '';
    state.collapsed = new Set(splitList(params.get('collapsed')));
    const smoothing = Number(params.get('smooth'));
    if (params.has('smooth') && smoothing >= 0 && smoothing <= 0.99) state.smoothing = smoothing;
    state.logy = new Set(splitList(params.get('logy')));
    const range = (params.get('x') ?? '').split('..').map(Number);
    if (range.length === 2 && range.every(Number.isFinite) && range[0] < range[1]) state.xRange = range;
    const step = Number(params.get('step'));
    if (params.has('step') && Number.isInteger(step)) state.mediaStep = step;
}

function writeUrl() {
    const params = new URLSearchParams();
    if (state.selected.size) params.set('runs', selectedPaths().map((path) => `${path}:${state.selected.get(path)}`).join(','));
    if (state.focused) params.set('focus', state.focused);
    if (state.hidden.size) params.set('hide', [...state.hidden].join(','));
    if (state.metricFilter) params.set('m', state.metricFilter);
    if (state.collapsed.size) params.set('collapsed', [...state.collapsed].join(','));
    if (state.smoothing !== 0.6) params.set('smooth', state.smoothing);
    if (state.logy.size) params.set('logy', [...state.logy].join(','));
    if (state.xRange) params.set('x', state.xRange.map((value) => Math.round(value)).join('..'));
    if (state.mediaStep !== null) params.set('step', state.mediaStep);
    const query = params.toString().replace(/%2F/g, '/').replace(/%2C/g, ',').replace(/%3A/g, ':');
    try {
        history.replaceState(null, '', query ? `?${query}` : location.pathname);
    } catch {
        // replaceState can throw in sandboxed frames; the URL just does not update
    }
}

const writeUrlDebounced = debounce(writeUrl, 300);
