// Entry point: boot, live polling, toolbar run chips, empty state, sections, theme and global keys.

import { forgetUnselected, pollMetrics, pollVisuals } from './data.js';
import { initLogConfig, pollLog } from './logconfig.js';
import { closeLightbox, cycleLightbox, initMedia, isLightboxOpen, renderMedia, stepMedia } from './media.js';
import { initMetrics, metricsDataChanged, resetZoom, setMetricFilter, toggleLogyAll, toggleLogyHovered } from './metrics.js';
import {
    deselectRun, emit, focusRun, indexTree, isLive, on, pruneSelection, readPalette, readUrl, runColor,
    runDash, selectedPaths, state, toggleHidden,
} from './state.js';
import { closeReport, initReport, isReportOpen, openReport } from './report.js';
import { focusTree, initTree } from './tree.js';
import { OfflineError, el, fetchJson, formatAge, icon, isTyping, loadPref, savePref, setCaret, shortName, swatch } from './util.js';

const POLL_MS = 5000;
const TREE_POLL_MS = 30000;
const BACKOFF_MS = [10000, 20000, 40000];
const SMOOTHING_STEP = 0.05;
const SIDEBAR_MIN_PX = 200;
const SIDEBAR_MAX_PX = 480;
const THEMES = ['auto', 'light', 'dark'];
const SECTIONS = { metrics: 'sec-metrics', media: 'sec-media', log: 'sec-log' };

const poll = {
    timer: null,
    running: false,
    rerun: false,
    lastSuccess: null,
    lastTreePoll: 0,
    failures: 0,
    nextAt: 0,
    error: null,
};

// ---- live polling ----

async function refreshTree() {
    const response = await fetchJson('/api/tree');
    state.root = response.root;
    indexTree(response.tree);
    poll.lastTreePoll = Date.now();
    document.getElementById('rootPath').textContent = response.root;
    document.getElementById('rootPath').title = response.root;
    emit('tree');
}

/** Poll metrics and visuals of every selected run, then the shown log, and push changes to the cards. */
async function pollSelectedRuns() {
    const paths = selectedPaths();
    const results = await Promise.all(paths.map((path) => Promise.all([pollMetrics(path), pollVisuals(path)])));
    if (results.some(([metricsChanged]) => metricsChanged)) metricsDataChanged();
    if (results.some(([, visualsChanged]) => visualsChanged)) renderMedia();
    await pollLog();
}

function schedulePoll(delay_ms) {
    clearTimeout(poll.timer);
    poll.nextAt = Date.now() + delay_ms;
    poll.timer = setTimeout(runPoll, delay_ms);
}

async function runPoll() {
    if (document.hidden) {
        renderPollStatus();
        return;  // resumed by the visibilitychange handler
    }
    if (poll.running) {
        poll.rerun = true;
        return;
    }
    poll.running = true;
    try {
        if (!state.treeLoaded || Date.now() - poll.lastTreePoll >= TREE_POLL_MS) await refreshTree();
        await pollSelectedRuns();
        poll.failures = 0;
        poll.error = null;
        poll.lastSuccess = Date.now();
        pulseDot();
    } catch (error) {
        poll.failures++;
        poll.error = error instanceof OfflineError ? 'server unreachable' : error.message;
        if (!(error instanceof OfflineError)) console.warn('poll failed', error);
    } finally {
        poll.running = false;
    }
    if (poll.rerun) {
        poll.rerun = false;
        schedulePoll(0);
    } else {
        schedulePoll(poll.failures ? BACKOFF_MS[Math.min(poll.failures, BACKOFF_MS.length) - 1] : POLL_MS);
    }
    renderPollStatus();
}

function pollNow() {
    schedulePoll(0);
}

function pulseDot() {
    const dot = document.getElementById('pollDot');
    dot.classList.remove('pulse');
    void dot.offsetWidth;  // restart the animation
    dot.classList.add('pulse');
}

function renderPollStatus() {
    const dot = document.getElementById('pollDot');
    const text = document.getElementById('pollText');
    const status = document.getElementById('pollStatus');
    dot.classList.toggle('paused', document.hidden);
    dot.classList.toggle('error', !document.hidden && poll.failures > 0);
    if (document.hidden) {
        text.textContent = 'paused';
        status.title = 'Polling pauses while the tab is hidden';
    } else if (poll.failures) {
        const seconds = Math.max(0, Math.ceil((poll.nextAt - Date.now()) / 1000));
        text.textContent = poll.error === 'server unreachable' ? `offline, retrying in ${seconds} s` : `error, retrying in ${seconds} s`;
        status.title = poll.error;
    } else if (poll.lastSuccess) {
        const seconds = (Date.now() - poll.lastSuccess) / 1000;
        text.textContent = `updated ${seconds < 1.5 ? 'just now' : `${formatAge(seconds)} ago`}`;
        status.title = 'Polling selected runs every 5 s, the run tree every 30 s';
    } else {
        text.textContent = 'connecting';
    }
}

// ---- toolbar run chips ----

function buildRunChip(path) {
    const run = state.runs.get(path);
    const chip = el('span', `chip${state.focused === path ? ' focused' : ''}${state.hidden.has(path) ? ' hidden' : ''}`);
    chip.style.setProperty('--c', runColor(path));
    chip.title = `${path}\nClick to show or hide, double-click to focus`;
    chip.tabIndex = 0;
    chip.dataset.path = path;
    chip.append(swatch(runColor(path), runDash(path) !== null), shortName(path));
    if (run && isLive(run)) {
        const live = el('span', 'live');
        live.title = 'live';
        chip.append(live);
    }
    const remove = el('button', 'x');
    remove.append(icon('close'));
    remove.setAttribute('aria-label', `Deselect ${shortName(path)}`);
    chip.append(remove);
    return chip;
}

function renderChips() {
    const chipsEl = document.getElementById('chips');
    const chips = selectedPaths().map(buildRunChip);
    if (chips.length > 1) {
        const clear = el('button', 'linkbtn', 'Clear all');
        clear.addEventListener('click', () => document.getElementById('clearSel').click());
        chips.push(el('span', 'spacer'), clear);
    }
    chipsEl.replaceChildren(...chips);
}

function initChips() {
    const chipsEl = document.getElementById('chips');
    chipsEl.addEventListener('click', (event) => {
        const chip = event.target.closest('.chip');
        if (!chip) return;
        if (event.target.closest('.x')) {
            deselectRun(chip.dataset.path);
            emit('selection');
            return;
        }
        toggleHidden(chip.dataset.path);
        emit('visibility');
    });
    chipsEl.addEventListener('dblclick', (event) => {
        const chip = event.target.closest('.chip');
        if (!chip) return;
        state.hidden.delete(chip.dataset.path);
        focusRun(chip.dataset.path);
        emit('selection');
    });
    chipsEl.addEventListener('keydown', (event) => {
        const chip = event.target.closest('.chip');
        if (!chip || (event.key !== 'Enter' && event.key !== ' ')) return;
        event.preventDefault();
        toggleHidden(chip.dataset.path);
        emit('visibility');
    });
}

// ---- empty state and sections ----

function renderEmptyState() {
    const empty = state.selected.size === 0;
    const emptyEl = document.getElementById('emptyState');
    emptyEl.hidden = !empty;
    for (const id of Object.values(SECTIONS)) document.getElementById(id).hidden = empty;
    if (!empty) return;
    const recent = [...state.runs.values()].filter((run) => run.mtime !== null).sort((a, b) => b.mtime - a.mtime).slice(0, 3);
    const buttons = recent.map((run) => {
        const button = el('button', 'btn recent');
        button.append(el('span', 'name', run.path), el('span', 'muted', `updated ${formatAge(Date.now() / 1000 - run.mtime)} ago`));
        button.addEventListener('click', () => {
            focusRun(run.path);
            emit('selection');
        });
        return button;
    });
    const list = document.getElementById('recentRuns');
    list.replaceChildren(...buttons);
    list.previousElementSibling.hidden = buttons.length === 0;
}

function applySectionState() {
    for (const [name, id] of Object.entries(SECTIONS)) {
        const section = document.getElementById(id);
        const collapsed = state.collapsed.has(name);
        section.classList.toggle('collapsed', collapsed);
        setCaret(section.querySelector('h2 .caret'), !collapsed);
        section.querySelector('h2').setAttribute('aria-expanded', !collapsed);
    }
}

function initSections() {
    for (const [name, id] of Object.entries(SECTIONS)) {
        const header = document.getElementById(id).querySelector('h2');
        const toggle = () => {
            if (state.collapsed.has(name)) state.collapsed.delete(name);
            else state.collapsed.add(name);
            applySectionState();
            emit('collapse');
        };
        header.addEventListener('click', toggle);
        header.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            toggle();
        });
    }
    applySectionState();
}

function jumpTo(targetId) {
    const section = document.getElementById(targetId).closest('.section');
    const name = Object.keys(SECTIONS).find((key) => SECTIONS[key] === section.id);
    if (state.collapsed.delete(name)) {
        applySectionState();
        emit('collapse');
    }
    document.getElementById(targetId).scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---- toolbar controls ----

const smoothInput = document.getElementById('smooth');

function setSmoothing(value) {
    state.smoothing = Math.round(Math.min(0.99, Math.max(0, value)) * 100) / 100;
    smoothInput.value = state.smoothing;
    document.getElementById('smoothVal').textContent = state.smoothing.toFixed(2);
    emit('smoothing');
}

function initToolbar() {
    smoothInput.value = state.smoothing;
    document.getElementById('smoothVal').textContent = state.smoothing.toFixed(2);
    smoothInput.addEventListener('input', () => setSmoothing(Number(smoothInput.value)));
    initChips();
}

// ---- theme ----

function applyTheme(theme) {
    state.theme = theme;
    if (theme === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', theme);
    document.getElementById('themeLabel').textContent = theme;
    readPalette();
    emit('theme');
}

function cycleTheme() {
    const theme = THEMES[(THEMES.indexOf(state.theme) + 1) % THEMES.length];
    savePref('theme', theme);
    applyTheme(theme);
}

// ---- sidebar resize ----

function setSidebarWidth(width_px) {
    const clamped = Math.min(SIDEBAR_MAX_PX, Math.max(SIDEBAR_MIN_PX, width_px));
    document.body.style.setProperty('--sidebar-w', `${clamped}px`);
    return clamped;
}

function initResizer() {
    setSidebarWidth(loadPref('sidebarWidth', 280));
    const resizer = document.getElementById('resizer');
    resizer.addEventListener('pointerdown', (event) => {
        resizer.setPointerCapture(event.pointerId);
        const move = (moveEvent) => setSidebarWidth(moveEvent.clientX);
        resizer.addEventListener('pointermove', move);
        resizer.addEventListener('pointerup', (upEvent) => {
            resizer.removeEventListener('pointermove', move);
            savePref('sidebarWidth', setSidebarWidth(upEvent.clientX));
        }, { once: true });
    });
}

// ---- keyboard ----

const helpEl = document.getElementById('help');
let smoothingMode = false;

function handleEscape() {
    if (isLightboxOpen()) return closeLightbox();
    if (isReportOpen()) return closeReport();
    if (helpEl.classList.contains('open')) return helpEl.classList.remove('open');
    const active = document.activeElement;
    if (active?.id === 'metricFilter' && active.value) return setMetricFilter('');
    if (active?.id === 'treeSearch' && active.value) {
        active.value = '';
        active.dispatchEvent(new Event('input'));
        return;
    }
    active?.blur();
}

const SHORTCUTS = {
    '/': () => document.getElementById('metricFilter').focus(),
    't': () => document.getElementById('treeSearch').focus(),
    'x': () => document.getElementById('clearSel').click(),
    's': () => (smoothingMode = true),
    'z': () => resetZoom(),
    'y': toggleLogyHovered,
    'Y': toggleLogyAll,
    '[': () => stepMedia(-1),
    ']': () => stepMedia(1),
    'd': cycleTheme,
    'r': openReport,
    '?': () => helpEl.classList.add('open'),
    '1': () => jumpTo('sec-metrics'),
    '2': () => jumpTo('sec-media'),
    '3': () => jumpTo('logCard'),
    '4': () => jumpTo('cfgCard'),
};

function handleKeydown(event) {
    if (event.key === 'Escape') {
        smoothingMode = false;
        handleEscape();
        return;
    }
    if (isLightboxOpen() && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
        event.preventDefault();
        cycleLightbox(event.key === 'ArrowRight' ? 1 : -1);
        return;
    }
    if (isReportOpen()) return;  // the dialog's checkboxes and selects are not text fields, keep the viewer keys off
    if (isTyping() || event.ctrlKey || event.metaKey || event.altKey) return;
    if (smoothingMode && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
        event.preventDefault();
        setSmoothing(state.smoothing + (event.key === 'ArrowRight' ? SMOOTHING_STEP : -SMOOTHING_STEP));
        return;
    }
    smoothingMode = false;
    if (event.key === 'ArrowDown' && document.activeElement === document.body) {
        event.preventDefault();
        focusTree();
        return;
    }
    const shortcut = SHORTCUTS[event.key];
    if (!shortcut) return;
    event.preventDefault();
    shortcut();
}

function initHelp() {
    document.getElementById('helpBtn').addEventListener('click', () => helpEl.classList.add('open'));
    helpEl.addEventListener('click', (event) => {
        if (event.target === helpEl || event.target.closest('.dialog-close')) helpEl.classList.remove('open');
    });
}

// ---- boot ----

function handleSelectionChange() {
    forgetUnselected(state.selected);
    renderChips();
    renderEmptyState();
    pollNow();
}

async function boot() {
    readUrl();
    applyTheme(state.theme);
    initResizer();
    initTree();
    initToolbar();
    initSections();
    initMetrics();
    initMedia();
    initLogConfig();
    initHelp();
    initReport();
    document.getElementById('themeBtn').addEventListener('click', cycleTheme);
    document.addEventListener('keydown', handleKeydown);
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if (state.theme === 'auto') applyTheme('auto');
    });
    document.addEventListener('visibilitychange', () => {
        renderPollStatus();
        if (!document.hidden) pollNow();
    });
    setInterval(renderPollStatus, 1000);

    on('selection', handleSelectionChange);
    on('visibility', renderChips);
    on('theme', renderChips);
    on('tree', () => {
        const before = state.selected.size;
        pruneSelection();
        if (state.selected.size !== before) emit('selection');
        renderChips();
        if (!state.selected.size) renderEmptyState();
    });

    try {
        await refreshTree();
    } catch (error) {
        poll.failures = 1;
        poll.error = error instanceof OfflineError ? 'server unreachable' : error.message;
        renderPollStatus();
    }
    emit('selection');
}

boot();
