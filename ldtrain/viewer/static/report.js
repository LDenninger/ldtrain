// Report dialog: choose runs, metrics, media and layout, then download the PDF rendered by POST /api/report.

import { groupPathOf } from './groups.js';
import { listMediaTags } from './media.js';
import { listMetricCards } from './metrics.js';
import { PALETTE_SIZE, selectedPaths, slotDash, state } from './state.js';
import { el, formatIteration, icon, setCaret, swatch } from './util.js';

const dialogEl = document.getElementById('report');
const formEl = document.getElementById('reportForm');
const titleInput = document.getElementById('reportTitle');
const runsEl = document.getElementById('reportRuns');
const filterInput = document.getElementById('reportFilter');
const stepSelect = document.getElementById('reportStep');
const errorEl = document.getElementById('reportError');
const generateButton = document.getElementById('reportGenerate');

// one entry per tab: its tree, pane and tab button, and the count shown on the tab
const TREES = {
    metrics: { tree: document.getElementById('reportMetrics'), pane: document.getElementById('reportPaneMetrics'),
        tab: document.getElementById('reportTabMetrics'), count: document.getElementById('reportMetricCount'), empty: 'No metrics logged yet for the selected runs.' },
    media: { tree: document.getElementById('reportMedia'), pane: document.getElementById('reportPaneMedia'),
        tab: document.getElementById('reportTabMedia'), count: document.getElementById('reportMediaCount'), empty: 'No media logged for the selected runs.' },
};

let lightPalette = [];
let activeTab = 'metrics';

// ---- palette ----

/** The run palette of the light theme: the PDF is printed on white whatever theme the viewer shows. */
function readLightPalette() {
    const root = document.documentElement;
    const theme = root.getAttribute('data-theme');
    root.setAttribute('data-theme', 'light');
    const style = getComputedStyle(root);
    const colors = Array.from({ length: PALETTE_SIZE }, (_, ii) => style.getPropertyValue(`--run-${ii}`).trim());
    if (theme === null) root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', theme);
    return colors;
}

function reportRun(path) {
    const slot = state.selected.get(path);
    return { path, color: lightPalette[slot % PALETTE_SIZE], dashed: slotDash(slot) !== null };
}

function checkbox(checked) {
    const box = el('input', 'cb');
    box.type = 'checkbox';
    box.checked = checked;
    return box;
}

// ---- runs ----

function renderRuns() {
    runsEl.replaceChildren(...selectedPaths().map((path) => {
        const run = reportRun(path);
        const lastIteration = state.runs.get(path)?.last_iteration;
        const box = checkbox(!state.hidden.has(path));
        box.value = path;
        const parent = path.includes('/') ? path.slice(0, path.lastIndexOf('/') + 1) : '';
        const label = el('label', 'rd-run');
        label.title = path;
        label.style.setProperty('--c', run.color);
        const pathEl = el('span', 'path');
        pathEl.append(el('span', 'parent', parent), el('span', 'name', path.slice(parent.length)));
        label.append(box, swatch(run.color, run.dashed), pathEl,
            el('span', 'iter num', lastIteration === null || lastIteration === undefined ? '' : formatIteration(lastIteration)));
        const item = el('li');
        item.append(label);
        return item;
    }));
    updateRunCount();
}

function updateRunCount() {
    const boxes = [...runsEl.querySelectorAll('input')];
    document.getElementById('reportRunCount').textContent = `${boxes.filter((box) => box.checked).length}/${boxes.length}`;
}

// ---- content trees ----

/**
 * Nest `/`-separated keys into groups to any depth, keeping the order they arrive in.
 * A group lists its own leaves before its subgroups, as the viewer's sections do.
 */
function buildTree(items) {
    const root = { path: '', name: '', leaves: [], groups: new Map() };
    for (const item of items) {
        const groupPath = groupPathOf(item.key);
        let node = root;
        if (groupPath) {
            const segments = groupPath.split('/');
            segments.forEach((segment, ii) => {
                const path = segments.slice(0, ii + 1).join('/');
                if (!node.groups.has(path)) node.groups.set(path, { path, name: segment, leaves: [], groups: new Map() });
                node = node.groups.get(path);
            });
        }
        node.leaves.push({ ...item, name: groupPath ? item.key.slice(groupPath.length + 1) : item.key });
    }
    return root;
}

function renderLeaf(leaf) {
    const item = el('li', 'tleaf');
    item.setAttribute('role', 'treeitem');
    item.dataset.key = leaf.key;
    const row = el('label', 'trow');
    row.title = leaf.key;
    const box = checkbox(leaf.checked);
    box.dataset.key = leaf.key;
    row.append(box, el('span', 'tname', leaf.name));
    if (leaf.meta) row.append(leaf.meta);
    item.append(row);
    return item;
}

function renderGroup(group) {
    const item = el('li', 'tgroup');
    item.setAttribute('role', 'treeitem');
    item.setAttribute('aria-expanded', 'true');
    const row = el('div', 'trow');
    const twist = el('button', 'twist');
    twist.type = 'button';
    twist.setAttribute('aria-label', `Collapse ${group.path}`);
    twist.append(icon('chevron-down', 'icon caret'));
    const label = el('label', 'tlabel');
    label.title = `${group.path}/`;
    label.append(checkbox(false), el('span', 'tname', group.name));
    row.append(twist, label, el('span', 'tcount num'));
    const children = el('ul', 'tchildren');
    children.setAttribute('role', 'group');
    children.append(...group.leaves.map(renderLeaf), ...[...group.groups.values()].map(renderGroup));
    item.append(row, children);
    return item;
}

function renderTree(name, items) {
    const { tree, empty } = TREES[name];
    const root = buildTree(items);
    const nodes = [...root.leaves.map(renderLeaf), ...[...root.groups.values()].map(renderGroup)];
    tree.replaceChildren(...(nodes.length ? nodes : [el('li', 'tempty', empty)]));
    applyFilter(name);
}

function leafBoxes(scope, visibleOnly = false) {
    return [...scope.querySelectorAll('input[data-key]')].filter((box) => !visibleOnly || !box.closest('li.tleaf').hidden);
}

/** Recompute every group's checkbox, count and visibility, and the tab's count, from the leaves. */
function refreshTree(name) {
    const { tree, count } = TREES[name];
    for (const groupEl of tree.querySelectorAll('li.tgroup')) {
        const boxes = leafBoxes(groupEl);
        const checkedCount = boxes.filter((box) => box.checked).length;
        const groupBox = groupEl.querySelector(':scope > .trow input');
        groupBox.checked = boxes.length > 0 && checkedCount === boxes.length;
        groupBox.indeterminate = checkedCount > 0 && checkedCount < boxes.length;
        groupEl.querySelector(':scope > .trow .tcount').textContent = `${checkedCount}/${boxes.length}`;
        groupEl.hidden = leafBoxes(groupEl, true).length === 0;
    }
    const boxes = leafBoxes(tree);
    count.textContent = boxes.length ? `${boxes.filter((box) => box.checked).length}/${boxes.length}` : '';
}

/** Keys matching the filter like the viewer's metric filter: a substring, or a /regex/. */
function keyMatches(key, filter) {
    if (!filter) return true;
    if (filter.length > 2 && filter.startsWith('/') && filter.endsWith('/')) {
        try {
            return new RegExp(filter.slice(1, -1)).test(key);
        } catch {
            return true;  // incomplete regex while typing
        }
    }
    return key.toLowerCase().includes(filter.toLowerCase());
}

function applyFilter(name) {
    const filter = filterInput.value.trim();
    for (const leaf of TREES[name].tree.querySelectorAll('li.tleaf')) leaf.hidden = !keyMatches(leaf.dataset.key, filter);
    refreshTree(name);
}

function toggleCollapsed(groupEl) {
    const collapsed = groupEl.classList.toggle('collapsed');
    groupEl.setAttribute('aria-expanded', String(!collapsed));
    const twist = groupEl.querySelector(':scope > .trow .twist');
    setCaret(twist.querySelector('.caret'), !collapsed);
    twist.setAttribute('aria-label', `${collapsed ? 'Expand' : 'Collapse'} ${groupEl.querySelector(':scope > .trow .tlabel').title.slice(0, -1)}`);
}

function initTree(name) {
    const { tree, tab } = TREES[name];
    tree.addEventListener('change', (event) => {
        const groupEl = event.target.closest('li.tgroup');
        if (groupEl && event.target === groupEl.querySelector(':scope > .trow input')) {
            for (const box of leafBoxes(groupEl, true)) box.checked = event.target.checked;  // only what the filter shows
        }
        refreshTree(name);
    });
    tree.addEventListener('click', (event) => {
        const twist = event.target.closest('.twist');
        if (twist) toggleCollapsed(twist.closest('li.tgroup'));
    });
    tab.addEventListener('click', () => selectTab(name));
}

function selectTab(name) {
    activeTab = name;
    for (const [key, entry] of Object.entries(TREES)) {
        entry.tab.setAttribute('aria-selected', String(key === name));
        entry.tab.tabIndex = key === name ? 0 : -1;
        entry.pane.hidden = key !== name;
    }
}

// ---- dialog ----

function renderMedia() {
    const { tags, steps, step } = listMediaTags();
    renderTree('media', tags.map(({ tag, kind }) => {
        const kindIcon = icon(kind === 'video' ? 'play' : 'image', 'icon kind');
        kindIcon.setAttribute('aria-label', kind);
        return { key: tag, checked: true, meta: kindIcon };
    }));
    const latest = el('option', '', steps.length ? `Latest (${steps[steps.length - 1].toLocaleString()})` : 'Latest');
    latest.value = '';
    stepSelect.replaceChildren(latest, ...[...steps].reverse().slice(1).map((value) => {
        const option = el('option', '', value.toLocaleString());
        option.value = value;
        return option;
    }));
    stepSelect.value = step === null || step === steps[steps.length - 1] ? '' : String(step);
    stepSelect.disabled = steps.length < 2;
    stepSelect.closest('.rd-step').hidden = tags.length === 0;
}

function renderViewNote() {
    const parts = [`smoothing ${state.smoothing.toFixed(2)}`];
    if (state.logy.size) parts.push(`log y on ${state.logy.size} metric${state.logy.size === 1 ? '' : 's'}`);
    const range = state.syncZoom ? state.xRange : null;
    if (range) parts.push(`iterations ${formatIteration(range[0])} to ${formatIteration(range[1])}`);
    document.getElementById('reportViewNote').textContent = `Charts use the viewer's ${parts.join(', ')}.`;
}

export function openReport() {
    if (!state.selected.size) return;
    lightPalette = readLightPalette();
    const paths = selectedPaths();
    titleInput.value = '';
    titleInput.placeholder = paths.length === 1 ? paths[0] : `Comparison of ${paths.length} runs`;
    filterInput.value = '';
    renderRuns();
    renderTree('metrics', listMetricCards().map(({ metric, visible }) => ({
        key: metric, checked: visible, meta: state.logy.has(metric) ? el('span', 'tag', 'log') : null,
    })));
    renderMedia();
    renderViewNote();
    selectTab('metrics');
    errorEl.hidden = true;
    dialogEl.classList.add('open');
    titleInput.focus();
}

export function closeReport() {
    dialogEl.classList.remove('open');
}

export function isReportOpen() {
    return dialogEl.classList.contains('open');
}

// ---- download ----

function checkedKeys(name) {
    return leafBoxes(TREES[name].tree).filter((box) => box.checked).map((box) => box.dataset.key);
}

function buildRequest() {
    const runs = new Set([...runsEl.querySelectorAll('input:checked')].map((box) => box.value));
    return {
        title: titleInput.value.trim(),
        runs: selectedPaths().filter((path) => runs.has(path)).map(reportRun),
        metrics: checkedKeys('metrics'),
        smoothing: state.smoothing,
        log_metrics: [...state.logy],
        x_range: state.syncZoom ? state.xRange : null,
        media_tags: checkedKeys('media'),
        media_step: stepSelect.value === '' ? null : Number(stepSelect.value),
        include_summary: document.getElementById('reportSummary').checked,
        include_config: document.getElementById('reportConfig').checked,
        orientation: formEl.elements.orientation.value,
    };
}

function showError(message) {
    errorEl.textContent = message;
    errorEl.hidden = false;
}

async function describeFailure(response) {
    try {
        const detail = (await response.json()).detail;
        if (Array.isArray(detail)) return detail.map((item) => `${item.loc.slice(1).join('.')}: ${item.msg}`).join('; ');
        return detail ?? response.statusText;
    } catch {
        return response.statusText;
    }
}

function saveBlob(blob, fileName) {
    const url = URL.createObjectURL(blob);
    const link = el('a');
    link.href = url;
    link.download = fileName;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function generate(event) {
    event.preventDefault();
    const body = buildRequest();
    if (!body.runs.length) return showError('Pick at least one run.');
    errorEl.hidden = true;
    generateButton.disabled = true;
    const label = generateButton.querySelector('span');
    label.textContent = 'Rendering…';
    try {
        const response = await fetch('/api/report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (!response.ok) return showError(`The report failed (${response.status}): ${await describeFailure(response)}`);
        const fileName = /filename="([^"]+)"/.exec(response.headers.get('Content-Disposition') ?? '')?.[1] ?? 'report.pdf';
        saveBlob(await response.blob(), fileName);
        closeReport();
    } catch (error) {
        showError(`The report failed: ${error.message}`);
    } finally {
        generateButton.disabled = false;
        label.textContent = 'Download PDF';
    }
}

export function initReport() {
    initTree('metrics');
    initTree('media');
    runsEl.addEventListener('change', updateRunCount);
    filterInput.addEventListener('input', () => Object.keys(TREES).forEach(applyFilter));
    for (const [id, checked] of [['reportAll', true], ['reportNone', false]]) {
        document.getElementById(id).addEventListener('click', () => {
            for (const box of leafBoxes(TREES[activeTab].tree, true)) box.checked = checked;
            refreshTree(activeTab);
        });
    }
    formEl.addEventListener('submit', generate);
    document.getElementById('reportCancel').addEventListener('click', closeReport);
    document.getElementById('reportClose').addEventListener('click', closeReport);
    dialogEl.addEventListener('click', (event) => {
        if (event.target === dialogEl) closeReport();
    });
}
