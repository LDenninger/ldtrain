// Report dialog: choose runs, metrics, media and layout, then download the PDF rendered by POST /api/report.

import { groupPathOf } from './groups.js';
import { listMediaTags } from './media.js';
import { listMetricCards } from './metrics.js';
import { on, PALETTE_SIZE, selectedPaths, slotDash, state } from './state.js';
import { el, formatIteration, swatch } from './util.js';

const dialogEl = document.getElementById('report');
const formEl = document.getElementById('reportForm');
const titleInput = document.getElementById('reportTitle');
const runsEl = document.getElementById('reportRuns');
const metricsEl = document.getElementById('reportMetrics');
const mediaEl = document.getElementById('reportMedia');
const stepSelect = document.getElementById('reportStep');
const errorEl = document.getElementById('reportError');
const generateButton = document.getElementById('reportGenerate');
const reportButton = document.getElementById('reportBtn');

let lightPalette = [];

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

// ---- checklists ----

function checkItem(value, checked, ...content) {
    const label = el('label');
    const box = el('input');
    box.type = 'checkbox';
    box.value = value;
    box.checked = checked;
    label.append(box, ...content);
    return label;
}

/** Check boxes under one heading per group path, a heading toggling every box of its group. */
function renderGroupedList(container, items) {
    const nodes = [];
    let currentGroup = null;
    for (const item of items) {
        const groupPath = groupPathOf(item.key);
        if (groupPath !== currentGroup) {
            currentGroup = groupPath;
            const heading = checkItem('', false, groupPath ? `${groupPath}/` : 'ungrouped');
            heading.classList.add('grp');
            heading.dataset.group = groupPath;
            nodes.push(heading);
        }
        const leaf = item.key.slice(groupPath ? groupPath.length + 1 : 0);
        const label = checkItem(item.key, item.checked, el('span', 'name', leaf), ...(item.extra ?? []));
        label.dataset.group = groupPath;
        nodes.push(label);
    }
    container.replaceChildren(...(nodes.length ? nodes : [el('div', 'muted', 'Nothing logged for the selected runs.')]));
    syncGroupBoxes(container);
}

function leafBoxes(container, group = null) {
    return [...container.querySelectorAll('label:not(.grp)')].filter((label) => group === null || label.dataset.group === group).map((label) => label.firstChild);
}

function syncGroupBoxes(container) {
    for (const heading of container.querySelectorAll('label.grp')) {
        const boxes = leafBoxes(container, heading.dataset.group);
        const checkedCount = boxes.filter((box) => box.checked).length;
        heading.firstChild.checked = checkedCount === boxes.length;
        heading.firstChild.indeterminate = checkedCount > 0 && checkedCount < boxes.length;
    }
    const counts = { reportMetrics: 'reportMetricCount', reportMedia: 'reportMediaCount' };
    const boxes = leafBoxes(container);
    document.getElementById(counts[container.id]).textContent = boxes.length ? `${boxes.filter((box) => box.checked).length} of ${boxes.length}` : '';
}

function checkedValues(container) {
    return leafBoxes(container).filter((box) => box.checked).map((box) => box.value);
}

function initChecklist(container) {
    container.addEventListener('change', (event) => {
        const heading = event.target.closest('label.grp');
        if (heading) for (const box of leafBoxes(container, heading.dataset.group)) box.checked = event.target.checked;
        syncGroupBoxes(container);
    });
}

// ---- dialog ----

function renderRuns() {
    const paths = selectedPaths();
    runsEl.replaceChildren(...paths.map((path) => {
        const run = reportRun(path);
        const label = checkItem(path, !state.hidden.has(path), swatch(run.color, run.dashed), el('span', 'name', path));
        label.title = path;
        return label;
    }));
}

function renderMedia() {
    const { tags, steps, step } = listMediaTags();
    renderGroupedList(mediaEl, tags.map(({ tag, kind }) => ({ key: tag, checked: true, extra: [el('span', 'kind', kind)] })));
    const latest = el('option', '', steps.length ? `Latest (${steps[steps.length - 1].toLocaleString()})` : 'Latest');
    latest.value = '';
    stepSelect.replaceChildren(latest, ...[...steps].reverse().slice(1).map((value) => {
        const option = el('option', '', value.toLocaleString());
        option.value = value;
        return option;
    }));
    stepSelect.value = step === null || step === steps[steps.length - 1] ? '' : String(step);
    stepSelect.disabled = steps.length < 2;
    document.getElementById('reportMediaSet').hidden = tags.length === 0;
}

function renderViewNote() {
    const parts = [`smoothing ${state.smoothing.toFixed(2)}`];
    if (state.logy.size) parts.push(`log y on ${state.logy.size} metric${state.logy.size === 1 ? '' : 's'}`);
    const range = state.syncZoom ? state.xRange : null;
    if (range) parts.push(`iterations ${formatIteration(range[0])} to ${formatIteration(range[1])}`);
    document.getElementById('reportViewNote').textContent = `Charts follow the viewer: ${parts.join(', ')}.`;
}

export function openReport() {
    if (!state.selected.size) return;
    lightPalette = readLightPalette();
    const paths = selectedPaths();
    titleInput.value = '';
    titleInput.placeholder = paths.length === 1 ? paths[0] : `Comparison of ${paths.length} runs`;
    renderRuns();
    renderGroupedList(metricsEl, listMetricCards().map(({ metric, visible }) => ({ key: metric, checked: visible })));
    renderMedia();
    renderViewNote();
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

function buildRequest() {
    const runs = checkedValues(runsEl);
    return {
        title: titleInput.value.trim(),
        runs: selectedPaths().filter((path) => runs.includes(path)).map(reportRun),
        metrics: checkedValues(metricsEl),
        smoothing: state.smoothing,
        log_metrics: [...state.logy],
        x_range: state.syncZoom ? state.xRange : null,
        media_tags: document.getElementById('reportMediaSet').hidden ? [] : checkedValues(mediaEl),
        media_step: stepSelect.value === '' ? null : Number(stepSelect.value),
        include_summary: document.getElementById('reportSummary').checked,
        include_config: document.getElementById('reportConfig').checked,
        orientation: document.getElementById('reportOrientation').value,
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
    const label = generateButton.lastChild;
    label.textContent = 'Rendering…';
    try {
        const response = await fetch('/api/report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (!response.ok) return showError(`Report failed (${response.status}): ${await describeFailure(response)}`);
        const fileName = /filename="([^"]+)"/.exec(response.headers.get('Content-Disposition') ?? '')?.[1] ?? 'report.pdf';
        saveBlob(await response.blob(), fileName);
        closeReport();
    } catch (error) {
        showError(`Report failed: ${error.message}`);
    } finally {
        generateButton.disabled = false;
        label.textContent = 'Download PDF';
    }
}

export function initReport() {
    reportButton.addEventListener('click', openReport);
    on('selection', () => (reportButton.disabled = state.selected.size === 0));
    initChecklist(metricsEl);
    initChecklist(mediaEl);
    formEl.addEventListener('submit', generate);
    formEl.addEventListener('click', (event) => {
        const all = event.target.closest('[data-all], [data-none]');
        if (!all) return;
        const container = document.getElementById(all.dataset.all ?? all.dataset.none);
        for (const box of leafBoxes(container)) box.checked = 'all' in all.dataset;
        syncGroupBoxes(container);
    });
    document.getElementById('reportCancel').addEventListener('click', closeReport);
    dialogEl.addEventListener('click', (event) => {
        if (event.target === dialogEl || event.target.closest('.dialog-close')) closeReport();
    });
}
