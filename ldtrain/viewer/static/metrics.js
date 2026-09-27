// Metric cards: one uPlot chart per metric column, overlaying every selected run.

import { peekRunData, pollMetrics } from './data.js';
import { GroupTree, createCardTitle, topGroupOf } from './groups.js';
import {
    drawOrder, emit, on, runColor, runDash, selectedPaths, state, toggleHidden, withAlpha,
} from './state.js';
import { el, formatIteration, formatValue, icon, nearestIndex, shortName, smoothEma, swatch } from './util.js';

const uPlot = window.uPlot;
const PLOT_PADDING_PX = 8;
const MAX_POINTS_WITH_MARKERS = 12;  // sparser series get hollow markers, denser ones lines only
const MIN_ZOOM_PX = 4;
const FRESH_HIGHLIGHT_MS = 2000;
const AXIS_FONT = '11px "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace';
const UNGROUPED = '';

const statusEl = document.getElementById('metricStatus');
const errorsEl = document.getElementById('metricErrors');
const groupsEl = document.getElementById('metricGroups');
const filterInput = document.getElementById('metricFilter');

const cards = new Map();   // metric -> card
const groupTree = new GroupTree(groupsEl);
let hoveredCard = null;
let firstLoadDone = false;
const themeColors = { axis: '#5a6470', grid: 'rgba(0,0,0,.07)', surface: '#ffffff' };

// ---- metric names and filter ----

/** Union of the metric columns of all selected runs, in order of first appearance. */
function metricNames() {
    const names = [];
    const seen = new Set();
    for (const path of selectedPaths()) {
        for (const column of peekRunData(path)?.metrics.columns ?? []) {
            if (column === 'iteration' || seen.has(column)) continue;
            seen.add(column);
            names.push(column);
        }
    }
    return names;
}

function metricMatches(metric) {
    const filter = state.metricFilter.trim();
    if (!filter) return true;
    if (filter.length > 2 && filter.startsWith('/') && filter.endsWith('/')) {
        try {
            return new RegExp(filter.slice(1, -1)).test(metric);
        } catch {
            return true;  // incomplete regex while typing
        }
    }
    return metric.toLowerCase().includes(filter.toLowerCase());
}

// ---- per-run series ----

/** Recompute each run's raw and smoothed series for a card, and whether log-y is possible. */
function computeRunSeries(card) {
    card.runSeries = new Map();
    card.pointCount = 0;
    let minValue = Infinity;
    for (const path of selectedPaths()) {
        const series = peekRunData(path)?.metrics.series.get(card.metric);
        if (!series || !series.x.length) continue;
        card.runSeries.set(path, { x: series.x, y: series.y, s: smoothEma(series.y, state.smoothing) });
        card.pointCount += series.x.length;
        for (const value of series.y) minValue = Math.min(minValue, value);
    }
    card.canLog = card.runSeries.size > 0 && minValue > 0;
}

/** Aligned uPlot data in draw order: [x, raw₀, smoothed₀, raw₁, smoothed₁, …]. */
function joinChartData(card) {
    if (!card.runKeys.length) return [[]];
    const tables = card.runKeys.map(({ path }) => {
        const series = card.runSeries.get(path);
        // copies: uPlot keeps references and the run arrays keep growing between polls
        return series ? [series.x.slice(), series.y.slice(), series.s.slice()] : [[], [], []];
    });
    return uPlot.join(tables);
}

function lastSmoothed(card, path) {
    const series = card.runSeries.get(path);
    return series ? series.s[series.s.length - 1] : null;
}

// ---- chart options ----

function currentRunKeys() {
    return drawOrder().map((path) => ({ path, key: `${path}|${state.selected.get(path)}|${path === state.focused}` }));
}

function seriesPair(card, path) {
    const focused = path === state.focused;
    const visible = !state.hidden.has(path);
    const dash = runDash(path) ?? undefined;
    const color = () => runColor(path);
    const pointCount = () => card.runSeries.get(path)?.x.length ?? 0;
    return [
        {
            label: `${shortName(path)} raw`, stroke: () => withAlpha(runColor(path), 0.25), width: 1, dash, spanGaps: true,
            auto: false, show: visible && state.smoothing > 0, points: { show: false },
        },
        {
            label: shortName(path), stroke: color, width: focused ? 2.2 : 1.6, dash, spanGaps: true, show: visible,
            points: { show: () => pointCount() <= MAX_POINTS_WITH_MARKERS, size: 6, width: 1.5, fill: () => themeColors.surface, stroke: color },
        },
    ];
}

function logSplits(u, axisIndex, min, max) {
    const powers = [];
    for (let exponent = Math.ceil(Math.log10(min)); exponent <= Math.floor(Math.log10(max)); exponent++) powers.push(10 ** exponent);
    if (powers.length >= 2) return powers;
    const splits = [];
    for (let exponent = Math.floor(Math.log10(min)); exponent <= Math.ceil(Math.log10(max)); exponent++) {
        for (const mantissa of [1, 2, 5]) {
            const value = mantissa * 10 ** exponent;
            if (value >= min && value <= max) splits.push(value);
        }
    }
    return splits.length ? splits : [min, max];
}

function axisOptions(size, values, extra = {}) {
    return {
        stroke: () => themeColors.axis, grid: { stroke: () => themeColors.grid, width: 1 }, ticks: { show: false },
        font: AXIS_FONT, size, values, ...extra,
    };
}

function chartOptions(card) {
    const series = [{}];
    for (const { path } of card.runKeys) series.push(...seriesPair(card, path));
    const yAxisExtra = card.logActive ? { splits: logSplits } : {};
    return {
        width: Math.max(0, card.plotEl.clientWidth - PLOT_PADDING_PX),
        height: card.plotEl.clientHeight,
        legend: { show: false },
        cursor: {
            sync: { key: 'ldtrain' },
            y: false,
            drag: { x: true, y: false, setScale: false },
            bind: { dblclick: () => null },
            points: { size: 6 },
        },
        scales: { x: { time: false }, y: card.logActive ? { distr: 3, log: 10 } : {} },
        axes: [
            axisOptions(26, (u, splits) => splits.map(formatIteration)),
            axisOptions(56, (u, splits) => splits.map(formatValue), yAxisExtra),
        ],
        series,
        plugins: [tooltipPlugin(card)],
        hooks: { setSelect: [(u) => handleDragSelect(card, u)] },
    };
}

// ---- tooltip ----

function tooltipRows(card, iteration) {
    const rows = [];
    for (const { path } of card.runKeys) {
        const series = card.runSeries.get(path);
        if (!series || state.hidden.has(path)) continue;
        if (iteration < series.x[0] || iteration > series.x[series.x.length - 1]) continue;
        const index = nearestIndex(series.x, iteration);
        rows.push({ path, raw: series.y[index], smoothed: series.s[index] });
    }
    return rows.sort((a, b) => b.smoothed - a.smoothed);
}

function renderTooltip(tooltip, card, iteration) {
    const header = el('div', 'it');
    header.append(el('b', '', `iteration ${iteration.toLocaleString()}`), el('span', '', card.metric));
    const table = el('table');
    for (const row of tooltipRows(card, iteration)) {
        const tr = el('tr', row.path === state.focused ? 'f' : '');
        const nameCell = el('td');
        nameCell.append(swatch(runColor(row.path), runDash(row.path) !== null), ` ${shortName(row.path)}`);
        tr.append(nameCell, el('td', 'v', formatValue(row.smoothed)));
        if (state.smoothing > 0) tr.append(el('td', 'r', formatValue(row.raw)));
        table.append(tr);
    }
    tooltip.replaceChildren(header, table);
}

function tooltipPlugin(card) {
    let tooltip = null;
    return {
        hooks: {
            init: (u) => {
                tooltip = el('div', 'tt');
                card.plotEl.append(tooltip);
                u.over.addEventListener('mouseenter', () => (hoveredCard = card));
                u.over.addEventListener('mouseleave', () => {
                    hoveredCard = null;
                    tooltip.style.display = 'none';
                });
                u.over.addEventListener('dblclick', () => resetZoom(state.syncZoom ? null : card));
            },
            setCursor: (u) => {
                const index = u.cursor.idx;
                if (hoveredCard !== card || index === null || index === undefined) {
                    tooltip.style.display = 'none';
                    return;
                }
                renderTooltip(tooltip, card, u.data[0][index]);
                tooltip.style.display = 'block';
                const wrap = card.plotEl.getBoundingClientRect();
                const over = u.over.getBoundingClientRect();
                let left = over.left - wrap.left + u.cursor.left + 14;
                if (left + tooltip.offsetWidth > wrap.width) left = over.left - wrap.left + u.cursor.left - tooltip.offsetWidth - 14;
                tooltip.style.left = `${Math.max(0, left)}px`;
                tooltip.style.top = `${Math.min(Math.max(0, u.cursor.top - 10), wrap.height - tooltip.offsetHeight)}px`;
            },
            destroy: () => tooltip?.remove(),
        },
    };
}

// ---- zoom ----

function cardRange(card) {
    return state.syncZoom ? state.xRange : card.xRange;
}

function applyZoom(card) {
    const chart = card.chart;
    if (!chart) return;
    const range = cardRange(card);
    if (range) chart.setScale('x', { min: range[0], max: range[1] });
    else chart.setData(chart.data, true);
    if (!range) card.newDataEl.hidden = true;
}

function handleDragSelect(card, u) {
    const { left, width } = u.select;
    if (width < MIN_ZOOM_PX) return;
    const range = [u.posToVal(left, 'x'), u.posToVal(left + width, 'x')];
    u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
    if (state.syncZoom) {
        state.xRange = range;
        for (const other of cards.values()) applyZoom(other);
    } else {
        card.xRange = range;
        applyZoom(card);
    }
    emit('zoom');
}

/** Reset the zoom of one card, or of every card when `onlyCard` is null. */
export function resetZoom(onlyCard = null) {
    if (onlyCard) {
        onlyCard.xRange = null;
        applyZoom(onlyCard);
    } else {
        state.xRange = null;
        for (const card of cards.values()) {
            card.xRange = null;
            applyZoom(card);
        }
    }
    emit('zoom');
}

export function setSyncZoom(enabled) {
    if (enabled === state.syncZoom) return;
    state.syncZoom = enabled;
    if (enabled) {
        resetZoom();
        return;
    }
    for (const card of cards.values()) card.xRange = state.xRange;
    state.xRange = null;
    emit('zoom');
}

// ---- chart lifecycle ----

function wantsLog(card) {
    return state.logy.has(card.metric) && card.canLog;
}

function createChart(card) {
    if (!uPlot || card.chart) return;
    computeRunSeries(card);
    card.runKeys = currentRunKeys();
    card.logActive = wantsLog(card);
    card.plotEl.classList.remove('skeleton');
    card.chart = new uPlot(chartOptions(card), joinChartData(card), card.plotEl);
    if (cardRange(card)) applyZoom(card);
}

function rebuildChart(card) {
    if (!card.chart) return;
    card.chart.destroy();
    card.chart = null;
    createChart(card);
}

/** Add and remove series in place so the chart keeps its zoom: only the changed tail is replaced. */
function syncSeries(card) {
    const desired = currentRunKeys();
    const current = card.runKeys;
    let common = 0;
    while (common < current.length && common < desired.length && current[common].key === desired[common].key) common++;
    for (let ii = current.length - 1; ii >= common; ii--) {
        card.chart.delSeries(2 + 2 * ii);
        card.chart.delSeries(1 + 2 * ii);
    }
    card.runKeys = desired;
    for (let ii = common; ii < desired.length; ii++) {
        for (const options of seriesPair(card, desired[ii].path)) card.chart.addSeries(options);
    }
}

/**
 * Push fresh data into a card's chart and texts. Never rebuilds unless the y distribution flips.
 * `fromPoll` marks newly logged rows, which raise the "new data" indicator on a zoomed card.
 */
function refreshCard(card, fromPoll = false) {
    const previousCount = card.pointCount;
    computeRunSeries(card);
    const chart = card.chart;
    if (chart && wantsLog(card) !== card.logActive) {
        rebuildChart(card);
    } else if (chart) {
        const zoomed = cardRange(card) !== null;
        chart.setData(joinChartData(card), !zoomed);
        if (fromPoll && zoomed && card.pointCount > previousCount) card.newDataEl.hidden = false;
    }
    renderCardText(card);
}

function setSeriesVisibility(card) {
    card.runKeys.forEach(({ path }, ii) => {
        const visible = !state.hidden.has(path);
        card.chart.setSeries(1 + 2 * ii, { show: visible && state.smoothing > 0 });
        card.chart.setSeries(2 + 2 * ii, { show: visible });
    });
}

// ---- card dom ----

function renderCardText(card) {
    const focused = state.focused;
    card.valueEl.replaceChildren();
    if (focused) {
        const value = card.runSeries.has(focused) ? formatValue(lastSmoothed(card, focused)) : 'n/a';
        card.valueEl.append(`${shortName(focused)} `, el('b', 'num', value));
    }
    const legend = selectedPaths().map((path) => {
        const item = el('span', `leg${state.hidden.has(path) ? ' hidden' : ''}`);
        item.tabIndex = 0;
        item.title = `${path}. Click to show or hide it in every chart`;
        const value = card.runSeries.has(path) ? formatValue(lastSmoothed(card, path)) : 'n/a';
        item.append(swatch(runColor(path), runDash(path) !== null), `${shortName(path)} `, el('b', 'num', value));
        item.dataset.path = path;
        return item;
    });
    card.footEl.replaceChildren(...legend);
    card.logButton.disabled = !card.canLog;
    card.logButton.title = card.canLog ? 'Log scale on the y axis (y)' : 'Log scale needs every value above zero';
    card.logButton.setAttribute('aria-pressed', wantsLog(card));
}

function createCardElement(card) {
    const article = el('article', 'card');
    article.dataset.metric = card.metric;
    const head = el('div', 'card-head');
    const title = createCardTitle(card.metric);
    card.valueEl = el('span', 'val');
    card.logButton = el('button', 'tbtn', 'Log');
    const wideButton = el('button', 'tbtn');
    wideButton.append(icon('expand'));
    wideButton.title = 'Full width';
    wideButton.setAttribute('aria-label', 'Full width');
    wideButton.setAttribute('aria-pressed', 'false');
    head.append(title, card.valueEl, card.logButton, wideButton);

    card.plotEl = el('div', 'plot skeleton');
    card.newDataEl = el('button', 'newdata');
    card.newDataEl.append(icon('refresh'), 'New data');
    card.newDataEl.title = 'New data arrived outside the zoomed range. Click to reset zoom.';
    card.newDataEl.hidden = true;
    card.plotEl.append(card.newDataEl);
    card.footEl = el('div', 'card-foot');
    article.append(head, card.plotEl, card.footEl);

    card.logButton.addEventListener('click', () => toggleLogy(card));
    wideButton.addEventListener('click', () => {
        const wide = article.classList.toggle('wide');
        wideButton.setAttribute('aria-pressed', wide);
    });
    card.newDataEl.addEventListener('click', () => resetZoom(state.syncZoom ? null : card));
    card.footEl.addEventListener('click', (event) => {
        const item = event.target.closest('.leg');
        if (!item) return;
        toggleHidden(item.dataset.path);
        emit('visibility');
    });
    card.footEl.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        event.target.closest('.leg')?.click();
    });
    return article;
}

const intersectionObserver = new IntersectionObserver((entries) => {
    for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const card = cards.get(entry.target.closest('.card').dataset.metric);
        if (card) createChart(card);
        intersectionObserver.unobserve(entry.target);
    }
}, { rootMargin: '300px 0px' });

const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
        const card = cards.get(entry.target.closest('.card').dataset.metric);
        card?.chart?.setSize({ width: Math.max(0, card.plotEl.clientWidth - PLOT_PADDING_PX), height: card.plotEl.clientHeight });
    }
});

function addCard(metric, highlight) {
    const card = { metric, group: topGroupOf(metric), chart: null, runKeys: [], runSeries: new Map(), pointCount: 0, canLog: false, logActive: false, xRange: null };
    card.el = createCardElement(card);
    cards.set(metric, card);
    computeRunSeries(card);
    renderCardText(card);
    if (highlight) {
        card.el.classList.add('fresh');
        setTimeout(() => card.el.classList.remove('fresh'), FRESH_HIGHLIGHT_MS);
    }
    intersectionObserver.observe(card.plotEl);
    resizeObserver.observe(card.plotEl);
}

function removeCard(card) {
    intersectionObserver.unobserve(card.plotEl);
    resizeObserver.unobserve(card.plotEl);
    card.chart?.destroy();
    card.el.remove();
    cards.delete(card.metric);
    if (hoveredCard === card) hoveredCard = null;
}

/** Create cards for new metric columns and drop cards no selected run has; re-sort only on request. */
function syncCards(resort) {
    const names = metricNames();
    const nameSet = new Set(names);
    for (const card of [...cards.values()]) {
        if (!nameSet.has(card.metric)) removeCard(card);
    }
    for (const metric of names) {
        if (!cards.has(metric)) addCard(metric, firstLoadDone);
    }
    groupTree.sync(names.map((metric) => ({ key: metric, el: cards.get(metric).el })), resort);
    if (selectedPaths().some((path) => peekRunData(path)?.metrics.loaded)) firstLoadDone = true;
}

// ---- section rendering ----

function renderGroupChips() {
    const chipsEl = document.getElementById('groupChips');
    const counts = new Map();
    for (const card of cards.values()) counts.set(card.group, (counts.get(card.group) ?? 0) + 1);
    chipsEl.replaceChildren(...[...counts].filter(([name]) => name !== UNGROUPED).map(([name, count]) => {
        const chip = el('button', 'gchip', `${name}/`);
        chip.append(el('span', 'count', String(count)));
        chip.title = `Show only ${name}/*`;
        chip.setAttribute('aria-pressed', state.metricFilter === `${name}/`);
        chip.addEventListener('click', () => {
            setMetricFilter(state.metricFilter === `${name}/` ? '' : `${name}/`);
        });
        return chip;
    }));
}

export function setMetricFilter(filter) {
    state.metricFilter = filter;
    filterInput.value = filter;
    applyFilter();
    emit('filter');
}

function renderStatus(visibleCount) {
    statusEl.replaceChildren();
    const paths = selectedPaths();
    if (!uPlot) {
        statusEl.append(el('div', 'strip error', 'uPlot failed to load from cdn.jsdelivr.net, charts need network access to it.'));
    }
    if (paths.length && !paths.some((path) => peekRunData(path)?.metrics.loaded)) {
        const skeletons = el('div', 'grid');
        for (let ii = 0; ii < 3; ii++) {
            const card = el('div', 'card');
            card.append(el('div', 'card-head'), el('div', 'plot skeleton'), el('div', 'card-foot'));
            skeletons.append(card);
        }
        statusEl.append(skeletons);
    } else if (paths.length && !cards.size) {
        statusEl.append(el('div', 'muted', 'No metrics logged yet for the selected runs.'));
    } else if (cards.size && !visibleCount) {
        const message = el('div', 'muted', `No metric matches "${state.metricFilter}". `);
        const clear = el('button', 'linkbtn', 'Clear filter');
        clear.addEventListener('click', () => setMetricFilter(''));
        message.append(clear);
        statusEl.append(message);
    }
}

function renderErrors() {
    const strips = [];
    for (const path of selectedPaths()) {
        const error = peekRunData(path)?.metrics.error;
        if (!error) continue;
        const strip = el('div', 'strip error', `${shortName(path)}: ${error} `);
        strip.title = path;
        const retry = el('button', 'linkbtn', 'Retry');
        retry.addEventListener('click', async () => {
            await pollMetrics(path).catch(() => false);
            metricsDataChanged();
        });
        strip.append(retry);
        strips.push(strip);
    }
    errorsEl.replaceChildren(...strips);
}

/** Show or hide cards and groups by filter and collapse state, and update the counts. */
function applyFilter() {
    for (const card of cards.values()) card.el.hidden = !metricMatches(card.metric);
    const visibleCount = groupTree.refresh();
    const groupCount = groupTree.topLevelCount;
    const shown = visibleCount === cards.size ? String(cards.size) : `${visibleCount} of ${cards.size}`;
    document.getElementById('metricCount').textContent = cards.size
        ? `${shown} metrics${groupCount > 1 ? ` in ${groupCount} groups` : ''}` : '';
    const allLog = cards.size > 0 && [...cards.values()].every((card) => state.logy.has(card.metric));
    document.getElementById('logyAll').setAttribute('aria-pressed', allLog);
    renderGroupChips();
    renderStatus(visibleCount);
}

// ---- log-y ----

function toggleLogy(card) {
    if (state.logy.has(card.metric)) state.logy.delete(card.metric);
    else state.logy.add(card.metric);
    refreshCard(card);
    applyFilter();
    emit('logy');
}

export function toggleLogyHovered() {
    if (hoveredCard) toggleLogy(hoveredCard);
}

export function toggleLogyAll() {
    const visible = [...cards.values()].filter((card) => !card.el.hidden);
    const enable = !visible.every((card) => state.logy.has(card.metric));
    for (const card of visible) {
        if (enable) state.logy.add(card.metric);
        else state.logy.delete(card.metric);
        refreshCard(card);
    }
    applyFilter();
    emit('logy');
}

// ---- reactions to state changes ----

/** Called by the poller when any selected run's metrics changed. */
export function metricsDataChanged() {
    syncCards(false);
    for (const card of cards.values()) refreshCard(card, true);
    renderErrors();
    applyFilter();
}

function handleSelection() {
    syncCards(true);
    for (const card of cards.values()) {
        if (card.chart) syncSeries(card);
        refreshCard(card);
    }
    renderErrors();
    applyFilter();
}

function handleVisibility() {
    for (const card of cards.values()) {
        if (card.chart) setSeriesVisibility(card);
        renderCardText(card);
    }
}

function handleSmoothing() {
    for (const card of cards.values()) {
        if (card.chart) setSeriesVisibility(card);
        refreshCard(card);
    }
}

function handleTheme() {
    const style = getComputedStyle(document.documentElement);
    themeColors.axis = style.getPropertyValue('--axis').trim();
    themeColors.grid = style.getPropertyValue('--grid').trim();
    themeColors.surface = style.getPropertyValue('--surface').trim();
    for (const card of cards.values()) {
        card.chart?.redraw(false, true);
        renderCardText(card);
    }
}

export function initMetrics() {
    filterInput.value = state.metricFilter;
    filterInput.addEventListener('input', () => setMetricFilter(filterInput.value));
    document.getElementById('logyAll').addEventListener('click', toggleLogyAll);
    document.getElementById('resetZoom').addEventListener('click', () => resetZoom());
    const syncButton = document.getElementById('syncZoom');
    syncButton.addEventListener('click', () => {
        setSyncZoom(!state.syncZoom);
        syncButton.setAttribute('aria-pressed', state.syncZoom);
    });
    handleTheme();
    on('selection', handleSelection);
    on('visibility', handleVisibility);
    on('smoothing', handleSmoothing);
    on('theme', handleTheme);
}
