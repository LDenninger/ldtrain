// Log card (tail with level coloring, filters and follow mode) and config card (highlighted raw text).

import { loadRunInfo } from './data.js';
import { on, selectedPaths, state } from './state.js';
import { ApiError, el, fetchJson, fetchRunFileText, icon, shortName } from './util.js';

const MAX_LOG_LINES = 2000;
const LEVEL_PREFIX = /^(DEBUG|WARNING|ERROR|CRITICAL|DEV) (?=\[)/;
const LOG_HEADER = /^((?:\[[^\]]*\])+)(.*)$/;
const LEVEL_LABELS = { WARNING: 'WARN', CRITICAL: 'CRIT' };
const LEVEL_CHIPS = { DEBUG: 'DEBUG', DEV: 'DEBUG', INFO: 'INFO', WARNING: 'WARN', ERROR: 'ERROR', CRITICAL: 'ERROR' };
const COPY_FEEDBACK_MS = 1500;

const logBody = document.getElementById('logBody');
const logRunSelect = document.getElementById('logRun');
const logRankSelect = document.getElementById('logRank');
const logError = document.getElementById('logError');
const newLinesButton = document.getElementById('logNewLines');
const followButton = document.getElementById('logFollow');
const cfgBody = document.getElementById('cfgBody');
const cfgRunSelect = document.getElementById('cfgRun');
const cfgError = document.getElementById('cfgError');
const cfgTitle = document.getElementById('cfgTitle');
const copyButton = document.getElementById('cfgCopy');

const log = {
    run: null,
    chosenByUser: false,
    rank: 0,
    ranks: [],
    offset: null,
    generation: 0,
    inFlight: false,
    lastLevel: 'INFO',
    chipsOn: { DEBUG: true, INFO: true, WARN: true, ERROR: true },
    filter: '',
    follow: true,
    pinned: true,
    unseen: 0,
};

const config = { run: null, chosenByUser: false, text: '', generation: 0 };

// ---- shared ----

function fillRunSelect(select, current) {
    const options = selectedPaths().map((path) => {
        const option = el('option', '', `${shortName(path)}${path === state.focused ? ' (focused)' : ''}`);
        option.value = path;
        option.title = path;
        option.selected = path === current;
        return option;
    });
    select.replaceChildren(...options);
    select.disabled = options.length === 0;
}

function showError(stripEl, message) {
    stripEl.textContent = message ?? '';
    stripEl.hidden = !message;
}

/** The run a single-run card should show: the user's pick while still selected, else the focused run. */
function pickRun(card) {
    if (card.chosenByUser && state.selected.has(card.run)) return card.run;
    card.chosenByUser = false;
    return state.focused;
}

// ---- log lines ----

function parseLogLine(text) {
    if (text.startsWith('    ')) return { level: log.lastLevel, continuation: true, header: '', message: text };
    const prefix = LEVEL_PREFIX.exec(text);
    const level = prefix ? prefix[1] : 'INFO';
    const rest = prefix ? text.slice(prefix[0].length) : text;
    const header = LOG_HEADER.exec(rest);
    log.lastLevel = level;
    return { level, continuation: false, header: header ? header[1] : '', message: header ? header[2] : rest };
}

function isLineVisible(line) {
    if (!log.chipsOn[line.dataset.chip]) return false;
    return !log.filter || line.textContent.toLowerCase().includes(log.filter);
}

function createLogLine(text) {
    const parsed = parseLogLine(text);
    const line = el('div', `logline ${parsed.level}${parsed.continuation ? ' cont' : ''}`);
    line.dataset.chip = LEVEL_CHIPS[parsed.level];
    line.append(el('span', 'lv', parsed.continuation ? '' : (LEVEL_LABELS[parsed.level] ?? parsed.level)));
    if (parsed.header) line.append(el('span', 't', parsed.header));
    line.append(parsed.message);
    line.hidden = !isLineVisible(line);
    return line;
}

function isScrolledToBottom() {
    return logBody.scrollHeight - logBody.scrollTop - logBody.clientHeight < 4;
}

function scrollLogToBottom() {
    logBody.scrollTop = logBody.scrollHeight;
    log.pinned = true;
    log.unseen = 0;
    newLinesButton.hidden = true;
}

function appendLogText(text) {
    const texts = text.split('\n');
    if (texts[texts.length - 1] === '') texts.pop();
    if (!texts.length) return;
    const fragment = document.createDocumentFragment();
    let visibleCount = 0;
    for (const lineText of texts.slice(-MAX_LOG_LINES)) {
        const line = createLogLine(lineText);
        if (!line.hidden) visibleCount++;
        fragment.append(line);
    }
    const stickToBottom = log.follow && log.pinned;
    logBody.append(fragment);
    while (logBody.childElementCount > MAX_LOG_LINES) logBody.firstElementChild.remove();
    if (stickToBottom) {
        scrollLogToBottom();
    } else if (visibleCount) {
        log.unseen += visibleCount;
        newLinesButton.replaceChildren(icon('arrow-down'), `${log.unseen} new line${log.unseen === 1 ? '' : 's'}`);
        newLinesButton.hidden = false;
    }
}

function applyLogFilters() {
    for (const line of logBody.children) {
        if (line.dataset.chip) line.hidden = !isLineVisible(line);
    }
    if (log.follow && log.pinned) scrollLogToBottom();
}

// ---- log polling ----

function resetLogBody(message) {
    logBody.replaceChildren();
    if (message) logBody.append(el('div', 'logline muted', message));
    log.unseen = 0;
    log.pinned = true;
    newLinesButton.hidden = true;
}

/** Fetch the log lines appended since the last call for the shown run and rank. */
export async function pollLog() {
    if (!log.run || !log.ranks.length || log.inFlight) return;
    const generation = log.generation;
    log.inFlight = true;
    try {
        const chunk = await fetchJson('/api/log', { run: log.run, rank: log.rank, offset: log.offset });
        if (generation !== log.generation) return;
        if (log.offset === null && chunk.truncated) {
            logBody.append(el('div', 'logline muted', 'Earlier lines not shown, the log is longer than 256 KB'));
        }
        log.offset = chunk.offset;
        showError(logError, null);
        appendLogText(chunk.text);
    } catch (error) {
        if (!(error instanceof ApiError)) throw error;
        if (generation === log.generation) showError(logError, `log unreadable: ${error.message}`);
    } finally {
        log.inFlight = false;
    }
}

async function showLog(run, rank) {
    log.run = run;
    log.generation++;
    log.offset = null;
    log.lastLevel = 'INFO';
    log.inFlight = false;
    showError(logError, null);
    fillRunSelect(logRunSelect, run);
    if (!run) {
        log.ranks = [];
        logRankSelect.replaceChildren();
        resetLogBody('No run selected.');
        return;
    }
    const generation = log.generation;
    resetLogBody();
    try {
        log.ranks = (await loadRunInfo(run)).ranks;
    } catch (error) {
        if (generation === log.generation) showError(logError, `run info unreadable: ${error.message}`);
        return;
    }
    if (generation !== log.generation) return;
    log.rank = log.ranks.includes(rank) ? rank : (log.ranks[0] ?? 0);
    logRankSelect.replaceChildren(...log.ranks.map((value) => {
        const option = el('option', '', `rank ${value}`);
        option.value = value;
        option.selected = value === log.rank;
        return option;
    }));
    logRankSelect.disabled = log.ranks.length < 2;
    if (!log.ranks.length) {
        resetLogBody('No log file in logs/ for this run.');
        return;
    }
    await pollLog().catch(() => undefined);  // an offline error is reported by the next regular poll
}

// ---- config ----

const YAML_KEY = /^(\s*(?:- )?)([^\s#:][^:#]*?)(:)(\s.*|)$/;
const YAML_SCALAR = /^([-+]?(\d[\d_]*\.?\d*|\.\d+)([eE][-+]?\d+)?|true|false|null|True|False|None|~)$/;

function highlightValue(rest) {
    const commentStart = rest.search(/\s#/);
    const value = commentStart >= 0 ? rest.slice(0, commentStart) : rest;
    const comment = commentStart >= 0 ? rest.slice(commentStart) : '';
    const trimmed = value.trim();
    const nodes = [value.slice(0, value.length - value.trimStart().length)];
    if (YAML_SCALAR.test(trimmed)) nodes.push(el('span', 'n', trimmed));
    else if (/^(".*"|'.*')$/.test(trimmed)) nodes.push(el('span', 's', trimmed));
    else nodes.push(trimmed);
    nodes.push(value.slice(value.trimEnd().length));
    if (comment) nodes.push(el('span', 'c', comment));
    return nodes;
}

function highlightLine(line) {
    if (/^\s*#/.test(line)) return [el('span', 'c', line)];
    const match = YAML_KEY.exec(line);
    if (!match) return [line];
    const [, indent, key, colon, rest] = match;
    return [indent, el('span', 'k', key), colon, ...highlightValue(rest)];
}

function renderConfigText(text) {
    const fragment = document.createDocumentFragment();
    for (const line of text.split('\n')) fragment.append(...highlightLine(line), '\n');
    cfgBody.replaceChildren(fragment);
}

async function showConfig(run) {
    config.run = run;
    config.text = '';
    config.generation++;
    const generation = config.generation;
    fillRunSelect(cfgRunSelect, run);
    showError(cfgError, null);
    cfgTitle.textContent = 'Config';
    copyButton.disabled = true;
    if (!run) {
        cfgBody.replaceChildren(el('span', 'muted', 'No run selected.'));
        return;
    }
    cfgBody.replaceChildren();
    try {
        const info = await loadRunInfo(run);
        if (generation !== config.generation) return;
        const configPath = info.config_files[0];
        if (!configPath) {
            cfgBody.replaceChildren(el('span', 'muted', 'No config file under configs/ for this run.'));
            return;
        }
        cfgTitle.textContent = shortName(configPath);
        cfgTitle.title = `${run}/${configPath}`;
        const text = await fetchRunFileText(run, configPath);
        if (generation !== config.generation) return;
        config.text = text;
        copyButton.disabled = false;
        renderConfigText(text);
    } catch (error) {
        if (generation === config.generation) showError(cfgError, `config unreadable: ${error.message}`);
    }
}

async function copyConfig() {
    try {
        await navigator.clipboard.writeText(config.text);
        copyButton.textContent = 'Copied';
    } catch {
        copyButton.textContent = 'Copy failed';
    }
    setTimeout(() => (copyButton.textContent = 'Copy'), COPY_FEEDBACK_MS);
}

// ---- wiring ----

function handleSelection() {
    const logRun = pickRun(log);
    if (logRun !== log.run) showLog(logRun, log.rank);
    else fillRunSelect(logRunSelect, log.run);
    const configRun = pickRun(config);
    if (configRun !== config.run) showConfig(configRun);
    else fillRunSelect(cfgRunSelect, config.run);
    document.getElementById('lcRun').textContent = state.focused ? `focused on ${shortName(state.focused)}` : '';
}

function buildLevelChips() {
    const chipsEl = document.getElementById('lvlChips');
    for (const chip of Object.keys(log.chipsOn)) {
        const button = el('button', `lvl ${chip}`, chip);
        button.setAttribute('aria-pressed', 'true');
        button.title = `Show ${chip} lines`;
        button.addEventListener('click', () => {
            log.chipsOn[chip] = !log.chipsOn[chip];
            button.setAttribute('aria-pressed', log.chipsOn[chip]);
            applyLogFilters();
        });
        chipsEl.append(button);
    }
}

export function initLogConfig() {
    buildLevelChips();
    logRunSelect.addEventListener('change', () => {
        log.chosenByUser = true;
        showLog(logRunSelect.value, log.rank);
    });
    logRankSelect.addEventListener('change', () => showLog(log.run, Number(logRankSelect.value)));
    cfgRunSelect.addEventListener('change', () => {
        config.chosenByUser = true;
        showConfig(cfgRunSelect.value);
    });
    document.getElementById('logFilter').addEventListener('input', (event) => {
        log.filter = event.target.value.trim().toLowerCase();
        applyLogFilters();
    });
    followButton.addEventListener('click', () => {
        log.follow = !log.follow;
        followButton.setAttribute('aria-pressed', log.follow);
        if (log.follow) scrollLogToBottom();
    });
    logBody.addEventListener('scroll', () => {
        log.pinned = isScrolledToBottom();
        if (log.pinned && log.follow) {
            log.unseen = 0;
            newLinesButton.hidden = true;
        }
    });
    newLinesButton.addEventListener('click', scrollLogToBottom);
    copyButton.addEventListener('click', copyConfig);
    on('selection', handleSelection);
}
