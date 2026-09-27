// Run tree in the sidebar: rendering, search filter, mouse selection and keyboard navigation.

import {
    clearSelection, emit, findNode, focusRun, isLive, on, runColor, runsBelow, selectRun,
    soloRun, state, toggleFolder, toggleRun,
} from './state.js';
import { el, formatAge, formatIteration, savePref, shortName } from './util.js';

const treeEl = document.getElementById('tree');
const INDENT_PX = 14;

// ---- rendering ----

function matchesFilter(node) {
    const query = state.treeFilter.trim().toLowerCase();
    return !query || runsBelow(node).some((run) => run.path.toLowerCase().includes(query));
}

function isExpanded(folder) {
    return !state.collapsedFolders.has(folder.path) || state.treeFilter.trim() !== '';
}

function buildCheckbox(checked, partial, color) {
    const box = el('span', `cb${checked ? ' on' : partial ? ' part' : ''}`, checked ? '✓' : '');
    box.style.setProperty('--c', color);
    return box;
}

function buildRunRow(run) {
    const selected = state.selected.has(run.path);
    const row = el('div', `row${state.focused === run.path ? ' focused' : ''}`);
    row.append(el('span', 'caret'), buildCheckbox(selected, false, selected ? runColor(run.path) : 'transparent'));
    const name = el('span', 'name', run.name);
    name.title = run.path;
    const meta = el('span', 'meta');
    meta.append(el('span', 'num', formatIteration(run.last_iteration)));
    const live = isLive(run);
    const status = el('span', `st ${live ? 'live' : 'done'}`, live ? '●' : '✓');
    status.title = run.mtime === null ? 'no metrics or log yet'
        : `${live ? 'live' : 'done'}: updated ${formatAge(Date.now() / 1000 - run.mtime)} ago`;
    meta.append(status);
    row.append(name, meta);
    return row;
}

function buildFolderRow(folder) {
    const runs = runsBelow(folder);
    const selectedCount = runs.filter((run) => state.selected.has(run.path)).length;
    const row = el('div', 'row folder');
    const all = selectedCount === runs.length;
    row.append(el('span', 'caret', isExpanded(folder) ? '▾' : '▸'), buildCheckbox(all, selectedCount > 0, 'var(--text-muted)'));
    row.append(el('span', 'name', folder.name), el('span', 'meta', `${selectedCount ? `${selectedCount}/` : ''}${runs.length}`));
    return row;
}

function buildItem(node, parentEl) {
    if (!matchesFilter(node)) return;
    const item = el('li');
    item.setAttribute('role', 'treeitem');
    item.setAttribute('aria-level', node.depth + 1);
    item.dataset.path = node.path;
    item.tabIndex = -1;
    const row = node.is_run ? buildRunRow(node) : buildFolderRow(node);
    row.style.paddingLeft = `${8 + node.depth * INDENT_PX}px`;
    item.append(row);
    parentEl.append(item);
    if (node.is_run) {
        item.setAttribute('aria-selected', state.selected.has(node.path));
        return;
    }
    item.setAttribute('aria-expanded', isExpanded(node));
    if (isExpanded(node)) {
        const group = el('ul');
        group.setAttribute('role', 'group');
        item.append(group);
        for (const child of node.children) buildItem(child, group);
    }
}

function renderEmptyRoot() {
    const hint = el('li', 'tree-empty');
    hint.append(el('div', '', 'No runs under'), el('div', 'num', state.root));
    hint.append(el('div', 'muted', 'A run is a folder holding one of:'));
    hint.append(el('pre', 'num', '<run>/metrics/metrics.csv\n<run>/logs/log.txt\n<run>/visuals/<step>/…'));
    treeEl.append(hint);
}

export function renderTree() {
    if (!state.tree) return;
    const activePath = treeEl.contains(document.activeElement) ? document.activeElement.dataset.path : null;
    treeEl.replaceChildren();
    if (state.runs.size === 0) renderEmptyRoot();
    for (const child of state.tree.children) buildItem(child, treeEl);

    const items = visibleItems();
    const target = items.find((item) => item.dataset.path === activePath)
        ?? items.find((item) => item.dataset.path === state.focused) ?? items[0];
    if (target) {
        target.tabIndex = 0;
        if (activePath !== null) target.focus({ preventScroll: true });
    }
    renderSummary();
}

function renderSummary() {
    const count = state.selected.size;
    const focused = state.focused ? shortName(state.focused) : '—';
    document.getElementById('selInfo').textContent = count ? `${count} selected · focus ${focused}` : 'nothing selected';
    const folderCount = [...state.folders.values()].filter((folder) => folder.path !== '').length;
    document.getElementById('treeStats').textContent = `${state.runs.size} runs · ${folderCount} folders`;
}

// ---- interaction ----

function visibleItems() {
    return [...treeEl.querySelectorAll('[role=treeitem]')];
}

function moveTabStop(item) {
    for (const other of visibleItems()) other.tabIndex = -1;
    item.tabIndex = 0;
    item.focus({ preventScroll: true });
    item.scrollIntoView({ block: 'nearest' });
}

function setFolderExpanded(folder, expanded) {
    if (expanded) state.collapsedFolders.delete(folder.path);
    else state.collapsedFolders.add(folder.path);
    savePref('collapsedFolders', [...state.collapsedFolders]);
    renderTree();
}

function selectRange(fromPath, toPath) {
    const runPaths = visibleItems().map((item) => item.dataset.path).filter((path) => state.runs.has(path));
    const [start, end] = [runPaths.indexOf(fromPath), runPaths.indexOf(toPath)].sort((a, b) => a - b);
    if (start < 0) return;
    for (const path of runPaths.slice(start, end + 1)) selectRun(path);
}

function handleClick(event) {
    const item = event.target.closest('[role=treeitem]');
    if (!item) return;
    const node = findNode(item.dataset.path);
    if (!node) return;
    moveTabStop(item);
    const onCheckbox = event.target.classList.contains('cb');

    if (!node.is_run) {
        if (onCheckbox) {
            toggleFolder(node);
            emit('selection');
        } else {
            setFolderExpanded(node, !isExpanded(node));
        }
        return;
    }
    if (onCheckbox || event.ctrlKey || event.metaKey) toggleRun(node.path);
    else if (event.altKey) soloRun(node.path);
    else if (event.shiftKey && state.lastClicked) selectRange(state.lastClicked, node.path);
    else focusRun(node.path);
    state.lastClicked = node.path;
    emit('selection');
}

function handleKeydown(event) {
    const item = document.activeElement?.closest('[role=treeitem]');
    if (!item || event.ctrlKey || event.metaKey || event.altKey) return;
    const items = visibleItems();
    const index = items.indexOf(item);
    const node = findNode(item.dataset.path);
    if (!node) return;

    switch (event.key) {
        case 'ArrowDown': if (items[index + 1]) moveTabStop(items[index + 1]); break;
        case 'ArrowUp': if (items[index - 1]) moveTabStop(items[index - 1]); break;
        case 'Home': moveTabStop(items[0]); break;
        case 'End': moveTabStop(items[items.length - 1]); break;
        case 'ArrowRight':
            if (!node.is_run && !isExpanded(node)) setFolderExpanded(node, true);
            else if (!node.is_run && items[index + 1]) moveTabStop(items[index + 1]);
            break;
        case 'ArrowLeft': {
            if (!node.is_run && isExpanded(node) && !state.treeFilter.trim()) {
                setFolderExpanded(node, false);
                break;
            }
            const parent = items.find((other) => other.dataset.path === node.parentPath);
            if (parent) moveTabStop(parent);
            break;
        }
        case ' ':
            if (node.is_run) toggleRun(node.path);
            else toggleFolder(node);
            emit('selection');
            break;
        case 'Enter':
            if (!node.is_run) return;
            focusRun(node.path);
            emit('selection');
            break;
        case 'o':
            if (!node.is_run) return;
            soloRun(node.path);
            emit('selection');
            break;
        default:
            return;
    }
    event.preventDefault();
    event.stopPropagation();
}

/** Move keyboard focus into the tree, onto its current tab stop. */
export function focusTree() {
    const target = visibleItems().find((item) => item.tabIndex === 0);
    target?.focus();
}

export function initTree() {
    treeEl.addEventListener('click', handleClick);
    treeEl.addEventListener('keydown', handleKeydown);
    const search = document.getElementById('treeSearch');
    search.addEventListener('input', () => {
        state.treeFilter = search.value;
        renderTree();
    });
    search.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowDown') {
            event.preventDefault();
            focusTree();
        }
    });
    document.getElementById('clearSel').addEventListener('click', () => {
        clearSelection();
        emit('selection');
    });
    on('tree', renderTree);
    on('selection', renderTree);
    on('theme', renderTree);
}
