// Nested card groups shared by the metric and media sections: one collapsible group per `/` prefix of a key.

import { emit, state } from './state.js';
import { el, icon, placeAt, setCaret } from './util.js';

const SEPARATOR = '/';
const UNGROUPED = '';

/** The group a key belongs to: everything before its last `/`, '' for an ungrouped key. */
export function groupPathOf(key) {
    const index = key.lastIndexOf(SEPARATOR);
    return index > 0 ? key.slice(0, index) : UNGROUPED;
}

/** The top-level group of a key: its first segment, '' for an ungrouped key. */
export function topGroupOf(key) {
    return groupPathOf(key).split(SEPARATOR)[0];
}

/** A card title with the group path muted: `train/loss/` + `total`. */
export function createCardTitle(key) {
    const title = el('span', 'card-title');
    title.title = key;
    const groupPath = groupPathOf(key);
    if (groupPath) title.append(el('span', 'pre', `${groupPath}${SEPARATOR}`), key.slice(groupPath.length + 1));
    else title.append(key);
    return title;
}

/**
 * Cards grouped by the `/` segments of their keys, one nested `.group` per prefix, into any depth.
 * A group lists its own cards in a grid first, then its subgroups. Ungrouped keys share one
 * top-level group. `collapsePrefix` namespaces the collapse keys kept in `state.collapsed`, and
 * `gridClass` is added to every group's card grid for section-specific card layout.
 */
export class GroupTree {
    constructor(container, collapsePrefix = '', gridClass = '') {
        this.container = container;
        this.collapsePrefix = collapsePrefix;
        this.gridClass = gridClass;
        this.groups = new Map();  // group path -> group
    }

    /** Number of top-level groups, the ungrouped one included. */
    get topLevelCount() {
        return this.container.children.length;
    }

    /**
     * Put every card into its group, creating missing groups, and drop groups left empty.
     * `items` are `{ key, el }` in sort order. With `resort`, groups and cards are re-ordered at
     * every level: a group sits where its first card sorts. Without it, new cards and groups are
     * appended at the end of their level.
     */
    sync(items, resort) {
        for (const { key, el: cardEl } of items) {
            cardEl.dataset.key = key;
            const grid = this.getGroup(groupPathOf(key)).grid;
            if (cardEl.parentNode !== grid) grid.append(cardEl);
        }
        if (resort) this.arrange(items);
        for (const [path, group] of [...this.groups].reverse()) {
            if (group.grid.children.length || group.subgroupsEl.children.length) continue;
            group.el.remove();
            this.groups.delete(path);
        }
    }

    /** Update the collapse state, visible counts and previews of every group from the cards' `hidden`. */
    refresh() {
        let visibleCount = 0;
        for (const groupEl of this.container.children) visibleCount += this.refreshGroup(this.groups.get(groupEl.dataset.path)).length;
        return visibleCount;
    }

    // ---- internals ----

    /** The element a group with this path is placed in: the container, or its parent's subgroups. */
    parentElOf(path) {
        return path.includes(SEPARATOR) ? this.getGroup(groupPathOf(path)).subgroupsEl : this.container;
    }

    getGroup(path) {
        if (this.groups.has(path)) return this.groups.get(path);
        const parentEl = this.parentElOf(path);
        const group = { path, el: el('div', 'group'), grid: el('div', `grid ${this.gridClass}`.trim()), subgroupsEl: el('div', 'subgroups') };
        group.el.dataset.path = path;
        const header = el('h3');
        header.tabIndex = 0;
        header.setAttribute('role', 'button');
        header.title = path || 'ungrouped';
        group.caretEl = icon('chevron-down', 'icon caret');
        group.countEl = el('span', 'count');
        group.previewEl = el('span', 'preview');
        const label = path.includes(SEPARATOR) ? path.slice(groupPathOf(path).length + 1) : path || 'ungrouped';
        header.append(group.caretEl, label, group.countEl, group.previewEl);
        group.headerEl = header;
        group.el.append(header, group.grid, group.subgroupsEl);
        header.addEventListener('click', () => this.toggle(path));
        header.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            this.toggle(path);
        });
        this.groups.set(path, group);
        parentEl.append(group.el);
        return group;
    }

    collapseKey(path) {
        return `${this.collapsePrefix}${path || 'ungrouped'}`;
    }

    toggle(path) {
        const key = this.collapseKey(path);
        if (state.collapsed.has(key)) state.collapsed.delete(key);
        else state.collapsed.add(key);
        this.refresh();
        emit('collapse');
    }

    /** Order the children of every level after the first card below them, touching only moved nodes. */
    arrange(items) {
        const order = new Map();  // parent element -> Set of children in order
        const push = (parentEl, node) => {
            if (!order.has(parentEl)) order.set(parentEl, new Set());
            order.get(parentEl).add(node);
        };
        for (const { key, el: cardEl } of items) {
            let path = groupPathOf(key);
            push(this.groups.get(path).grid, cardEl);
            for (;;) {
                const parentEl = this.parentElOf(path);
                push(parentEl, this.groups.get(path).el);
                if (parentEl === this.container) break;
                path = groupPathOf(path);
            }
        }
        for (const [parentEl, nodes] of order) [...nodes].forEach((node, index) => placeAt(parentEl, node, index));
    }

    /** Refresh one group and its subgroups, returning the keys of its visible cards. */
    refreshGroup(group) {
        const keys = [...group.grid.children].filter((cardEl) => !cardEl.hidden).map((cardEl) => cardEl.dataset.key);
        for (const childEl of group.subgroupsEl.children) keys.push(...this.refreshGroup(this.groups.get(childEl.dataset.path)));
        const collapsed = state.collapsed.has(this.collapseKey(group.path));
        group.el.hidden = keys.length === 0;
        group.el.classList.toggle('collapsed', collapsed);
        group.headerEl.setAttribute('aria-expanded', !collapsed);
        setCaret(group.caretEl, !collapsed);
        group.countEl.textContent = String(keys.length);
        group.previewEl.textContent = keys.map((key) => key.slice(group.path ? group.path.length + 1 : 0)).join(', ');
        return keys;
    }
}
