// Media cards: one card per visual tag, one tile per selected run, a shared step slider and a lightbox.

import { peekRunData } from './data.js';
import { emit, on, runColor, selectedPaths, state } from './state.js';
import { el, icon, runFileUrl, shortName } from './util.js';

const mediaEl = document.getElementById('media');
const statusEl = document.getElementById('mediaStatus');
const lightboxEl = document.getElementById('lightbox');

const cards = new Map();  // tag -> card
let steps = [];           // ascending union of the steps of all selected runs
let entriesByRun = new Map();  // path -> Map(tag -> [{ step, file }] ascending)
let lightbox = null;      // { tag, path } while open

// ---- visual index ----

function collectVisuals() {
    const stepSet = new Set();
    const tagKinds = new Map();
    entriesByRun = new Map();
    for (const path of selectedPaths()) {
        const byTag = new Map();
        const visualSteps = peekRunData(path)?.visuals.steps ?? new Map();
        for (const step of [...visualSteps.keys()].sort((a, b) => a - b)) {
            for (const file of visualSteps.get(step)) {
                stepSet.add(step);
                tagKinds.set(file.tag, file.kind);
                if (!byTag.has(file.tag)) byTag.set(file.tag, []);
                byTag.get(file.tag).push({ step, file });
            }
        }
        entriesByRun.set(path, byTag);
    }
    steps = [...stepSet].sort((a, b) => a - b);
    return [...tagKinds.entries()].sort((a, b) => a[0].localeCompare(b[0]));
}

/** The slider step: the latest one when following, else the chosen step snapped to a known step. */
function currentStep() {
    if (state.mediaStep === null) return steps[steps.length - 1] ?? 0;
    return steps.filter((step) => step <= state.mediaStep).pop() ?? steps[0] ?? state.mediaStep;
}

/** The run's file for `tag` at the latest step ≤ `step`, null if it has none that early. */
function resolveEntry(path, tag, step) {
    const entries = entriesByRun.get(path)?.get(tag) ?? [];
    for (let ii = entries.length - 1; ii >= 0; ii--) {
        if (entries[ii].step <= step) return entries[ii];
    }
    return null;
}

// ---- tiles ----

function createMediaElement(kind, url, alt) {
    if (kind === 'video') {
        const video = el('video');
        Object.assign(video, { controls: true, loop: true, muted: true, playsInline: true, preload: 'metadata', src: url });
        video.addEventListener('error', () => video.replaceWith(createVideoFallback(url)));
        return video;
    }
    const image = el('img');
    image.alt = alt;
    image.src = url;
    return image;
}

/** Shown when the browser cannot decode a video, e.g. an mp4 encoded with the MPEG-4 Part 2 codec. */
function createVideoFallback(url) {
    const fallback = el('div', 'video-error', 'The browser cannot play this video codec. ');
    const link = el('a', '', 'download');
    link.href = url;
    link.download = '';
    fallback.append(link);
    return fallback;
}

/** Swap an image's source only once the new one has loaded, so polling never flashes a blank tile. */
function swapImage(image, url, alt) {
    const preload = new Image();
    preload.onload = () => {
        image.src = url;
        image.alt = alt;
    };
    preload.src = url;
}

function updateTile(tile, path, tag, kind, step) {
    const entry = resolveEntry(path, tag, step);
    tile.el.style.setProperty('--c', runColor(path));
    tile.el.style.opacity = state.hidden.has(path) ? 0.35 : 1;
    tile.nameEl.textContent = shortName(path);
    tile.nameEl.title = path;
    tile.el.classList.toggle('missing', !entry);
    if (!entry) {
        tile.filePath = null;
        tile.wrapEl.replaceChildren('No visual yet');
        tile.stepEl.textContent = '';
        return;
    }
    const url = runFileUrl(path, entry.file.path);
    const alt = `${tag} of ${shortName(path)} at step ${entry.step}`;
    const current = tile.wrapEl.firstElementChild;
    if (tile.filePath !== entry.file.path) {
        if (kind === 'image' && current?.tagName === 'IMG') swapImage(current, url, alt);
        else tile.wrapEl.replaceChildren(createMediaElement(kind, url, alt));
        tile.filePath = entry.file.path;
    }
    const offStep = entry.step !== step;
    tile.stepEl.className = `stp num${offStep ? ' off' : ''}`;
    tile.stepEl.textContent = `${offStep ? '≤ ' : ''}${entry.step.toLocaleString()}`;
    tile.stepEl.title = offStep ? 'No visual at this step, showing the closest earlier one' : '';
}

function createTile(card, path) {
    const tile = { el: el('figure', 'tile'), wrapEl: el('div', 'imgwrap'), nameEl: el('span', 'name'), stepEl: el('span', 'stp num'), filePath: null };
    const caption = el('figcaption');
    caption.append(tile.nameEl, tile.stepEl);
    tile.el.append(tile.wrapEl, caption);
    tile.el.addEventListener('click', (event) => {
        if (event.target.closest('video, a') || !tile.filePath) return;
        openLightbox(card.tag, path);
    });
    return tile;
}

// ---- cards ----

/**
 * Put `node` at child position `index` of `parent`, touching the DOM only when it is elsewhere.
 * Re-inserting a node detaches it first, which ends a slider drag in progress and pauses a playing video.
 */
function placeAt(parent, node, index) {
    const current = parent.children[index] ?? null;
    if (current !== node) parent.insertBefore(node, current);
}

function createCard(tag, kind) {
    const card = { tag, kind, el: el('article', 'card wide'), tiles: new Map() };
    const head = el('div', 'card-head');
    const [prefix, ...rest] = tag.split('/');
    const title = el('span', 'card-title');
    if (rest.length) title.append(el('span', 'pre', `${prefix}/`), rest.join('/'));
    else title.append(tag);
    const kindEl = el('span', 'kind');
    kindEl.append(icon(kind === 'video' ? 'play' : 'image'), kind);
    const controls = el('div', 'media-head-ctl');
    card.slider = el('input');
    card.slider.type = 'range';
    card.slider.min = 0;
    card.slider.setAttribute('aria-label', `Step for ${tag}`);
    card.stepEl = el('span', 'num step-value');
    card.latestButton = el('button', 'tbtn');
    card.latestButton.append(icon('refresh'), 'Latest');
    card.latestButton.title = 'Follow the latest step';
    controls.append(el('span', '', 'Step'), card.slider, card.stepEl, card.latestButton);
    head.append(title, kindEl, controls);
    card.tilesEl = el('div', 'tiles');
    card.el.append(head, card.tilesEl);

    card.slider.addEventListener('input', () => {
        const index = Number(card.slider.value);
        setMediaStep(index >= steps.length - 1 ? null : steps[index]);
    });
    card.latestButton.addEventListener('click', () => setMediaStep(null));
    return card;
}

function updateCard(card) {
    const step = currentStep();
    card.slider.max = Math.max(0, steps.length - 1);
    card.slider.value = Math.max(0, steps.indexOf(step));
    card.slider.disabled = steps.length < 2;
    card.stepEl.textContent = step.toLocaleString();
    card.latestButton.setAttribute('aria-pressed', state.mediaStep === null);

    const paths = selectedPaths();
    for (const [path, tile] of [...card.tiles]) {
        if (!paths.includes(path)) {
            tile.el.remove();
            card.tiles.delete(path);
        }
    }
    paths.forEach((path, index) => {
        if (!card.tiles.has(path)) card.tiles.set(path, createTile(card, path));
        const tile = card.tiles.get(path);
        placeAt(card.tilesEl, tile.el, index);  // keeps tiles in slot order
        updateTile(tile, path, card.tag, card.kind, step);
    });
}

function renderStatus(tagCount) {
    const paths = selectedPaths();
    const strips = paths.filter((path) => peekRunData(path)?.visuals.error)
        .map((path) => el('div', 'strip error', `${shortName(path)}: visuals unreadable: ${peekRunData(path).visuals.error}`));
    if (paths.length && !paths.some((path) => peekRunData(path)?.visuals.loaded)) {
        const skeleton = el('div', 'card wide');
        skeleton.append(el('div', 'card-head'), el('div', 'plot skeleton'));
        strips.push(skeleton);
    } else if (paths.length && !tagCount) {
        strips.push(el('div', 'muted', 'No media logged for the selected runs.'));
    }
    statusEl.replaceChildren(...strips);
    document.getElementById('mediaCount').textContent = tagCount ? `${tagCount} tag${tagCount === 1 ? '' : 's'}` : '';
}

/** Reconcile cards and tiles with the current visuals and step. Called on poll and selection. */
export function renderMedia() {
    const tags = collectVisuals();
    const tagNames = new Set(tags.map(([tag]) => tag));
    for (const [tag, card] of [...cards]) {
        if (!tagNames.has(tag)) {
            card.el.remove();
            cards.delete(tag);
        }
    }
    tags.forEach(([tag, kind], index) => {
        if (!cards.has(tag)) cards.set(tag, createCard(tag, kind));
        const card = cards.get(tag);
        placeAt(mediaEl, card.el, index);
        updateCard(card);
    });
    renderStatus(tags.length);
    if (lightbox) renderLightbox();
}

function setMediaStep(step) {
    state.mediaStep = step;
    renderMedia();
    emit('media');
}

/** Move the shared step by `delta` steps ([ and ] keys). */
export function stepMedia(delta) {
    if (!steps.length) return;
    const index = Math.min(steps.length - 1, Math.max(0, steps.indexOf(currentStep()) + delta));
    setMediaStep(index === steps.length - 1 ? null : steps[index]);
}

// ---- lightbox ----

function openLightbox(tag, path) {
    lightbox = { tag, path };
    lightboxEl.classList.add('open');
    renderLightbox();
    lightboxEl.querySelector('.lb-close').focus();
}

export function closeLightbox() {
    lightbox = null;
    lightboxEl.classList.remove('open');
    lightboxEl.querySelector('.lb-media').replaceChildren();
}

export function isLightboxOpen() {
    return lightbox !== null;
}

function renderLightbox() {
    const card = cards.get(lightbox.tag);
    const step = currentStep();
    const entry = card ? resolveEntry(lightbox.path, lightbox.tag, step) : null;
    const mediaBox = lightboxEl.querySelector('.lb-media');
    const caption = lightboxEl.querySelector('.lb-caption');
    caption.style.setProperty('--c', runColor(lightbox.path));
    if (!entry) {
        mediaBox.replaceChildren(el('div', 'muted', 'No visual yet'));
        caption.replaceChildren(el('span', '', lightbox.tag), el('span', 'muted', lightbox.path));
        return;
    }
    const current = mediaBox.firstElementChild;
    const url = runFileUrl(lightbox.path, entry.file.path);
    if (current?.dataset.url !== url) {
        const media = createMediaElement(card.kind, url, `${lightbox.tag} of ${lightbox.path} at step ${entry.step}`);
        media.dataset.url = url;
        mediaBox.replaceChildren(media);
    }
    caption.replaceChildren(el('span', '', lightbox.tag), el('span', 'muted', lightbox.path), el('span', 'num', `step ${entry.step.toLocaleString()}`));
}

/** Show the neighbouring selected run in the lightbox (← and →). */
export function cycleLightbox(delta) {
    const paths = selectedPaths();
    if (!lightbox || !paths.length) return;
    const index = (paths.indexOf(lightbox.path) + delta + paths.length) % paths.length;
    lightbox.path = paths[index];
    renderLightbox();
}

export function initMedia() {
    lightboxEl.addEventListener('click', (event) => {
        if (event.target === lightboxEl || event.target.closest('.lb-close')) closeLightbox();
        else if (event.target.closest('.lb-prev')) cycleLightbox(-1);
        else if (event.target.closest('.lb-next')) cycleLightbox(1);
    });
    on('selection', () => {
        if (lightbox && !state.selected.has(lightbox.path)) closeLightbox();
        renderMedia();
    });
    on('visibility', renderMedia);
    on('theme', renderMedia);
}
