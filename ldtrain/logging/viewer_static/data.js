// Per-run data fetched from the API: incremental metrics, visuals by step, and run info.

import { ApiError, fetchJson } from './util.js';

const runData = new Map();  // path -> RunData

function createRunData() {
    return {
        metrics: {
            columns: [],
            series: new Map(),  // metric -> { x: number[], y: number[] }, non-null rows only
            offset: 0,
            fileId: '',
            loaded: false,
            error: null,
        },
        visuals: {
            steps: new Map(),   // step -> [{ tag, kind, path }]
            lastStep: null,
            loaded: false,
            error: null,
        },
        info: null,
        infoPromise: null,
    };
}

export function getRunData(path) {
    if (!runData.has(path)) runData.set(path, createRunData());
    return runData.get(path);
}

export function peekRunData(path) {
    return runData.get(path) ?? null;
}

/** Forget runs that are no longer selected, so selecting them again starts from a fresh read. */
export function forgetUnselected(selected) {
    for (const path of [...runData.keys()]) {
        if (!selected.has(path)) runData.delete(path);
    }
}

function describeError(error) {
    return error instanceof ApiError ? `${error.status}: ${error.message}` : error.message;
}

// ---- metrics ----

function appendMetricRows(metrics, chunk) {
    const iterations = chunk.data.iteration ?? [];
    for (const column of chunk.columns) {
        if (column === 'iteration') continue;
        if (!metrics.series.has(column)) metrics.series.set(column, { x: [], y: [] });
        const series = metrics.series.get(column);
        const values = chunk.data[column];
        for (let ii = 0; ii < values.length; ii++) {
            if (values[ii] !== null && iterations[ii] !== null) {
                series.x.push(iterations[ii]);
                series.y.push(values[ii]);
            }
        }
    }
}

/**
 * Fetch the metric rows appended since the last call.
 *
 * Returns whether the run's metrics changed. An OfflineError propagates, any other
 * failure is stored on the run and shown on its cards.
 */
export async function pollMetrics(path) {
    const data = getRunData(path);
    const metrics = data.metrics;
    let chunk;
    try {
        chunk = await fetchJson('/api/metrics', { run: path, offset: metrics.offset, file_id: metrics.fileId });
    } catch (error) {
        if (!(error instanceof ApiError)) throw error;
        if (runData.get(path) !== data) return false;
        const changed = metrics.error === null;
        metrics.error = describeError(error);
        metrics.loaded = true;
        return changed;
    }
    if (runData.get(path) !== data) return false;  // deselected while the request was in flight

    const hadError = metrics.error !== null || !metrics.loaded;
    metrics.error = chunk.warning;
    metrics.loaded = true;
    metrics.offset = chunk.offset;
    metrics.fileId = chunk.file_id;
    const rowCount = (chunk.data.iteration ?? []).length;
    if (chunk.reset) {
        metrics.series = new Map();
        metrics.columns = chunk.columns;
    }
    appendMetricRows(metrics, chunk);
    return chunk.reset || rowCount > 0 || hadError;
}

// ---- visuals ----

export async function pollVisuals(path) {
    const data = getRunData(path);
    const visuals = data.visuals;
    let response;
    try {
        response = await fetchJson('/api/visuals', { run: path, since_step: visuals.lastStep ?? 0 });
    } catch (error) {
        if (!(error instanceof ApiError)) throw error;
        if (runData.get(path) !== data) return false;
        visuals.error = describeError(error);
        return true;
    }
    if (runData.get(path) !== data) return false;

    let changed = !visuals.loaded || visuals.error !== null;
    visuals.loaded = true;
    visuals.error = null;
    for (const { step, files } of response.steps) {
        const previous = visuals.steps.get(step);
        if (!previous || previous.length !== files.length) changed = true;
        visuals.steps.set(step, files);
        visuals.lastStep = Math.max(visuals.lastStep ?? step, step);
    }
    return changed;
}

// ---- run info ----

/** Ranks and config files of a run, fetched once per selection. */
export function loadRunInfo(path) {
    const data = getRunData(path);
    if (!data.infoPromise) {
        data.infoPromise = fetchJson('/api/run', { run: path }).then(
            (info) => (data.info = info),
            (error) => {
                data.infoPromise = null;  // retry on the next request
                throw error;
            },
        );
    }
    return data.infoPromise;
}
