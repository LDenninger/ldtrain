"""Tests for `ldtrain.viewer.report`: the PDF endpoint, the report HTML, video grids and the chart helpers."""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ldtrain.tests.conftest import RunTree
from ldtrain.utils import files
from ldtrain.viewer.report import (
    MAX_GRID_FRAMES, ReportRequest, ReportRun, build_report_html, log_splits, render_video_grid, smooth_ema, tile_frames,
)
from ldtrain.viewer.server import create_app

RUN_1 = 'proj_a/run_1'
RUN_2 = 'proj_a/run_2'
TAGS = ['val/pred', 'depth', 'val/rollout']


@pytest.fixture(scope='module')
def client(run_root: RunTree) -> TestClient:
    return TestClient(create_app(run_root.root))


def report_body(**overrides: object) -> dict:
    body = {'runs': [{'path': RUN_1, 'color': '#0b6fb8'}, {'path': RUN_2, 'color': '#d55e00', 'dashed': True}],
            'metrics': ['train/loss', 'val/psnr'], 'media_tags': TAGS}
    return {**body, **overrides}


def request_for(run_root: RunTree, **overrides: object) -> tuple[list[Path], ReportRequest]:
    body = report_body(**overrides)
    request = ReportRequest(**{**body, 'runs': [ReportRun(**run) for run in body['runs']]})
    return [run_root.root / run.path for run in request.runs], request

#---------------------------------------------------------------------
# endpoint
#---------------------------------------------------------------------

def test_report_endpoint_returns_a_pdf(client: TestClient) -> None:
    response = client.post('/api/report', json=report_body(title='lr sweep: run 1 & 2'))
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/pdf'
    assert response.headers['content-disposition'] == 'attachment; filename="lr_sweep_run_1_2.pdf"'
    assert response.content.startswith(b'%PDF-')


def test_report_file_name_defaults_to_the_run(client: TestClient) -> None:
    response = client.post('/api/report', json=report_body(runs=[{'path': RUN_1, 'color': '#0b6fb8'}], orientation='landscape'))
    assert response.status_code == 200
    assert response.headers['content-disposition'] == 'attachment; filename="proj_a_run_1.pdf"'


@pytest.mark.parametrize(('overrides', 'status'), [
    ({'runs': []}, 422),
    ({'runs': [{'path': RUN_1, 'color': 'red'}]}, 422),
    ({'runs': [{'path': RUN_1, 'color': '#0b6fb8"/><script>'}]}, 422),
    ({'orientation': 'sideways'}, 422),
    ({'smoothing': 1.5}, 422),
    ({'runs': [{'path': '../outside', 'color': '#0b6fb8'}]}, 404),
    ({'runs': [{'path': 'proj_a', 'color': '#0b6fb8'}]}, 404),
])
def test_report_rejects_invalid_requests(client: TestClient, overrides: dict, status: int) -> None:
    assert client.post('/api/report', json=report_body(**overrides)).status_code == status

#---------------------------------------------------------------------
# report content
#---------------------------------------------------------------------

def test_report_holds_the_chosen_sections(run_root: RunTree) -> None:
    page = build_report_html(*request_for(run_root, metrics=['train/loss', 'val/psnr', 'no/such_metric'], title='<b>sweep</b>'))
    assert '<h1>&lt;b&gt;sweep&lt;/b&gt;</h1>' in page
    assert page.count('<article class="card"><div class="card-head">') == 2  # one chart per logged metric, the unknown one skipped
    assert page.count('<svg') == 2
    assert 'train/lr' not in page
    assert '<h3>train</h3>' in page and '<h3>val</h3>' in page
    assert 'Final logged value of each run' in page
    assert 'class="sw dash"' in page  # run_2 is dashed
    assert 'data:image/png;base64,' in page  # the rollout video as a frame grid
    assert 'visuals/00000030/val/pred.png' in page
    assert 'lr: 0.001' in page  # run_1's YAML config


def test_report_leaves_out_what_is_not_chosen(run_root: RunTree) -> None:
    page = build_report_html(*request_for(run_root, metrics=[], media_tags=[], include_summary=False, include_config=False))
    assert '<svg' not in page and 'Summary' not in page and 'Media' not in page and 'Configs' not in page
    assert 'Comparison of 2 runs' in page


def test_media_step_shows_the_latest_visual_at_or_below_it(run_root: RunTree) -> None:
    page = build_report_html(*request_for(run_root, media_step=10))
    assert 'visuals/00000000/val/pred.png' in page and 'visuals/00000030' not in page
    assert 'No visual' in page  # the rollout is logged at step 30 only, run_2 has no media


def test_single_run_summary_has_min_and_max(run_root: RunTree) -> None:
    page = build_report_html(*request_for(run_root, runs=[{'path': RUN_1, 'color': '#0b6fb8'}], metrics=['train/lr']))
    assert '<th>Final</th><th>Min</th><th>Max</th>' in page
    assert page.count('<td class="num">0.001</td>') == 3

#---------------------------------------------------------------------
# video grids
#---------------------------------------------------------------------

def test_tile_frames_lays_out_row_major_with_gaps() -> None:
    frames = np.stack([np.full((16, 24, 3), index, dtype=np.uint8) for index in range(10)])  # (10, 16, 24, 3)
    grid = tile_frames(frames)
    assert grid.shape == (4 * 18 - 2, 3 * 26 - 2, 3)  # 3 columns, 4 rows, 2 px gaps
    assert grid[0, 0, 0] == 0 and grid[0, 26, 0] == 1 and grid[18, 0, 0] == 3 and grid[-1, -1, 0] == 255


def test_tile_frames_downscales_wide_grids() -> None:
    grid = tile_frames(np.zeros((4, 600, 800, 3), dtype=np.uint8))
    assert grid.shape[1] <= 2048


def test_long_videos_are_subsampled(tmp_path: Path) -> None:
    video_path = tmp_path / 'long.webm'
    files.save_video(np.random.default_rng(0).integers(0, 255, (100, 16, 16, 3), dtype=np.uint8), video_path, fps=10, fourcc='VP80')
    image, detail = render_video_grid(video_path)
    assert image.startswith('<img src="data:image/png;base64,')
    assert detail == f'{MAX_GRID_FRAMES} of 100 frames, evenly spaced, row by row'


def test_unreadable_video_is_reported_in_place(tmp_path: Path) -> None:
    video_path = tmp_path / 'broken.webm'
    video_path.write_bytes(b'not a video')
    image, detail = render_video_grid(video_path)
    assert 'class="empty"' in image and detail == ''

#---------------------------------------------------------------------
# chart helpers
#---------------------------------------------------------------------

def test_smooth_ema_matches_the_viewer() -> None:
    np.testing.assert_allclose(smooth_ema(np.array([1.0, 3.0]), 0.5), [1.0, 1.75 / 0.75])
    values = np.array([2.0, 5.0])
    assert smooth_ema(values, 0) is values


def test_log_splits_match_the_viewer() -> None:
    np.testing.assert_allclose(log_splits(0.3, 2.4), [0.5, 1, 2])
    np.testing.assert_allclose(log_splits(1e-3, 10), [1e-3, 1e-2, 1e-1, 1, 10])
