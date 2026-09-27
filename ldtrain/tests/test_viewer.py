"""API tests for `ldtrain.viewer.server` over a run written by the real ldtrain API."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import ldtrain
from ldtrain.viewer.server import create_app

RUN = 'project/run_a'


@pytest.fixture(scope='module')
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    root = tmp_path_factory.mktemp('runs')
    ldtrain.initialize(run_dir=root / RUN, color=False)
    for step in range(3):
        ldtrain.log_metrics({'train/loss': 1.0 / (step + 1), 'lr': 0.1}, step=step)
    ldtrain.log_images({'val/pred': np.zeros((8, 8, 3), np.uint8)}, step=2)
    ldtrain.info('viewer test line')
    ldtrain.finish()
    ldtrain.initialize()  # detach logging from the temporary run
    return TestClient(create_app(root))


def test_tree_lists_the_run(client: TestClient) -> None:
    tree = client.get('/api/tree').json()['tree']
    assert tree['num_runs'] == 1
    run = tree['children'][0]['children'][0]
    assert run['path'] == RUN and run['is_run'] and run['last_iteration'] == 2.0


def test_run_metrics_visuals_log(client: TestClient) -> None:
    info = client.get('/api/run', params={'run': RUN}).json()
    assert info['ranks'] == [0] and info['has_metrics'] and info['has_visuals']

    metrics = client.get('/api/metrics', params={'run': RUN}).json()
    assert metrics['columns'] == ['iteration', 'train/loss', 'lr']
    assert metrics['data']['iteration'] == [0.0, 1.0, 2.0]

    steps = client.get('/api/visuals', params={'run': RUN}).json()['steps']
    assert steps == [{'step': 2, 'files': [{'tag': 'val/pred', 'kind': 'image', 'path': 'visuals/00000002/val/pred.png'}]}]
    image = client.get('/api/file', params={'run': RUN, 'path': steps[0]['files'][0]['path']})
    assert image.status_code == 200 and image.headers['content-type'] == 'image/png'

    assert 'viewer test line' in client.get('/api/log', params={'run': RUN}).json()['text']


@pytest.mark.parametrize('params', [
    {'run': '../'}, {'run': 'project'}, {'run': '/etc'},
    {'run': RUN, 'path': '../../../etc/passwd'}, {'run': RUN, 'path': 'metrics/missing.csv'}])
def test_rejects_paths_outside_root_and_non_runs(client: TestClient, params: dict) -> None:
    endpoint = '/api/file' if 'path' in params else '/api/metrics'
    assert client.get(endpoint, params=params).status_code == 404


def test_serves_the_frontend(client: TestClient) -> None:
    assert client.get('/').status_code == 200
