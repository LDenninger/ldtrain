"""API tests for `ldtrain.viewer.server` over the shared run tree written by the ldtrain API."""
import pytest
from fastapi.testclient import TestClient

from ldtrain.tests.conftest import RunTree
from ldtrain.viewer.server import STATIC_DIR, create_app

RUN = 'proj_a/run_1'


@pytest.fixture(scope='module')
def client(run_root: RunTree) -> TestClient:
    return TestClient(create_app(run_root.root))


def test_tree_lists_every_run_with_its_last_iteration(client: TestClient, run_root: RunTree) -> None:
    body = client.get('/api/tree').json()
    assert body['root'] == str(run_root.root)
    tree = body['tree']
    assert tree['num_runs'] == 3 and [child['name'] for child in tree['children']] == ['proj_a', 'proj_b']
    runs = {run['path']: run for project in tree['children'] for run in project['children']}
    assert set(runs) == set(run_root.steps)
    for path, steps in run_root.steps.items():
        assert runs[path]['is_run'] and runs[path]['last_iteration'] == steps - 1 and runs[path]['mtime'] is not None


def test_run_info(client: TestClient) -> None:
    assert client.get('/api/run', params={'run': RUN}).json() == {
        'path': RUN, 'ranks': [0], 'config_files': ['configs/config.yaml'], 'has_metrics': True, 'has_visuals': True}
    info = client.get('/api/run', params={'run': 'proj_a/run_2'}).json()
    assert info['ranks'] == [0, 1] and info['config_files'] == [] and not info['has_visuals']


def test_metrics_full_then_incremental(client: TestClient, run_root: RunTree) -> None:
    first = client.get('/api/metrics', params={'run': RUN}).json()
    assert first['columns'] == ['iteration', 'train/loss', 'train/lr', 'val/psnr'] and first['reset']
    assert first['data']['iteration'] == list(range(40)) and first['data']['val/psnr'][10] == 22.5 and first['data']['val/psnr'][11] is None
    again = client.get('/api/metrics', params={'run': RUN, 'offset': first['offset'], 'file_id': first['file_id']}).json()
    assert all(values == [] for values in again['data'].values()) and not again['reset'] and again['offset'] == first['offset']
    stale = client.get('/api/metrics', params={'run': RUN, 'offset': first['offset'], 'file_id': 'other'}).json()
    assert stale['reset'] and stale['data']['iteration'] == list(range(40))


def test_visuals_and_files(client: TestClient) -> None:
    steps = client.get('/api/visuals', params={'run': RUN}).json()['steps']
    assert [step['step'] for step in steps] == [0, 30]
    assert sorted((f['tag'], f['kind']) for f in steps[1]['files']) == [('depth', 'image'), ('val/pred', 'image'), ('val/rollout', 'video')]
    assert [step['step'] for step in client.get('/api/visuals', params={'run': RUN, 'since_step': 1}).json()['steps']] == [30]
    image = client.get('/api/file', params={'run': RUN, 'path': 'visuals/00000030/val/pred.png'})
    assert image.status_code == 200 and image.headers['content-type'] == 'image/png'
    video = client.get('/api/file', params={'run': RUN, 'path': 'visuals/00000030/val/rollout.webm'})
    assert video.status_code == 200 and video.headers['content-type'] == 'video/webm'
    config = client.get('/api/file', params={'run': RUN, 'path': 'configs/config.yaml'})
    assert config.status_code == 200 and config.text.startswith('model:')


def test_log_by_rank_and_offset(client: TestClient) -> None:
    chunk = client.get('/api/log', params={'run': RUN}).json()
    assert 'starting run_1' in chunk['text'] and 'WARNING' in chunk['text'] and chunk['text'].endswith('finished\n')
    assert not chunk['truncated']
    follow = client.get('/api/log', params={'run': RUN, 'offset': chunk['offset']}).json()
    assert follow['text'] == '' and follow['offset'] == chunk['offset']
    rank1 = client.get('/api/log', params={'run': 'proj_a/run_2', 'rank': 1}).json()
    assert 'hello from rank 1' in rank1['text'] and 'starting' not in rank1['text']


@pytest.mark.parametrize('endpoint, params', [
    ('/api/metrics', {'run': '../'}), ('/api/metrics', {'run': 'proj_a'}), ('/api/metrics', {'run': '/etc'}),
    ('/api/run', {'run': 'proj_a/missing'}), ('/api/visuals', {'run': '..'}), ('/api/log', {'run': 'proj_a/../..'}),
    ('/api/file', {'run': RUN, 'path': '../../../etc/passwd'}), ('/api/file', {'run': RUN, 'path': 'metrics/missing.csv'}),
    ('/api/file', {'run': RUN, 'path': '/etc/passwd'})])
def test_rejects_paths_outside_root_and_non_runs(client: TestClient, endpoint: str, params: dict) -> None:
    assert client.get(endpoint, params=params).status_code == 404


def test_serves_the_frontend_and_every_static_file(client: TestClient) -> None:
    index = client.get('/')
    assert index.status_code == 200 and index.headers['content-type'].startswith('text/html')
    for path in sorted(STATIC_DIR.iterdir()):
        response = client.get(f'/static/{path.name}')
        assert response.status_code == 200, path.name
        expected = {'.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html'}[path.suffix]
        assert response.headers['content-type'].startswith(expected), path.name
