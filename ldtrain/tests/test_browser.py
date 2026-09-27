"""Browser tests: the viewer frontend in headless Chromium over the shared run tree, one URL state per test."""
import re
import shutil
import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn
from playwright.sync_api import Page, expect

import ldtrain
from ldtrain.tests.conftest import RunTree
from ldtrain.viewer.server import create_app

pytestmark = pytest.mark.browser

RUN_1 = 'proj_a/run_1'
RUN_2 = 'proj_a/run_2'
POLL_WAIT_MS = 12000


@pytest.fixture(scope='module')
def viewer_url(run_root: RunTree) -> Iterator[str]:
    """Serve the shared tree with uvicorn on a free port for the whole module."""
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(run_root.root), host='127.0.0.1', port=port, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    yield f'http://127.0.0.1:{port}'
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def browser_context_args(browser_context_args: dict) -> dict:
    return {**browser_context_args, 'permissions': ['clipboard-read', 'clipboard-write'], 'viewport': {'width': 1400, 'height': 900}}


@pytest.fixture
def view(page: Page) -> Iterator[Page]:
    """A page that fails the test on any browser console error or uncaught exception."""
    errors: list[str] = []
    page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
    page.on('pageerror', lambda error: errors.append(str(error)))
    yield page
    assert errors == [], errors


def open_view(view: Page, viewer_url: str, query: str = '') -> Page:
    view.goto(f'{viewer_url}/?{query}')
    expect(view.locator('#tree [role=treeitem]').first).to_be_visible()
    return view


def open_runs(view: Page, viewer_url: str, *runs: str, extra: str = '') -> Page:
    query = 'runs=' + ','.join(f'{run}:{ii}' for ii, run in enumerate(runs)) + f'&focus={runs[0]}' + extra
    open_view(view, viewer_url, query)
    expect(view.locator('#metricGroups .card .uplot canvas').first).to_be_visible()
    return view


#---------------------------------------------------------------------
# tree and selection
#---------------------------------------------------------------------

def test_empty_state_lists_recent_runs(view: Page, viewer_url: str) -> None:
    open_view(view, viewer_url)
    expect(view.locator('#emptyState h1')).to_have_text('Pick runs to compare')
    expect(view.locator('#recentRuns .recent')).to_have_count(3)
    expect(view.locator('#treeStats')).to_have_text('3 runs, 2 folders')


def test_click_selects_and_cross_removes(view: Page, viewer_url: str) -> None:
    open_view(view, viewer_url)
    view.locator(f'[role=treeitem][data-path="{RUN_1}"] .row').click()
    expect(view.locator('#chips .chip')).to_have_count(1)
    expect(view.locator(f'[role=treeitem][data-path="{RUN_1}"] .row')).to_have_class('row run selected focused')
    expect(view.locator('#selInfo')).to_have_text('1 selected, focus on run_1')
    expect(view).to_have_url(re.compile(f'runs={RUN_1}'))
    expect(view).to_have_url(re.compile(f'focus={RUN_1}'))
    row = view.locator(f'[role=treeitem][data-path="{RUN_1}"] .row')
    row.hover()
    row.locator('.x').click()
    expect(view.locator('#chips .chip')).to_have_count(0)
    expect(view.locator('#emptyState')).to_be_visible()


def test_folder_click_selects_every_run_below(view: Page, viewer_url: str) -> None:
    open_view(view, viewer_url)
    view.locator('[role=treeitem][data-path="proj_a"] > .row .name').click()
    expect(view.locator('#chips .chip')).to_have_count(2)
    expect(view.locator('[role=treeitem][data-path="proj_a"] > .row .meta')).to_have_text('2/2')
    view.locator('[role=treeitem][data-path="proj_a"] > .row .name').click()
    expect(view.locator('#chips .chip')).to_have_count(0)


def test_filter_and_keyboard_in_tree(view: Page, viewer_url: str) -> None:
    open_view(view, viewer_url)
    view.keyboard.press('t')
    view.keyboard.type('run_3')
    expect(view.locator('#tree [role=treeitem]')).to_have_count(2)  # proj_b and its run
    view.keyboard.press('ArrowDown')  # from the filter into the tree
    assert view.evaluate('document.activeElement.getAttribute("role")') == 'treeitem'
    view.keyboard.press('End')
    view.keyboard.press('Space')
    expect(view.locator('#chips .chip')).to_have_text(['run_3'])
    view.keyboard.press('x')
    expect(view.locator('#chips .chip')).to_have_count(0)


#---------------------------------------------------------------------
# metrics
#---------------------------------------------------------------------

def test_metric_cards_groups_and_filter(view: Page, viewer_url: str) -> None:
    open_runs(view, viewer_url, RUN_1, RUN_2)
    expect(view.locator('#metricGroups .card')).to_have_count(3)
    expect(view.locator('#metricCount')).to_have_text('3 metrics in 2 groups')
    expect(view.locator('#chips .chip')).to_have_count(2)
    expect(view.locator('.card[data-metric="train/loss"] .card-foot .leg')).to_have_count(2)
    view.locator('#groupChips .gchip', has_text='val/').click()
    expect(view.locator('#metricGroups .card:visible')).to_have_count(1)
    expect(view).to_have_url(re.compile('m=val'))
    view.locator('#metricFilter').fill('/lr$/')
    expect(view.locator('#metricGroups .card:visible')).to_have_count(1)
    expect(view.locator('#metricGroups .card:visible')).to_have_attribute('data-metric', 'train/lr')
    view.keyboard.press('Escape')
    expect(view.locator('#metricGroups .card:visible')).to_have_count(3)


def test_log_scale_smoothing_and_zoom(view: Page, viewer_url: str) -> None:
    open_runs(view, viewer_url, RUN_1)
    card = view.locator('.card[data-metric="train/loss"]')
    card.locator('.tbtn', has_text='Log').click()
    expect(card.locator('.tbtn', has_text='Log')).to_have_attribute('aria-pressed', 'true')
    expect(view).to_have_url(re.compile('logy=train'))
    view.locator('#logyAll').click()
    expect(view.locator('#logyAll')).to_have_attribute('aria-pressed', 'true')
    expect(view.locator('.card[data-metric="val/psnr"] .tbtn', has_text='Log')).to_have_attribute('aria-pressed', 'true')

    before = card.locator('.card-foot .leg b').first.inner_text()
    view.locator('#smooth').press('Home')
    expect(view.locator('#smoothVal')).to_have_text('0.00')
    assert card.locator('.card-foot .leg b').first.inner_text() != before
    expect(view).to_have_url(re.compile('smooth=0'))

    over = card.locator('.u-over')
    box = over.bounding_box()
    assert box is not None
    view.mouse.move(box['x'] + box['width'] * 0.2, box['y'] + box['height'] / 2)
    view.mouse.down()
    view.mouse.move(box['x'] + box['width'] * 0.6, box['y'] + box['height'] / 2, steps=8)
    view.mouse.up()
    expect(view).to_have_url(re.compile('x='))
    view.locator('#resetZoom').click()
    expect(view).not_to_have_url(re.compile('x='))


def test_legend_click_hides_a_run_everywhere(view: Page, viewer_url: str) -> None:
    open_runs(view, viewer_url, RUN_1, RUN_2)
    view.locator('.card[data-metric="train/loss"] .card-foot .leg', has_text='run_2').click()
    expect(view.locator('.card[data-metric="train/lr"] .card-foot .leg', has_text='run_2')).to_have_class('leg hidden')
    expect(view.locator(f'#chips .chip[data-path="{RUN_2}"]')).to_have_class('chip hidden')
    expect(view).to_have_url(re.compile(f'hide={RUN_2}'))


def test_new_rows_arrive_by_polling(view: Page, viewer_url: str, run_root: RunTree) -> None:
    run_dir = run_root.root / 'proj_b' / 'run_live'
    ldtrain.initialize(run_dir=run_dir, color=False, log_level='warning')
    ldtrain.log_metrics({'loss': 1.0}, step=0)
    ldtrain.finish()
    try:
        open_runs(view, viewer_url, 'proj_b/run_live', extra='&smooth=0')
        value = view.locator('.card[data-metric="loss"] .card-foot .leg b')
        expect(value).to_have_text('1')
        ldtrain.initialize(run_dir=run_dir, color=False, log_level='warning')
        ldtrain.log_metrics({'loss': 0.25}, step=1)
        ldtrain.finish()
        expect(value).to_have_text('0.25', timeout=POLL_WAIT_MS)
        expect(view.locator('#pollText')).to_contain_text('updated')
    finally:
        ldtrain.initialize(color=False)
        shutil.rmtree(run_dir)


#---------------------------------------------------------------------
# media
#---------------------------------------------------------------------

def test_media_cards_step_slider_and_lightbox(view: Page, viewer_url: str) -> None:
    open_runs(view, viewer_url, RUN_1, RUN_2)
    expect(view.locator('#media .card')).to_have_count(3)
    expect(view.locator('#mediaCount')).to_have_text('3 tags')
    pred = view.locator('#media .card', has=view.locator('.card-title', has_text='pred'))
    expect(pred.locator('.tile')).to_have_count(2)
    expect(pred.locator('.tile').first.locator('.stp')).to_have_text('30')
    expect(pred.locator('.tile').nth(1)).to_have_class('tile missing')
    view.keyboard.press('[')
    expect(pred.locator('.tile').first.locator('.stp')).to_have_text('0')
    expect(view).to_have_url(re.compile('step=0'))
    view.keyboard.press(']')
    expect(pred.locator('.tile').first.locator('.stp')).to_have_text('30')
    pred.locator('.tile').first.click()
    expect(view.locator('#lightbox')).to_have_class('overlay lightbox open')
    expect(view.locator('#lightbox .lb-caption')).to_contain_text('val/pred')
    expect(view.locator('#lightbox .lb-caption')).to_contain_text('step 30')
    view.keyboard.press('Escape')
    expect(view.locator('#lightbox')).not_to_have_class('overlay lightbox open')


#---------------------------------------------------------------------
# log and config
#---------------------------------------------------------------------

def test_log_levels_filter_and_follow(view: Page, viewer_url: str) -> None:
    open_runs(view, viewer_url, RUN_1, RUN_2)
    lines = view.locator('#logBody .logline:visible')
    expect(lines).to_have_count(5)  # initialized, starting, warning, error, finished
    expect(lines.last).to_contain_text('finished')
    expect(view.locator('#logRun')).to_have_value(RUN_1)
    view.locator('#lvlChips .lvl.WARN').click()
    expect(view.locator('#logBody .logline:visible')).to_have_count(4)
    view.locator('#logFilter').fill('finished')
    expect(view.locator('#logBody .logline:visible')).to_have_count(1)
    view.locator('#logFollow').click()
    expect(view.locator('#logFollow')).to_have_attribute('aria-pressed', 'false')
    view.locator('#logRun').select_option(RUN_2)
    expect(view.locator('#logRank option')).to_have_count(2)
    view.locator('#logRank').select_option('1')
    view.locator('#logFilter').fill('')
    expect(view.locator('#logBody')).to_contain_text('hello from rank 1')


def test_config_is_highlighted_and_copies(view: Page, viewer_url: str) -> None:
    open_runs(view, viewer_url, RUN_1)
    expect(view.locator('#cfgTitle')).to_have_text('config.yaml')
    expect(view.locator('#cfgBody .k')).to_have_count(3)
    expect(view.locator('#cfgBody .n')).to_have_text(['0.001'])
    view.locator('#cfgCopy').click()
    expect(view.locator('#cfgCopy')).to_have_text('Copied')
    assert view.evaluate('navigator.clipboard.readText()').startswith('model:')


#---------------------------------------------------------------------
# state, theme and help
#---------------------------------------------------------------------

def test_url_state_survives_reload(view: Page, viewer_url: str) -> None:
    open_runs(view, viewer_url, RUN_1, RUN_2, extra='&m=train/&smooth=0.3&collapsed=media')
    view.reload()
    expect(view.locator('#chips .chip')).to_have_count(2)
    expect(view.locator('#metricFilter')).to_have_value('train/')
    expect(view.locator('#smoothVal')).to_have_text('0.30')
    expect(view.locator('#sec-media')).to_have_class('section collapsed')
    expect(view.locator('#metricGroups .card:visible')).to_have_count(2)


def test_theme_cycles_and_help_opens(view: Page, viewer_url: str) -> None:
    open_view(view, viewer_url)
    view.locator('#themeBtn').click()
    expect(view.locator('html')).to_have_attribute('data-theme', 'light')
    view.keyboard.press('d')
    expect(view.locator('html')).to_have_attribute('data-theme', 'dark')
    view.keyboard.press('d')
    expect(view.locator('html')).not_to_have_attribute('data-theme', 'dark')
    view.keyboard.press('?')
    expect(view.locator('#help')).to_have_class('overlay open')
    view.keyboard.press('Escape')
    expect(view.locator('#help')).not_to_have_class('overlay open')
