"""Web viewer over a root directory of ldtrain runs.

`create_app` builds a read-only FastAPI app: a JSON API backed by `reader` plus the
static single-page frontend in `static/`. Run it with
`python -m ldtrain.viewer <root_dir> [--host 127.0.0.1] [--port 8765]` or the `ldtrain-viewer`
command and open the printed URL. It binds to localhost by default; reach a cluster node through an SSH
tunnel (`ssh -L 8765:localhost:8765 <node>`).
"""
import argparse
import socket
import sys
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ldtrain.viewer import reader
from ldtrain.viewer.reader import PathOutsideRoot

STATIC_DIR = Path(__file__).parent / 'static'

#---------------------------------------------------------------------
# app
#---------------------------------------------------------------------

def create_app(root_dir: str | Path) -> FastAPI:
    """Build the viewer app serving every run below `root_dir`.

    Args:
        root_dir: Directory scanned for runs. Requests can never read outside it.

    Raises:
        NotADirectoryError: If `root_dir` is not a directory.
    """
    root_dir = Path(root_dir).resolve()
    if not root_dir.is_dir():
        raise NotADirectoryError(root_dir)
    app = FastAPI(title='ldtrain viewer', docs_url=None, redoc_url=None)

    def resolve_run(run: str) -> Path:
        try:
            run_dir = reader.resolve_inside(root_dir, run)
        except PathOutsideRoot:
            raise HTTPException(status_code=404, detail=f'{run} is outside the root')
        if not reader.is_run_dir(run_dir):
            raise HTTPException(status_code=404, detail=f'{run} is not a run')
        return run_dir

    @app.get('/api/tree')
    def get_tree() -> dict:
        return {'root': str(root_dir), 'tree': asdict(reader.scan_tree(root_dir))}

    @app.get('/api/run')
    def get_run(run: str) -> dict:
        return asdict(reader.read_run_info(resolve_run(run), run))

    @app.get('/api/metrics')
    def get_metrics(run: str, offset: int = 0, file_id: str = '') -> dict:
        return asdict(reader.read_metrics(resolve_run(run), offset, file_id))

    @app.get('/api/visuals')
    def get_visuals(run: str, since_step: int = 0) -> dict:
        return {'steps': [asdict(step) for step in reader.list_visuals(resolve_run(run), since_step)]}

    @app.get('/api/log')
    def get_log(run: str, rank: int = 0, offset: int | None = None) -> dict:
        return asdict(reader.read_log(resolve_run(run), rank, offset))

    @app.get('/api/file')
    def get_file(run: str, path: str) -> FileResponse:
        run_dir = resolve_run(run)
        try:
            file_path = reader.resolve_inside(run_dir, path)
        except PathOutsideRoot:
            raise HTTPException(status_code=404, detail=f'{path} is outside the run')
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail=f'{path} does not exist')
        return FileResponse(file_path)

    @app.get('/')
    def get_index() -> FileResponse:
        return FileResponse(STATIC_DIR / 'index.html')

    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
    return app

#---------------------------------------------------------------------
# script
#---------------------------------------------------------------------

def viewer(root_dir: str, host: str, port: int) -> None:
    """Serve the viewer until Ctrl-C, reporting what it found and where it listens."""
    root = Path(root_dir).expanduser()
    if not root.is_dir():
        sys.exit(f'error: {root} is not a directory')
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # like uvicorn, so TIME_WAIT sockets do not count
            probe.bind((host, port))
    except OSError as error:
        sys.exit(f'error: cannot listen on {host}:{port} ({error.strerror}), pick another port with --port')

    app = create_app(root)
    num_runs = reader.scan_tree(root.resolve()).num_runs
    url_host = 'localhost' if host in ('127.0.0.1', '0.0.0.0') else host
    print('ldtrain viewer')
    print(f'  runs   {root.resolve()}  ({num_runs} found, rescanned every 30 s)')
    print(f'  open   http://{url_host}:{port}')
    print('  stop   Ctrl-C')
    if num_runs == 0:
        print('  note   a run is a folder holding metrics/metrics.csv, logs/log.txt or visuals/<step>/')
    if host in ('127.0.0.1', 'localhost'):
        print(f'  tunnel ssh -L {port}:localhost:{port} {socket.gethostname()}   (from another machine)')
    sys.stdout.flush()
    try:
        uvicorn.run(app, host=host, port=port, log_level='warning')
    except KeyboardInterrupt:
        pass
    print('ldtrain viewer stopped')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='ldtrain-viewer', description='Serve the ldtrain run viewer over a directory of runs.')
    parser.add_argument('root_dir', help='directory holding the runs, scanned recursively')
    parser.add_argument('--host', default='127.0.0.1', help='interface to bind, 0.0.0.0 exposes it to the network')
    parser.add_argument('--port', type=int, default=8765)
    return parser.parse_args()

