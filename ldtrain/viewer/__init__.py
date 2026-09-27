"""Readers of a run directory and the browser viewer over it.

`reader` parses run directories with the standard library only. `server` needs the `viewer`
extra (FastAPI and uvicorn) and is never imported by training code.
"""
from ldtrain.viewer import reader

__all__ = ['reader']
