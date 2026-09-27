<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo-light.svg" alt="ldtrain" height="60">
  </picture>
</h1>

Log deep-learning runs as human-readable files and compare them in a local web viewer.

ldtrain adds logging to an existing training script with one `initialize()` call. Every run is a
directory of plain files, text logs, a CSV of metrics, PNG images and WebM videos, which stay
readable with `cat`, a spreadsheet or an image viewer. The optional viewer shows every run below a
directory in one dashboard.

## ✨ Features

- **Human-readable runs:** logs, metrics, images and videos are written as standard files, with no
  database and no binary event format.
- **Non-blocking logging:** console, log-file and metric writes run on background threads, so a slow
  disk or a stalled terminal does not stall a training step.
- **Distributed training:** under `torchrun` the rank is read from the environment, the NCCL
  process group is set up, each rank writes its own log file and rank 0 writes the run data.
- **Run viewer:** a browser dashboard with a nested run tree, metrics of several runs overlaid in
  one chart, smoothing, synced zoom, images and videos by step, logs and configs, updated live
  while a run trains.
- **I/O utilities:** `ldtrain.utils.io` reads and writes images, videos, JSON and YAML.

## 📦 Install

```bash
git clone https://github.com/LDenninger/ldtrain.git
pip install -e "./ldtrain[viewer]"
```

Omit `[viewer]` in environments that only train, and use `[dev]` to also install the test tools.

### Dependencies

Python 3.12 or newer. PyTorch, NumPy, OpenCV, PyYAML and OmegaConf are installed as dependencies.
The viewer adds FastAPI and uvicorn.

## 🚀 Usage

```python
import numpy as np

import ldtrain

ldtrain.initialize(run_dir='runs/demo/lr1e-3')

for step in range(2000):
    loss = 1.0 / (step + 1)
    ldtrain.log_metrics({'train/loss': loss, 'train/lr': 1e-3}, step=step)
    if step % 1000 == 0:
        prediction = np.random.rand(64, 64, 3)          # (H, W, 3), uint8 or float in [0, 1]
        rollout = np.random.rand(8, 64, 64, 3)          # (T, H, W, 3)
        ldtrain.log_images({'val/pred': prediction}, step=step)
        ldtrain.log_videos({'val/rollout': rollout}, step=step)
        ldtrain.info(f'step {step}: loss {loss:.4f}')
```

Calling `initialize()` again reinitializes logging. An existing run directory is reused with a
warning, and its files are appended to, so a resumed run continues in place. Under `torchrun`,
distributed mode is enabled automatically when `WORLD_SIZE` is greater than 1.

### Viewer

```bash
python -m ldtrain.logging.viewer runs --port 8080
```

Open `http://127.0.0.1:8080`. The viewer binds to localhost only. To use it on a remote
machine, forward the port with `ssh -L 8080:localhost:8080 <host>`.

## Run directory

```
runs/demo/lr1e-3
├── checkpoints                 # your model checkpoints, not written by ldtrain
├── configs                     # your config files, shown by the viewer (.yaml, .yml, .json, .txt)
├── logs
│   ├── log.txt                 # console log of rank 0
│   └── log.rank<r>.txt         # console log of rank r > 0
├── metrics
│   └── metrics.csv             # one row per iteration, one column per metric
└── visuals
    └── 00001000                # one folder per iteration that logged media
        └── val
            ├── pred.png
            └── rollout.webm    # VP8, plays in every current browser
```

ldtrain writes `logs`, `metrics` and `visuals`. After `initialize()`, the paths of all five folders
are available as `ldtrain.core.run_config.checkpoint_dir`, `config_dir`, `log_dir`, `metrics_dir`
and `visual_dir`, so checkpoints and configs can be saved into the run as well.

`metrics.csv` loads with `pandas.read_csv('runs/demo/lr1e-3/metrics/metrics.csv', index_col='iteration')`.

## 🤝 Contributing

Questions and bug reports go to the [GitHub issues](https://github.com/LDenninger/ldtrain/issues).
Pull requests are welcome, and larger changes are best discussed in an issue first. By submitting a
contribution you agree that it is licensed under this project's license and may be included in
commercial licenses granted by the licensor.

## License

[PolyForm Noncommercial 1.0.0](LICENSE) (`PolyForm-Noncommercial-1.0.0`) © 2026 LDenninger

Personal, research, educational and other noncommercial use is permitted under the license
terms. Commercial use requires a separate license: contact luis.denninger@gmail.com.
