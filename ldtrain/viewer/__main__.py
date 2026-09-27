"""Run the viewer: `python -m ldtrain.viewer <root_dir> [--host HOST] [--port PORT]`."""
from ldtrain.viewer.server import parse_args, viewer


def main() -> None:
    """Entry point of the `ldtrain-viewer` command."""
    args = parse_args()
    viewer(**vars(args))


if __name__ == '__main__':
    main()
