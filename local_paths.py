"""Load machine-specific paths from the ignored local configuration."""

from pathlib import Path

import yaml


_CONFIG_PATH = Path(__file__).with_name(".local_paths.yaml")


def get_path(name):
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Create {_CONFIG_PATH} from .local_paths.example.yaml "
            "and fill in the machine-specific paths."
        )
    with _CONFIG_PATH.open() as config_file:
        paths = yaml.safe_load(config_file) or {}
    try:
        return paths[name]
    except KeyError as error:
        raise KeyError(f"Missing {name!r} in {_CONFIG_PATH}") from error