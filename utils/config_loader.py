# utils/config_loader.py
import yaml
from pathlib import Path
from configs.base_config import BaseConfig, PathsConfig


def load_config(yaml_path=None, task_name='Test', creat_dir=True):
    """
    Load configuration, supporting YAML override and automatic path handling.

    Args:
        yaml_path (str, optional): Path to the YAML configuration file.
        task_name (str): Task name, default is 'Test'.
        creat_dir (bool): Whether to create directories if they do not exist.

    Returns:
        BaseConfig: The loaded configuration object.
    """
    dummy_root = ""
    config = BaseConfig(
        task_name=task_name,
        paths=PathsConfig(root_dir=dummy_root)
    )

    if yaml_path:
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file {yaml_path} does not exist. Please check the path.")
        with open(yaml_path, encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            _update_config(config, yaml_config)

    config.update_all_paths(create_dir=creat_dir)

    return config


def _update_config(base, update):
    """
    Enhanced configuration update that supports both dataclasses and dictionaries.

    Args:
        base: The base configuration object.
        update: The dictionary containing updates.
    """
    for k, v in update.items():
        if isinstance(v, dict):
            if hasattr(base, k):
                sub_obj = getattr(base, k)
                if isinstance(sub_obj, dict):
                    sub_obj.update(v)
                else:
                    _update_config(sub_obj, v)
            else:
                setattr(base, k, v)
        else:
            if hasattr(base, k):
                orig_type = type(getattr(base, k))
                if orig_type == tuple and isinstance(v, list):
                    v = tuple(v)
            setattr(base, k, v)