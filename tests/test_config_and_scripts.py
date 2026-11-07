"""Smoke tests for config.yaml and the scripts."""
import py_compile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_has_published_defaults():
    with open(REPO_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config == {
        "run_path": "C:/z/configure11",
        "rainfall_num": [1030, 2030, 3030, 5030, 10030],
        "config_num": 2,
        "idx_config": 1,
        "pop_size": 100,
        "num_generations": 100,
    }


def test_scripts_compile():
    for name in ("configure11_main.py", "configure11_functions.py"):
        py_compile.compile(str(REPO_ROOT / name), doraise=True)
