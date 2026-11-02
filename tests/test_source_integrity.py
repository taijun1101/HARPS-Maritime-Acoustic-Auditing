"""Lightweight tests that do not need the model or dataset installed."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SmokeTests(unittest.TestCase):
    def test_runner_parses(self):
        ast.parse((ROOT / 'run_experiments.py').read_text(encoding='utf-8'))


    def test_downloaders_parse(self):
        for name in ('download_ast.py', 'download_shipsear.py', 'check_dataset.py', 'check_environment.py'):
            ast.parse((ROOT / 'scripts' / name).read_text(encoding='utf-8'))


    def test_reference_environment_exists(self):
        self.assertTrue((ROOT / 'environment' / 'environment.json').is_file())
