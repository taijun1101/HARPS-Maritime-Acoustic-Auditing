"""AST-level structural parity: compares every original function/class body."""
import ast
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'reference'/'original_run_experiments.py'

def definitions(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    return {n.name: ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}

class ModularEquivalenceTests(unittest.TestCase):
    def test_every_original_definition_preserved(self):
        original=definitions(ORIGINAL)
        modular={}
        for path in (ROOT/'harps').glob('*.py'):
            for name,value in definitions(path).items():
                self.assertNotIn(name,modular, f'duplicate definition: {name}')
                modular[name]=value
        self.assertEqual(original,modular)
