#!/usr/bin/env python3
"""Check the active Python environment against the recorded experiment environment."""
import json
import platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def main():
    ref = json.loads((Path(__file__).resolve().parents[1] / 'environment' / 'environment.json').read_text())
    mismatches = []
    print('Python:', platform.python_version(), '(reference:', ref['python']['version'].split()[0] + ')')
    if platform.python_version() != ref['python']['version'].split()[0]:
        mismatches.append('Python')
    for name, expected in ref['packages'].items():
        if expected is None:
            continue
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = '<not installed>'
        ok = actual == expected
        print(f'{name:20} installed={actual:18} expected={expected:18} {"OK" if ok else "MISMATCH"}')
        if not ok:
            mismatches.append(name)
    if mismatches:
        raise SystemExit('Environment mismatch: ' + ', '.join(mismatches))
    print('Recorded Python and package versions match.')

if __name__ == '__main__':
    main()
