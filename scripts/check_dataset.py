#!/usr/bin/env python3
"""Validate the directory convention expected by the unchanged HARPS runner."""
import argparse
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', required=True)
    p.add_argument('--allow_non_12_classes', action='store_true')
    args = p.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    if not root.is_dir():
        p.error(f'Dataset directory not found: {root}')
    files = sorted(root.rglob('*.wav'))
    if not files:
        p.error(f'No .wav files found under {root}')
    bad = [str(x.relative_to(root)) for x in files if len(x.relative_to(root).parts) < 3]
    if bad:
        p.error('Files violate group/class_name/file.wav convention: ' + repr(bad[:5]))
    counts = Counter(x.relative_to(root).parts[1] for x in files)
    print(f'Data root: {root}\nTotal WAV clips: {len(files)}\nClasses: {len(counts)}')
    for name, n in sorted(counts.items()):
        print(f'  {name}: {n}')
    if len(counts) != 12 and not args.allow_non_12_classes:
        p.error('Expected 12 classes; inspect the extraction and preprocessing structure.')
    print('Structure check passed. This does NOT establish dataset equivalence to the paper.')

if __name__ == '__main__':
    main()
