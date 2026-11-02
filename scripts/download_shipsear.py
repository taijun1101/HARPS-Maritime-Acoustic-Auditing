#!/usr/bin/env python3
"""Download the DS3500 repository; optionally extract its ShipsEar.zip archive."""
import argparse
from pathlib import Path
from zipfile import ZipFile
from huggingface_hub import snapshot_download


def safe_extract(zip_path: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f'Unsafe ZIP member: {member.filename}')
            if member.is_dir():
                continue
            # Reject symlink entries encoded in Unix mode bits.
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(f'Symlink ZIP member not supported: {member.filename}')
        archive.extractall(destination)
    print(f'Extracted to: {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo_id', default='peng7554/DS3500')
    parser.add_argument('--output_dir', default='./data/DS3500')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.repo_id, repo_type='dataset', revision=args.revision,
                      local_dir=str(root), max_workers=args.max_workers)
    print(f'Dataset downloaded to: {root}')
    if args.extract:
        archive = root / 'ShipsEar.zip'
        if not archive.is_file():
            raise FileNotFoundError(f'Expected archive not found: {archive}')
        safe_extract(archive, root / 'ShipsEar')

if __name__ == '__main__':
    main()
