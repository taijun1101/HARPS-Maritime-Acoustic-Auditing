#!/usr/bin/env python3
"""Download the exact Hugging Face AST model repository used by HARPS."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo_id', default='MIT/ast-finetuned-audioset-10-10-0.4593')
    parser.add_argument('--output_dir', default='./ast_weight')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--max_workers', type=int, default=8)
    args = parser.parse_args()
    dest = Path(args.output_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    result = snapshot_download(repo_id=args.repo_id, repo_type='model', revision=args.revision,
                               local_dir=str(dest), max_workers=args.max_workers)
    print(f'AST downloaded to: {result}')

if __name__ == '__main__':
    main()
