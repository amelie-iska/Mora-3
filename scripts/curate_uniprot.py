#!/usr/bin/env python
"""Curate an offline, checksum-pinned UniProt search snapshot (at most 64 rows)."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mora3.data.uniprot import curate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = curate(args.source, args.output, expected_sha256=args.sha256, release=args.release)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
