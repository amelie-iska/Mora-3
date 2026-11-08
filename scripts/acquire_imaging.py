#!/usr/bin/env python3
"""Acquire explicitly registered public assets into a fresh output directory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mora3.data.imaging import archive_members, download_asset, file_sha256, validate_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path('configs/imaging_datasets.json'))
    parser.add_argument('--dataset', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-free-gib', type=float, default=200)
    args = parser.parse_args()
    if args.minimum_free_gib < 0:
        parser.error('Minimum free space cannot be negative')
    catalog = validate_catalog(json.loads(args.catalog.read_text()))
    selected = [x for x in catalog['datasets'] if x['id'] in args.dataset]
    missing = set(args.dataset) - {x['id'] for x in selected}
    if missing:
        parser.error('Unknown dataset IDs: ' + ', '.join(sorted(missing)))
    for source in selected:
        if Path(source['id']).name != source['id'] or source['id'] in {'.', '..'}:
            parser.error('Unsafe dataset ID')
        for asset in source.get('assets', []):
            if Path(asset['name']).name != asset['name'] or asset['name'] in {'.', '..'}:
                parser.error('Unsafe asset name')
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema='mora3.imaging_acquisition.v1',
                   catalog_sha256=file_sha256(args.catalog), datasets=[])
    for source in selected:
        result = dict(id=source['id'], training_ready=False, assets=[])
        if not source.get('assets'):
            result.update(status='source_resolution_pending', reason='No verified public assets registered')
        else:
            folder = args.output / source['id']
            folder.mkdir()
            for asset in source['assets']:
                try:
                    receipt = download_asset(asset['url'], folder / asset['name'],
                        max_bytes=asset['bytes'], expected_bytes=asset['bytes'],
                        expected_sha256=asset.get('sha256'), min_free_gib=args.minimum_free_gib)
                    if asset.get('md5'):
                        with Path(receipt['path']).open('rb') as stream:
                            actual = hashlib.file_digest(stream, 'md5').hexdigest()
                        if actual != asset['md5']:
                            raise ValueError('Upstream MD5 mismatch; downloaded artifact retained for inspection')
                        receipt.update(md5=actual, checksum_origin='upstream-md5-and-local-sha256')
                    if asset['name'].endswith('.zip'):
                        inventory = archive_members(receipt['path'])
                        (folder / (asset['name'] + '.inventory.json')).write_text(json.dumps(inventory, indent=2))
                    verified = bool(asset.get('sha256') or asset.get('md5'))
                    receipt['status'] = 'downloaded_verified' if verified else 'downloaded_size_checked'
                    (folder / (asset['name'] + '.receipt.json')).write_text(json.dumps(receipt, indent=2))
                    result['assets'].append(receipt)
                except Exception as exc:
                    result['assets'].append(dict(name=asset['name'], status='failed',
                                                error_type=type(exc).__name__, reason=str(exc)))
            statuses = {x['status'] for x in result['assets']}
            result['status'] = ('failed' if 'failed' in statuses else
                                'downloaded_verified' if statuses == {'downloaded_verified'} else
                                'downloaded_size_checked')
        summary['datasets'].append(result)
        (args.output / 'acquisition_summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(dict(id=result['id'], status=result['status'])), flush=True)
    return 0 if all(x['status'].startswith('downloaded_') for x in summary['datasets']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
