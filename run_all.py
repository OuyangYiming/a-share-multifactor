"""Reproduce implemented price, reference-data and pilot-eligibility audits."""

import argparse
from pathlib import Path

from src.example import load_example, write_results


def main() -> None:
    """Run implemented data modules; later research stages remain unimplemented."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Require existing local data; never connect to the provider.')
    parser.add_argument('--stage', choices=['stage0', 'stage1-pilot', 'reference', 'sample-audit', 'all'], default='all',
                        help='Select implemented modules (default: all).')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    daily = None
    if args.stage in ['stage0', 'all']:
        frame, metadata = load_example(root, offline=args.offline)
        write_results(frame, metadata, root)
        print(f'Stage 0: {len(frame)} rows, {frame.index.min().date()} to {frame.index.max().date()}')
        print(f'Raw SHA256: {metadata["sha256"]}')
        print('Outputs: results/figures/example_close.png and results/tables/example_summary.csv')
    if args.stage in ['stage1-pilot', 'all']:
        from src.data import run_pilot
        daily = run_pilot(root, offline=args.offline)
        print(f'Stage 1 pilot: {daily["code"].nunique()} stocks, {len(daily)} paired daily records.')
        print('Full-market universe and historical status filters are not complete; see results/stage1_audit.json.')
    if args.stage == 'reference':
        from src.reference_data import run_reference
        print(run_reference(root, offline=args.offline))
    if args.stage in ['sample-audit', 'all']:
        from src.sample_audit import run_sample_audit
        audit = run_sample_audit(root, offline=args.offline, daily=daily)
        print(f'Pilot eligibility: {audit["known_new_stock_exclusions"]} known new-stock exclusions, '
              f'{audit["unknown_rows"]} unresolved rows, {audit["eligible_rows"]} established eligible rows.')
        print('Unknown historical status is not treated as normal. See results/stage1_progress.json.')


if __name__ == '__main__':
    main()
