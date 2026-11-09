"""Apply eligibility rules to the pilot without inventing historical statuses."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.eligibility import classify_eligibility, summarize_eligibility


def audit_sample(root: Path, daily: pd.DataFrame, calendar: pd.DatetimeIndex,
                 listings: pd.DataFrame) -> dict:
    """Audit supplied pilot prices and listing dates; ST/suspension stay unknown.

    The listing dates are historical event dates from the current exchange
    master, not historical ST or industry information. The caller obtains and
    validates the calendar and market snapshots before invoking this function.
    """
    if not {'code', 'listing_date'}.issubset(listings.columns):
        raise ValueError('Listing metadata must contain code and listing_date.')
    if listings['code'].duplicated().any():
        raise ValueError('Duplicate listing metadata code.')
    if set(daily.columns) & {'listing_date', 'is_st', 'is_suspended', 'st_asof', 'suspension_asof'}:
        raise ValueError('Pilot audit expects price-only input; use classify_eligibility for supplied historical statuses.')
    records = daily.merge(listings[['code', 'listing_date']], on='code',
                          how='left', validate='many_to_one')
    # No historical flags were supplied by these price/reference endpoints.
    for column in ('is_st', 'is_suspended'):
        records[column] = pd.Series(pd.NA, index=records.index, dtype='boolean')
    for column in ('st_asof', 'suspension_asof'):
        records[column] = pd.NaT
    classified = classify_eligibility(records, calendar)
    annual = summarize_eligibility(classified)
    observed_volume = daily['volume'].gt(0) & daily['amount'].notna()
    vwap = daily['amount'].div(daily['volume'].where(observed_volume))
    # A diagnostic tolerance for displayed-price/volume rounding, not a filter.
    outside = observed_volume & (vwap.lt(daily['low'] * .995) | vwap.gt(daily['high'] * 1.005))
    quality = pd.DataFrame({'code': daily['code'], 'checked_rows': observed_volume,
                           'vwap_outside_range': outside}).groupby('code').sum().reset_index()
    quality['relative_tolerance'] = .005
    output = root / 'data/stage1/processed'
    tables = root / 'results/tables'
    output.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    classified.to_csv(output / 'pilot_eligibility.csv', index=False, date_format='%Y-%m-%d')
    annual.to_csv(tables / 'pilot_sample_summary.csv', index=False)
    quality.to_csv(tables / 'pilot_price_volume_consistency.csv', index=False)
    counts = classified['eligibility_status'].value_counts()
    report = {
        'scope': 'Five preselected stocks, observed rows only; not full-market universe',
        'input_rows': len(classified),
        'input_stocks': int(classified['code'].nunique()),
        'known_new_stock_exclusions': int(counts.get('excluded_new_stock', 0)),
        'unknown_rows': int(counts.get('unknown', 0)),
        'eligible_rows': int(counts.get('eligible', 0)),
        'st_status_obtained': False, 'suspension_status_obtained': False,
        'vwap_outside_range': int(outside.sum()),
        'vwap_relative_tolerance': .005,
        'external_price_validation_completed': False,
        'full_stage1_completed': False,
        'interpretation': 'Zero eligible rows means the supplied evidence cannot establish eligibility; it does not mean there were no tradable stocks.',
    }
    (root / 'results/pilot_eligibility_audit.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def run_sample_audit(root: Path, offline: bool = False,
                     daily: pd.DataFrame | None = None) -> dict:
    """Rebuild verified inputs and run the pilot audit without filling unknowns."""
    from src.data import run_pilot
    from src.reference_data import run_reference

    if daily is None:
        daily = run_pilot(root, offline=offline)
    reference = run_reference(root, offline=offline)
    directory = root / 'data/reference/processed'
    sessions = pd.read_csv(directory / 'trading_calendar.csv', parse_dates=['date'])
    listings = pd.read_csv(directory / 'pilot_listing_dates.csv', parse_dates=['listing_date'])
    report = audit_sample(root, daily, pd.DatetimeIndex(sessions['date']), listings)
    progress = {'reference_data': reference, 'pilot_eligibility': report,
                'full_stage1_completed': False, 'remaining_data_fields': reference['still_missing'],
                'remaining_acceptance': ['independent_quote_comparison', 'historical_universe_coverage',
                                         'user_explanation_of_core_modules']}
    (root / 'results/stage1_progress.json').write_text(json.dumps(progress, indent=2) + '\n', encoding='utf-8')
    return report


def main() -> None:
    """Run the full pilot-input and eligibility audit chain."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    result = run_sample_audit(Path(__file__).resolve().parents[1], offline=args.offline)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
