"""Stage 1 pilot only: audit five specified stocks, not an all-A universe."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

PILOT_SYMBOLS = ('sh600000', 'sz000001', 'sh600519', 'sz000002', 'sz300750')
START_DATE, END_DATE = '20150101', '20251231'
VENDOR_FIELDS = ['date', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'amount']
PRICE_FIELDS = ['open', 'close', 'high', 'low']
ACTIVITY_FIELDS = ['volume', 'turnover', 'amount']
SOURCE_URL = 'https://github.com/akfamily/akshare/blob/main/akshare/stock/quote/stock_hist_tx.py'


def request_parameters(symbol: str, adjust: str) -> dict:
    """Restrict requests to the five predefined stocks and two price conventions."""
    if symbol not in PILOT_SYMBOLS or adjust not in ('', 'hfq'):
        raise ValueError('This pilot supports only the fixed five stocks and none/hfq.')
    return dict(symbol=symbol, start_date=START_DATE, end_date=END_DATE, adjust=adjust)


def validate_vendor_frame(raw: pd.DataFrame, symbol: str, adjust: str) -> pd.DataFrame:
    """Validate the SDK table; retain absent activity values without filling.

    Code and adjustment are request metadata, not independently verified vendor
    attributes. Missing or nonpositive OHLC and invalid price ranges are errors.
    """
    request_parameters(symbol, adjust)
    if not isinstance(raw, pd.DataFrame) or raw.empty or not set(VENDOR_FIELDS).issubset(raw.columns):
        raise ValueError('Expected a nonempty table with all required vendor fields.')
    frame = raw[VENDOR_FIELDS].copy()
    frame['date'] = pd.to_datetime(frame['date'], format='%Y-%m-%d', errors='raise')
    if frame['date'].isna().any() or frame['date'].duplicated().any():
        raise ValueError('Missing or duplicate date for the requested stock.')
    if not frame['date'].between('2015-01-01', '2025-12-31').all():
        raise ValueError('Date outside the pilot interval.')
    if not frame['date'].eq(frame['date'].dt.normalize()).all():
        raise ValueError('Daily dates must not contain time components.')
    for field in PRICE_FIELDS + ACTIVITY_FIELDS:
        frame[field] = pd.to_numeric(frame[field].replace('', np.nan), errors='raise')
        present = frame[field].dropna().to_numpy(dtype=float)
        if not np.isfinite(present).all():
            raise ValueError(f'Nonfinite {field}.')
        if field in PRICE_FIELDS:
            if frame[field].isna().any() or (frame[field] <= 0).any():
                raise ValueError(f'Missing or nonpositive {field}.')
        elif (frame[field] < 0).any():
            raise ValueError(f'Negative {field}.')
    if ((frame['low'] > frame[['open', 'close']].min(axis=1)) |
            (frame['high'] < frame[['open', 'close']].max(axis=1))).any():
        raise ValueError('OHLC values violate the low/high range.')
    frame['code'] = symbol[:2] + '.' + symbol[2:]
    frame['adjustment'] = adjust or 'none'
    return frame.sort_values('date').reset_index(drop=True)


def fetch_vendor_frame(symbol: str, adjust: str) -> pd.DataFrame:
    """Fetch sequentially with a 15-second connect/read timeout per HTTP request.

    This timeout bounds inactive HTTP operations, not the entire multiwindow
    download. The SDK's prior process-local setting is restored even on failure.
    """
    import akshare as ak

    parameters = request_parameters(symbol, adjust)
    previous = ak.get_timeout()
    try:
        ak.set_timeout(15)
        return ak.stock_zh_a_hist_tx(**parameters)
    finally:
        ak.set_timeout(previous)


def load_snapshot(root: Path, symbol: str, adjust: str, offline: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load a checksum-verified SDK CSV, or download and cache a validated table.

    A CSV without its metadata, altered content, or a changed public reference
    checksum is an error. Existing snapshots are never silently refreshed.
    """
    parameters = request_parameters(symbol, adjust)
    cache = root / 'data/stage1/raw' / f'{symbol}_{START_DATE}_{END_DATE}_{adjust or "none"}.csv'
    record = cache.with_suffix('.json')
    if cache.exists() != record.exists():
        raise ValueError('Incomplete snapshot: cache and metadata must both exist.')
    if cache.exists():
        payload = cache.read_bytes()
        metadata = json.loads(record.read_text(encoding='utf-8'))
        if metadata.get('request') != parameters or metadata.get('source') != 'AKShare / Tencent':
            raise ValueError('Snapshot request or source mismatch.')
        if metadata.get('sha256') != sha256(payload).hexdigest():
            raise ValueError('Snapshot checksum mismatch.')
        raw = pd.read_csv(cache, dtype=str, keep_default_na=False)
    else:
        if offline:
            raise FileNotFoundError(f'Offline cache is missing: {cache.name}')
        raw = fetch_vendor_frame(symbol, adjust)
        validate_vendor_frame(raw, symbol, adjust)
        payload = raw.to_csv(index=False, lineterminator='\n').encode('utf-8')
        metadata = {
            'source': 'AKShare / Tencent', 'endpoint': 'stock_zh_a_hist_tx',
            'source_url': SOURCE_URL, 'akshare_version': version('akshare'),
            'request': parameters, 'cache_file': cache.name,
            'sha256': sha256(payload).hexdigest(),
            'fetched_utc': datetime.now(timezone.utc).isoformat(),
            'representation': 'SDK DataFrame serialized as CSV; not raw HTTP bytes',
            'identity_and_adjustment': 'Added from request; external validation pending',
        }
    reference = root / 'results/stage1_provenance.json'
    if reference.exists():
        pinned = json.loads(reference.read_text(encoding='utf-8'))
        match = next((m for m in pinned['snapshots'] if m['request'] == parameters), None)
        if match is None or match['sha256'] != metadata['sha256']:
            raise ValueError('Public reference checksum mismatch; preserve and review both snapshots.')
    normalized = validate_vendor_frame(raw, symbol, adjust)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix('.tmp')
        temporary.write_bytes(payload)
        temporary.replace(cache)
        record.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return normalized, metadata


def align_adjustments(unadjusted: pd.DataFrame, backward: pd.DataFrame) -> pd.DataFrame:
    """Join identical stock/date keys; preserve unadjusted activity measurements.

    Missing dates are an audit failure, not a reason to drop observations. The
    backward-adjusted prices are analysis fields, not executable trade prices.
    """
    keys = ['date', 'code']
    for table, label in [(unadjusted, 'none'), (backward, 'hfq')]:
        if (table.empty or table['code'].nunique() != 1 or table.duplicated(keys).any()
                or not table['adjustment'].eq(label).all()):
            raise ValueError('Expected one stock, unique keys and the correct adjustment.')
    left_keys = pd.MultiIndex.from_frame(unadjusted[keys])
    right_keys = pd.MultiIndex.from_frame(backward[keys])
    if len(left_keys.difference(right_keys)) or len(right_keys.difference(left_keys)):
        raise ValueError('Unadjusted and backward-adjusted date/code sets differ.')
    prices = backward[keys + PRICE_FIELDS].rename(columns={k: 'hfq_' + k for k in PRICE_FIELDS})
    return unadjusted.drop(columns='adjustment').merge(prices, on=keys, validate='one_to_one').sort_values(keys)


def build_coverage(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count records for every fixed stock/year, including zero-observation years.

    These counts describe downloaded rows, not complete exchange calendars,
    eligible research universes, or the numbers removed by exclusion rules.
    """
    codes = [s[:2] + '.' + s[2:] for s in PILOT_SYMBOLS]
    grid = pd.MultiIndex.from_product([range(2015, 2026), codes], names=['year', 'code'])
    grouped = daily.assign(year=daily['date'].dt.year).groupby(['year', 'code'])
    detail = grouped.agg(rows=('date', 'size'), first_date=('date', 'min'), last_date=('date', 'max'))
    detail = detail.reindex(grid)
    detail['rows'] = detail['rows'].fillna(0).astype(int)
    detail = detail.reset_index()
    annual = detail.assign(has_observations=detail['rows'].gt(0)).groupby('year').agg(
        pilot_stocks_with_observations=('has_observations', 'sum'), observed_rows=('rows', 'sum'))
    annual['predefined_pilot_stocks'] = len(PILOT_SYMBOLS)
    return detail, annual.reset_index()


def write_audit(root: Path, daily: pd.DataFrame, snapshots: list[dict]) -> None:
    """Write measured coverage and an explicit capability-gap audit for the pilot."""
    results = root / 'results'
    tables = results / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    detail, annual = build_coverage(daily)
    detail.to_csv(tables / 'stage1_stock_year_coverage.csv', index=False, date_format='%Y-%m-%d')
    annual.to_csv(tables / 'stage1_year_coverage.csv', index=False)
    fields = []
    for field in PRICE_FIELDS + ACTIVITY_FIELDS + ['hfq_' + x for x in PRICE_FIELDS]:
        fields.append({'field': field, 'present_rows': int(daily[field].notna().sum()),
                       'missing_rows': int(daily[field].isna().sum()),
                       'total_rows': len(daily), 'status': 'measured_sdk_values'})
    for field in ['is_st', 'is_suspended', 'listing_date', 'delisting_date', 'industry_asof',
                  'total_market_cap', 'pe_asof', 'pb_asof', 'limit_up_price', 'limit_down_price']:
        fields.append({'field': field, 'present_rows': 0, 'missing_rows': len(daily),
                       'total_rows': len(daily), 'status': 'not_obtained_by_this_pilot'})
    pd.DataFrame(fields).to_csv(tables / 'stage1_field_coverage.csv', index=False)
    missing = daily.assign(year=daily['date'].dt.year).groupby(['year', 'code'])[ACTIVITY_FIELDS].agg(
        lambda series: int(series.isna().sum()))
    missing.rename(columns={k: 'missing_' + k for k in ACTIVITY_FIELDS}).to_csv(tables / 'stage1_activity_missing.csv')
    flags = daily.groupby('code').agg(
        observed_rows=('date', 'size'), first_observed_date=('date', 'min'),
        last_observed_date=('date', 'max'), zero_volume_rows=('volume', lambda s: int(s.eq(0).sum())))
    flags.to_csv(tables / 'stage1_stock_summary.csv', date_format='%Y-%m-%d')
    provenance = {'scope': 'Five preselected stocks, no universe exclusions or external price verification',
                  'snapshots': snapshots}
    (results / 'stage1_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    audit = {
        'scope': 'stage1_five_stock_pilot', 'stocks_requested': len(PILOT_SYMBOLS),
        'stocks_observed': int(daily['code'].nunique()), 'paired_daily_rows': len(daily),
        'duplicate_date_code': int(daily.duplicated(['date', 'code']).sum()),
        'missing_ohlc': int(daily[PRICE_FIELDS + ['hfq_' + x for x in PRICE_FIELDS]].isna().sum().sum()),
        'date_sets_aligned_for_all_pairs': True, 'external_price_comparison_completed': False,
        'full_market_pipeline_completed': False, 'st_suspension_new_stock_exclusions_applied': False,
        'blocking_fields': ['historical_stock_universe_including_delisted', 'is_st', 'is_suspended',
                            'listing_date_and_exchange_calendar', 'point_in_time_industry',
                            'daily_total_market_cap', 'point_in_time_pe_pb', 'benchmark_series'],
        'zero_volume_is_not_a_verified_suspension_flag': True,
    }
    (results / 'stage1_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run_pilot(root: Path, offline: bool = False) -> pd.DataFrame:
    """Read ten snapshots sequentially and audit their five paired price histories."""
    frames, snapshots = [], []
    for symbol in PILOT_SYMBOLS:
        pair = []
        for adjust in ('', 'hfq'):
            frame, metadata = load_snapshot(root, symbol, adjust, offline=offline)
            pair.append(frame)
            snapshots.append(metadata)
            print(f'{symbol} {adjust or "none"}: {len(frame)} rows', flush=True)
            if not offline:
                time.sleep(0.4)
        frames.append(align_adjustments(*pair))
    daily = pd.concat(frames, ignore_index=True).sort_values(['date', 'code'])
    if daily.duplicated(['date', 'code']).any():
        raise ValueError('Duplicate combined (date, code).')
    destination = root / 'data/stage1/processed'
    destination.mkdir(parents=True, exist_ok=True)
    daily.to_csv(destination / 'pilot_daily.csv', index=False, date_format='%Y-%m-%d')
    write_audit(root, daily, snapshots)
    return daily


def main() -> None:
    """Expose a separate stage-1 pilot command without changing stage-0 behavior."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Require all ten cached snapshots; never connect.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    daily = run_pilot(root, offline=args.offline)
    print(f'Pilot only: {daily["code"].nunique()} stocks, {len(daily)} paired stock/date records. Stage 1 is not complete.')


if __name__ == '__main__':
    main()
