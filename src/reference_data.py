"""Reference-data pilot: trading calendar, two indices and five listing dates."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data import PILOT_SYMBOLS

SPECS = {
    'calendar': {
        'endpoint': 'tool_trade_date_hist_sina', 'parameters': {},
        'source': 'AKShare / Sina trading calendar',
        'source_url': 'https://finance.sina.com.cn/realstock/company/klc_td_sh.txt',
    },
    'hs300': {
        'endpoint': 'stock_zh_index_daily_tx',
        'parameters': {'symbol': 'sh000300', 'start_date': '20150101', 'end_date': '20251231'},
        'source': 'AKShare / Tencent index history', 'source_url': 'https://gu.qq.com/sh000300/zs',
    },
    'zz500': {
        'endpoint': 'stock_zh_index_daily_tx',
        'parameters': {'symbol': 'sh000905', 'start_date': '20150101', 'end_date': '20251231'},
        'source': 'AKShare / Tencent index history', 'source_url': 'https://gu.qq.com/sh000905/zs',
    },
    'sh_listings': {
        'endpoint': 'stock_info_sh_name_code', 'parameters': {'symbol': '主板A股'},
        'source': 'AKShare / Shanghai Stock Exchange current security list',
        'source_url': 'https://www.sse.com.cn/assortment/stock/list/share/',
    },
    'sz_listings': {
        'endpoint': 'stock_info_sz_name_code', 'parameters': {'symbol': 'A股列表'},
        'source': 'AKShare / Shenzhen Stock Exchange current security list',
        'source_url': 'https://www.szse.cn/market/product/stock/list/index.html',
    },
}
PRICE_FIELDS = ['open', 'close', 'high', 'low']
LISTING_FIELDS = {
    'sh': ['证券代码', '证券简称', '上市日期'],
    'sz': ['A股代码', 'A股简称', 'A股上市日期'],
}


def checked_dates(values: pd.Series) -> pd.Series:
    """Parse strict calendar dates, refusing missing values or time components."""
    dates = pd.to_datetime(values, format='%Y-%m-%d', errors='raise')
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError('Missing date or non-midnight time component.')
    return dates


def validate_calendar(raw: pd.DataFrame) -> pd.DataFrame:
    """Return sorted unique trading dates without inventing or filling sessions."""
    if not isinstance(raw, pd.DataFrame) or raw.empty or 'trade_date' not in raw:
        raise ValueError('Expected nonempty trading calendar with trade_date.')
    calendar = pd.DataFrame({'date': checked_dates(raw['trade_date'])})
    if calendar['date'].duplicated().any():
        raise ValueError('Duplicate trading calendar date.')
    return calendar.sort_values('date').reset_index(drop=True)


def validate_benchmark(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Check index OHLC and unique dates inside the explicit research interval."""
    if symbol not in ('sh000300', 'sh000905'):
        raise ValueError('Unsupported benchmark symbol.')
    if not isinstance(raw, pd.DataFrame) or raw.empty or not set(['date'] + PRICE_FIELDS).issubset(raw):
        raise ValueError('Expected nonempty benchmark OHLC table.')
    table = raw[['date'] + PRICE_FIELDS].copy()
    table['date'] = checked_dates(table['date'])
    if table['date'].duplicated().any() or not table['date'].between('2015-01-01', '2025-12-31').all():
        raise ValueError('Duplicate or out-of-range benchmark date.')
    for field in PRICE_FIELDS:
        table[field] = pd.to_numeric(table[field], errors='raise')
        if not np.isfinite(table[field].to_numpy(dtype=float)).all() or table[field].le(0).any():
            raise ValueError('Benchmark prices must be positive and finite.')
    if ((table['low'] > table[['open', 'close']].min(axis=1)) |
            (table['high'] < table[['open', 'close']].max(axis=1))).any():
        raise ValueError('Benchmark OHLC violates the high/low range.')
    table['code'] = symbol[:2] + '.' + symbol[2:]
    return table[['date', 'code'] + PRICE_FIELDS].sort_values('date').reset_index(drop=True)


def validate_source_schema(raw: pd.DataFrame, key: str) -> None:
    """Check a source before caching; current exchange masters are not universes."""
    if key not in SPECS:
        raise ValueError('Unknown reference source.')
    if key == 'calendar':
        validate_calendar(raw)
    elif key in ('hs300', 'zz500'):
        validate_benchmark(raw, SPECS[key]['parameters']['symbol'])
    else:
        required = LISTING_FIELDS[key[:2]]
        if not isinstance(raw, pd.DataFrame) or raw.empty or not set(required).issubset(raw):
            raise ValueError('Exchange master is empty or missing listing fields.')


def extract_listings(sh: pd.DataFrame, sz: pd.DataFrame) -> pd.DataFrame:
    """Extract exactly five static A-share listing dates from exchange masters.

    Only name/date are retained. Current industry, ST status and share capital
    are deliberately excluded from this historical-reference output.
    """
    pieces = []
    for market, raw in [('sh', sh), ('sz', sz)]:
        key = market + '_listings'
        validate_source_schema(raw, key)
        code_field, name_field, date_field = LISTING_FIELDS[market]
        normalized_codes = raw[code_field].astype(str).str.strip().str.zfill(6)
        targets = [s[2:] for s in PILOT_SYMBOLS if s.startswith(market)]
        chosen = raw.loc[normalized_codes.isin(targets), [code_field, name_field, date_field]].copy()
        chosen[code_field] = normalized_codes.loc[chosen.index]
        if set(chosen[code_field]) != set(targets) or chosen[code_field].duplicated().any():
            raise ValueError('Missing or duplicate target stock in exchange master.')
        chosen = chosen.rename(columns={code_field: 'code', name_field: 'name', date_field: 'listing_date'})
        chosen['listing_date'] = checked_dates(chosen['listing_date'])
        if not chosen['listing_date'].between('1990-01-01', '2025-12-31').all():
            raise ValueError('Listing date is outside the plausible pilot interval.')
        if chosen['name'].isna().any() or chosen['name'].astype(str).str.strip().eq('').any():
            raise ValueError('Missing target security name.')
        chosen['code'] = market + '.' + chosen['code']
        chosen['source_endpoint'] = SPECS[key]['endpoint']
        pieces.append(chosen)
    return pd.concat(pieces, ignore_index=True).sort_values('code').reset_index(drop=True)


def fetch_reference(key: str) -> pd.DataFrame:
    """Call a fixed official SDK endpoint with bounded HTTP inactivity timeout."""
    import akshare as ak

    if key not in SPECS:
        raise ValueError('Unknown reference source.')
    spec = SPECS[key]
    previous = ak.get_timeout()
    try:
        ak.set_timeout(15)
        return getattr(ak, spec['endpoint'])(**spec['parameters'])
    finally:
        ak.set_timeout(previous)


def load_reference_snapshot(root: Path, key: str, offline: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load or fetch a reference table, enforcing local and public checksums."""
    if key not in SPECS:
        raise ValueError('Unknown reference source.')
    spec = SPECS[key]
    request = {'endpoint': spec['endpoint'], 'parameters': spec['parameters']}
    cache = root / 'data/reference/raw' / (key + '.csv')
    record = cache.with_suffix('.json')
    if cache.exists() != record.exists():
        raise ValueError('Incomplete reference cache: CSV and metadata must both exist.')
    if cache.exists():
        payload = cache.read_bytes()
        metadata = json.loads(record.read_text(encoding='utf-8'))
        if metadata.get('key') != key or metadata.get('request') != request or metadata.get('source') != spec['source']:
            raise ValueError('Reference cache request/source mismatch.')
        if sha256(payload).hexdigest() != metadata.get('sha256'):
            raise ValueError('Reference cache checksum mismatch.')
        raw = pd.read_csv(cache, dtype=str, keep_default_na=False)
    else:
        if offline:
            raise FileNotFoundError('Offline reference cache missing: ' + key)
        raw = fetch_reference(key)
        validate_source_schema(raw, key)
        payload = raw.to_csv(index=False, lineterminator='\n').encode('utf-8')
        metadata = dict(spec, key=key, request=request, sha256=sha256(payload).hexdigest(),
                        akshare_version=version('akshare'),
                        fetched_utc=datetime.now(timezone.utc).isoformat(),
                        representation='SDK DataFrame as CSV, not raw HTTP response bytes')
    public_reference = root / 'results/reference_provenance.json'
    if public_reference.exists():
        reference = json.loads(public_reference.read_text(encoding='utf-8'))
        expected = next((m for m in reference['snapshots'] if m['key'] == key), None)
        if expected is None or expected['sha256'] != metadata['sha256'] or expected['request'] != request:
            raise ValueError('Public reference checksum/request mismatch; preserve old snapshots.')
    validate_source_schema(raw, key)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix('.tmp')
        temporary.write_bytes(payload)
        temporary.replace(cache)
        record.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return raw, metadata


def check_reference_alignment(calendar: pd.DataFrame, benchmarks: pd.DataFrame,
                              listings: pd.DataFrame) -> pd.DataFrame:
    """Require exact 2015-2025 index/calendar agreement and calendar coverage of IPOs.

    Agreement between two vendors is a cross-check, not exchange certification
    of every historical date. Session 1 is the listing session; session 60 is
    the first session satisfying the explicit at-least-60-session convention.
    """
    if calendar['date'].min() > listings['listing_date'].min() or calendar['date'].max() < pd.Timestamp('2025-12-31'):
        raise ValueError('Trading calendar does not cover listing dates and the research end.')
    if set(benchmarks['code']) != {'sh.000300', 'sh.000905'} or benchmarks.duplicated(['date', 'code']).any():
        raise ValueError('Both unique benchmark series are required.')
    research_dates = pd.DatetimeIndex(calendar.loc[calendar['date'].between('2015-01-01', '2025-12-31'), 'date'])
    for code, table in benchmarks.groupby('code'):
        observed = pd.DatetimeIndex(table['date'])
        if len(observed.difference(research_dates)) or len(research_dates.difference(observed)):
            raise ValueError(f'Benchmark {code} dates differ from the trading calendar.')
    sessions = pd.DatetimeIndex(calendar['date'])
    rows = []
    for item in listings.itertuples(index=False):
        if item.listing_date not in sessions:
            raise ValueError('Listing date is absent from calendar: ' + item.code)
        position = sessions.get_loc(item.listing_date)
        if position + 59 >= len(sessions):
            raise ValueError('Insufficient calendar to locate the 60th listing session.')
        rows.append({'code': item.code, 'listing_date': item.listing_date,
                     'session_60_date': sessions[position + 59],
                     'convention': 'IPO session is 1; eligible from session 60'})
    return pd.DataFrame(rows)


def write_reference_results(root: Path, calendar: pd.DataFrame, benchmarks: pd.DataFrame,
                            listings: pd.DataFrame, thresholds: pd.DataFrame, metadata: list[dict]) -> dict:
    """Save reference inputs privately and publish measured audit summaries only."""
    destination = root / 'data/reference/processed'
    tables = root / 'results/tables'
    destination.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for filename, table in [('trading_calendar.csv', calendar), ('benchmark_daily.csv', benchmarks),
                            ('pilot_listing_dates.csv', listings)]:
        table.to_csv(destination / filename, index=False, date_format='%Y-%m-%d')
    thresholds.to_csv(tables / 'reference_listing_60th_session.csv', index=False, date_format='%Y-%m-%d')
    listings.to_csv(tables / 'reference_listing_dates.csv', index=False, date_format='%Y-%m-%d')
    summary = benchmarks.groupby('code').agg(rows=('date', 'size'), first_date=('date', 'min'),
                                             last_date=('date', 'max'), first_close=('close', 'first'),
                                             last_close=('close', 'last'))
    summary.to_csv(tables / 'reference_benchmark_summary.csv', date_format='%Y-%m-%d')
    calendar.assign(year=calendar['date'].dt.year).groupby('year').size().rename('trading_sessions').to_csv(
        tables / 'reference_calendar_year_counts.csv')
    benchmarks.assign(year=benchmarks['date'].dt.year).groupby(['year', 'code']).size().rename('rows').to_csv(
        tables / 'reference_benchmark_year_counts.csv')
    manifest = {'scope': 'Reference inputs only; current exchange masters are not historical stock universes',
                'snapshots': metadata}
    (root / 'results/reference_provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    audit = {
        'scope': 'calendar_two_indices_five_listing_dates',
        'calendar_source_first_date': str(calendar['date'].min().date()),
        'calendar_source_last_date': str(calendar['date'].max().date()), 'calendar_sessions': len(calendar),
        'calendar_sessions_2015_2025': int(calendar['date'].between('2015-01-01', '2025-12-31').sum()),
        'benchmarks': int(benchmarks['code'].nunique()), 'benchmark_rows': len(benchmarks),
        'listing_dates': len(listings), 'calendar_index_date_sets_equal_2015_2025': True,
        'calendar_covers_pilot_listing_dates': True,
        'pre_2015_cross_exchange_calendar_independently_verified': False,
        'calendar_sdk_correction': 'AKShare appends 1992-05-04 to Sina dates; no project-side date filling',
        'benchmark_convention': 'Vendor index OHLC via SDK qfq request; no total-return claim',
        'security_master_scope': 'Current exchange lists used only for the five historical listing dates',
        'independent_quote_software_price_check_completed': False,
        'full_historical_stock_universe_completed': False,
        'still_missing': ['historical_ST', 'suspension_status', 'historical_and_delisted_universe',
                          'point_in_time_industry', 'daily_total_market_cap', 'point_in_time_PE_PB'],
    }
    (root / 'results/reference_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return audit


def run_reference(root: Path, offline: bool = False) -> dict:
    """Fetch reference sources serially and return their measured audit summary."""
    raw, metadata = {}, []
    for key in SPECS:
        table, item = load_reference_snapshot(root, key, offline=offline)
        raw[key] = table
        metadata.append(item)
        print(f'Reference {key}: {len(table)} SDK rows', flush=True)
        if not offline:
            time.sleep(0.4)
    calendar = validate_calendar(raw['calendar'])
    benchmarks = pd.concat([validate_benchmark(raw[key], SPECS[key]['parameters']['symbol'])
                            for key in ('hs300', 'zz500')], ignore_index=True).sort_values(['date', 'code'])
    listings = extract_listings(raw['sh_listings'], raw['sz_listings'])
    thresholds = check_reference_alignment(calendar, benchmarks, listings)
    return write_reference_results(root, calendar, benchmarks, listings, thresholds, metadata)


def main() -> None:
    """Run reference acquisition separately from the stage-0 and price pilots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Require all cached reference snapshots.')
    args = parser.parse_args()
    audit = run_reference(Path(__file__).resolve().parents[1], offline=args.offline)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
