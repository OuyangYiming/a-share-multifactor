"""Stage 0: download and validate one real stock's daily history."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ['date', 'code', 'close', 'adjustflag']
REQUEST = dict(symbol='sh600000', start_date='20150101',
               end_date='20251231', adjust='hfq')
CODE = 'sh.600000'


def validate_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate the fixed example and return a sorted, date-indexed copy.

    No missing prices are filled. Duplicate keys, wrong adjustment flags,
    mixed stocks and observations outside the requested period are errors.
    """
    if list(raw.columns) != FIELDS or raw.empty:
        raise ValueError('Expected nonempty daily data with the requested fields.')
    frame = raw.copy()
    frame['date'] = pd.to_datetime(frame['date'], format='%Y-%m-%d', errors='raise')
    if frame['date'].isna().any():
        raise ValueError('Missing date.')
    if frame.duplicated(['date', 'code']).any():
        raise ValueError('Duplicate (date, code).')
    if not frame['code'].eq(CODE).all():
        raise ValueError('Unexpected stock code.')
    if not frame['adjustflag'].astype(str).eq('1').all():
        raise ValueError('Expected backward-adjusted prices (adjustflag=1).')
    if not frame['date'].between('2015-01-01', '2025-12-31').all():
        raise ValueError('Date outside the requested interval.')
    for column in ['close']:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f'Missing or infinite {column}.')
    if (frame['close'] <= 0).any():
        raise ValueError('Nonpositive price.')
    return frame.sort_values('date').set_index('date')


def fetch_daily() -> pd.DataFrame:
    """Read AKShare's Tencent daily endpoint using explicit backward adjustment.

    The SDK combines date windows and filters the requested period. The code
    and adjustment flag below describe the request, not independent vendor
    evidence of correct identity or adjustment; external price checks remain.
    """
    import akshare as ak

    data = ak.stock_zh_a_hist_tx(**REQUEST)
    if not isinstance(data, pd.DataFrame) or not {'date', 'close'}.issubset(data.columns):
        raise ValueError('Unexpected AKShare response schema.')
    raw = data[['date', 'close']].copy()
    raw['date'] = pd.to_datetime(raw['date'], errors='raise').dt.strftime('%Y-%m-%d')
    raw['code'] = CODE
    raw['adjustflag'] = '1'
    return raw[FIELDS]


def load_example(root: Path, offline: bool = False) -> tuple[pd.DataFrame, dict]:
    """Use the local snapshot, or fetch it; enforce the reference checksum.

    The reference JSON is committed, but raw market data stay in ignored data/.
    If the provider revises history, stop instead of silently changing results.
    """
    cache = root / 'data/raw/sh600000_20150101_20251231_hfq.csv'
    reference = root / 'config/example_snapshot.json'
    expected = json.loads(reference.read_text(encoding='utf-8')) if reference.exists() else None
    if expected and (expected['request'] != REQUEST or expected['fields'] != FIELDS):
        raise ValueError('Reference snapshot parameters do not match this example.')
    if cache.exists():
        payload = cache.read_bytes()
        raw = pd.read_csv(cache, dtype=str, keep_default_na=False)
    else:
        if offline:
            raise FileNotFoundError('No cache. Run once online with: python run_all.py')
        raw = fetch_daily()
        validate_daily(raw)
        payload = raw.to_csv(index=False, lineterminator='\n').encode('utf-8')
    digest = sha256(payload).hexdigest()
    if expected and digest != expected['sha256']:
        raise ValueError('Snapshot checksum mismatch. Preserve the old data and review the change; do not silently replace the reference.')
    frame = validate_daily(raw)
    if expected is None:
        expected = {
            'source': 'AKShare / Tencent',
            'source_url': 'https://akshare.akfamily.xyz/data/stock/stock.html',
            'endpoint': 'stock_zh_a_hist_tx',
            'akshare_version': version('akshare'), 'request': REQUEST,
            'fields': FIELDS, 'sha256': digest,
            'snapshot_recorded_utc': datetime.now(timezone.utc).isoformat(),
            'note': 'Single-stock stage-0 example; not a research universe.'}
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix('.tmp')
        temporary.write_bytes(payload)
        temporary.replace(cache)
    return frame, expected


def write_results(frame: pd.DataFrame, metadata: dict, root: Path) -> None:
    """Write descriptive tables, provenance and a monochrome adjusted-close plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results = root / 'results'
    (results / 'figures').mkdir(parents=True, exist_ok=True)
    (results / 'tables').mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{
        'code': CODE, 'rows': len(frame),
        'first_date': frame.index.min().strftime('%Y-%m-%d'),
        'last_date': frame.index.max().strftime('%Y-%m-%d'),
        'duplicate_date_code': 0, 'missing_close': int(frame['close'].isna().sum()),
        'first_close_hfq': frame['close'].iloc[0], 'last_close_hfq': frame['close'].iloc[-1],
        'adjustflag': '1', 'raw_sha256': metadata['sha256']}])
    summary.to_csv(results / 'tables/example_summary.csv', index=False, lineterminator='\n')
    annual = frame.groupby(frame.index.year).agg(rows=('close', 'size'))
    annual.index.name = 'year'
    annual.to_csv(results / 'tables/example_year_counts.csv', lineterminator='\n')
    (results / 'provenance.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
    ax.plot(frame.index, frame['close'], color='black', linewidth=0.85)
    ax.set(title='Shanghai Pudong Development Bank (sh.600000)',
           xlabel='Date', ylabel='Backward-adjusted close (CNY scale)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='0.85', linewidth=0.5)
    fig.suptitle('AKShare / Tencent daily history | 2015-2025 | backward adjusted', fontsize=10)
    fig.savefig(results / 'figures/example_close.png', dpi=180)
    plt.close(fig)
