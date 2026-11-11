"""Check completed MVP outputs, build the existing report, and package delivery.

This command never downloads prices or runs a backtest. Use --check-only for
read-only readiness checks. Raw daily/adjustment caches are not packaged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd


QUINTILES = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
PORTFOLIOS = QUINTILES + ['benchmark', 'long_short', 'long_short_gross']
FACTORS = ['momentum', 'reversal', 'low_volatility', 'turnover', 'score']
PERFORMANCE = ['portfolio', 'total_return', 'cagr', 'annual_vol', 'sharpe',
               'max_drawdown', 'mean_monthly_turnover',
               'mean_monthly_one_way_turnover', 'annualized_one_way_turnover']
TABLE_COLUMNS = {
    'nav': ['date'] + PORTFOLIOS,
    'performance': PERFORMANCE,
    'rank_ic': ['date', 'factor', 'rank_ic', 'n'],
    'ic_summary': ['factor', 'mean_rank_ic', 'ic_std', 'ic_ir', 'positive_fraction', 'months'],
    'sample_summary': ['signal_date', 'constituents', 'eligible', 'excluded'],
    'trades': ['date', 'portfolio', 'code', 'side', 'notional', 'cost', 'reason', 'pre_trade_nav'],
    'execution_summary': ['portfolio', 'executed_orders', 'blocked_orders', 'total_notional', 'total_cost'],
    'turnover': ['date', 'portfolio', 'pre_trade_nav', 'buy_notional', 'sell_notional',
                 'cost', 'turnover', 'one_way_turnover'],
    'long_short_comparison': PERFORMANCE + ['cost_bps', 'target_long_exposure',
                                          'target_short_exposure', 'total_cost'],
    'benchmark_comparison': ['portfolio', 'cagr', 'benchmark_cagr', 'cagr_difference',
                             'relative_wealth_return', 'tracking_error', 'information_ratio'],
    'stale_valuations': ['portfolio', 'code', 'stale_days'],
    'missing_labels': ['scope', 'pairs', 'valid_pairs', 'missing_pairs', 'missing_stocks'],
}
CSV_COLUMNS = {f'results/tables/mvp_{name}.csv': columns
               for name, columns in TABLE_COLUMNS.items()}
CSV_COLUMNS.update({
    'data/mvp/signals.csv': ['date', 'code'] + FACTORS,
    'data/mvp/forward_returns.csv': ['date', 'code', 'forward_return'],
})
MEMBERS = 'data/mvp/constituents.csv'
MANIFEST = 'data/mvp/manifest.csv'
FIGURE = 'results/figures/mvp_nav.png'
SOURCE_FILES = [
    'README.md', '.gitignore', 'requirements.txt', 'pytest.ini',
    '.github/workflows/tests.yml', 'run_all.py', 'src/__init__.py', 'src/data.py',
    'src/factors.py', 'src/evaluate.py', 'src/backtest.py', 'tests/test_mvp.py',
    'tests/test_lookahead_canary.py',
    'notebooks/mvp_walkthrough.ipynb', 'report/build_report.py',
    'scripts/finalize_delivery.py', 'scripts/build_manifest.py',
]
REPORT_FILES = ['report/report.tex', 'report/report.pdf']


def fingerprint(paths):
    """Observe input size, timestamps and identity, without reading price data."""
    result = {}
    for path in paths:
        stat = path.stat()
        result[path] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    return result


def assert_unchanged(before):
    changed = []
    for path, expected in before.items():
        try:
            current = fingerprint([path])[path]
        except OSError:
            current = None
        if current != expected:
            changed.append(str(path))
    if changed:
        raise ValueError('Inputs changed during finalization; no new delivery accepted:\n'
                         + '\n'.join(changed))


def require_keys(frame, columns, label):
    keys = frame[columns]
    if keys.isna().any().any() or any(keys[column].astype(str).str.strip().eq('').any()
                                    for column in columns):
        raise ValueError(f'{label}: missing keys in {columns}.')
    if keys.duplicated().any():
        raise ValueError(f'{label}: duplicate keys in {columns}.')


def require_set(frame, column, expected, label):
    require_keys(frame, [column], label)
    if set(frame[column]) != set(expected):
        raise ValueError(f'{label}: expected {column} values {list(expected)}.')


def load_report_module(root):
    spec = importlib.util.spec_from_file_location('_mvp_delivery_report', root / 'report/build_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_delivery(root):
    """Require the entire agreed output set; return existing report data only."""
    root = Path(root).resolve()
    relative_files = list(CSV_COLUMNS) + [MEMBERS, MANIFEST, FIGURE] + SOURCE_FILES
    missing = [name for name in relative_files if not (root / name).is_file()]
    if missing:
        raise ValueError('Delivery inputs are incomplete; nothing generated:\n' + '\n'.join(missing))
    paths = [root / name for name in relative_files]
    before = fingerprint(paths)
    frames, errors = {}, []
    required = {**CSV_COLUMNS, MEMBERS: ['signal_date', 'code'],
                MANIFEST: ['code', 'start_date', 'end_date', 'row_count',
                           'download_time', 'download_time_source']}
    for name, columns in required.items():
        try:
            frame = pd.read_csv(root / name, dtype={'code': str})
            absent = sorted(set(columns) - set(frame.columns))
            if absent:
                errors.append(f'{name}: missing columns {absent}.')
            elif frame.empty and name != 'results/tables/mvp_stale_valuations.csv':
                errors.append(f'{name}: no records.')
            else:
                frames[name] = frame
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            errors.append(f'{name}: {exc}')
    if errors:
        raise ValueError('Delivery CSV requirements are not met:\n' + '\n'.join(errors))
    tables = {name: frames[f'results/tables/mvp_{name}.csv'] for name in TABLE_COLUMNS}
    sample = tables['sample_summary']
    sample_dates = pd.DatetimeIndex(pd.to_datetime(sample['signal_date'], errors='raise'))
    expected_months = pd.period_range('2017-12', '2025-11', freq='M')
    if (sample_dates.hasnans or not sample_dates.is_unique
            or not sample_dates.to_period('M').equals(expected_months)):
        raise ValueError('Sample dates must cover each of the 96 signal months exactly once in order.')

    members = frames[MEMBERS]
    members['signal_date'] = pd.to_datetime(members['signal_date'], errors='raise')
    require_keys(members, ['signal_date', 'code'], MEMBERS)
    if set(members.signal_date) != set(sample_dates):
        raise ValueError('Historical membership dates differ from the sample dates.')
    codes = sorted(members.code.unique())
    if any(re.fullmatch(r'(sh|sz|bj)\.\d{6}', code) is None for code in codes):
        raise ValueError('Membership contains an invalid BaoStock security code.')
    cache_paths = [root / 'data/mvp/daily' / f'{code}.csv' for code in codes]
    absent = [path.name for path in cache_paths if not path.is_file() or path.stat().st_size == 0]
    if absent:
        raise ValueError(f'Download is incomplete for {len(absent)} historical members:\n' + '\n'.join(absent))
    before.update(fingerprint(cache_paths))
    manifest = frames[MANIFEST]
    require_set(manifest, 'code', codes, MANIFEST)
    row_counts = pd.to_numeric(manifest.row_count, errors='raise')
    if row_counts.isna().any() or (row_counts <= 0).any() or (row_counts % 1 != 0).any():
        raise ValueError('Manifest row counts must be positive integers.')
    start = pd.to_datetime(manifest.start_date, errors='raise')
    end = pd.to_datetime(manifest.end_date, errors='raise')
    downloaded = pd.to_datetime(manifest.download_time, errors='raise', utc=True)
    if start.isna().any() or end.isna().any() or downloaded.isna().any() or (end < start).any():
        raise ValueError('Manifest contains missing or invalid dates.')
    if not manifest.download_time_source.eq('daily_file_mtime').all():
        raise ValueError('Manifest must identify its timestamp source as daily_file_mtime.')

    for name in ('performance', 'execution_summary'):
        require_set(tables[name], 'portfolio', PORTFOLIOS, name)
    require_set(tables['long_short_comparison'], 'portfolio', ['long_short', 'long_short_gross'],
                'long_short_comparison')
    require_set(tables['benchmark_comparison'], 'portfolio', QUINTILES, 'benchmark_comparison')
    require_set(tables['missing_labels'], 'scope', ['all', 'complete_windows', 'terminal_window'],
                'missing_labels')
    require_set(tables['ic_summary'], 'factor', FACTORS, 'ic_summary')
    nav = tables['nav']
    values = nav[PORTFOLIOS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('All eight NAV accounts must have positive finite values.')
    turnover = tables['turnover']
    turnover['date'] = pd.to_datetime(turnover['date'], errors='raise')
    require_keys(turnover, ['date', 'portfolio'], 'turnover')
    if set(turnover.portfolio) != set(PORTFOLIOS):
        raise ValueError('Turnover must include all eight accounts.')
    for portfolio, group in turnover.groupby('portfolio'):
        months = pd.DatetimeIndex(group.date).sort_values().to_period('M')
        if not months.equals(pd.period_range('2018-01', '2025-12', freq='M')):
            raise ValueError(f'Turnover for {portfolio} must contain all 96 execution months.')

    signals = frames['data/mvp/signals.csv']
    forward = frames['data/mvp/forward_returns.csv']
    for label, frame in [('signals', signals), ('forward_returns', forward)]:
        frame['date'] = pd.to_datetime(frame['date'], errors='raise')
        require_keys(frame, ['date', 'code'], label)
        if set(frame.date) != set(sample_dates):
            raise ValueError(f'{label}: expected the same 96 signal dates as sample_summary.')
    signal_keys = pd.MultiIndex.from_frame(signals[['date', 'code']])
    forward_keys = pd.MultiIndex.from_frame(forward[['date', 'code']])
    if len(signal_keys) != len(forward_keys) or len(signal_keys.difference(forward_keys)):
        raise ValueError('Signals and forward labels must have identical one-to-one date/code keys.')
    member_keys = pd.MultiIndex.from_frame(members[['signal_date', 'code']])
    if len(signal_keys.difference(member_keys)):
        raise ValueError('A signal security is absent from its historical member snapshot.')
    eligible = signals.groupby('date').size().reindex(sample_dates).to_numpy()
    if not np.array_equal(eligible, sample.eligible.to_numpy()):
        raise ValueError('Signal counts disagree with the sample summary.')
    terminal = forward.loc[forward.date.eq(sample_dates[-1]), 'forward_return']
    if terminal.empty or terminal.notna().any():
        raise ValueError('All final-window forward labels must remain missing.')
    last_ic = tables['rank_ic'].loc[
        pd.to_datetime(tables['rank_ic'].date).eq(sample_dates[-1])]
    if len(last_ic) != len(FACTORS) or last_ic.rank_ic.notna().any() or not last_ic.n.eq(0).all():
        raise ValueError('Final-window IC must be missing with zero complete pairs.')
    stale = tables['stale_valuations']
    if not stale.empty:
        require_keys(stale, ['portfolio', 'code'], 'stale_valuations')
        if not set(stale.portfolio).issubset(PORTFOLIOS):
            raise ValueError('Unknown account in stale valuation records.')

    report = load_report_module(root)
    data = report.read_inputs(root)
    assert_unchanged(before)
    return {'root': root, 'files': relative_files, 'fingerprint': before,
            'report': report, 'data': data, 'stocks': len(codes)}


def write_archive(entries, target):
    """Write supplied files and exact-content checksums to a temporary archive."""
    checksums = []
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name, path in sorted(entries.items()):
            digest = hashlib.sha256()
            with Path(path).open('rb') as incoming, bundle.open(name, 'w', force_zip64=True) as outgoing:
                for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
                    digest.update(chunk)
                    outgoing.write(chunk)
            checksums.append(f'{digest.hexdigest()}  {name}')
        note = (
            'A-share multifactor research delivery.\n\n'
            'Start with report/report.pdf and README.md. Editable source: report/report.tex.\n'
            'Results: results/tables/*.csv and results/figures/mvp_nav.png.\n'
            'Signals and forward labels: data/mvp/. Teaching: notebooks/mvp_walkthrough.ipynb.\n'
            'Raw daily prices, adjustment caches, logs and .git are excluded.\n'
            'Recomputing research requires the separate price cache or downloading it with run_all.py.\n'
            'finalize_delivery.py only checks outputs, builds the report and packages files.\n'
            'SHA256SUMS.txt identifies the exact packaged source and result bytes.\n'
        ).encode('utf-8')
        bundle.writestr('DELIVERY.txt', note)
        checksums.append(f'{hashlib.sha256(note).hexdigest()}  DELIVERY.txt')
        bundle.writestr('SHA256SUMS.txt', '\n'.join(checksums) + '\n')


def finalize(prepared, output, engine='xelatex'):
    root, output = prepared['root'], Path(output).resolve()
    inputs = set(prepared['fingerprint'])
    protected = inputs | {root / name for name in REPORT_FILES}
    if output.suffix.lower() != '.zip' or output in protected:
        raise ValueError('Output must be a .zip path and must not overwrite an input or report.')
    if output.exists() and (not output.is_file() or any(os.path.samefile(output, path)
                                                       for path in protected if path.exists())):
        raise ValueError('Output aliases a protected file or is not a regular file.')
    assert_unchanged(prepared['fingerprint'])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip = None
    with tempfile.TemporaryDirectory(prefix='.delivery-', dir=root / 'report') as temporary:
        staging = Path(temporary)
        tex, pdf = staging / 'report.tex', staging / 'report.pdf'
        tex.write_text(prepared['report'].render_tex(prepared['data']), encoding='utf-8')
        prepared['report'].compile_tex(tex, pdf, engine)
        if not pdf.is_file() or pdf.stat().st_size == 0:
            raise ValueError('Report compilation did not create a new PDF.')
        assert_unchanged(prepared['fingerprint'])
        entries = {name: root / name for name in prepared['files']}
        entries.update({'report/report.tex': tex, 'report/report.pdf': pdf})
        descriptor, name = tempfile.mkstemp(prefix=f'.{output.stem}-', suffix='.zip', dir=output.parent)
        os.close(descriptor)
        temporary_zip = Path(name)
        try:
            write_archive(entries, temporary_zip)
            assert_unchanged(prepared['fingerprint'])
            backups = {}
            for relative in REPORT_FILES:
                destination = root / relative
                backup = staging / ('old-' + destination.name)
                if destination.exists():
                    shutil.copyfile(destination, backup)
                    backups[destination] = backup
                else:
                    backups[destination] = None
            replaced = []
            try:
                for generated, relative in [(tex, REPORT_FILES[0]), (pdf, REPORT_FILES[1])]:
                    destination = root / relative
                    os.replace(generated, destination)
                    replaced.append(destination)
                os.replace(temporary_zip, output)
            except OSError:
                for destination in reversed(replaced):
                    backup = backups[destination]
                    if backup is None:
                        destination.unlink(missing_ok=True)
                    else:
                        os.replace(backup, destination)
                raise
        finally:
            temporary_zip.unlink(missing_ok=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--check-only', action='store_true', help='Read inputs only; do not compile or package.')
    parser.add_argument('--xelatex', default='xelatex', help='XeLaTeX executable name or absolute path.')
    parser.add_argument('--output', type=Path, help='Default: <root>/delivery/a-share-multifactor-delivery.zip.')
    args = parser.parse_args(argv)
    try:
        prepared = check_delivery(args.root)
        print(f'READY: 14 result CSVs, 8 accounts, 96 months, {prepared["stocks"]} member caches.')
        if not args.check_only:
            target = args.output or prepared['root'] / 'delivery/a-share-multifactor-delivery.zip'
            print(f'DELIVERY: {finalize(prepared, target, args.xelatex)}')
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
