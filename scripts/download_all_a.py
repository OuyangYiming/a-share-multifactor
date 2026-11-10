"""Download historical Shanghai/Shenzhen A shares without changing the CSI800 run."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data import (BaoStockClient, DATA_START, DATA_END, FIELDS, _Socket,
                      _save, _session_blocks, load_calendar)

# 689 is a depositary-receipt code range, rather than an ordinary A share.
A_SHARE_PATTERN = r'(?:sh\.6(?!89)\d{5}|sz\.(?:00|30)\d{4})'
PRICE_COLUMNS = FIELDS.split(',')


class DownloadSocket(_Socket):
    """Allow larger responses while retaining a finite per-request deadline."""
    def send(self, payload):
        self.deadline = time.monotonic() + 120
        self.connection.sendall(payload)
        return len(payload)

    def recv(self, count):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('BaoStock response exceeded 120 seconds')
        self.connection.settimeout(min(45, remaining))
        value = self.connection.recv(count)
        if not value:
            raise ConnectionError('BaoStock connection closed')
        return value


class DownloadClient(BaoStockClient):
    def connect(self):
        import baostock.common.context as context
        super().connect()
        context.default_socket = DownloadSocket(context.default_socket.connection)


@contextmanager
def download_lock(folder):
    """OS-managed lock is released even when an interrupted process exits."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.download.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if sys.platform == 'win32':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == 'win32':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def select_a_shares(snapshot):
    if not {'code', 'tradeStatus', 'code_name'}.issubset(snapshot):
        raise ValueError('Historical snapshot is missing required columns')
    if snapshot.empty or snapshot.code.duplicated().any():
        raise ValueError('Historical snapshot is empty or contains duplicate codes')
    result = snapshot.loc[snapshot.code.str.fullmatch(A_SHARE_PATTERN)].copy()
    if result.empty or not result.tradeStatus.isin(['0', '1']).all():
        raise ValueError('Invalid A-share snapshot or trading status')
    # Names can be current names. Neither names nor current listing status filter history.
    return result.sort_values('code').reset_index(drop=True)


def validate_raw(table, code, first, last):
    if not set(PRICE_COLUMNS).issubset(table):
        raise ValueError(f'{code}: incomplete daily schema')
    table = table[PRICE_COLUMNS].copy()
    if table.empty:
        return table
    dates = pd.to_datetime(table.date, errors='raise')
    if (not table.code.eq(code).all() or dates.duplicated().any()
            or not dates.between(pd.Timestamp(first), pd.Timestamp(last)).all()):
        raise ValueError(f'{code}: wrong code, duplicate date or out-of-window row')
    for field in PRICE_COLUMNS[2:]:
        values = pd.to_numeric(table[field].replace('', np.nan), errors='raise')
        if not np.isfinite(values.dropna()).all():
            raise ValueError(f'{code}: nonfinite {field}')
        if field in ('isST', 'tradestatus') and not values.dropna().isin([0, 1]).all():
            raise ValueError(f'{code}: invalid {field}')
    return table.sort_values('date').reset_index(drop=True)


def uncovered_blocks(calendar, first, last, intervals):
    sessions = calendar[(calendar >= pd.Timestamp(first)) & (calendar <= pd.Timestamp(last))]
    covered = np.zeros(len(sessions), dtype=bool)
    for start, end in intervals:
        covered |= (sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))
    gaps = np.flatnonzero(~covered)
    if not len(gaps):
        return []
    groups = np.split(gaps, np.flatnonzero(np.diff(gaps) > 1) + 1)
    return [block for group in groups for block in _session_blocks(
        calendar, sessions[group[0]], sessions[group[-1]])]


def validate_adjustments(adjustments, code):
    required = ['code', 'dividOperateDate', 'backAdjustFactor']
    if not set(required).issubset(adjustments):
        raise ValueError(f'{code}: incomplete adjustment schema')
    factors = adjustments[required].copy()
    factors['date'] = pd.to_datetime(factors.dividOperateDate, errors='raise')
    factors['backAdjustFactor'] = pd.to_numeric(factors.backAdjustFactor, errors='raise')
    if (not factors.code.eq(code).all() or factors.date.isna().any()
            or factors.date.duplicated().any()
            or not np.isfinite(factors.backAdjustFactor).all()
            or factors.backAdjustFactor.le(0).any()
            or factors.date.gt(pd.Timestamp(DATA_END)).any()):
        raise ValueError(f'{code}: invalid adjustment events')
    return factors


def adjusted_daily(pieces, adjustments, code, first, last):
    factors = validate_adjustments(adjustments, code)
    table = pd.concat(pieces, ignore_index=True)
    table = table.loc[table.date.between(first, last)].copy()
    if table.empty:
        raise ValueError(f'{code}: no returned daily rows in requested window')
    for field in PRICE_COLUMNS[2:]:
        table[field] = pd.to_numeric(table[field].replace('', np.nan), errors='raise')
    if table.drop_duplicates().duplicated(['date', 'code']).any():
        raise ValueError(f'{code}: conflicting overlapping cache rows')
    table = table.drop_duplicates(['date', 'code']).sort_values('date').reset_index(drop=True)
    table['date'] = pd.to_datetime(table.date)
    if adjustments.empty:
        table['adj_close'] = table.close
    else:
        aligned = pd.merge_asof(table[['date']], factors.sort_values('date')[
            ['date', 'backAdjustFactor']], on='date', direction='backward')
        table['adj_close'] = table.close * aligned.backAdjustFactor.fillna(1.0)
    return table


class Downloader:
    def __init__(self, root, offline=False):
        self.root, self.offline = Path(root), offline
        self.folder = self.root / 'data/all_a'
        self.client = DownloadClient()
        self.requests = 0

    def cached(self, path, endpoint, validator=None, **kwargs):
        if path.exists():
            table = pd.read_csv(path, dtype=str, keep_default_na=False)
            if validator:
                validator(table)
            return table
        if self.offline:
            raise FileNotFoundError(f'Missing cache: {path.relative_to(self.root)}')
        if self.requests >= 10000:
            raise RuntimeError('Per-run request budget reached; resume in a later session')
        started = time.monotonic()
        self.requests += 1
        try:
            table = self.client.query(endpoint, **kwargs)
            if validator:
                validator(table)
            _save(table, path)
        except Exception as exc:
            self.log(endpoint, kwargs, started, 0, f'{type(exc).__name__}: {exc}')
            raise
        self.log(endpoint, kwargs, started, len(table), '')
        return table

    def log(self, endpoint, kwargs, started, rows, error):
        row = pd.DataFrame([{'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                             'endpoint': endpoint, 'parameters': repr(kwargs),
                             'seconds': time.monotonic() - started, 'rows': rows,
                             'error': error}])
        path = self.folder / 'download_log.csv'
        row.to_csv(path, mode='a', header=not path.exists(), index=False)

    def universe(self, calendar):
        dates = pd.Series(calendar).groupby(calendar.to_period('M')).max().loc['2017-12':'2025-11']
        frames = []
        for number, day in enumerate(dates, 1):
            day = day.strftime('%Y-%m-%d')
            raw = self.cached(self.folder / 'universe' / f'{day}.csv', 'query_all_stock',
                              select_a_shares, day=day)
            selected = select_a_shares(raw)
            selected.insert(0, 'signal_date', day)
            frames.append(selected)
            print(f'UNIVERSE {number}/{len(dates)} date={day} A_shares={len(selected)}', flush=True)
        members = pd.concat(frames, ignore_index=True)
        _save(members, self.folder / 'constituents.csv')
        summary = members.groupby('signal_date').agg(
            listed_a_shares=('code', 'nunique'),
            snapshot_suspended=('tradeStatus', lambda s: int(s.eq('0').sum())))
        _save(summary.reset_index(), self.folder / 'sample_summary.csv')
        return members

    def stock(self, code, first, calendar):
        raw_folder = self.folder / 'raw'
        raw_folder.mkdir(parents=True, exist_ok=True)
        # Copy original responses, never treat a late-starting ready file as full history.
        for source in sorted((self.root / 'data/mvp/raw').glob(f'{code}_*.csv')):
            target = raw_folder / source.name
            if not target.exists():
                _, start, end = source.stem.split('_')
                validate_raw(pd.read_csv(source, dtype=str, keep_default_na=False), code, start, end)
                shutil.copyfile(source, target)
        intervals, pieces = [], []
        for path in sorted(raw_folder.glob(f'{code}_*.csv')):
            _, start, end = path.stem.split('_')
            pieces.append(validate_raw(pd.read_csv(path, dtype=str, keep_default_na=False), code, start, end))
            intervals.append((start, end))
        missing = uncovered_blocks(calendar, first, DATA_END, intervals)
        for start, end in missing:
            validator = lambda t, s=start, e=end: validate_raw(t, code, s, e)
            table = self.cached(raw_folder / f'{code}_{start}_{end}.csv',
                                'query_history_k_data_plus', validator,
                                code=code, fields=FIELDS, start_date=start, end_date=end,
                                frequency='d', adjustflag='3')
            pieces.append(table[PRICE_COLUMNS])
        adjustment_path = self.folder / 'adjustments' / f'{code}.csv'
        original = self.root / 'data/mvp/adjustments' / f'{code}.csv'
        if not adjustment_path.exists() and original.exists():
            validate_adjustments(pd.read_csv(original, dtype=str, keep_default_na=False), code)
            adjustment_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, adjustment_path)
        adjustments = self.cached(adjustment_path, 'query_adjust_factor',
                                  lambda t: validate_adjustments(t, code),
                                  code=code, start_date='1990-01-01', end_date=DATA_END)
        table = adjusted_daily(pieces, adjustments, code, first, DATA_END)
        _save(table, self.folder / 'daily' / f'{code}.csv')
        return table, len(missing)

    def prices(self, members, calendar, limit=None):
        first_signals = pd.to_datetime(members.groupby('code').signal_date.min()).sort_index()
        positions = calendar.get_indexer(pd.DatetimeIndex(first_signals))
        if (positions < 252).any():
            raise ValueError('Calendar must include 252 pre-signal sessions')
        starts = pd.Series(calendar[positions - 252].strftime('%Y-%m-%d'), index=first_signals.index)
        starts.iloc[0] = DATA_START
        manifest_path = self.folder / 'manifest.csv'
        records = (pd.read_csv(manifest_path, keep_default_na=False).set_index('code').to_dict('index')
                   if manifest_path.exists() else {})
        records = {code: record for code, record in records.items() if code in starts.index}
        signal_dates = members.groupby('code').signal_date.agg(list)
        failures, consecutive = {}, 0
        started = time.monotonic()
        selected = starts.iloc[:limit] if limit is not None else starts
        for number, (code, first) in enumerate(selected.items(), 1):
            try:
                table, blocks = self.stock(code, first, calendar)
                actual_first, actual_last = table.date.min(), table.date.max()
                internal = calendar[(calendar >= actual_first) & (calendar <= actual_last)]
                records[code] = {'requested_start': first, 'requested_end': DATA_END,
                                 'first_date': actual_first.strftime('%Y-%m-%d'),
                                 'last_date': actual_last.strftime('%Y-%m-%d'), 'rows': len(table),
                                 'missing_sessions_between_first_last': len(internal.difference(table.date)),
                                 'missing_signal_date_rows': len(pd.DatetimeIndex(
                                     pd.to_datetime(signal_dates[code])).difference(table.date)),
                                 'last_member_signal_date': max(signal_dates[code]),
                                 'ends_before_research_end': bool(actual_last < pd.Timestamp(DATA_END)),
                                 'nonpositive_close_rows': int(table.close.le(0).sum())}
                consecutive = 0
                print(f'DAILY {number}/{len(selected)} code={code} rows={len(table)} '
                      f'new_blocks={blocks} elapsed={time.monotonic()-started:.1f}s', flush=True)
            except Exception as exc:
                failures[code] = f'{type(exc).__name__}: {exc}'
                records.pop(code, None)
                consecutive += 1
                print(f'DEFERRED {code}: {failures[code]}', flush=True)
            if number % 10 == 0 or number == len(selected) or consecutive >= 3:
                _save(pd.DataFrame.from_dict(records, orient='index').rename_axis('code').reset_index(), manifest_path)
                _save(pd.DataFrame(list(failures.items()), columns=['code', 'error']), self.folder / 'pending.csv')
            if consecutive >= 3:
                raise RuntimeError('Three consecutive securities failed; saved progress, resume later')
        if failures:
            raise RuntimeError(f'{len(failures)} securities remain pending; rerun to retry')
        if limit is not None and len(selected) < len(starts):
            print(f'PARTIAL: processed {len(selected)} of {len(starts)} securities (--limit)', flush=True)
        else:
            print(f'COMPLETE: {len(starts)} securities; {sum(r["rows"] for r in records.values())} daily rows', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'universe', 'prices'], default='all')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--limit', type=int, help='Process only the first N codes for a smoke run.')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    downloader = Downloader(ROOT, args.offline)
    with download_lock(downloader.folder):
        try:
            # This shares the unchanged calendar, not the CSI800 constituent selection.
            calendar = load_calendar(ROOT, downloader.client, args.offline)
            if args.stage in ('all', 'universe'):
                members = downloader.universe(calendar)
            else:
                members = pd.read_csv(downloader.folder / 'constituents.csv', dtype=str)
                expected = pd.Series(calendar).groupby(calendar.to_period('M')).max().loc['2017-12':'2025-11']
                if set(members.signal_date) != set(expected.dt.strftime('%Y-%m-%d')):
                    raise ValueError('Incomplete monthly universe; run --stage universe first')
            if args.stage in ('all', 'prices'):
                downloader.prices(members, calendar, args.limit)
        finally:
            downloader.client.close()


if __name__ == '__main__':
    main()
