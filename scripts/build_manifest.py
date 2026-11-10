"""Inventory completed daily caches without network access or backtesting.

The downloader did not persist per-stock completion timestamps. download_time
therefore records the daily CSV's last-write time, in UTC+08:00, as explicitly
identified by download_time_source. It is not a historical HTTP/API receipt.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
import re
import tempfile


COLUMNS = ['code', 'start_date', 'end_date', 'row_count', 'download_time',
           'download_time_source']
LOCAL_TIME = timezone(timedelta(hours=8))


def build_manifest(root):
    folder = Path(root).resolve() / 'data/mvp'
    with (folder / 'constituents.csv').open(encoding='utf-8-sig', newline='') as stream:
        expected = {row['code'] for row in csv.DictReader(stream)}
    if not expected or any(re.fullmatch(r'(sh|sz|bj)\.\d{6}', code) is None
                           for code in expected):
        raise ValueError('Historical constituents contain no codes or invalid codes.')
    paths = sorted((folder / 'daily').glob('*.csv'))
    actual = {path.stem for path in paths}
    if actual != expected:
        raise ValueError(f'Daily cache set differs from constituents: '
                         f'missing={sorted(expected - actual)}, extra={sorted(actual - expected)}')
    records, observed = [], {}
    for path in paths:
        before = path.stat()
        dates = set()
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'date', 'code'}.issubset(reader.fieldnames or []):
                raise ValueError(f'{path}: missing date/code columns.')
            for row in reader:
                if row['code'] != path.stem:
                    raise ValueError(f'{path}: unexpected code {row["code"]!r}.')
                day = date.fromisoformat(row['date'])
                if day in dates:
                    raise ValueError(f'{path}: duplicate date {day}.')
                dates.add(day)
        if not dates:
            raise ValueError(f'{path}: empty daily cache.')
        observed[path] = (before.st_size, before.st_mtime_ns)
        records.append({
            'code': path.stem,
            'start_date': min(dates).isoformat(),
            'end_date': max(dates).isoformat(),
            'row_count': len(dates),
            'download_time': datetime.fromtimestamp(before.st_mtime, LOCAL_TIME).isoformat(timespec='microseconds'),
            'download_time_source': 'daily_file_mtime',
        })
    for path, signature in observed.items():
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != signature:
            raise ValueError(f'Cache changed during manifest generation: {path}')
    target = folder / 'manifest.csv'
    descriptor, name = tempfile.mkstemp(prefix='.manifest-', suffix='.tmp', dir=folder)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    target, records = build_manifest(args.root)
    print(f'MANIFEST: {target}; stocks={len(records)}; rows={sum(row["row_count"] for row in records)}')


if __name__ == '__main__':
    main()
