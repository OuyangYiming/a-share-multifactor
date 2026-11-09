"""Offline tests with invented reference tables; no fixture is market evidence."""

from hashlib import sha256
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src import reference_data as ref


@pytest.fixture
def calendar_raw():
    """Supply two deliberately constructed dates in reverse order."""
    return pd.DataFrame({'trade_date': ['2015-01-06', '2015-01-05']})


@pytest.fixture
def benchmark_raw():
    """Supply two invented OHLC rows unrelated to real index levels."""
    return pd.DataFrame({'date': ['2015-01-06', '2015-01-05'], 'open': [10, 11],
                         'close': [11, 10], 'high': [12, 12], 'low': [9, 9]})


@pytest.fixture
def exchange_masters():
    """Create fictitious current masters with extra columns that must be ignored."""
    sh = pd.DataFrame({'证券代码': ['600000', '600519', '600001'],
                       '证券简称': ['test a', 'test b', 'unrelated'],
                       '上市日期': ['1999-01-04', '2001-01-04', '2000-01-04'],
                       '当前行业': ['invented'] * 3})
    sz = pd.DataFrame({'A股代码': ['000001', '000002', '300750'],
                       'A股简称': ['test c', 'test d', 'test e'],
                       'A股上市日期': ['1991-01-04', '1991-01-07', '2018-01-04'],
                       '所属行业': ['invented'] * 3})
    return sh, sz


def test_calendar_sorts_without_changing_input(calendar_raw):
    """Keep every supplied date exactly once and return Timestamp dates."""
    before = calendar_raw.copy()
    actual = ref.validate_calendar(calendar_raw)
    assert actual['date'].tolist() == [pd.Timestamp('2015-01-05'), pd.Timestamp('2015-01-06')]
    pd.testing.assert_frame_equal(calendar_raw, before)


@pytest.mark.parametrize('values', [[], [''], ['bad-date'], [None],
                                  ['2015-01-05', '2015-01-05'], ['2015-01-05 01:00:00']])
def test_calendar_rejects_invalid_observations(values):
    """Missing, duplicate and non-date values cannot silently become sessions."""
    with pytest.raises(ValueError):
        ref.validate_calendar(pd.DataFrame({'trade_date': values}))


def test_benchmark_conversion_and_identity(benchmark_raw):
    """Normalize price fields without treating an index code as a stock code."""
    actual = ref.validate_benchmark(benchmark_raw, 'sh000300')
    assert actual['code'].tolist() == ['sh.000300', 'sh.000300']
    assert actual['close'].tolist() == [10, 11]
    assert actual.columns.tolist() == ['date', 'code', 'open', 'close', 'high', 'low']


@pytest.mark.parametrize(('field', 'value'), [
    ('close', 0), ('open', -1), ('high', 10), ('low', 11), ('close', np.nan),
    ('close', np.inf), ('date', '2014-12-31'), ('date', '2026-01-01'), ('date', ''),
])
def test_benchmark_rejects_invalid_prices_or_dates(benchmark_raw, field, value):
    """Reject implausible levels and out-of-window dates before caching."""
    bad = benchmark_raw.astype(object)
    bad.loc[0, field] = value
    with pytest.raises(ValueError):
        ref.validate_benchmark(bad, 'sh000300')


def test_benchmark_rejects_duplicate_keys_and_wrong_symbol(benchmark_raw):
    """Refuse duplicate observations and unsupported index identity."""
    with pytest.raises(ValueError):
        ref.validate_benchmark(pd.concat([benchmark_raw, benchmark_raw.iloc[[0]]]), 'sh000300')
    with pytest.raises(ValueError):
        ref.validate_benchmark(benchmark_raw, 'sh600000')


def test_extracts_only_required_listing_facts(exchange_masters):
    """Retain five target dates, excluding current industry and unrelated securities."""
    actual = ref.extract_listings(*exchange_masters)
    assert len(actual) == 5
    assert set(actual['code']) == {'sh.600000', 'sh.600519', 'sz.000001', 'sz.000002', 'sz.300750'}
    assert set(actual.columns) == {'code', 'name', 'listing_date', 'source_endpoint'}
    assert str(actual['listing_date'].dtype).startswith('datetime64')


@pytest.mark.parametrize('case', ['missing_target', 'duplicate_target', 'missing_date', 'invalid_date', 'future_date'])
def test_listing_targets_fail_closed(exchange_masters, case):
    """Missing dates or ambiguous current records cannot become historical IPO facts."""
    sh, sz = [x.copy() for x in exchange_masters]
    if case == 'missing_target':
        sz = sz.iloc[:-1]
    elif case == 'duplicate_target':
        sz = pd.concat([sz, sz.iloc[[0]]], ignore_index=True)
    else:
        sz.loc[0, 'A股上市日期'] = {'missing_date': '', 'invalid_date': 'invalid', 'future_date': '2026-01-01'}[case]
    with pytest.raises(ValueError):
        ref.extract_listings(sh, sz)


def test_reference_cache_roundtrip_and_public_pin(tmp_path, monkeypatch, calendar_raw):
    """Use cached bytes offline and reject a jointly changed cache and sidecar."""
    fetch = Mock(return_value=calendar_raw)
    monkeypatch.setattr(ref, 'fetch_reference', fetch)
    _, metadata = ref.load_reference_snapshot(tmp_path, 'calendar')
    results = tmp_path / 'results'
    results.mkdir()
    (results / 'reference_provenance.json').write_text(json.dumps({'snapshots': [metadata]}), encoding='utf-8')
    ref.load_reference_snapshot(tmp_path, 'calendar', offline=True)
    assert fetch.call_count == 1
    cache = tmp_path / 'data/reference/raw/calendar.csv'
    payload = cache.read_bytes().replace(b'2015-01-06', b'2015-01-07')
    cache.write_bytes(payload)
    metadata['sha256'] = sha256(payload).hexdigest()
    cache.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
    with pytest.raises(ValueError, match='Public reference'):
        ref.load_reference_snapshot(tmp_path, 'calendar', offline=True)


def test_missing_or_partial_cache_never_calls_source(tmp_path, monkeypatch):
    """Offline and incomplete snapshots are errors, not permission to fetch."""
    fetch = Mock(side_effect=AssertionError('must not connect'))
    monkeypatch.setattr(ref, 'fetch_reference', fetch)
    with pytest.raises(FileNotFoundError):
        ref.load_reference_snapshot(tmp_path, 'calendar', offline=True)
    path = tmp_path / 'data/reference/raw/calendar.csv'
    path.parent.mkdir(parents=True)
    path.write_text('trade_date\n2015-01-05\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Incomplete'):
        ref.load_reference_snapshot(tmp_path, 'calendar')
    fetch.assert_not_called()


def test_bad_download_is_not_cached(tmp_path, monkeypatch):
    """Validation happens before any invalid response can become a snapshot."""
    monkeypatch.setattr(ref, 'fetch_reference', Mock(return_value=pd.DataFrame({'wrong': [1]})))
    with pytest.raises(ValueError):
        ref.load_reference_snapshot(tmp_path, 'calendar')
    assert not (tmp_path / 'data/reference/raw/calendar.csv').exists()


@pytest.mark.parametrize('fails', [False, True])
def test_timeout_is_restored_after_fetch(monkeypatch, calendar_raw, fails):
    """Successful and failed SDK calls both restore the previous local timeout."""
    state = {'timeout': 7}

    def set_timeout(value):
        """Track the fake SDK's process-local setting."""
        state['timeout'] = value

    def fetch_calendar():
        """Verify the configured bound without making any network request."""
        assert state['timeout'] == 15
        if fails:
            raise TimeoutError('constructed network failure')
        return calendar_raw

    fake = SimpleNamespace(get_timeout=lambda: state['timeout'], set_timeout=set_timeout,
                           tool_trade_date_hist_sina=fetch_calendar)
    monkeypatch.setitem(sys.modules, 'akshare', fake)
    if fails:
        with pytest.raises(TimeoutError):
            ref.fetch_reference('calendar')
    else:
        pd.testing.assert_frame_equal(ref.fetch_reference('calendar'), calendar_raw)
    assert state['timeout'] == 7


@pytest.fixture
def aligned_reference():
    """Use a wholly invented calendar with a session on every calendar day."""
    calendar = pd.DataFrame({'date': pd.date_range('2015-01-01', '2025-12-31')})
    benchmarks = pd.concat([calendar.assign(code=code) for code in ('sh.000300', 'sh.000905')], ignore_index=True)
    listings = pd.DataFrame({'code': ['sh.600000'], 'listing_date': [pd.Timestamp('2015-01-01')]})
    return calendar, benchmarks, listings


def test_sixtieth_session_uses_explicit_inclusive_convention(aligned_reference):
    """With daily synthetic sessions, January 1 counted as 1 makes March 1 day 60."""
    actual = ref.check_reference_alignment(*aligned_reference)
    assert actual.loc[0, 'session_60_date'] == pd.Timestamp('2015-03-01')


def test_calendar_index_gaps_are_not_filled(aligned_reference):
    """Fail when one index lacks a calendar session instead of inventing a close."""
    calendar, benchmarks, listings = aligned_reference
    with pytest.raises(ValueError, match='differ'):
        ref.check_reference_alignment(calendar, benchmarks.iloc[1:], listings)


def test_alignment_requires_both_indices(aligned_reference):
    """A single available index must not count as a complete two-index audit."""
    calendar, benchmarks, listings = aligned_reference
    with pytest.raises(ValueError, match='Both unique'):
        ref.check_reference_alignment(calendar, benchmarks[benchmarks['code'].eq('sh.000300')], listings)


def test_insufficient_pre_ipo_calendar_fails(aligned_reference):
    """A calendar that begins after the known listing date cannot measure age."""
    calendar, benchmarks, listings = aligned_reference
    listings.loc[0, 'listing_date'] = pd.Timestamp('2014-12-31')
    with pytest.raises(ValueError, match='cover'):
        ref.check_reference_alignment(calendar, benchmarks, listings)
