"""Offline tests for the five-stock pilot, using constructed values only.

Fixtures are not market observations. Files written by these tests stay under
pytest's temporary directory and must never be used as research results.
"""

from hashlib import sha256
import json
import sys
from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest

from src import data


@pytest.fixture
def vendor_frame():
    """Return two invented OHLCV rows in reverse chronological order."""
    return pd.DataFrame([
        ['2015-01-06', '12', '13', '14', '11', '200', '0.02', '2600'],
        ['2015-01-05', '10', '11', '12', '9', '100', '0.01', '1100'],
    ], columns=['date', 'open', 'close', 'high', 'low',
                'volume', 'turnover', 'amount'])


@pytest.fixture
def adjusted_pair(vendor_frame):
    """Build matching unadjusted and backward-adjusted synthetic tables."""
    adjusted = vendor_frame.copy()
    for column in ['open', 'close', 'high', 'low']:
        adjusted[column] = pd.to_numeric(adjusted[column]) * 2
    return (data.validate_vendor_frame(vendor_frame, 'sh600000', ''),
            data.validate_vendor_frame(adjusted, 'sh600000', 'hfq'))


def test_validation_normalizes_without_mutating(vendor_frame):
    """Sort dates and convert quantities while retaining the source frame."""
    before = vendor_frame.copy(deep=True)
    actual = data.validate_vendor_frame(vendor_frame, 'sh600000', '')
    assert actual['date'].tolist() == [pd.Timestamp('2015-01-05'),
                                       pd.Timestamp('2015-01-06')]
    assert actual['code'].tolist() == ['sh.600000', 'sh.600000']
    assert actual['adjustment'].tolist() == ['none', 'none']
    assert actual['close'].tolist() == [11, 13]
    assert actual['volume'].tolist() == [100, 200]
    assert actual['turnover'].tolist() == [0.01, 0.02]
    assert actual['amount'].tolist() == [1100, 2600]
    pd.testing.assert_frame_equal(vendor_frame, before)


@pytest.mark.parametrize('case', ['empty', 'missing_open', 'missing_volume'])
def test_validation_rejects_missing_contract(vendor_frame, case):
    """An empty response or absent required field cannot form a valid table."""
    if case == 'empty':
        vendor_frame = vendor_frame.iloc[:0]
    else:
        vendor_frame = vendor_frame.drop(columns=case.removeprefix('missing_'))
    with pytest.raises(ValueError):
        data.validate_vendor_frame(vendor_frame, 'sh600000', '')


def test_validation_rejects_duplicate_dates(vendor_frame):
    """Do not silently deduplicate repeated stock-date observations."""
    repeated = pd.concat([vendor_frame, vendor_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        data.validate_vendor_frame(repeated, 'sh600000', '')


@pytest.mark.parametrize('value', ['2014-12-31', '2026-01-01', '', 'invalid-date'])
def test_validation_rejects_invalid_dates(vendor_frame, value):
    """Reject observations outside the fixed research interval and bad dates."""
    vendor_frame.loc[0, 'date'] = value
    with pytest.raises(ValueError):
        data.validate_vendor_frame(vendor_frame, 'sh600000', '')


@pytest.mark.parametrize(('column', 'value'), [
    ('open', '0'), ('close', '-1'), ('high', np.nan), ('low', 'inf'),
    ('close', ''), ('open', 'not-a-number'),
])
def test_validation_rejects_invalid_prices(vendor_frame, column, value):
    """OHLC values must be finite, positive and present."""
    vendor_frame.loc[0, column] = value
    with pytest.raises(ValueError):
        data.validate_vendor_frame(vendor_frame, 'sh600000', '')


@pytest.mark.parametrize(('column', 'value'), [
    ('open', '15'), ('close', '10'), ('high', '10'), ('low', '15'),
])
def test_validation_rejects_impossible_ohlc(vendor_frame, column, value):
    """Opening and closing prices must lie inside the reported daily range."""
    vendor_frame.loc[0, column] = value
    with pytest.raises(ValueError):
        data.validate_vendor_frame(vendor_frame, 'sh600000', '')


@pytest.mark.parametrize(('column', 'value'), [
    ('volume', '-1'), ('amount', '-1'), ('turnover', '-0.01'),
    ('volume', 'inf'), ('amount', '-inf'), ('turnover', 'inf'),
])
def test_validation_rejects_invalid_activity(vendor_frame, column, value):
    """Available trading quantities cannot be negative or infinite."""
    vendor_frame.loc[0, column] = value
    with pytest.raises(ValueError):
        data.validate_vendor_frame(vendor_frame, 'sh600000', '')


def test_validation_preserves_activity_missingness(vendor_frame):
    """Missing volume, turnover and amount remain missing instead of zero."""
    columns = ['volume', 'turnover', 'amount']
    vendor_frame.loc[0, columns] = np.nan
    actual = data.validate_vendor_frame(vendor_frame, 'sh600000', '')
    affected = actual.loc[actual['date'].eq(pd.Timestamp('2015-01-06')), columns]
    assert affected.isna().all().all()
    assert actual.loc[actual['date'].eq(pd.Timestamp('2015-01-05')), 'volume'].item() == 100


def test_validation_accepts_zero_trading_quantities(vendor_frame):
    """Zero activity is retained and is not automatically labelled suspension."""
    vendor_frame.loc[0, ['volume', 'turnover', 'amount']] = '0'
    actual = data.validate_vendor_frame(vendor_frame, 'sh600000', '')
    affected = actual.loc[actual['date'].eq(pd.Timestamp('2015-01-06'))]
    assert affected['volume'].item() == 0
    assert not {'is_st', 'isST', 'tradestatus', 'is_suspended'}.intersection(actual.columns)


def test_alignment_joins_by_keys_and_keeps_unadjusted_activity(adjusted_pair):
    """Align by date/code, never row position or adjusted trading amounts."""
    unadjusted, backward = adjusted_pair
    unadjusted_before = unadjusted.copy(deep=True)
    backward = backward.iloc[::-1].copy()
    backward['amount'] = 9999
    actual = data.align_adjustments(unadjusted, backward).sort_values('date')
    assert 'adjustment' not in actual.columns
    assert actual['close'].tolist() == [11, 13]
    assert actual['hfq_close'].tolist() == [22, 26]
    assert actual['hfq_open'].tolist() == [20, 24]
    assert actual['hfq_high'].tolist() == [24, 28]
    assert actual['hfq_low'].tolist() == [18, 22]
    assert actual['amount'].tolist() == [1100, 2600]
    pd.testing.assert_frame_equal(unadjusted, unadjusted_before)


@pytest.mark.parametrize('case', [
    'missing_date', 'different_date', 'different_code',
    'mixed_codes', 'wrong_unadjusted_flag', 'wrong_backward_flag',
])
def test_alignment_rejects_incompatible_tables(adjusted_pair, case):
    """Reject mismatched histories rather than drop observations in an inner join."""
    unadjusted, backward = adjusted_pair
    if case == 'missing_date':
        backward = backward.iloc[:1]
    elif case == 'different_date':
        backward.loc[backward.index[0], 'date'] = pd.Timestamp('2015-01-07')
    elif case == 'different_code':
        backward['code'] = 'sz.000001'
    elif case == 'mixed_codes':
        unadjusted.loc[unadjusted.index[0], 'code'] = 'sz.000001'
        backward.loc[backward.index[0], 'code'] = 'sz.000001'
    elif case == 'wrong_unadjusted_flag':
        unadjusted['adjustment'] = 'hfq'
    else:
        backward['adjustment'] = 'none'
    with pytest.raises(ValueError):
        data.align_adjustments(unadjusted, backward)


def snapshot_paths(root, adjust=''):
    """Return the documented test-only cache and sidecar paths."""
    label = adjust or 'none'
    cache = root / f'data/stage1/raw/sh600000_20150101_20251231_{label}.csv'
    return cache, cache.with_suffix('.json')


def test_first_fetch_records_reusable_snapshot(tmp_path, monkeypatch, vendor_frame):
    """Persist exact bytes and metadata, then reuse them without another fetch."""
    fetch = Mock(return_value=vendor_frame)
    monkeypatch.setattr(data, 'fetch_vendor_frame', fetch)
    first, metadata = data.load_snapshot(tmp_path, 'sh600000', '')
    cache, sidecar = snapshot_paths(tmp_path)
    assert cache.exists() and sidecar.exists()
    assert sha256(cache.read_bytes()).hexdigest() == metadata['sha256']
    assert metadata['request']['symbol'] == 'sh600000'
    assert metadata['request']['adjust'] == ''
    assert metadata['source']
    pd.testing.assert_frame_equal(first, data.validate_vendor_frame(vendor_frame, 'sh600000', ''))
    second, stored = data.load_snapshot(tmp_path, 'sh600000', '', offline=True)
    pd.testing.assert_frame_equal(first, second)
    assert metadata == stored
    fetch.assert_called_once_with('sh600000', '')


@pytest.mark.parametrize('offline', [False, True])
def test_cache_hit_does_not_connect(tmp_path, monkeypatch, vendor_frame, offline):
    """Verified cache reuse never invokes the data service in either mode."""
    monkeypatch.setattr(data, 'fetch_vendor_frame', Mock(return_value=vendor_frame))
    expected, metadata = data.load_snapshot(tmp_path, 'sh600000', 'hfq')
    fetch = Mock(side_effect=AssertionError('Network forbidden in unit tests'))
    monkeypatch.setattr(data, 'fetch_vendor_frame', fetch)
    actual, actual_metadata = data.load_snapshot(tmp_path, 'sh600000', 'hfq', offline=offline)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual_metadata == metadata
    fetch.assert_not_called()


def test_missing_offline_cache_never_fetches(tmp_path, monkeypatch):
    """Offline mode fails explicitly when no local input exists."""
    fetch = Mock(side_effect=AssertionError('Network forbidden in unit tests'))
    monkeypatch.setattr(data, 'fetch_vendor_frame', fetch)
    with pytest.raises(FileNotFoundError):
        data.load_snapshot(tmp_path, 'sh600000', '', offline=True)
    fetch.assert_not_called()
    assert not list(tmp_path.rglob('*.csv'))


@pytest.mark.parametrize('missing', ['csv', 'json'])
def test_incomplete_cache_pair_is_not_repaired_silently(tmp_path, monkeypatch, vendor_frame, missing):
    """A missing cache or sidecar stops processing instead of rebasing history."""
    monkeypatch.setattr(data, 'fetch_vendor_frame', Mock(return_value=vendor_frame))
    data.load_snapshot(tmp_path, 'sh600000', '')
    cache, sidecar = snapshot_paths(tmp_path)
    removed, retained = (cache, sidecar) if missing == 'csv' else (sidecar, cache)
    retained_bytes = retained.read_bytes()
    removed.unlink()
    fetch = Mock(side_effect=AssertionError('Network forbidden in unit tests'))
    monkeypatch.setattr(data, 'fetch_vendor_frame', fetch)
    with pytest.raises(ValueError):
        data.load_snapshot(tmp_path, 'sh600000', '')
    assert not removed.exists()
    assert retained.read_bytes() == retained_bytes
    fetch.assert_not_called()


def test_tampered_csv_is_preserved_and_rejected(tmp_path, monkeypatch, vendor_frame):
    """Checksum validation detects changed input before regenerating any output."""
    monkeypatch.setattr(data, 'fetch_vendor_frame', Mock(return_value=vendor_frame))
    data.load_snapshot(tmp_path, 'sh600000', '')
    cache, sidecar = snapshot_paths(tmp_path)
    original_sidecar = sidecar.read_bytes()
    tampered = cache.read_bytes() + b'\n'
    cache.write_bytes(tampered)
    fetch = Mock(side_effect=AssertionError('Network forbidden in unit tests'))
    monkeypatch.setattr(data, 'fetch_vendor_frame', fetch)
    with pytest.raises(ValueError):
        data.load_snapshot(tmp_path, 'sh600000', '')
    assert cache.read_bytes() == tampered
    assert sidecar.read_bytes() == original_sidecar
    fetch.assert_not_called()


def test_wrong_request_metadata_is_rejected(tmp_path, monkeypatch, vendor_frame):
    """A cache labelled with a different adjustment cannot satisfy this request."""
    monkeypatch.setattr(data, 'fetch_vendor_frame', Mock(return_value=vendor_frame))
    data.load_snapshot(tmp_path, 'sh600000', '')
    _, sidecar = snapshot_paths(tmp_path)
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    metadata['request']['adjust'] = 'qfq'
    sidecar.write_text(json.dumps(metadata), encoding='utf-8')
    fetch = Mock(side_effect=AssertionError('Network forbidden in unit tests'))
    monkeypatch.setattr(data, 'fetch_vendor_frame', fetch)
    with pytest.raises(ValueError):
        data.load_snapshot(tmp_path, 'sh600000', '')
    fetch.assert_not_called()


def test_invalid_download_does_not_create_snapshot(tmp_path, monkeypatch, vendor_frame):
    """Bad source prices are not saved as a successful data acquisition."""
    vendor_frame.loc[0, 'close'] = '1000'
    monkeypatch.setattr(data, 'fetch_vendor_frame', Mock(return_value=vendor_frame))
    with pytest.raises(ValueError):
        data.load_snapshot(tmp_path, 'sh600000', '')
    assert not list(tmp_path.rglob('*.csv'))
    assert not list(tmp_path.rglob('*.json'))


def test_network_failure_does_not_create_snapshot(tmp_path, monkeypatch):
    """A timeout propagates without creating empty or partially valid data."""
    fetch = Mock(side_effect=TimeoutError('constructed timeout'))
    monkeypatch.setattr(data, 'fetch_vendor_frame', fetch)
    with pytest.raises(TimeoutError, match='constructed timeout'):
        data.load_snapshot(tmp_path, 'sh600000', '')
    assert not list(tmp_path.rglob('*.csv'))
    assert not list(tmp_path.rglob('*.json'))


def test_coverage_counts_observed_stocks_and_retains_zero_years():
    """Distinguish the fixed pilot size from stocks with actual rows each year."""
    daily = pd.DataFrame({
        'date': pd.to_datetime(['2015-01-05', '2015-01-06', '2020-01-02']),
        'code': ['sh.600000', 'sh.600000', 'sz.300750'],
    })
    detail, annual = data.build_coverage(daily)
    assert len(detail) == 55
    assert not detail.duplicated(['year', 'code']).any()
    assert set(detail['year']) == set(range(2015, 2026))
    assert set(detail['code']) == {
        'sh.600000', 'sz.000001', 'sh.600519', 'sz.000002', 'sz.300750'}
    absent = detail.loc[(detail['year'] == 2015) & (detail['code'] == 'sz.300750')]
    assert absent['rows'].item() == 0
    assert absent[['first_date', 'last_date']].isna().all().all()
    observed = detail.loc[(detail['year'] == 2015) & (detail['code'] == 'sh.600000')]
    assert observed['rows'].item() == 2
    assert observed['first_date'].item() == pd.Timestamp('2015-01-05')
    assert observed['last_date'].item() == pd.Timestamp('2015-01-06')
    annual = annual.set_index('year')
    assert annual.loc[2015, 'pilot_stocks_with_observations'] == 1
    assert annual.loc[2015, 'observed_rows'] == 2
    assert annual.loc[2020, 'pilot_stocks_with_observations'] == 1
    assert annual.loc[2020, 'observed_rows'] == 1
    assert annual.loc[2016, 'pilot_stocks_with_observations'] == 0
    assert annual.loc[2016, 'observed_rows'] == 0
    assert annual['predefined_pilot_stocks'].eq(5).all()
    assert annual['observed_rows'].sum() == len(daily)


def write_public_pin(root, snapshots):
    """Write an independent checksum reference inside the pytest temp directory."""
    reference = root / 'results/stage1_provenance.json'
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(json.dumps({'snapshots': snapshots}), encoding='utf-8')
    return reference


def test_public_pin_rejects_coordinated_cache_and_sidecar_changes(tmp_path, monkeypatch, vendor_frame):
    """Updating a local sidecar cannot authorize changing pinned market input."""
    monkeypatch.setattr(data, 'fetch_vendor_frame', Mock(return_value=vendor_frame))
    _, metadata = data.load_snapshot(tmp_path, 'sh600000', '')
    reference = write_public_pin(tmp_path, [metadata])
    reference_bytes = reference.read_bytes()
    cache, sidecar = snapshot_paths(tmp_path)
    changed = pd.read_csv(cache, dtype=str)
    changed.loc[0, 'close'] = '13.5'  # Still valid OHLC: the public hash is the guard.
    changed_payload = changed.to_csv(index=False, lineterminator='\n').encode('utf-8')
    cache.write_bytes(changed_payload)
    metadata['sha256'] = sha256(changed_payload).hexdigest()
    sidecar.write_text(json.dumps(metadata), encoding='utf-8')
    sidecar_bytes = sidecar.read_bytes()
    fetch = Mock(side_effect=AssertionError('Network forbidden in unit tests'))
    monkeypatch.setattr(data, 'fetch_vendor_frame', fetch)
    with pytest.raises(ValueError):
        data.load_snapshot(tmp_path, 'sh600000', '', offline=True)
    assert cache.read_bytes() == changed_payload
    assert sidecar.read_bytes() == sidecar_bytes
    assert reference.read_bytes() == reference_bytes
    fetch.assert_not_called()


def test_public_pin_requires_matching_request(tmp_path, monkeypatch, vendor_frame):
    """A valid local pair is insufficient when the public record omits its request."""
    monkeypatch.setattr(data, 'fetch_vendor_frame', Mock(return_value=vendor_frame))
    _, metadata = data.load_snapshot(tmp_path, 'sh600000', '')
    metadata['request']['adjust'] = 'hfq'
    write_public_pin(tmp_path, [metadata])
    fetch = Mock(side_effect=AssertionError('Network forbidden in unit tests'))
    monkeypatch.setattr(data, 'fetch_vendor_frame', fetch)
    with pytest.raises(ValueError):
        data.load_snapshot(tmp_path, 'sh600000', '', offline=True)
    fetch.assert_not_called()


def test_matching_public_pin_allows_offline_reuse(tmp_path, monkeypatch, vendor_frame):
    """An unchanged snapshot matching its public pin remains usable offline."""
    monkeypatch.setattr(data, 'fetch_vendor_frame', Mock(return_value=vendor_frame))
    expected, metadata = data.load_snapshot(tmp_path, 'sh600000', '')
    write_public_pin(tmp_path, [metadata])
    fetch = Mock(side_effect=AssertionError('Network forbidden in unit tests'))
    monkeypatch.setattr(data, 'fetch_vendor_frame', fetch)
    actual, actual_metadata = data.load_snapshot(tmp_path, 'sh600000', '', offline=True)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual_metadata == metadata
    fetch.assert_not_called()


@pytest.mark.parametrize('fails', [False, True])
def test_fetch_bounds_http_and_restores_timeout(monkeypatch, vendor_frame, fails):
    """Use a bounded SDK timeout and restore caller settings on success or error."""
    api = Mock()
    api.get_timeout.return_value = 7
    api.stock_zh_a_hist_tx.return_value = vendor_frame
    if fails:
        api.stock_zh_a_hist_tx.side_effect = TimeoutError('constructed SDK timeout')
    monkeypatch.setitem(sys.modules, 'akshare', api)
    if fails:
        with pytest.raises(TimeoutError, match='constructed SDK timeout'):
            data.fetch_vendor_frame('sh600000', 'hfq')
    else:
        actual = data.fetch_vendor_frame('sh600000', 'hfq')
        pd.testing.assert_frame_equal(actual, vendor_frame)
    assert api.method_calls == [
        call.get_timeout(),
        call.set_timeout(15),
        call.stock_zh_a_hist_tx(symbol='sh600000', start_date='20150101',
                               end_date='20251231', adjust='hfq'),
        call.set_timeout(7),
    ]
