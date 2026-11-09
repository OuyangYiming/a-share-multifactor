"""Offline tests only: all fixture values are constructed, not market data.

Test tables are confined to pytest temporary directories. They must never be
used as research observations or written into the project's results/ folder.
"""

from hashlib import sha256
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src import example


@pytest.fixture
def raw_daily():
    """Return two deliberately invented rows in reverse date order."""
    return pd.DataFrame([
        ['2015-01-06', 'sh.600000', '12.5', '1'],
        ['2015-01-05', 'sh.600000', '10', '1'],
    ], columns=example.FIELDS)


def test_validation_sorts_converts_and_preserves_input(raw_daily):
    """Dates become the index; numeric columns convert without changing input."""
    before = raw_daily.copy(deep=True)
    actual = example.validate_daily(raw_daily)
    assert isinstance(actual.index, pd.DatetimeIndex)
    assert actual.index.name == 'date'
    assert list(actual.index) == [pd.Timestamp('2015-01-05'), pd.Timestamp('2015-01-06')]
    assert actual['close'].tolist() == [10.0, 12.5]
    pd.testing.assert_frame_equal(raw_daily, before)


@pytest.mark.parametrize('case', ['empty', 'missing_field', 'extra_field', 'reordered_fields'])
def test_validation_rejects_wrong_table_shape(raw_daily, case):
    """Refuse empty input or a schema different from the requested API fields."""
    if case == 'empty':
        raw_daily = raw_daily.iloc[:0]
    elif case == 'missing_field':
        raw_daily = raw_daily.drop(columns='close')
    elif case == 'extra_field':
        raw_daily['unexpected'] = 'value'
    else:
        raw_daily = raw_daily[list(reversed(example.FIELDS))]
    with pytest.raises(ValueError, match='requested fields'):
        example.validate_daily(raw_daily)


def test_validation_rejects_duplicate_stock_date(raw_daily):
    """A duplicate observation must fail rather than be silently discarded."""
    duplicate = pd.concat([raw_daily, raw_daily.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match='Duplicate'):
        example.validate_daily(duplicate)


@pytest.mark.parametrize(('column', 'value'), [
    ('code', 'sz.000001'), ('code', ''),
    ('date', '2014-12-31'), ('date', '2026-01-01'),
    ('date', 'not-a-date'), ('date', ''),
    ('adjustflag', '2'), ('adjustflag', '3'),
    ('close', 'nan'), ('close', 'inf'), ('close', ''), ('close', 'not-a-number'),
    ('close', '0'), ('close', '-1'),
])
def test_validation_rejects_invalid_values(raw_daily, column, value):
    """Reject wrong identity/period/adjustment and invalid financial quantities."""
    raw_daily.loc[0, column] = value
    with pytest.raises(ValueError):
        example.validate_daily(raw_daily)


@pytest.fixture
def fake_api(monkeypatch, raw_daily):
    """Inject a fake SDK module so tests cannot contact the real data service."""
    source = raw_daily[['date', 'close']].copy()
    source['date'] = pd.to_datetime(source['date']).dt.date
    source['close'] = pd.to_numeric(source['close'])
    source['open'] = 1.0  # Extra vendor columns are not included in the snapshot.
    api = SimpleNamespace(stock_zh_a_hist_tx=Mock(return_value=source))
    monkeypatch.setitem(sys.modules, 'akshare', api)
    return api


def test_fetch_uses_fixed_request_and_normalizes_schema(fake_api, raw_daily):
    """Request explicit backward adjustment and normalize dates and identifiers."""
    before = fake_api.stock_zh_a_hist_tx.return_value.copy(deep=True)
    result = example.fetch_daily()
    expected = raw_daily.copy()
    expected['close'] = pd.to_numeric(expected['close'])
    pd.testing.assert_frame_equal(result, expected)
    fake_api.stock_zh_a_hist_tx.assert_called_once_with(
        symbol='sh600000', start_date='20150101', end_date='20251231', adjust='hfq')
    pd.testing.assert_frame_equal(fake_api.stock_zh_a_hist_tx.return_value, before)


@pytest.mark.parametrize('response', [None, [], {'date': [], 'close': []},
                                    pd.DataFrame({'date': ['2015-01-05']}),
                                    pd.DataFrame({'close': [10.0]})])
def test_unexpected_api_schema_writes_nothing(fake_api, tmp_path, response):
    """Do not accept a changed vendor response type or a missing required field."""
    fake_api.stock_zh_a_hist_tx.return_value = response
    with pytest.raises(ValueError, match='Unexpected AKShare response schema'):
        example.load_example(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_empty_api_response_writes_nothing(fake_api, tmp_path):
    """An empty but correctly shaped response cannot become a valid snapshot."""
    fake_api.stock_zh_a_hist_tx.return_value = pd.DataFrame(columns=['date', 'close'])
    with pytest.raises(ValueError, match='nonempty daily data'):
        example.load_example(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_invalid_api_date_writes_nothing(fake_api, tmp_path):
    """A malformed vendor date must not be normalized into a cached record."""
    fake_api.stock_zh_a_hist_tx.return_value = pd.DataFrame(
        {'date': ['not-a-date'], 'close': [10.0]})
    with pytest.raises(ValueError):
        example.load_example(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_api_network_exception_writes_nothing(fake_api, tmp_path):
    """A failed request propagates to the caller without writing a partial cache."""
    fake_api.stock_zh_a_hist_tx.side_effect = TimeoutError('constructed timeout')
    with pytest.raises(TimeoutError, match='constructed timeout'):
        example.load_example(tmp_path)
    assert list(tmp_path.iterdir()) == []


def install_snapshot(root, raw_daily):
    """Write a constructed cache/reference pair solely under a pytest temp dir."""
    cache = root / 'data/raw/sh600000_20150101_20251231_hfq.csv'
    reference = root / 'config/example_snapshot.json'
    payload = raw_daily.to_csv(index=False, lineterminator='\n').encode('utf-8')
    metadata = {
        'request': example.REQUEST.copy(), 'fields': example.FIELDS.copy(),
        'sha256': sha256(payload).hexdigest(), 'source': 'unit-test-fixture-only',
    }
    cache.parent.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    cache.write_bytes(payload)
    reference.write_text(json.dumps(metadata), encoding='utf-8')
    return cache, reference, metadata


@pytest.mark.parametrize('offline', [False, True])
def test_cache_hit_never_fetches(tmp_path, monkeypatch, raw_daily, offline):
    """Both default and offline runs reuse a verified local snapshot."""
    _, _, metadata = install_snapshot(tmp_path, raw_daily)
    fetch = Mock(side_effect=AssertionError('Network access forbidden in this test'))
    monkeypatch.setattr(example, 'fetch_daily', fetch)
    frame, actual_metadata = example.load_example(tmp_path, offline=offline)
    pd.testing.assert_frame_equal(frame, example.validate_daily(raw_daily))
    assert actual_metadata == metadata
    fetch.assert_not_called()


def test_offline_cache_miss_never_fetches(tmp_path, monkeypatch):
    """Missing offline data cause a clear error without a network fallback."""
    fetch = Mock(side_effect=AssertionError('Network access forbidden in this test'))
    monkeypatch.setattr(example, 'fetch_daily', fetch)
    with pytest.raises(FileNotFoundError, match='No cache'):
        example.load_example(tmp_path, offline=True)
    fetch.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_cache_tampering_is_detected_without_overwrite(tmp_path, monkeypatch, raw_daily):
    """A changed value cannot silently replace a committed reference checksum."""
    cache, reference, _ = install_snapshot(tmp_path, raw_daily)
    original_reference = reference.read_bytes()
    tampered = cache.read_bytes().replace(b'12.5', b'99.5')
    cache.write_bytes(tampered)
    fetch = Mock(side_effect=AssertionError('Network access forbidden in this test'))
    monkeypatch.setattr(example, 'fetch_daily', fetch)
    with pytest.raises(ValueError, match='checksum mismatch'):
        example.load_example(tmp_path)
    assert cache.read_bytes() == tampered
    assert reference.read_bytes() == original_reference
    fetch.assert_not_called()


def test_provider_revision_does_not_replace_reference(tmp_path, monkeypatch, raw_daily):
    """Refetched history must match the reference before any raw file is saved."""
    cache, reference, _ = install_snapshot(tmp_path, raw_daily)
    original_reference = reference.read_bytes()
    cache.unlink()
    changed = raw_daily.copy()
    changed.loc[0, 'close'] = '99.5'
    monkeypatch.setattr(example, 'fetch_daily', Mock(return_value=changed))
    with pytest.raises(ValueError, match='checksum mismatch'):
        example.load_example(tmp_path)
    assert not cache.exists()
    assert reference.read_bytes() == original_reference


def test_reference_request_mismatch_fails_before_download(tmp_path, monkeypatch, raw_daily):
    """A snapshot for another adjustment regime cannot be accepted."""
    _, reference, metadata = install_snapshot(tmp_path, raw_daily)
    metadata['request']['adjust'] = 'qfq'
    reference.write_text(json.dumps(metadata), encoding='utf-8')
    fetch = Mock(side_effect=AssertionError('Network access forbidden in this test'))
    monkeypatch.setattr(example, 'fetch_daily', fetch)
    with pytest.raises(ValueError, match='parameters do not match'):
        example.load_example(tmp_path)
    fetch.assert_not_called()


def test_first_fetch_creates_reusable_snapshot(tmp_path, monkeypatch, raw_daily):
    """A first download records exact bytes; a second offline run is identical."""
    fetch = Mock(return_value=raw_daily)
    monkeypatch.setattr(example, 'fetch_daily', fetch)
    monkeypatch.setattr(example, 'version', Mock(return_value='test-version'))
    first, metadata = example.load_example(tmp_path)
    cache = tmp_path / 'data/raw/sh600000_20150101_20251231_hfq.csv'
    assert sha256(cache.read_bytes()).hexdigest() == metadata['sha256']
    assert metadata['request']['adjust'] == 'hfq'
    assert metadata['akshare_version'] == 'test-version'
    assert metadata['endpoint'] == 'stock_zh_a_hist_tx'
    assert not cache.with_suffix('.tmp').exists()
    second, second_metadata = example.load_example(tmp_path, offline=True)
    pd.testing.assert_frame_equal(first, second)
    assert metadata == second_metadata
    fetch.assert_called_once_with()


def test_invalid_first_download_writes_nothing(tmp_path, monkeypatch, raw_daily):
    """Invalid observations never become a cached or referenced snapshot."""
    raw_daily.loc[0, 'close'] = ''
    monkeypatch.setattr(example, 'fetch_daily', Mock(return_value=raw_daily))
    with pytest.raises(ValueError):
        example.load_example(tmp_path)
    assert list(tmp_path.iterdir()) == []
