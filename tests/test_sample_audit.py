"""Constructed fixtures only; integration checks never download market data."""

import pandas as pd
import pytest

from src.sample_audit import audit_sample


def test_unknown_status_never_becomes_eligible(tmp_path):
    """Known listing age excludes day 59; unknown historical flags block day 60."""
    sessions = pd.bdate_range('2020-01-01', periods=61)
    daily = pd.DataFrame({'date': sessions[58:60], 'code': 'test-stock',
                          'low': 9., 'high': 11., 'volume': 100., 'amount': 1000.})
    listings = pd.DataFrame({'code': ['test-stock'], 'listing_date': [sessions[0]]})
    result = audit_sample(tmp_path, daily, sessions, listings)
    assert result['known_new_stock_exclusions'] == 1
    assert result['unknown_rows'] == 1
    assert result['eligible_rows'] == 0
    assert result['vwap_outside_range'] == 0
    summary = pd.read_csv(tmp_path / 'results/tables/pilot_sample_summary.csv')
    assert summary.loc[0, 'input_rows'] == summary.loc[0, ['excluded_new_stock', 'unknown']].sum()
    assert not (tmp_path / 'results/tables/sample_summary.csv').exists()


def test_missing_listing_stays_unknown_and_vwap_is_only_diagnostic(tmp_path):
    """No listing data cannot certify age; a unit mismatch is flagged, not deleted."""
    sessions = pd.bdate_range('2020-01-01', periods=2)
    daily = pd.DataFrame({'date': sessions, 'code': 'test-stock',
                          'low': 9., 'high': 11., 'volume': 100., 'amount': [100000., 1000.]})
    listings = pd.DataFrame({'code': [], 'listing_date': pd.to_datetime([])})
    result = audit_sample(tmp_path, daily, sessions, listings)
    assert result['input_rows'] == 2
    assert result['unknown_rows'] == 2
    assert result['vwap_outside_range'] == 1


def test_duplicate_listing_metadata_rejected(tmp_path):
    """A metadata join must not multiply stock-day observations."""
    listings = pd.DataFrame({'code': ['test-stock'] * 2, 'listing_date': ['2020-01-01'] * 2})
    with pytest.raises(ValueError, match='Duplicate listing'):
        audit_sample(tmp_path, pd.DataFrame(), pd.bdate_range('2020-01-01', periods=2), listings)


def test_supplied_status_cannot_be_silently_overwritten(tmp_path):
    """The price-only pilot must not erase historical flags supplied by a caller."""
    listings = pd.DataFrame({'code': ['test-stock'], 'listing_date': ['2020-01-01']})
    with pytest.raises(ValueError, match='price-only input'):
        audit_sample(tmp_path, pd.DataFrame({'is_st': [True]}),
                     pd.bdate_range('2020-01-01', periods=2), listings)
