"""Constructed examples only; no test value is a real market observation."""

import numpy as np
import pandas as pd
import pytest

from src.eligibility import classify_eligibility, summarize_eligibility, STATUSES


@pytest.fixture
def calendar():
    """Return an invented session calendar with weekends and two extra closures."""
    return pd.bdate_range('2020-01-02', periods=100).difference(
        pd.DatetimeIndex(['2020-01-06', '2020-01-07']))


def make_records(calendar, positions=(58, 59), code='test.stock'):
    """Build hypothetical records whose historical states were available that day."""
    dates = calendar[list(positions)]
    return pd.DataFrame({
        'date': dates, 'code': code, 'listing_date': calendar[0],
        'is_st': False, 'is_suspended': False,
        'st_asof': dates, 'suspension_asof': dates,
    })


def test_59th_and_60th_sessions_have_different_outcomes(calendar):
    """The 59th session is excluded; the 60th satisfies the listing-age rule."""
    records = make_records(calendar)
    before = records.copy(deep=True)
    actual = classify_eligibility(records, calendar)
    assert actual['listing_trading_days'].tolist() == [59, 60]
    assert actual['eligibility_status'].tolist() == ['excluded_new_stock', 'eligible']
    assert actual['eligible'].tolist() == [False, True]
    assert actual['unknown_fields'].tolist() == ['', '']
    assert str(actual['listing_trading_days'].dtype) == 'Int64'
    pd.testing.assert_frame_equal(records, before)


def test_age_uses_exchange_sessions_not_calendar_days_or_observation_rows(calendar):
    """Weekends and calendar closures do not count; omitted stock rows still age."""
    records = make_records(calendar, positions=(2, 59))
    actual = classify_eligibility(records, calendar)
    assert calendar[2] == pd.Timestamp('2020-01-08')
    assert actual['listing_trading_days'].tolist() == [3, 60]
    assert actual.loc[1, 'eligible']


def test_listing_on_closed_day_counts_from_next_session(calendar):
    """A listing date on a non-session does not add a nonexistent trading day."""
    records = make_records(calendar, positions=(2,))
    records['listing_date'] = pd.Timestamp('2020-01-04')
    result = classify_eligibility(records, calendar)
    assert result['listing_trading_days'].item() == 1


def test_listing_before_calendar_keeps_exact_age_unknown(calendar):
    """A truncated calendar is not a replacement for the listing's full history."""
    records = make_records(calendar, positions=(59,))
    records['listing_date'] = pd.Timestamp('2019-01-01')
    result = classify_eligibility(records, calendar)
    assert pd.isna(result['listing_trading_days'].item())
    assert result['listing_age_unknown'].item()
    assert result['eligibility_status'].item() == 'unknown'
    assert pd.isna(result['eligible'].item())


@pytest.mark.parametrize('column', ['is_st', 'is_suspended', 'listing_date',
                                    'st_asof', 'suspension_asof'])
def test_missing_required_fact_never_passes_as_normal(calendar, column):
    """Missing state, availability date or listing date prevents eligibility."""
    records = make_records(calendar, positions=(59,))
    records[column] = pd.NA
    result = classify_eligibility(records, calendar)
    assert result['eligibility_status'].item() == 'unknown'
    assert pd.isna(result['eligible'].item())
    assert result['unknown_fields'].item()


def test_missing_statuses_remain_missing_even_when_age_excludes(calendar):
    """Known young age excludes a row without turning unavailable states False."""
    records = make_records(calendar)
    for column in ['is_st', 'is_suspended', 'st_asof', 'suspension_asof']:
        records[column] = pd.NA
    result = classify_eligibility(records, calendar)
    assert result['eligibility_status'].tolist() == ['excluded_new_stock', 'unknown']
    assert result[['is_st', 'is_suspended']].isna().all().all()
    assert result[['st_unknown', 'suspension_unknown']].all().all()
    assert result['unknown_fields'].tolist() == ['st,suspension', 'st,suspension']
    assert result['eligible'].iloc[0] == False
    assert pd.isna(result['eligible'].iloc[1])


def test_mutually_exclusive_priority_and_unknown_details(calendar):
    """Known ST precedes suspension and new-stock rules; unknowns remain visible."""
    records = pd.concat([make_records(calendar, positions=(0,), code=f'test.{i}')
                         for i in range(5)], ignore_index=True)
    records['is_st'] = pd.Series([True, False, pd.NA, pd.NA, False], dtype='boolean')
    records['is_suspended'] = pd.Series([True, True, True, pd.NA, False], dtype='boolean')
    records.loc[3:, 'listing_date'] = pd.NaT
    result = classify_eligibility(records, calendar)
    assert result['eligibility_status'].tolist() == [
        'excluded_st', 'excluded_suspended', 'excluded_suspended', 'unknown', 'unknown']
    assert result.loc[2, 'st_unknown']
    assert result.loc[2, 'unknown_fields'] == 'st'
    assert result.loc[3, 'unknown_fields'] == 'st,suspension,listing_age'


def test_true_state_without_asof_is_not_a_verified_exclusion(calendar):
    """A true flag of unknown timing is not accepted as historical evidence."""
    records = make_records(calendar, positions=(59,))
    records['is_st'] = True
    records['st_asof'] = pd.NaT
    result = classify_eligibility(records, calendar)
    assert result['eligibility_status'].item() == 'unknown'
    assert result['is_st'].item() == True
    assert result['st_unknown'].item()


@pytest.mark.parametrize('column', ['st_asof', 'suspension_asof'])
@pytest.mark.parametrize('flag_missing', [False, True])
def test_future_asof_rejected_even_for_unknown_flag(calendar, column, flag_missing):
    """Any supplied future availability date is rejected before classification."""
    records = make_records(calendar, positions=(59,))
    records[column] = records['date'] + pd.Timedelta(days=1)
    if flag_missing:
        records['is_st' if column == 'st_asof' else 'is_suspended'] = pd.NA
    with pytest.raises(ValueError, match='future information'):
        classify_eligibility(records, calendar)


def test_past_asof_allowed_for_status_applicable_on_observation_date(calendar):
    """Historical effective states may be known before the observation date."""
    records = make_records(calendar, positions=(59,))
    records[['st_asof', 'suspension_asof']] = calendar[0]
    assert classify_eligibility(records, calendar)['eligible'].item()


def test_future_listing_date_rejected(calendar):
    """A price observation before its claimed listing date is an invalid record."""
    records = make_records(calendar, positions=(59,))
    records['listing_date'] = calendar[60]
    with pytest.raises(ValueError, match='Listing date is after'):
        classify_eligibility(records, calendar)


@pytest.mark.parametrize('value', ['False', 'True', '0', '1', 2, -1, np.inf])
def test_invalid_flag_values_rejected(calendar, value):
    """Arbitrary truthy strings and non-binary quantities cannot become states."""
    records = make_records(calendar, positions=(59,))
    records['is_st'] = value
    with pytest.raises(ValueError, match='only bool'):
        classify_eligibility(records, calendar)


@pytest.mark.parametrize('value', [0, 0.0, np.int64(0), False, np.bool_(False)])
def test_numeric_binary_and_boolean_flags_are_normalized(calendar, value):
    """Explicit numeric zero and Boolean False are valid known normal states."""
    records = make_records(calendar, positions=(59,))
    records['is_st'] = value
    result = classify_eligibility(records, calendar)
    assert str(result['is_st'].dtype) == 'boolean'
    assert result['eligible'].item()


@pytest.mark.parametrize('case', ['empty', 'unsorted', 'duplicate', 'missing', 'timezone', 'time_component'])
def test_bad_calendar_structure_rejected(calendar, case):
    """An invalid session index cannot define listing age."""
    records = make_records(calendar, positions=(59,))
    if case == 'empty':
        calendar = pd.DatetimeIndex([])
    elif case == 'unsorted':
        calendar = calendar[::-1]
    elif case == 'duplicate':
        calendar = calendar.append(calendar[-1:])
    elif case == 'missing':
        calendar = calendar.append(pd.DatetimeIndex([pd.NaT]))
    elif case == 'timezone':
        calendar = calendar.tz_localize('Asia/Shanghai')
    else:
        calendar = calendar + pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        classify_eligibility(records, calendar)


def test_off_calendar_observation_rejected(calendar):
    """A row on a declared closed day cannot be classified as a trading record."""
    records = make_records(calendar, positions=(59,))
    records['date'] = pd.Timestamp('2020-01-04')
    with pytest.raises(ValueError, match='absent from'):
        classify_eligibility(records, calendar)


def test_duplicate_stock_date_rejected(calendar):
    """Duplicate rows must not inflate exclusion counts."""
    records = make_records(calendar, positions=(59,))
    records = pd.concat([records, records], ignore_index=True)
    with pytest.raises(ValueError, match='Duplicate'):
        classify_eligibility(records, calendar)


def test_asof_columns_are_mandatory(calendar):
    """Omitting the provenance contract does not imply contemporaneous states."""
    records = make_records(calendar).drop(columns='st_asof')
    with pytest.raises(ValueError, match='required columns'):
        classify_eligibility(records, calendar)


def test_annual_summary_conserves_records_and_counts_stocks_separately(calendar):
    """Every input stock-day has exactly one outcome, including unknowns."""
    records = pd.concat([make_records(calendar, positions=(0, 59, 60), code='test.a'),
                         make_records(calendar, positions=(59, 60), code='test.b')], ignore_index=True)
    records['is_st'] = records['is_st'].astype('boolean')
    records.loc[1, 'is_st'] = True
    records.loc[2, 'is_suspended'] = True
    records.loc[3, 'is_st'] = pd.NA
    classified = classify_eligibility(records, calendar)
    summary = summarize_eligibility(classified)
    assert summary['year'].tolist() == [2020]
    row = summary.iloc[0]
    assert row['input_rows'] == 5
    assert row['input_stocks'] == 2
    assert row['eligible_stocks'] == 1
    assert row[list(STATUSES)].tolist() == [1, 1, 1, 1, 1]
    assert row[list(STATUSES)].sum() == row['input_rows']


def test_appended_or_changed_future_records_do_not_change_past_results(calendar):
    """The filter is row-local and never reads future rows to infer past facts."""
    records = make_records(calendar, positions=(58, 59, 60))
    past = classify_eligibility(records.iloc[:2], calendar)
    combined = classify_eligibility(records, calendar)
    pd.testing.assert_frame_equal(past, combined.iloc[:2])
    changed = records.copy()
    changed.loc[2, 'is_st'] = True
    changed.loc[2, 'listing_date'] = calendar[20]
    pd.testing.assert_frame_equal(past, classify_eligibility(changed, calendar).iloc[:2])


def test_input_order_and_nonunique_dataframe_index_are_preserved(calendar):
    """Only stock-date keys must be unique; arbitrary caller row labels survive."""
    records = make_records(calendar, positions=(59, 58))
    records.index = ['same', 'same']
    records['extra'] = [4, 5]
    actual = classify_eligibility(records, calendar)
    assert actual.index.tolist() == ['same', 'same']
    assert actual['date'].tolist() == records['date'].tolist()
    assert actual['extra'].tolist() == [4, 5]
    assert actual['listing_trading_days'].tolist() == [60, 59]


def test_empty_input_has_valid_empty_outputs(calendar):
    """An empty supplied sample is distinct from a failed data download."""
    records = make_records(calendar).iloc[:0]
    result = classify_eligibility(records, calendar)
    assert result.empty
    assert str(result['eligible'].dtype) == 'boolean'
    assert summarize_eligibility(result).empty


def test_summary_rejects_invalid_status(calendar):
    """Malformed outputs cannot silently disappear from the annual totals."""
    result = classify_eligibility(make_records(calendar), calendar)
    result.loc[0, 'eligibility_status'] = 'not_a_category'
    with pytest.raises(ValueError, match='Unrecognized'):
        summarize_eligibility(result)
