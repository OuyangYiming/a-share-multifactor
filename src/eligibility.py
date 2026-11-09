"""Point-in-time eligibility rules; unavailable historical facts stay unknown.

This module only classifies supplied records. It does not fetch historical
statuses, reconstruct a security universe, or certify the calendar's source.
"""

from numbers import Real

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    'date', 'code', 'listing_date', 'is_st', 'is_suspended',
    'st_asof', 'suspension_asof',
)
STATUSES = ('excluded_st', 'excluded_suspended', 'excluded_new_stock',
            'unknown', 'eligible')
SUMMARY_COLUMNS = ('year', 'input_rows', 'input_stocks', *STATUSES, 'eligible_stocks')
MIN_LISTING_SESSIONS = 60


def _daily_dates(values: pd.Series, name: str, allow_missing: bool) -> pd.Series:
    """Parse daily dates without accepting timestamps, time zones or numeric dates."""
    if any(isinstance(value, (Real, np.bool_)) and not pd.isna(value)
           for value in values):
        raise ValueError(f'{name} must contain dates, not numeric values.')
    parsed = pd.to_datetime(values, errors='raise')
    if parsed.dt.tz is not None or not parsed.dropna().eq(parsed.dropna().dt.normalize()).all():
        raise ValueError(f'{name} must contain timezone-naive dates at midnight.')
    if not allow_missing and parsed.isna().any():
        raise ValueError(f'{name} cannot contain missing dates.')
    return parsed


def _nullable_flags(values: pd.Series, name: str) -> pd.Series:
    """Accept bool or numeric 0/1 and missing values; never coerce arbitrary text."""
    normalized = []
    for value in values:
        if value is None or value is pd.NA or value is pd.NaT:
            normalized.append(pd.NA)
        elif isinstance(value, (bool, np.bool_)):
            normalized.append(bool(value))
        elif isinstance(value, Real) and pd.isna(value):
            normalized.append(pd.NA)
        elif isinstance(value, Real) and value in (0, 1):
            normalized.append(bool(value))
        else:
            raise ValueError(f'{name} accepts only bool, numeric 0/1, or missing values.')
    return pd.Series(pd.array(normalized, dtype='boolean'), index=values.index, name=name)


def validate_calendar(trading_calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Check calendar structure; the caller must verify complete exchange coverage.

    Every date between the first and last session is covered by the caller's
    source, including closed days. Missing sessions cannot be inferred here.
    """
    if not isinstance(trading_calendar, pd.DatetimeIndex) or trading_calendar.empty:
        raise ValueError('Supply a nonempty validated DatetimeIndex of trading sessions.')
    if (trading_calendar.tz is not None or trading_calendar.hasnans or
            not trading_calendar.is_unique or not trading_calendar.is_monotonic_increasing or
            not trading_calendar.equals(trading_calendar.normalize())):
        raise ValueError('Trading sessions must be unique, increasing, timezone-naive daily dates.')
    return trading_calendar.copy()


def classify_eligibility(records: pd.DataFrame, trading_calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Classify supplied stock-days using only status information known that day.

    Required columns are date, code, listing_date, is_st, is_suspended, st_asof
    and suspension_asof. The as-of dates certify availability of historical
    status applicable to the row date; a current status is not a substitute.
    Missing status or its missing as-of date makes that status unknown. Any
    supplied as-of date after the observation date is an error, even when the
    corresponding flag is missing.

    Listing age is the number of exchange sessions in [listing_date, date],
    inclusive. Exact age is unknown when listing_date predates the calendar.
    A stock passes the age rule on its 60th exchange session. The observation
    window's row count is never used as listing age.

    Mutually exclusive priority: known ST, known suspension, known age < 60,
    remaining unknowns, otherwise eligible. Thus a known exclusion can decide
    a record even when another fact is unavailable; unknown flags are retained.
    The input index, order and other columns are preserved without mutation.
    """
    calendar = validate_calendar(trading_calendar)
    if not isinstance(records, pd.DataFrame) or not set(REQUIRED_COLUMNS).issubset(records.columns):
        raise ValueError('Eligibility input is missing required columns, including historical as-of dates.')
    if not records.columns.is_unique:
        raise ValueError('Input column names must be unique.')
    frame = records.copy(deep=True)
    for column in ('date', 'listing_date', 'st_asof', 'suspension_asof'):
        frame[column] = _daily_dates(frame[column], column, allow_missing=column != 'date')
    if not frame['code'].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError('Every record requires a nonempty string stock code.')
    if frame.duplicated(['date', 'code']).any():
        raise ValueError('Duplicate (date, code) eligibility records.')
    if not frame['date'].isin(calendar).all():
        raise ValueError('Observation date is absent from the supplied trading calendar.')
    if (frame['listing_date'] > frame['date']).any():
        raise ValueError('Listing date is after the observation date.')
    for column in ('st_asof', 'suspension_asof'):
        if (frame[column] > frame['date']).any():
            raise ValueError(f'{column} is after the observation date (future information).')
    for column in ('is_st', 'is_suspended'):
        frame[column] = _nullable_flags(frame[column], column)

    frame['st_unknown'] = frame['is_st'].isna() | frame['st_asof'].isna()
    frame['suspension_unknown'] = frame['is_suspended'].isna() | frame['suspension_asof'].isna()
    exact_age = frame['listing_date'].notna() & frame['listing_date'].ge(calendar[0])
    ages = pd.array(calendar.searchsorted(frame['date'], side='right') -
                    calendar.searchsorted(frame['listing_date'], side='left'), dtype='Int64')
    ages[~exact_age.to_numpy()] = pd.NA
    frame['listing_trading_days'] = ages
    frame['listing_age_unknown'] = ~exact_age

    excluded_st = frame['is_st'] & ~frame['st_unknown']
    excluded_suspended = frame['is_suspended'] & ~frame['suspension_unknown']
    excluded_new = frame['listing_trading_days'].lt(MIN_LISTING_SESSIONS) & exact_age
    unknown = frame[['st_unknown', 'suspension_unknown', 'listing_age_unknown']].any(axis=1)
    frame['eligibility_status'] = np.select(
        [excluded_st.to_numpy(dtype=bool), excluded_suspended.to_numpy(dtype=bool),
         excluded_new.to_numpy(dtype=bool), unknown.to_numpy(dtype=bool)],
        list(STATUSES[:-1]), default='eligible')
    frame['eligible'] = pd.array(frame['eligibility_status'].eq('eligible'), dtype='boolean')
    frame.loc[frame['eligibility_status'].eq('unknown'), 'eligible'] = pd.NA
    frame['unknown_fields'] = [
        ','.join(name for name, missing in zip(('st', 'suspension', 'listing_age'), flags) if missing)
        for flags in frame[['st_unknown', 'suspension_unknown', 'listing_age_unknown']].itertuples(index=False, name=None)
    ]
    return frame


def summarize_eligibility(classified: pd.DataFrame) -> pd.DataFrame:
    """Count mutually exclusive outcomes by year; counts are stock-days, not stocks.

    input_stocks and eligible_stocks count unique codes separately. Only years
    with supplied records are emitted. Missing market records are not inferred.
    """
    required = {'date', 'code', 'eligibility_status'}
    if not isinstance(classified, pd.DataFrame) or not required.issubset(classified.columns):
        raise ValueError('Expected the output of classify_eligibility.')
    dates = _daily_dates(classified['date'], 'date', allow_missing=False)
    if classified.duplicated(['date', 'code']).any():
        raise ValueError('Duplicate (date, code) cannot be counted.')
    if not classified['eligibility_status'].isin(STATUSES).all():
        raise ValueError('Unrecognized or missing eligibility status.')
    rows = []
    for year, group in classified.assign(_year=dates.dt.year).groupby('_year', sort=True):
        counts = group['eligibility_status'].value_counts()
        row = {'year': int(year), 'input_rows': len(group), 'input_stocks': group['code'].nunique()}
        row.update({status: int(counts.get(status, 0)) for status in STATUSES})
        row['eligible_stocks'] = group.loc[group['eligibility_status'].eq('eligible'), 'code'].nunique()
        if sum(row[status] for status in STATUSES) != row['input_rows']:
            raise ValueError('Eligibility accounting does not conserve the input records.')
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).astype('int64')
