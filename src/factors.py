"""MVP: four fixed monthly factors and an equal-weight percentile-rank score."""

import numpy as np
import pandas as pd

FACTOR_COLUMNS = ['momentum', 'reversal', 'low_volatility', 'turnover']
SIGNAL_COLUMNS = ['date', 'code'] + FACTOR_COLUMNS + ['score']


def _keys(table: pd.DataFrame, date_column: str, name: str) -> pd.DataFrame:
    """Copy a table with unique daily-date/string-code keys, refusing ambiguity."""
    if not isinstance(table, pd.DataFrame) or not table.columns.is_unique or not {date_column, 'code'}.issubset(table):
        raise ValueError(f'{name} requires unique columns including {date_column} and code.')
    data = table.copy()
    if pd.api.types.is_numeric_dtype(data[date_column].dtype) and data[date_column].notna().any():
        raise ValueError(f'{name} dates must not be numeric timestamps.')
    dates = pd.to_datetime(data[date_column], errors='raise')
    if dates.dt.tz is not None or dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError(f'{name} dates must be nonmissing, timezone-naive calendar dates.')
    data[date_column] = dates
    if not data['code'].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError(f'{name} needs nonempty string codes.')
    data['code'] = data['code'].str.strip()
    if data.duplicated([date_column, 'code']).any():
        raise ValueError(f'{name} has duplicate date/code keys.')
    return data


def build_signals(prices: pd.DataFrame, constituents: pd.DataFrame) -> pd.DataFrame:
    """Build four causal factors at the explicitly supplied constituent month ends.

    The sorted union of all supplied price dates is the common session grid.
    Its market-calendar completeness is the caller's responsibility. Suspended
    observations are retained; missing rows are reindexed as NA, never filled.
    Membership applies only at the signal date, not throughout the lookback.

    All price factors use adj_close. A stock is scored only if it is a member,
    isST=0, tradestatus=1, and all four factors are finite on the signal date.
    Higher values are favorable for every factor. Average-tie percentile ranks
    are computed on this eligible cross-section and averaged with equal weights.
    Signals use the close of t; this function does not authorize trading at t.
    """
    daily = _keys(prices, 'date', 'prices')
    members = _keys(constituents, 'signal_date', 'constituents')
    required = ['adj_close', 'turn', 'isST', 'tradestatus']
    if not set(required).issubset(daily):
        raise ValueError('Prices require adj_close, turn, isST and tradestatus.')
    for field in required:
        values = pd.to_numeric(daily[field].replace('', np.nan), errors='raise').astype(float)
        if not np.isfinite(values.dropna()).all():
            raise ValueError(f'Nonfinite supplied {field}.')
        if field == 'adj_close' and values.dropna().le(0).any():
            raise ValueError('Observed adj_close must be positive.')
        if field == 'turn' and values.dropna().lt(0).any():
            raise ValueError('Observed turn must be nonnegative.')
        if field in ('isST', 'tradestatus') and not values.dropna().isin([0, 1]).all():
            raise ValueError(f'{field} must be 0, 1 or missing.')
        daily[field] = values
    if daily.empty or members.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    calendar = pd.DatetimeIndex(daily['date'].unique()).sort_values()
    # Future constituent dates are not evaluated against a truncated price table.
    members = members.loc[members['signal_date'].between(calendar.min(), calendar.max())]
    month_ends = pd.Series(calendar).groupby(calendar.to_period('M')).max()
    if not members['signal_date'].isin(month_ends).all():
        raise ValueError('Constituent signal dates must be month-end sessions in the common grid.')
    rows = []
    requested_codes = set(members['code'])
    for code, stock in daily.loc[daily['code'].isin(requested_codes)].groupby('code', sort=True):
        frame = stock.set_index('date').reindex(calendar)
        close = frame['adj_close']
        factors = pd.DataFrame(index=calendar)
        factors['momentum'] = close.shift(21) / close.shift(252) - 1
        factors['reversal'] = -(close / close.shift(21) - 1)
        signal_dates = pd.DatetimeIndex(members.loc[members['code'].eq(code), 'signal_date'])
        selected = factors.reindex(signal_dates).copy()
        positions = calendar.get_indexer(signal_dates)
        # Direct signal windows keep older rolling-roundoff from splitting true ties.
        for factor, values, window in (
            ('low_volatility', close.pct_change(fill_method=None).to_numpy(), 60),
            ('turnover', frame['turn'].to_numpy(), 20),
        ):
            complete = positions >= window - 1
            samples = values[positions[complete, None] - np.arange(window - 1, -1, -1)]
            selected[factor] = np.nan
            selected.loc[complete, factor] = -(
                samples.std(axis=1, ddof=1) if factor == 'low_volatility'
                else samples.mean(axis=1)
            )
        active = frame.reindex(signal_dates)
        eligible = (active['isST'].eq(0) & active['tradestatus'].eq(1) &
                    np.isfinite(selected[FACTOR_COLUMNS]).all(axis=1))
        selected = selected.loc[eligible]
        selected.index.name = 'date'
        selected = selected.reset_index()
        selected['code'] = code
        rows.append(selected)
    if not rows:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    signals = pd.concat(rows, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    if signals.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    ranks = signals.groupby('date')[FACTOR_COLUMNS].rank(method='average', pct=True)
    signals['score'] = ranks.mean(axis=1)
    return signals[SIGNAL_COLUMNS]
