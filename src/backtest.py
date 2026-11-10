"""Monthly portfolios with next-session opening execution."""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import numpy as np
import pandas as pd


PORTFOLIOS = ('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'benchmark')
TRADE_COLUMNS = ('date', 'portfolio', 'code', 'side', 'notional', 'cost', 'reason', 'pre_trade_nav')
FORWARD_COLUMNS = ('date', 'code', 'forward_return')
REBALANCE_COLUMNS = ('date', 'portfolio', 'pre_trade_nav', 'buy_notional',
                     'sell_notional', 'cost', 'turnover', 'one_way_turnover')
STALE_COLUMNS = ('portfolio', 'code', 'stale_days')
_EPS = 1e-12


def transaction_cost(notional: float, cost_bps: float) -> float:
    """Charge a nonnegative one-way notional at the supplied basis-point rate."""
    notional, cost_bps = float(notional), float(cost_bps)
    if not np.isfinite([notional, cost_bps]).all() or min(notional, cost_bps) < 0:
        raise ValueError('Notional and cost_bps must be finite and nonnegative.')
    return notional * cost_bps / 10000.0


def _flag(value):
    """Parse known binary daily status; missing or ambiguous values stay unknown."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        return {'0': False, '1': True}.get(value.strip())
    if value in (0, 1):
        return bool(value)
    return None


def _positive(value) -> bool:
    try:
        return bool(np.isfinite(float(value)) and float(value) > 0)
    except (TypeError, ValueError):
        return False


def _trade_block_reason(row, side: str, date) -> str:
    if side not in ('buy', 'sell'):
        raise ValueError('side must be buy or sell.')
    if row is None:
        return 'missing_price_row'
    status = _flag(row.get('tradestatus'))
    if status is not True:
        return 'suspended' if status is False else 'missing_tradestatus'
    if not _positive(row.get('open')):
        return 'missing_open'
    if not _positive(row.get('preclose')):
        return 'missing_preclose'
    is_st = _flag(row.get('isST'))
    if is_st is None:
        return 'missing_ST_status'
    code = str(row.get('code', '')).lower().replace('.', '')
    digits = code[-6:]
    if len(digits) != 6 or not digits.isdigit():
        return 'invalid_code'
    growth_board = digits.startswith('30')
    twenty_percent = digits.startswith(('688', '689')) or (
        growth_board and pd.Timestamp(date) >= pd.Timestamp('2020-08-24'))
    rate = Decimal('0.20') if twenty_percent else Decimal('0.05' if is_st else '0.10')
    try:
        previous = Decimal(str(row['preclose']))
        opening = Decimal(str(row['open'])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        upper = (previous * (1 + rate)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        lower = (previous * (1 - rate)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return 'invalid_raw_price'
    if side == 'buy' and opening >= upper:
        return 'limit_up'
    if side == 'sell' and opening <= lower:
        return 'limit_down'
    return ''


def can_trade(row, side: str, date) -> bool:
    """Gate an opening order using raw prices and the historical board/ST rule.

    Prices at or beyond the rounded upper limit block buys; prices at or below
    the lower limit block sells. Unknown status blocks both directions. Daily
    total volume is not available at the open and is not an execution gate.
    """
    return not _trade_block_reason(row, side, date)


def _dates(values, label: str) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors='raise'))
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError(f'{label} requires nonmissing, timezone-naive daily dates.')
    return dates


def _next_session(calendar: pd.DatetimeIndex, date):
    position = calendar.searchsorted(pd.Timestamp(date), side='right')
    return calendar[position] if position < len(calendar) else None


def _next_month_open(calendar: pd.DatetimeIndex, execution_date):
    month_start = pd.Timestamp(execution_date).to_period('M').to_timestamp() + pd.offsets.MonthBegin(1)
    position = calendar.searchsorted(month_start)
    if position >= len(calendar) or calendar[position].to_period('M') != month_start.to_period('M'):
        return None
    return calendar[position]


def _record(trades, date, portfolio, code, side, notional=0.0, cost=0.0,
            reason='rebalance', *, pre_trade_nav):
    trades.append((date, portfolio, code, side, float(notional), float(cost),
                   reason, float(pre_trade_nav)))


def _record_rebalance(records, trades, first_trade, date, portfolio, equity):
    if records is None:
        return
    orders = trades[first_trade:]
    bought = sum(order[4] for order in orders if order[3] == 'buy')
    sold = sum(order[4] for order in orders if order[3] == 'sell')
    cost = sum(order[5] for order in orders)
    turnover = (bought + sold) / equity
    records.append((date, portfolio, float(equity), bought, sold, cost,
                    turnover, turnover / 2.0))


def _rebalance(positions, cash, target_indices, opening_marks, rows, codes,
               date, portfolio, cost_bps, trades, rebalance_records=None):
    """Keep blocked holdings, sell excess first, then scale all affordable buys."""
    equity = cash + float(positions @ opening_marks)
    if not _positive(equity):
        raise ArithmeticError('Pre-trade NAV must be positive and finite.')
    first_trade = len(trades)
    desired = np.zeros(len(codes), dtype=float)
    if len(target_indices):
        desired[target_indices] = equity / len(target_indices)
    for index in np.flatnonzero(positions > _EPS):
        current_value = positions[index] * opening_marks[index]
        excess = current_value - desired[index]
        if excess <= _EPS:
            continue
        code = codes[index]
        row = rows.get(code)
        reason = _trade_block_reason(row, 'sell', date)
        if not reason and not _positive(row.get('_adj_open')):
            reason = 'missing_adjusted_open'
        if reason:
            _record(trades, date, portfolio, code, 'sell', reason=reason, pre_trade_nav=equity)
            continue
        price = float(row['_adj_open'])
        quantity = min(positions[index], excess / price)
        notional = quantity * price
        cost = transaction_cost(notional, cost_bps)
        positions[index] -= quantity
        if abs(positions[index]) < _EPS:
            positions[index] = 0.0
        cash += notional - cost
        _record(trades, date, portfolio, code, 'sell', notional, cost, pre_trade_nav=equity)
    pending = []
    for index in target_indices:
        deficit = desired[index] - positions[index] * opening_marks[index]
        if deficit <= _EPS:
            continue
        code = codes[index]
        row = rows.get(code)
        reason = _trade_block_reason(row, 'buy', date)
        if not reason and not _positive(row.get('_adj_open')):
            reason = 'missing_adjusted_open'
        if reason:
            _record(trades, date, portfolio, code, 'buy', reason=reason, pre_trade_nav=equity)
            continue
        pending.append((index, deficit, float(row['_adj_open'])))
    required = sum(notional + transaction_cost(notional, cost_bps) for _, notional, _ in pending)
    scale = min(1.0, max(cash, 0.0) / required) if required > 0 else 0.0
    for index, requested, price in pending:
        notional = requested * scale
        if notional <= _EPS:
            continue
        cost = transaction_cost(notional, cost_bps)
        positions[index] += notional / price
        cash -= notional + cost
        _record(trades, date, portfolio, codes[index], 'buy', notional, cost, pre_trade_nav=equity)
    if cash < -_EPS or (positions < -_EPS).any():
        raise ArithmeticError('Execution would create borrowing or a short position.')
    _record_rebalance(rebalance_records, trades, first_trade, date, portfolio, equity)
    return max(cash, 0.0)


def _rebalance_long_short(positions, cash, long_indices, short_indices, opening_marks,
                          rows, codes, date, portfolio, cost_bps, trades, rebalance_records):
    """Target each leg at 100% of pre-trade equity; keep actual signed holdings.

    This research account assumes unlimited stock borrow, no borrow charges or
    interest, and immediately usable short proceeds. Buy/sell gates apply to the
    actual order direction, including covers and short sales. Costs, blocked
    orders and subsequent price moves can move exposure away from the target.
    """
    equity = cash + float(positions @ opening_marks)
    if not _positive(equity):
        raise ArithmeticError(f'{portfolio} has nonpositive or nonfinite opening equity at {date}.')
    first_trade = len(trades)
    desired = np.zeros(len(codes))
    desired[long_indices] = equity / len(long_indices)
    desired[short_indices] = -equity / len(short_indices)
    for index in np.flatnonzero(positions * opening_marks - desired > _EPS):
        code = codes[index]
        row = rows.get(code)
        reason = _trade_block_reason(row, 'sell', date)
        if not reason and not _positive(row.get('_adj_open')):
            reason = 'missing_adjusted_open'
        if reason:
            _record(trades, date, portfolio, code, 'sell', reason=reason, pre_trade_nav=equity)
            continue
        notional = positions[index] * opening_marks[index] - desired[index]
        cost = transaction_cost(notional, cost_bps)
        positions[index] -= notional / float(row['_adj_open'])
        cash += notional - cost
        _record(trades, date, portfolio, code, 'sell', notional, cost, pre_trade_nav=equity)
    pending = []
    for index in np.flatnonzero(desired - positions * opening_marks > _EPS):
        code = codes[index]
        row = rows.get(code)
        reason = _trade_block_reason(row, 'buy', date)
        if not reason and not _positive(row.get('_adj_open')):
            reason = 'missing_adjusted_open'
        if reason:
            _record(trades, date, portfolio, code, 'buy', reason=reason, pre_trade_nav=equity)
            continue
        pending.append((index, desired[index] - positions[index] * opening_marks[index],
                        float(row['_adj_open'])))
    required = sum(value + transaction_cost(value, cost_bps) for _, value, _ in pending)
    scale = min(1.0, max(cash, 0.0) / required) if required > 0 else 0.0
    for index, requested, price in pending:
        notional = requested * scale
        if notional <= _EPS:
            continue
        cost = transaction_cost(notional, cost_bps)
        positions[index] += notional / price
        cash -= notional + cost
        _record(trades, date, portfolio, codes[index], 'buy', notional, cost, pre_trade_nav=equity)
    if cash < -_EPS:
        raise ArithmeticError('Long-short execution would create negative cash.')
    cash = max(cash, 0.0)
    if not _positive(cash + float(positions @ opening_marks)):
        raise ArithmeticError(f'{portfolio} has nonpositive or nonfinite equity after costs at {date}.')
    _record_rebalance(rebalance_records, trades, first_trade, date, portfolio, equity)
    return cash


def run_backtest(prices: pd.DataFrame, signals: pd.DataFrame, calendar: pd.DatetimeIndex,
                 start='2018-01-01', end='2025-12-31', cost_bps=15.0,
                 include_long_short=False):
    """Return closing NAVs, opening order records, and independent forward labels.

    Each portfolio starts with cash 1. Signals must already describe the eligible
    historical membership at month-end sessions and contain at least five finite
    scores per month. Execution requires the immediately following calendar month.
    Quantile ties are broken by code. The benchmark uses those same members and
    the same execution/cost rules. Blocked attempts have zero notional and cost.
    Future labels use observed adjusted opens only, without transaction gates or
    costs; their date is the signal date. Incomplete labels remain NaN.

    include_long_short adds independent monthly Q5-long/Q1-short accounts:
    long_short pays cost_bps; long_short_gross pays zero costs. Each leg targets
    100% of pre-trade equity, with signed positions held between rebalances.
    rebalance_summary(nav) includes zero-trade months and the initial purchase;
    stale_valuation_summary(nav) counts held stock-days using a prior close.
    """
    transaction_cost(0, cost_bps)
    if cost_bps >= 10000:
        raise ValueError('One-way trading cost must be below 100 percent.')
    portfolios = PORTFOLIOS + (('long_short', 'long_short_gross') if include_long_short else ())
    portfolio_costs = [float(cost_bps)] * len(PORTFOLIOS)
    if include_long_short:
        portfolio_costs += [float(cost_bps), 0.0]
    if not isinstance(calendar, pd.DatetimeIndex):
        raise TypeError('calendar must be a DatetimeIndex.')
    calendar = _dates(calendar, 'calendar')
    if calendar.empty or not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError('calendar must be nonempty, sorted and unique.')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end:
        raise ValueError('start must not follow end.')
    sessions = calendar[(calendar >= start) & (calendar <= end)]
    if sessions.empty:
        raise ValueError('No trading sessions in the requested interval.')
    price_fields = ['date', 'code', 'open', 'close', 'preclose', 'adj_close',
                    'isST', 'tradestatus']
    if not set(price_fields).issubset(prices) or not {'date', 'code', 'score'}.issubset(signals):
        raise ValueError('Prices or signals are missing required columns.')
    if not prices.columns.is_unique or not signals.columns.is_unique:
        raise ValueError('Duplicate input column names.')
    signal = signals[['date', 'code', 'score']].copy()
    signal['date'] = _dates(signal['date'], 'signal dates')
    if signal.empty or signal[['date', 'code']].duplicated().any():
        raise ValueError('Signals must be nonempty with unique date/code keys.')
    for frame, label in [(signal, 'signal'), (prices, 'price')]:
        if frame['code'].isna().any() or frame['code'].astype(str).str.strip().eq('').any():
            raise ValueError(f'Missing {label} security code.')
    signal['code'] = signal['code'].astype(str)
    signal['score'] = pd.to_numeric(signal['score'], errors='raise')
    if not np.isfinite(signal['score'].to_numpy(dtype=float)).all():
        raise ValueError('Signal scores must be finite.')
    signal_dates = pd.DatetimeIndex(signal['date'].unique()).sort_values()
    if signal_dates.to_period('M').duplicated().any():
        raise ValueError('Only one signal date per calendar month is supported.')
    if not signal_dates.isin(calendar).all():
        raise ValueError('Each signal date must be in the supplied trading calendar.')
    month_ends = pd.Series(calendar).groupby(calendar.to_period('M')).max()
    if not signal_dates.isin(month_ends).all():
        raise ValueError('Signal dates must be month-end sessions in the supplied trading calendar.')
    codes = sorted(signal['code'].unique())
    code_index = {code: index for index, code in enumerate(codes)}
    schedule, signal_groups, opening_dates = {}, [], set()
    for signal_date, group in signal.groupby('date', sort=True):
        if len(group) < 5:
            raise ValueError('At least five eligible securities are required per signal month.')
        execution_date = _next_session(calendar, signal_date)
        if execution_date is not None and execution_date.to_period('M') != signal_date.to_period('M') + 1:
            raise ValueError('The trading calendar is missing the month immediately after a signal.')
        exit_date = _next_month_open(calendar, execution_date) if execution_date is not None else None
        if exit_date is not None and exit_date > end:
            exit_date = None
        signal_groups.append((signal_date, group, execution_date, exit_date))
        if execution_date is not None and start <= execution_date <= end:
            if execution_date in schedule:
                raise ValueError('Multiple signals target the same opening session.')
            ordered = group.sort_values(['score', 'code'], kind='stable')['code'].map(code_index).to_numpy()
            schedule[execution_date] = list(np.array_split(ordered, 5)) + [ordered]
            opening_dates.add(execution_date)
            if exit_date is not None:
                opening_dates.add(exit_date)
    table = prices[price_fields].copy()
    table['date'] = _dates(table['date'], 'price dates')
    table['code'] = table['code'].astype(str)
    table = table.loc[table['date'].between(start, end) & table['code'].isin(code_index)].copy()
    if table.duplicated(['date', 'code']).any():
        raise ValueError('Duplicate daily price date/code key.')
    if not table['date'].isin(sessions).all():
        raise ValueError('A price observation is outside the supplied trading sessions.')
    for field in ('open', 'close', 'preclose', 'adj_close'):
        table[field] = pd.to_numeric(table[field], errors='raise')
    raw_close, adjusted_close = table['close'].to_numpy(dtype=float), table['adj_close'].to_numpy(dtype=float)
    adjustment = np.full(len(table), np.nan)
    valid = np.isfinite(raw_close) & (raw_close > 0) & np.isfinite(adjusted_close) & (adjusted_close > 0)
    np.divide(adjusted_close, raw_close, out=adjustment, where=valid)
    adjusted_open = table['open'].to_numpy(dtype=float) * adjustment
    table['_adj_open'] = np.where(np.isfinite(adjusted_open) & (adjusted_open > 0), adjusted_open, np.nan)
    close_grid = np.full((len(sessions), len(codes)), np.nan)
    row_indices = sessions.get_indexer(table['date'])
    column_indices = table['code'].map(code_index).to_numpy(dtype=int)
    close_grid[row_indices, column_indices] = np.where(
        np.isfinite(adjusted_close) & (adjusted_close > 0), adjusted_close, np.nan)
    open_grid = None
    if include_long_short:
        open_grid = np.full_like(close_grid, np.nan)
        open_grid[row_indices, column_indices] = table['_adj_open'].to_numpy(dtype=float)
    openings = {}
    for date, part in table.loc[table['date'].isin(opening_dates)].groupby('date', sort=False):
        records = part.drop(columns='date').set_index('code').to_dict('index')
        for code, record in records.items():
            record['code'] = code
        openings[date] = records
    forward_records = []
    for signal_date, group, execution_date, exit_date in signal_groups:
        entry_rows = openings.get(execution_date, {})
        exit_rows = openings.get(exit_date, {})
        for code in group['code']:
            entry_price = entry_rows.get(code, {}).get('_adj_open', np.nan)
            exit_price = exit_rows.get(code, {}).get('_adj_open', np.nan)
            complete = execution_date is not None and start <= execution_date <= end and exit_date is not None
            future_return = (exit_price / entry_price - 1.0) if complete and _positive(entry_price) and _positive(exit_price) else np.nan
            forward_records.append((signal_date, code, future_return))
    positions = np.zeros((len(portfolios), len(codes)))
    cash = np.ones(len(portfolios))
    min_cash = cash.copy()
    last_marks = np.full(len(codes), np.nan)
    nav_values = np.empty((len(sessions), len(portfolios)))
    stale_counts = np.zeros_like(positions, dtype=int)
    trades, rebalances = [], []
    for day_index, date in enumerate(sessions):
        if include_long_short:
            opening_marks = np.where(np.isfinite(open_grid[day_index]), open_grid[day_index], last_marks)
            opening_marks = np.nan_to_num(opening_marks, nan=0.0)
            opening_equity = cash[len(PORTFOLIOS):] + positions[len(PORTFOLIOS):] @ opening_marks
            if not np.isfinite(opening_equity).all() or (opening_equity <= 0).any():
                raise ArithmeticError(f'Long-short opening equity must be positive and finite at {date}.')
        if date in schedule:
            rows = openings.get(date, {})
            if not include_long_short:
                opening_marks = last_marks.copy()
                for code, row in rows.items():
                    if _positive(row['_adj_open']):
                        opening_marks[code_index[code]] = row['_adj_open']
                opening_marks = np.nan_to_num(opening_marks, nan=0.0)
            for portfolio_index, portfolio in enumerate(PORTFOLIOS):
                cash[portfolio_index] = _rebalance(
                    positions[portfolio_index], cash[portfolio_index], schedule[date][portfolio_index],
                    opening_marks, rows, codes, date, portfolio, cost_bps, trades, rebalances)
            for portfolio_index in range(len(PORTFOLIOS), len(portfolios)):
                cash[portfolio_index] = _rebalance_long_short(
                    positions[portfolio_index], cash[portfolio_index], schedule[date][4], schedule[date][0],
                    opening_marks, rows, codes, date, portfolios[portfolio_index],
                    portfolio_costs[portfolio_index], trades, rebalances)
            min_cash = np.minimum(min_cash, cash)
        current_marks = close_grid[day_index]
        present = np.isfinite(current_marks)
        last_marks[present] = current_marks[present]
        held = np.abs(positions) > _EPS
        if (held & ~np.isfinite(last_marks)).any():
            raise ArithmeticError('A held position has no known valuation price.')
        stale_counts += held & ~present
        nav_values[day_index] = cash + positions @ np.nan_to_num(last_marks, nan=0.0)
        if include_long_short:
            closing_equity = nav_values[day_index, len(PORTFOLIOS):]
            if not np.isfinite(closing_equity).all() or (closing_equity <= 0).any():
                raise ArithmeticError(f'Long-short closing equity must be positive and finite at {date}.')
    nav = pd.DataFrame(nav_values, index=sessions.rename('date'), columns=portfolios)
    nav.attrs['initial_nav'] = 1.0
    nav.attrs['ending_cash'] = dict(zip(portfolios, cash.tolist()))
    nav.attrs['min_cash'] = dict(zip(portfolios, min_cash.tolist()))
    nav.attrs['cost_bps'] = float(cost_bps)
    nav.attrs['portfolio_cost_bps'] = dict(zip(portfolios, portfolio_costs))
    nav.attrs['ending_positions'] = {
        portfolio: {codes[index]: float(positions[p, index])
                    for index in np.flatnonzero(np.abs(positions[p]) > _EPS)}
        for p, portfolio in enumerate(portfolios)}
    nav.attrs['rebalances'] = rebalances
    nav.attrs['stale_valuations'] = [
        (portfolio, codes[index], int(stale_counts[p, index]))
        for p, portfolio in enumerate(portfolios)
        for index in np.flatnonzero(stale_counts[p])]
    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    forward = pd.DataFrame(forward_records, columns=FORWARD_COLUMNS).sort_values(['date', 'code']).reset_index(drop=True)
    return nav, trade_frame, forward


def rebalance_summary(nav: pd.DataFrame) -> pd.DataFrame:
    """Export the rebalance ledger from an unmodified run_backtest NAV result.

    turnover = (actual buys + actual sells) / pre-trade opening NAV;
    one_way_turnover = turnover / 2. Initial establishment, zero-trade months
    and blocked attempts are retained; blocked notional contributes zero.
    This is per-rebalance turnover, not an annualized value. Export this table
    before saving NAV to CSV, because CSV does not retain DataFrame attrs.
    """
    if 'rebalances' not in nav.attrs:
        raise ValueError('Rebalance ledger is missing; use the original run_backtest NAV result.')
    result = pd.DataFrame(nav.attrs['rebalances'], columns=REBALANCE_COLUMNS)
    if not result.empty:
        if not result['date'].isin(nav.index).all() or not result['portfolio'].isin(nav.columns).all():
            raise ValueError('Rebalance ledger does not match the supplied NAV dates/portfolios.')
        if result.duplicated(['date', 'portfolio']).any():
            raise ValueError('Duplicate date/portfolio in the rebalance ledger.')
    return result


def stale_valuation_summary(nav: pd.DataFrame) -> pd.DataFrame:
    """Export stock-day counts of nonzero long/short holdings using a prior close.

    A held security with a missing row or invalid adjusted close contributes
    one day. Counts describe valuation carry-forward, not verified delistings.
    Use the original NAV result; attrs do not survive CSV serialization.
    """
    if 'stale_valuations' not in nav.attrs:
        raise ValueError('Stale valuation counts are missing; use the original run_backtest NAV result.')
    return pd.DataFrame(nav.attrs['stale_valuations'], columns=STALE_COLUMNS)


def performance_summary(nav: pd.DataFrame) -> pd.DataFrame:
    """Summarize NAV from initial cash 1, including first-session costs/returns.

    CAGR uses 252 sessions/year. Volatility is sample daily standard deviation;
    Sharpe uses a zero risk-free rate. Drawdown includes the initial cash value.
    """
    if nav.empty:
        raise ValueError('NAV must not be empty.')
    initial = float(nav.attrs.get('initial_nav', 1.0))
    if not np.isfinite(initial) or initial <= 0:
        raise ValueError('Initial NAV must be positive and finite.')
    records = []
    for portfolio in nav.columns:
        values = nav[portfolio].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('NAV values must be positive and finite.')
        with_initial = np.r_[initial, values]
        returns = with_initial[1:] / with_initial[:-1] - 1.0
        volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else np.nan
        total_return = values[-1] / initial - 1.0
        records.append({
            'portfolio': portfolio,
            'total_return': float(total_return),
            'cagr': float((values[-1] / initial) ** (252.0 / len(values)) - 1.0),
            'annual_vol': volatility * np.sqrt(252.0),
            'sharpe': float(np.mean(returns) / volatility * np.sqrt(252.0)) if volatility > 0 else np.nan,
            'max_drawdown': float(np.min(with_initial / np.maximum.accumulate(with_initial) - 1.0)),
        })
    return pd.DataFrame(records)
