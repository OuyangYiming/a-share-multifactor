"""Synthetic negative control: future labels must not enter a historical signal.

Run explicitly: python -B -m pytest -q -p no:cacheprovider tests/test_lookahead_canary.py
The repository's pytest.ini collects this file alongside test_mvp.py by default.
No market data, downloads, production configuration, or existing tests are used
or changed. The artificial ten-stock weekday calendar is not an exchange calendar.
High IC/returns alone are not a leakage detector: the second test verifies that
the historical-invariance assertion rejects the deliberately contaminated data.
"""

from functools import lru_cache
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import pytest

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import performance_summary, run_backtest
from src.evaluate import calculate_rank_ic
from src.factors import FACTOR_COLUMNS, build_signals


COST_BPS = 15.0
CUTOFF = pd.Timestamp('2021-12-31')


def _synthetic_inputs():
    """Generate ten stocks, 96 signal months, and the unchanged research window."""
    rng = np.random.default_rng(20261110)
    calendar = pd.bdate_range('2016-12-01', '2025-12-31')
    periods = calendar.to_period('M')
    codes = [f'sh.{600000 + i:06d}' for i in range(10)]
    daily_gross = np.ones((len(calendar), len(codes)))
    # Each new month receives an independently shuffled, symmetric return grid.
    # Its rank is unrelated to the preceding price history by construction.
    for month in periods.unique():
        rows = periods == month
        monthly_return = rng.permutation(np.linspace(-0.04, 0.04, len(codes)))
        daily_gross[rows] = (1.0 + monthly_return) ** (1.0 / rows.sum())
    opening = np.empty_like(daily_gross)
    closing = np.empty_like(daily_gross)
    level = np.linspace(20.0, 29.0, len(codes))
    for day in range(len(calendar)):
        opening[day] = level
        level = level * daily_gross[day]
        closing[day] = level
    prices = pd.concat([
        pd.DataFrame({'date': calendar, 'code': code, 'open': opening[:, i],
                      'close': closing[:, i], 'preclose': opening[:, i],
                      'adj_close': closing[:, i], 'turn': 1.0 + i / 10.0,
                      'isST': 0, 'tradestatus': 1})
        for i, code in enumerate(codes)
    ], ignore_index=True)
    month_ends = pd.Series(calendar).groupby(periods).max().loc['2017-12':'2025-11']
    members = pd.MultiIndex.from_product(
        [month_ends.to_numpy(), codes], names=['signal_date', 'code']).to_frame(index=False)
    return prices, members, calendar


def _contaminate(signals, forward):
    """DELIBERATELY WRONG: inject future-return ranks before recomputing score.

    A fixed, conspicuous coefficient makes this a deterministic canary, not
    parameter selection. All four original factor directions remain unchanged.
    The final month has no complete label and is left untouched.
    """
    mixed = signals.merge(forward, on=['date', 'code'], how='left', validate='one_to_one')
    future_rank = mixed.groupby('date')['forward_return'].rank(method='average', pct=True)
    complete = future_rank.notna()
    for factor in FACTOR_COLUMNS:
        mixed.loc[complete, factor] += 100.0 * future_rank.loc[complete]
    mixed['score'] = mixed.groupby('date')[FACTOR_COLUMNS].rank(
        method='average', pct=True).mean(axis=1)
    return mixed[signals.columns].copy()


def _change_future_only(prices):
    """Reverse future cross-stock paths, rescaled to each stock's cutoff price."""
    changed = prices.copy(deep=True)
    codes = sorted(prices.code.unique())
    fields = ['open', 'close', 'preclose', 'adj_close']
    for destination, source in zip(codes, reversed(codes)):
        target_rows = prices.code.eq(destination) & prices.date.gt(CUTOFF)
        source_rows = prices.code.eq(source) & prices.date.gt(CUTOFF)
        target_anchor = prices.loc[prices.code.eq(destination) & prices.date.eq(CUTOFF), 'close'].iloc[0]
        source_anchor = prices.loc[prices.code.eq(source) & prices.date.eq(CUTOFF), 'close'].iloc[0]
        changed.loc[target_rows, fields] = (
            prices.loc[source_rows, fields].to_numpy() * target_anchor / source_anchor)
    return changed


def _past(signals):
    return signals.loc[signals.date.le(CUTOFF)].reset_index(drop=True)


@lru_cache(maxsize=1)
def run_canary_experiment():
    """Run the real factor/evaluation/accounting functions on artificial prices."""
    prices, members, calendar = _synthetic_inputs()
    baseline = build_signals(prices, members)
    nav, trades, forward = run_backtest(
        prices, baseline, calendar, cost_bps=COST_BPS, include_long_short=True)
    contaminated = _contaminate(baseline, forward)
    bad_nav, bad_trades, bad_forward = run_backtest(
        prices, contaminated, calendar, cost_bps=COST_BPS, include_long_short=True)
    pd.testing.assert_frame_equal(forward, bad_forward)

    changed_prices = _change_future_only(prices)
    pd.testing.assert_frame_equal(prices.loc[prices.date.le(CUTOFF)],
                                  changed_prices.loc[changed_prices.date.le(CUTOFF)])
    changed_baseline = build_signals(changed_prices, members)
    _, _, changed_forward = run_backtest(
        changed_prices, changed_baseline, calendar, cost_bps=COST_BPS)
    changed_contaminated = _contaminate(changed_baseline, changed_forward)

    records = []
    for name, signal, value, orders, changed in (
        ('causal_baseline', baseline, nav, trades, changed_baseline),
        ('future_contaminated', contaminated, bad_nav, bad_trades, changed_contaminated),
    ):
        ic = calculate_rank_ic(signal, forward)
        score_ic = ic.loc[ic.factor.eq('score'), 'rank_ic'].dropna()
        perf = performance_summary(value).set_index('portfolio')
        earlier = _past(signal)
        altered = _past(changed)
        differences = (earlier[FACTOR_COLUMNS + ['score']].to_numpy()
                       != altered[FACTOR_COLUMNS + ['score']].to_numpy()).any(axis=1)
        records.append({
            'scenario': name, 'stocks': signal.code.nunique(),
            'signal_months': signal.date.nunique(), 'valid_ic_months': len(score_ic),
            'mean_score_rank_ic': score_ic.mean(), 'q5_final_nav': value.Q5.iloc[-1],
            'q5_total_return': perf.loc['Q5', 'total_return'],
            'q5_cagr': perf.loc['Q5', 'cagr'],
            'long_short_total_return': perf.loc['long_short', 'total_return'],
            'benchmark_total_return': perf.loc['benchmark', 'total_return'],
            'past_signal_rows_changed': int(differences.sum()), 'cost_bps': COST_BPS,
        })
        # Both treatments use the identical fee rule and completely fill orders.
        paid = orders.loc[orders.portfolio.ne('long_short_gross')]
        np.testing.assert_allclose(paid.cost, paid.notional * COST_BPS / 10000.0,
                                   rtol=1e-12, atol=1e-14)
        assert orders.reason.eq('rebalance').all()
    return {'results': pd.DataFrame(records), 'baseline': baseline,
            'changed_baseline': changed_baseline, 'contaminated': contaminated,
            'changed_contaminated': changed_contaminated, 'forward': forward,
            'nav': nav, 'bad_nav': bad_nav}


def test_deliberate_future_leak_inflates_rank_ic_and_returns():
    """The injected oracle outperforms the causal control under identical execution."""
    experiment = run_canary_experiment()
    results = experiment['results'].set_index('scenario')
    clean, leaked = results.loc['causal_baseline'], results.loc['future_contaminated']
    assert (results.stocks == 10).all()
    assert (results.signal_months == 96).all()
    assert (results.valid_ic_months == 95).all()
    assert abs(clean.mean_score_rank_ic) < 0.2
    assert leaked.mean_score_rank_ic > 0.99
    assert leaked.mean_score_rank_ic > clean.mean_score_rank_ic + 0.5
    assert leaked.q5_total_return > clean.q5_total_return + 1.0
    assert leaked.long_short_total_return > clean.long_short_total_return + 1.0
    # No signal beyond the fixed window is invented to fill the final IC label.
    last = experiment['forward'].date.max()
    assert experiment['forward'].loc[experiment['forward'].date.eq(last), 'forward_return'].isna().all()
    for value in (experiment['nav'], experiment['bad_nav']):
        assert value.index.min() == pd.Timestamp('2018-01-01')
        assert value.index.max() == pd.Timestamp('2025-12-31')
    # All stocks remain in the common benchmark regardless of the leaked ordering.
    np.testing.assert_allclose(experiment['nav'].benchmark, experiment['bad_nav'].benchmark,
                               rtol=1e-12, atol=1e-14)


def test_future_perturbation_rejects_contaminated_historical_signals():
    """A genuine temporal-invariance check passes the control and rejects the leak."""
    experiment = run_canary_experiment()
    pd.testing.assert_frame_equal(_past(experiment['baseline']),
                                  _past(experiment['changed_baseline']), check_exact=True)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(_past(experiment['contaminated']),
                                      _past(experiment['changed_contaminated']), check_exact=True)
    rows = experiment['results'].set_index('scenario').past_signal_rows_changed
    assert rows.loc['causal_baseline'] == 0
    assert rows.loc['future_contaminated'] > 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True,
                        help='Explicit output CSV outside the repository, e.g. ../walkthrough/lookahead_canary_results.csv')
    args = parser.parse_args()
    result = run_canary_experiment()['results']
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open('x', encoding='utf-8', newline='') as handle:
        result.to_csv(handle, index=False, float_format='%.15g')
    print(result.to_string(index=False, float_format=lambda value: f'{value:.9f}'))
