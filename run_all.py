"""Run monthly BaoStock data, four factors, RankIC and costed quintile portfolios."""
import argparse
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_research_data
from src.factors import build_signals
from src.evaluate import calculate_rank_ic, summarize_ic
from src.backtest import (run_backtest, performance_summary, rebalance_summary,
                          stale_valuation_summary)


def benchmark_comparison(nav):
    """Compare each net long-only quintile with the same-universe benchmark."""
    initial = pd.DataFrame([{column: 1.0 for column in nav}],
                           index=[nav.index[0] - pd.Timedelta(days=1)])
    returns = pd.concat([initial, nav]).pct_change(fill_method=None).iloc[1:]
    stats = performance_summary(nav).set_index('portfolio')
    records = []
    for portfolio in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5'):
        active = returns[portfolio] - returns['benchmark']
        tracking_error = active.std(ddof=1) * np.sqrt(252)
        records.append({
            'portfolio': portfolio,
            'cagr': stats.loc[portfolio, 'cagr'],
            'benchmark_cagr': stats.loc['benchmark', 'cagr'],
            'cagr_difference': stats.loc[portfolio, 'cagr'] - stats.loc['benchmark', 'cagr'],
            'relative_wealth_return': nav[portfolio].iloc[-1] / nav['benchmark'].iloc[-1] - 1,
            'tracking_error': tracking_error,
            'information_ratio': active.mean() * 252 / tracking_error if tracking_error > 0 else np.nan,
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Use existing BaoStock CSV caches.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    tables = root / 'results/tables'
    figures = root / 'results/figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    prices, constituents, calendar = load_research_data(root, offline=args.offline)
    print(f'DATA: {prices.code.nunique()} stocks, {len(prices):,} daily rows, {constituents.signal_date.nunique()} monthly universes', flush=True)
    observed = pd.DatetimeIndex(prices['date'].unique()).sort_values()
    missing, extra = calendar.difference(observed), observed.difference(calendar)
    if len(missing) or len(extra):
        raise ValueError(f'Price/calendar mismatch: missing={missing.tolist()}, extra={extra.tolist()}')
    signals = build_signals(prices, constituents)
    expected = pd.Series(calendar).groupby(calendar.to_period('M')).max().loc['2017-12':'2025-11']
    actual = pd.DatetimeIndex(signals['date'].unique()).sort_values()
    if len(expected) != 96 or not actual.equals(pd.DatetimeIndex(expected)):
        raise ValueError('Signal dates must equal all 96 month ends from 2017-12 through 2025-11')
    print(f'SIGNALS: {len(signals):,} eligible stock-months', flush=True)
    nav, trades, forward = run_backtest(prices, signals, calendar, cost_bps=15.0,
                                       include_long_short=True)
    ic = calculate_rank_ic(signals, forward)
    ic_summary = summarize_ic(ic)
    performance = performance_summary(nav)
    turnover = rebalance_summary(nav)
    turnover_summary = turnover.groupby('portfolio').agg(
        mean_monthly_turnover=('turnover', 'mean'),
        mean_monthly_one_way_turnover=('one_way_turnover', 'mean'))
    turnover_summary['annualized_one_way_turnover'] = 12 * turnover_summary['mean_monthly_one_way_turnover']
    performance = performance.merge(turnover_summary, on='portfolio', validate='one_to_one')
    comparison = benchmark_comparison(nav)
    long_short = performance.loc[performance.portfolio.isin(['long_short_gross', 'long_short'])].copy()
    long_short['cost_bps'] = long_short.portfolio.map(nav.attrs['portfolio_cost_bps'])
    long_short['target_long_exposure'] = 1.0
    long_short['target_short_exposure'] = -1.0
    long_short['total_cost'] = long_short.portfolio.map(trades.groupby('portfolio').cost.sum()).fillna(0.0)
    last_signal = signals.date.max()
    missing_labels = []
    for scope, selected in [('all', forward), ('complete_windows', forward.loc[forward.date.ne(last_signal)]),
                            ('terminal_window', forward.loc[forward.date.eq(last_signal)])]:
        missing = selected.loc[selected.forward_return.isna()]
        missing_labels.append({'scope': scope, 'pairs': len(selected),
                               'valid_pairs': int(selected.forward_return.notna().sum()),
                               'missing_pairs': len(missing), 'missing_stocks': missing.code.nunique()})
    sample = constituents.groupby('signal_date').code.nunique().rename('constituents').to_frame()
    sample['eligible'] = signals.groupby('date').code.nunique()
    sample['excluded'] = sample['constituents'] - sample['eligible']
    sample.index.name = 'signal_date'
    nav.to_csv(tables / 'mvp_nav.csv', index_label='date')
    performance.to_csv(tables / 'mvp_performance.csv', index=False)
    ic.to_csv(tables / 'mvp_rank_ic.csv', index=False)
    ic_summary.to_csv(tables / 'mvp_ic_summary.csv', index=False)
    sample.to_csv(tables / 'mvp_sample_summary.csv')
    trades.to_csv(tables / 'mvp_trades.csv', index=False)
    turnover.to_csv(tables / 'mvp_turnover.csv', index=False)
    comparison.to_csv(tables / 'mvp_benchmark_comparison.csv', index=False)
    long_short.to_csv(tables / 'mvp_long_short_comparison.csv', index=False)
    stale_valuation_summary(nav).to_csv(tables / 'mvp_stale_valuations.csv', index=False)
    pd.DataFrame(missing_labels).to_csv(tables / 'mvp_missing_labels.csv', index=False)
    signals.to_csv(root / 'data/mvp/signals.csv', index=False)
    forward.to_csv(root / 'data/mvp/forward_returns.csv', index=False)
    execution = trades.groupby('portfolio').agg(
        executed_orders=('reason', lambda s: int(s.eq('rebalance').sum())),
        blocked_orders=('reason', lambda s: int(s.ne('rebalance').sum())),
        total_notional=('notional', 'sum'), total_cost=('cost', 'sum'))
    execution.to_csv(tables / 'mvp_execution_summary.csv')
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    colors = ['#b8c9dd', '#829bb8', '#557999', '#2d597b', '#b43d35']
    for name, color in zip(['Q1','Q2','Q3','Q4','Q5'], colors):
        ax.plot(nav.index, nav[name], label=f'{name}' + (' (highest score)' if name == 'Q5' else ''), color=color, linewidth=2 if name=='Q5' else 1.3)
    ax.plot(nav.index, nav['benchmark'], label='Eligible universe equal weight', color='#222222', linestyle='--', linewidth=1.6)
    ax.set(title='Monthly four-factor quintiles | 2018-2025 | 15 bps per side',
           xlabel='', ylabel='Net asset value (initial = 1)')
    ax.axhline(1, color='gray', linewidth=.6, alpha=.5)
    ax.grid(axis='y', alpha=.18)
    ax.legend(loc='upper left', ncol=2, frameon=False)
    fig.savefig(figures / 'mvp_nav.png', dpi=180)
    plt.close(fig)
    print('\nPERFORMANCE\n'+performance.to_string(index=False, float_format=lambda x:f'{x:.6f}'), flush=True)
    print('\nRANK_IC\n'+ic_summary.to_string(index=False, float_format=lambda x:f'{x:.6f}'), flush=True)
    print('\nEXECUTION\n'+execution.to_string(float_format=lambda x:f'{x:.6f}'), flush=True)
    print('\nLONG_SHORT\n'+long_short.to_string(index=False, float_format=lambda x:f'{x:.6f}'), flush=True)
    print('\nBENCHMARK_COMPARISON\n'+comparison.to_string(index=False, float_format=lambda x:f'{x:.6f}'), flush=True)
    print(f'\nCOMPLETE in {time.monotonic()-started:.1f}s; figure: results/figures/mvp_nav.png', flush=True)


if __name__ == '__main__':
    main()
