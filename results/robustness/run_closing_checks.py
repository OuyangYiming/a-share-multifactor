"""Offline closing analyses; frozen MVP inputs, exclusive output creation."""
from pathlib import Path
import argparse
import gc
import hashlib
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.backtest import run_backtest, performance_summary
from src.factors import FACTOR_COLUMNS

def read(path, **kwargs):
    return pd.read_csv(ROOT / path, float_precision='round_trip', **kwargs)

def save(frame, name, index=False):
    with (OUT / name).open('x', encoding='utf-8', newline='') as stream:
        frame.to_csv(stream, index=index)

def inputs():
    signals = read('data/mvp/signals.csv', parse_dates=['date'])
    nav = read('results/tables/mvp_nav.csv', parse_dates=['date'], index_col='date')
    members = read('data/mvp/constituents.csv', parse_dates=['signal_date'])
    pieces = [pd.read_csv(p, dtype=str) for p in sorted((ROOT / 'data/mvp/calendar').glob('*.csv'))]
    days = pd.concat(pieces, ignore_index=True)
    calendar = pd.DatetimeIndex(pd.to_datetime(days.loc[days.is_trading_day.eq('1'), 'calendar_date'])).sort_values()
    assert calendar.is_unique
    # Use exactly the original daily-cache loader's float parsing.
    prices = pd.concat([pd.read_csv(p, parse_dates=['date'], dtype={'code': str})
                        for p in sorted((ROOT / 'data/mvp/daily').glob('*.csv'))], ignore_index=True)
    return prices, signals, calendar, nav, members

def missing(prices, signals, calendar, nav, members):
    stale = read('results/tables/mvp_stale_valuations.csv')
    manifest = read('data/mvp/manifest.csv')
    early = set(manifest.loc[manifest.end_date.lt('2025-12-31'), 'code'])
    known = set(stale.code)
    excluded = sorted(early | known)
    sessions = nav.index
    trades = read('results/tables/mvp_trades.csv', parse_dates=['date'])
    trades = trades.loc[trades.reason.eq('rebalance') & trades.portfolio.eq('benchmark') & trades.code.isin(excluded)]
    rows = []
    for code in excluded:
        stock = prices.loc[prices.code.eq(code)].sort_values('date').set_index('date')
        valid = np.isfinite(stock.adj_close) & stock.adj_close.gt(0)
        last_valid = stock.index[valid][-1]
        last_record = stock.index[-1]
        amounts = trades.loc[trades.code.eq(code)].copy()
        opening = stock.open * stock.adj_close / stock.close
        amounts['quantity'] = amounts.notional / amounts.date.map(opening)
        assert np.isfinite(amounts.quantity).all()
        amounts.loc[amounts.side.eq('sell'), 'quantity'] *= -1
        holdings = amounts.groupby('date').quantity.sum().reindex(sessions, fill_value=0).cumsum()
        observed = stock.adj_close.reindex(sessions)
        missing_dates = sessions[(holdings > 1e-10) & (~np.isfinite(observed) | observed.le(0))]
        recorded = stale.loc[stale.code.eq(code) & stale.portfolio.eq('benchmark'), 'stale_days'].sum()
        assert len(missing_dates) == recorded, (code, len(missing_dates), recorded)
        later = members.loc[members.code.eq(code) & members.signal_date.gt(last_record)]
        tail = sessions[sessions > last_valid]
        rows.append(dict(code=code,known_stale=code in known,ended_early=code in early,
                         last_valid_date=last_valid.date(),last_record_date=last_record.date(),
                         last_tradestatus=stock.tradestatus.iloc[-1],
                         no_valid_price_days_after_last=len(tail),
                         first_missing_tail_date=tail[0].date() if len(tail) else '',
                         last_missing_tail_date=tail[-1].date() if len(tail) else '',
                         benchmark_stale_days=len(missing_dates),
                         first_held_stale_date=missing_dates[0].date() if len(missing_dates) else '',
                         last_held_stale_date=missing_dates[-1].date() if len(missing_dates) else '',
                         max_account_stale_days=int(stale.loc[stale.code.eq(code),'stale_days'].max()) if code in known else 0,
                         later_member_months=later.signal_date.nunique(),
                         observed_type='最后记录后无行情' if last_record < sessions[-1] else '存在缺价记录',
                         economic_cause='无法判断'))
    save(pd.DataFrame(rows), 'missing_price_inventory.csv')
    baseline, orders, labels = run_backtest(prices, signals, calendar, cost_bps=15, include_long_short=True)
    assert baseline.index.equals(nav.index) and baseline.columns.equals(nav.columns)
    delta = float(np.max(np.abs(baseline.to_numpy()-nav.to_numpy())))
    assert delta < 1e-10, delta
    print('BASELINE_MAX_DIFFERENCE', delta, flush=True)
    del baseline, orders, labels
    selected = signals.loc[~signals.code.isin(excluded)].copy()
    # Keep original score, then form quintiles among the remaining securities.
    result, orders, labels = run_backtest(prices, selected, calendar, cost_bps=15, include_long_short=True)
    before = performance_summary(nav).set_index('portfolio')
    after = performance_summary(result).set_index('portfolio')
    impact = before[['cagr','sharpe','max_drawdown']].add_prefix('before_').join(
        after[['cagr','sharpe','max_drawdown']].add_prefix('after_'))
    for field in ['cagr','sharpe','max_drawdown']:
        impact[field+'_change'] = impact['after_'+field]-impact['before_'+field]
    impact['excluded_codes'] = len(excluded)
    save(impact.reset_index(), 'missing_price_impact.csv')
    save(result, 'missing_price_excluded_nav.csv', index=True)
    excess = pd.DataFrame([{'version':version,'q5_minus_benchmark_cagr':stats.loc['Q5','cagr']-stats.loc['benchmark','cagr'],
                           'q4_cagr':stats.loc['Q4','cagr'],'q5_cagr':stats.loc['Q5','cagr']}
                          for version,stats in [('original',before),('exclude_union',after)]])
    save(excess, 'missing_price_excess_comparison.csv')
    print(impact.to_string(), flush=True)
    print(excess.to_string(index=False), flush=True)

def interval(nav, first, last, label):
    selected = nav.loc[first:last, ['Q5','benchmark','long_short']].copy()
    prior = nav.loc[nav.index < selected.index[0]]
    initial = prior.iloc[-1] if len(prior) else pd.Series(1.,index=nav.columns)
    selected = selected.div(initial[selected.columns],axis='columns')
    selected.attrs = {'initial_nav':1.}
    result = performance_summary(selected)
    result.insert(0,'period',label)
    return result

def robustness(prices, signals, calendar, nav, members):
    correlations = [g[FACTOR_COLUMNS].rank(method='average').corr().to_numpy()
                    for _,g in signals.groupby('date',sort=True)]
    matrix = pd.DataFrame(np.mean(correlations,axis=0),index=FACTOR_COLUMNS,columns=FACTOR_COLUMNS)
    matrix.index.name='factor'
    save(matrix,'factor_rank_correlation.csv',index=True)
    singles=[]
    for factor in FACTOR_COLUMNS:
        variant=signals.copy(); variant['score']=variant[factor]
        result,orders,labels=run_backtest(prices,variant,calendar,cost_bps=15)
        assert np.allclose(result.benchmark,nav.benchmark,rtol=0,atol=1e-10)
        stats=performance_summary(result).query("portfolio != 'benchmark'")
        stats.insert(0,'factor',factor);singles.append(stats)
        save(result,f'closing_single_{factor}_nav.csv',index=True)
        del result,orders,labels;gc.collect()
        print('SINGLE_DONE',factor,flush=True)
    single=pd.concat(singles,ignore_index=True)
    save(single,'single_factor_performance.csv')
    costs=[]
    for bps in [5,15,30]:
        if bps==15: result=nav
        else: result,orders,labels=run_backtest(prices,signals,calendar,cost_bps=bps,include_long_short=True)
        stats=performance_summary(result).query("portfolio in ['Q5','benchmark','long_short']")
        stats.insert(0,'cost_bps',bps);costs.append(stats)
        if bps!=15:
            save(result,f'closing_cost_{bps}bps_nav.csv',index=True)
            del result,orders,labels;gc.collect()
        print('COST_DONE',bps,flush=True)
    cost=pd.concat(costs,ignore_index=True);save(cost,'cost_sensitivity.csv')
    periods=[interval(nav,str(y),str(y),str(y)) for y in range(2018,2026)]
    periods += [interval(nav,str(a),str(b),f'{a}-{b}') for a,b in [(2018,2021),(2022,2025)]]
    period=pd.concat(periods,ignore_index=True);save(period,'period_performance.csv')
    month=nav[['Q5','benchmark']].groupby(nav.index.to_period('M')).last()
    previous=month.shift();previous.iloc[0]=1.
    returns=month/previous-1;returns['excess']=returns.Q5-returns.benchmark
    returns.index=returns.index.astype(str);returns.index.name='month'
    save(returns,'monthly_excess.csv',index=True)
    values=returns.excess;se=values.std(ddof=1)/np.sqrt(len(values))
    summary=pd.DataFrame([dict(months=len(values),mean=values.mean(),standard_error=se,
                               t_value=values.mean()/se,win_rate=values.gt(0).mean())])
    save(summary,'monthly_excess_summary.csv')
    ic=read('results/tables/mvp_rank_ic.csv')
    significance=[]
    for factor,g in ic.groupby('factor'):
        x=g.rank_ic.dropna(); se=x.std(ddof=1)/np.sqrt(len(x))
        significance.append(dict(factor=factor,months=len(x),mean_ic=x.mean(),iid_standard_error=se,
                                 iid_t_value=x.mean()/se))
    save(pd.DataFrame(significance),'ic_mean_statistics.csv')
    for name,frame in [('CORRELATION',matrix),('SINGLE',single),('COST',cost),('PERIOD',period),('MONTHLY_EXCESS',summary)]:
        print(name+'\n'+frame.to_string(index=name=='CORRELATION'),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=['missing','robustness']);args=parser.parse_args()
    protected=list((ROOT/'results/tables').glob('mvp_*.csv'))+list((ROOT/'src').glob('*.py'))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={p:digest(p) for p in protected}
    start=time.monotonic();data=inputs()
    print('INPUT',len(data[0]),'rows;',len(data[1]),'signals',flush=True)
    (missing if args.step=='missing' else robustness)(*data)
    assert before=={p:digest(p) for p in protected}
    print(f'COMPLETE {args.step}: {time.monotonic()-start:.2f}s; frozen inputs unchanged',flush=True)

if __name__=='__main__':main()
