"""Build an editable XeLaTeX report from completed MVP result files.

Run from the project root: python report/build_report.py
Use --tex-only to export the source without invoking XeLaTeX.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image


PORTFOLIOS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'benchmark']
FACTORS = ['momentum', 'reversal', 'low_volatility', 'turnover', 'score']
NAMES = {'benchmark': '等权基准', 'momentum': '动量', 'reversal': '反转',
         'low_volatility': '低波动', 'turnover': '换手率', 'score': '综合分数',
         'long_short': '多空扣费后', 'long_short_gross': '多空零费用'}
REQUIRED = {
    'nav': ['date'] + PORTFOLIOS,
    'performance': ['portfolio', 'total_return', 'cagr', 'annual_vol', 'sharpe', 'max_drawdown'],
    'rank_ic': ['date', 'factor', 'rank_ic', 'n'],
    'ic_summary': ['factor', 'mean_rank_ic', 'ic_std', 'ic_ir', 'positive_fraction', 'months'],
    'sample_summary': ['signal_date', 'constituents', 'eligible', 'excluded'],
    'trades': ['date', 'portfolio', 'code', 'side', 'notional', 'cost', 'reason'],
    'execution_summary': ['portfolio', 'executed_orders', 'blocked_orders', 'total_notional', 'total_cost'],
}


def escape(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(char, char) for char in str(value))


def number(value, digits=3):
    return '--' if pd.isna(value) else f'{float(value):.{digits}f}'


def percent(value, digits=2):
    return '--' if pd.isna(value) else f'{float(value) * 100:.{digits}f}' + r'\%'


def tabular(headers, rows, alignment=None):
    alignment = alignment or 'l' + 'r' * (len(headers) - 1)
    lines = [r'\begin{center}\small', r'\setlength{\tabcolsep}{5pt}',
             r'\renewcommand{\arraystretch}{1.18}', r'\begin{tabular}{' + alignment + '}',
             r'\toprule', ' & '.join(headers) + r' \\', r'\midrule']
    lines.extend(' & '.join(row) + r' \\' for row in rows)
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}', r'\end{center}'])


def read_inputs(root):
    tables = root / 'results' / 'tables'
    paths = {name: tables / f'mvp_{name}.csv' for name in REQUIRED}
    paths['figure'] = root / 'results' / 'figures' / 'mvp_nav.png'
    missing = [str(path.relative_to(root)) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError('正式输入尚未齐备，未生成报告：\n' + '\n'.join(missing))
    with Image.open(paths['figure']) as image:
        image.verify()
    data = {}
    for name, columns in REQUIRED.items():
        frame = pd.read_csv(paths[name])
        if frame.empty or not set(columns).issubset(frame):
            raise ValueError(f'{paths[name].name}: 空表或缺少必要列 {columns}')
        data[name] = frame
    nav = data['nav'].set_index('date')
    nav.index = pd.to_datetime(nav.index, errors='raise')
    if (not nav.index.is_unique or not nav.index.is_monotonic_increasing
            or nav.index.min().year != 2018 or nav.index.max().year != 2025
            or not np.isfinite(nav[PORTFOLIOS]).all().all() or nav[PORTFOLIOS].le(0).any().any()):
        raise ValueError('mvp_nav.csv 必须含按日期递增的 2018-2025 年六组合正净值.')
    data['nav'] = nav[PORTFOLIOS]
    sample = data['sample_summary']
    sample['signal_date'] = pd.to_datetime(sample['signal_date'])
    months = sample['signal_date'].dt.to_period('M')
    if (months.duplicated().any() or not pd.PeriodIndex(months).equals(
            pd.period_range('2017-12', '2025-11', freq='M'))):
        raise ValueError('样本汇总必须完整覆盖 2017-12 至 2025-11 的 96 个信号月.')
    if not sample['constituents'].sub(sample['eligible']).eq(sample['excluded']).all():
        raise ValueError('样本汇总的成员、合格和排除数量不一致.')
    for name, key, expected in [('performance', 'portfolio', PORTFOLIOS),
                                ('execution_summary', 'portfolio', PORTFOLIOS),
                                ('ic_summary', 'factor', FACTORS)]:
        allowed = set(expected) | ({'long_short', 'long_short_gross'} if key == 'portfolio' else set())
        observed = set(data[name][key])
        if data[name][key].duplicated().any() or not set(expected).issubset(observed) or not observed.issubset(allowed):
            raise ValueError(f'mvp_{name}.csv 的 {key} 不完整或重复.')
        data[name] = data[name].set_index(key).loc[expected]
    ic = data['rank_ic']
    ic['date'] = pd.to_datetime(ic['date'])
    expected_pairs = pd.MultiIndex.from_product([sample['signal_date'], FACTORS])
    actual_pairs = pd.MultiIndex.from_frame(ic[['date', 'factor']])
    if actual_pairs.has_duplicates or len(expected_pairs.difference(actual_pairs)) or len(actual_pairs) != len(expected_pairs):
        raise ValueError('月度 IC 必须逐月包含四个因子与综合分数.')
    trades = data['trades']
    trades['date'] = pd.to_datetime(trades['date'])
    filled = trades.loc[trades['portfolio'].isin(PORTFOLIOS) &
                        trades['reason'].eq('rebalance') & trades['notional'].gt(0)]
    if filled.empty:
        raise ValueError('成交表没有有效成交，不能推断成本口径.')
    rates = filled['cost'] / filled['notional'] * 10000
    if not np.isfinite(rates).all() or not np.allclose(rates, rates.iloc[0], rtol=1e-8, atol=1e-8):
        raise ValueError('各笔成交费率不一致，请先确认报告口径.')
    data['cost_bps'] = float(rates.iloc[0])
    if not np.allclose(data['performance']['total_return'], nav[PORTFOLIOS].iloc[-1].to_numpy() - 1):
        raise ValueError('绩效总收益与净值终值不一致，请重新生成同一次运行的结果.')
    for stem in ('benchmark_comparison', 'long_short_comparison', 'stale_valuations', 'missing_labels'):
        path = tables / f'mvp_{stem}.csv'
        if path.exists():
            data[stem] = pd.read_csv(path)
    return data


def nav_plot(nav):
    ticks = [int(np.flatnonzero(nav.index.year == year)[0]) for year in sorted(set(nav.index.year))]
    labels = [str(nav.index[index].year) for index in ticks]
    styles = ['thin,densely dotted', 'thin,dashed', 'thin,dashdotted',
              'semithick,dash pattern=on 5pt off 2pt on 1pt off 2pt',
              'thick,solid', 'semithick,dash pattern=on 8pt off 3pt']
    lines = [r'\begin{center}\begin{tikzpicture}', r'\begin{axis}[',
             r'width=\textwidth,height=7.3cm,scale only axis=false,',
             r'tick label style={font=\small},label style={font=\small},',
             r'axis line style={black},ymajorgrids=true,grid style={black!15},',
             r'ylabel={净值},xlabel={交易日期},xmin=0,xmax=' + str(len(nav) - 1) + ',',
             'xtick={' + ','.join(map(str, ticks)) + '},xticklabels={' + ','.join(labels) + '},',
             r'legend style={font=\small,draw=none,at={(0.5,1.04)},anchor=south},legend columns=3]']
    for code, style in zip(PORTFOLIOS, styles):
        coords = ' '.join(f'({i},{value:.8g})' for i, value in enumerate(nav[code]))
        lines += [r'\addplot[black,no marks,' + style + '] coordinates {' + coords + '};',
                  r'\addlegendentry{' + NAMES.get(code, code) + '}']
    return '\n'.join(lines + [r'\end{axis}\end{tikzpicture}',
                              r'\par\small 图 1. 月度分层组合与等权基准的扣费净值.', r'\end{center}'])


def quintile_interpretation(performance):
    annual = performance.loc[PORTFOLIOS[:5], 'cagr']
    if not np.isfinite(annual).all():
        raise ValueError('分层年化收益必须是有限值，才能判断单调性.')
    differences = np.diff(annual.to_numpy())
    if (differences > 0).all():
        relation = '随得分严格递增'
    elif (differences < 0).all():
        relation = '随得分严格递减'
    else:
        relation = '未呈严格单调关系'
    leaders = '、'.join(annual.index[annual.eq(annual.max())])
    return ('Q1 至 Q5 的年化收益' + relation + '. 最高年化组为 ' + escape(leaders)
            + '（' + percent(annual.max()) + '），Q5 为 ' + percent(annual['Q5'])
            + '，等权基准为 ' + percent(performance.at['benchmark', 'cagr']) + '.')


def optional_comparisons(data):
    """Render the agreed optional schemas without guessing exposure conventions."""
    sections = []
    if 'benchmark_comparison' in data:
        frame = data['benchmark_comparison']
        required = {'portfolio', 'cagr_difference', 'relative_wealth_return', 'tracking_error', 'information_ratio'}
        if not required.issubset(frame):
            raise ValueError('基准比较文件不符合已约定列定义.')
        rows = [[escape(row.portfolio), number(row.cagr_difference * 100, 2), percent(row.relative_wealth_return),
                 percent(row.tracking_error), number(row.information_ratio)]
                for row in frame.itertuples(index=False)]
        sections.append(r'\paragraph{相对等权基准.} 年化差为组合与等权基准的年化收益率之差，'
                        '相对财富收益为终值之比减一. 跟踪误差为日收益率差的样本标准差乘以 '
                        r'$\sqrt{252}$；信息比率为日收益率差的均值乘以 252 后除以跟踪误差.'
                        + tabular(['组合', '年化差/百分点', '相对财富收益', '跟踪误差', '信息比率'], rows))
    if 'long_short_comparison' in data:
        frame = data['long_short_comparison']
        required = {'portfolio', 'total_return', 'cagr', 'sharpe', 'max_drawdown', 'cost_bps',
                    'target_long_exposure', 'target_short_exposure', 'total_cost'}
        if not required.issubset(frame) or frame.empty:
            raise ValueError('多空比较文件不符合已约定列定义.')
        if (frame['target_long_exposure'].nunique() != 1 or frame['target_short_exposure'].nunique() != 1):
            raise ValueError('多空对照使用了不同的目标敞口，不能直接并列.')
        rows = [[NAMES.get(row.portfolio, escape(row.portfolio)), number(row.cost_bps, 2),
                 percent(row.total_return), percent(row.cagr), number(row.sharpe), percent(row.max_drawdown)]
                for row in frame.itertuples(index=False)]
        accounts = frame.set_index('portfolio')
        if accounts.index.has_duplicates or not {'long_short', 'long_short_gross'}.issubset(accounts.index):
            raise ValueError('多空对照须包含唯一的扣费后和零费用账户.')
        net, gross = accounts.at['long_short', 'cagr'], accounts.at['long_short_gross', 'cagr']
        if not np.isfinite([net, gross]).all():
            raise ValueError('多空年化收益必须是有限值，才能比较成本影响.')
        difference = gross - net
        cost_effect = ('两者年化收益相同' if difference == 0 else
                       '扣费后比零费用' + ('低 ' if difference > 0 else '高 ')
                       + number(abs(difference) * 100, 2) + ' 个百分点')
        sections.append(r'\paragraph{多空扣费对照.} 每次月度调仓的目标多头敞口为净值的 '
                        + percent(frame['target_long_exposure'].iloc[0]) + '，目标空头敞口为 '
                        + percent(frame['target_short_exposure'].iloc[0])
                        + '. 多头对应 Q5，空头对应 Q1；多头证券数量为正，空头为负，月内按实际持仓估值. '
                        '扣费后与零费用账户分别逐日记账，'
                        '表内零费用结果来自独立运行，并非在扣费后净值上机械加回累计费用.'
                        + tabular(['组合', '费率/bps', '总收益', '年化收益', 'Sharpe', '最大回撤'], rows)
                        + '零费用多空账户年化收益为 ' + percent(gross) + '，扣费后为 ' + percent(net)
                        + '；' + cost_effect + '. 该差异体现本成交及持仓设定下费用对复利路径的影响.')
    return '\n'.join(sections)


def observed_coverage(data):
    parts = []
    if 'missing_labels' in data:
        frame = data['missing_labels'].set_index('scope')
        if 'complete_windows' not in frame.index or frame.index.has_duplicates:
            raise ValueError('缺失标签汇总缺少唯一的 complete_windows 行.')
        row = frame.loc['complete_windows']
        parts.append(f'完整持有窗口的标签共 {int(row["pairs"]):,} 对，其中有效 '
                     f'{int(row["valid_pairs"]):,} 对、缺失 {int(row["missing_pairs"]):,} 对.')
    if 'stale_valuations' in data:
        frame = data['stale_valuations']
        if not {'portfolio', 'code', 'stale_days'}.issubset(frame):
            raise ValueError('持仓估值延用统计缺少必要列.')
        parts.append(f'持仓沿用历史收盘价的累计数量为 {int(frame["stale_days"].sum()):,} 个组合--股票--交易日，'
                     f'涉及 {frame["code"].nunique():,} 只证券. 该计数逐组合累加，未把缺少行情直接解释为退市.')
    return (r'\paragraph{数据覆盖统计.} ' + ' '.join(parts)) if parts else ''


def render_tex(data):
    nav, performance, ic = data['nav'], data['performance'], data['ic_summary']
    sample, execution = data['sample_summary'], data['execution_summary']
    months = len(sample)
    start, end = nav.index.min().strftime('%Y-%m-%d'), nav.index.max().strftime('%Y-%m-%d')
    performance_rows = [[NAMES.get(code, code)] + [percent(performance.at[code, field])
                        if field != 'sharpe' else number(performance.at[code, field])
                        for field in ['total_return', 'cagr', 'annual_vol', 'sharpe', 'max_drawdown']]
                        for code in PORTFOLIOS]
    ic_rows = [[NAMES[code], number(ic.at[code, 'mean_rank_ic'], 4), number(ic.at[code, 'ic_std'], 4),
                number(ic.at[code, 'ic_ir']), percent(ic.at[code, 'positive_fraction']),
                str(int(ic.at[code, 'months']))] for code in FACTORS]
    execution_rows = [[NAMES.get(code, code), f'{int(execution.at[code, "executed_orders"]):,}',
                       f'{int(execution.at[code, "blocked_orders"]):,}',
                       number(execution.at[code, 'total_notional'], 3),
                       number(execution.at[code, 'total_cost'], 4)] for code in PORTFOLIOS]
    execution_headers = ['组合', '成交笔数', '受阻笔数', '成交额合计', '成本合计']
    turnover_text = ''
    if 'mean_monthly_one_way_turnover' in performance:
        execution_headers.append('月均单边换手')
        for code, row in zip(PORTFOLIOS, execution_rows):
            row.append(percent(performance.at[code, 'mean_monthly_one_way_turnover']))
        turnover_text = ('月度双边换手率为当月买卖成交额之和除以调仓前净值，单边换手率为其一半. '
                         '表中月均单边换手为各次调仓的算术平均，受阻订单的零成交额不增加换手率.')
    annual_rows, previous = [], pd.Series(1.0, index=PORTFOLIOS)
    for year, group in nav.groupby(nav.index.year):
        returns = group.iloc[-1] / previous - 1
        annual_rows.append([str(year), percent(returns['Q1']), percent(returns['Q5']),
                            percent(returns['benchmark']), number((returns['Q5'] - returns['benchmark']) * 100, 2)])
        previous = group.iloc[-1]
    avg_n = data['rank_ic'].loc[data['rank_ic']['factor'].eq('score') & data['rank_ic']['rank_ic'].notna(), 'n'].mean()
    base_trades = data['trades'].loc[data['trades']['portfolio'].isin(PORTFOLIOS)]
    filled = base_trades.loc[base_trades['reason'].eq('rebalance')]
    blocked = base_trades.loc[base_trades['reason'].ne('rebalance')]
    q5, benchmark = performance.loc['Q5'], performance.loc['benchmark']
    replacements = {
        '@@START@@': start, '@@END@@': end, '@@DAYS@@': f'{len(nav):,}', '@@MONTHS@@': str(months),
        '@@ELIGIBLE_MIN@@': str(int(sample['eligible'].min())), '@@ELIGIBLE_MAX@@': str(int(sample['eligible'].max())),
        '@@ELIGIBLE_MEAN@@': number(sample['eligible'].mean(), 1), '@@STOCK_MONTHS@@': f'{int(sample["eligible"].sum()):,}',
        '@@MEMBER_MIN@@': str(int(sample['constituents'].min())), '@@MEMBER_MAX@@': str(int(sample['constituents'].max())),
        '@@EXCLUDED_MEAN@@': number(sample['excluded'].mean(), 1), '@@COST@@': number(data['cost_bps'], 2),
        '@@Q5_RETURN@@': percent(q5['total_return']), '@@BENCH_RETURN@@': percent(benchmark['total_return']),
        '@@Q5_CAGR@@': percent(q5['cagr']), '@@Q5_DD@@': percent(q5['max_drawdown']),
        '@@SCORE_IC@@': number(ic.at['score', 'mean_rank_ic'], 4), '@@SCORE_MONTHS@@': str(int(ic.at['score', 'months'])),
        '@@PAIR_MEAN@@': number(avg_n, 1), '@@FILLED@@': f'{len(filled):,}', '@@BLOCKED@@': f'{len(blocked):,}',
        '@@PERFORMANCE@@': tabular(['组合', '总收益', '年化收益', '年化波动', 'Sharpe', '最大回撤'], performance_rows),
        '@@IC@@': tabular(['因子', '均值', '标准差', 'ICIR', '正值比例', '有效月数'], ic_rows),
        '@@EXECUTION@@': tabular(execution_headers, execution_rows), '@@TURNOVER@@': turnover_text,
        '@@ANNUAL@@': tabular(['年份', 'Q1', 'Q5', '等权基准', '收益差/百分点'], annual_rows),
        '@@FIGURE@@': nav_plot(nav), '@@OPTIONAL@@': optional_comparisons(data),
        '@@COVERAGE@@': observed_coverage(data),
        '@@QUINTILE_INTERPRETATION@@': quintile_interpretation(performance),
    }
    source = r'''% !TEX program = xelatex
% Generated from completed MVP CSV files by build_report.py.
\IfFileExists{aomart.cls}{\documentclass[a4paper]{aomart}}{\documentclass[11pt,a4paper]{amsart}}
\usepackage[a4paper,margin=25mm,headheight=15pt,headsep=7mm]{geometry}
\usepackage{fontspec,xeCJK,amsmath,amssymb,booktabs,array,pgfplots,needspace}
\IfFontExistsTF{Noto Serif CJK SC}{\setCJKmainfont{Noto Serif CJK SC}}
 {\IfFontExistsTF{SimSun}{\setCJKmainfont{SimSun}[AutoFakeBold=2]}{\setCJKmainfont{FandolSong-Regular}[AutoFakeBold=2]}}
\usepackage{hyperref}
\hypersetup{hidelinks,pdftitle={A股四因子月度研究},pdfauthor={}}
\pgfplotsset{compat=1.18}
\makeatletter
\@ifclassloaded{aomart}{\copyrightnote{}\doinumber{}\let\ps@firstpage\ps@plain}{}
\renewcommand{\abstractname}{摘要}
\renewcommand\section{\@startsection{section}{1}{\z@}{1.3\baselineskip}{.5\baselineskip}{\normalfont\fontsize{13}{16}\selectfont\bfseries\raggedright}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}{.6\baselineskip}{-1em}{\normalfont\bfseries}}
\makeatother
\setlength{\parindent}{2em}
\setlength{\parskip}{0pt}
\AtBeginDocument{\fontsize{11}{14}\selectfont}
\pagestyle{plain}
\title{A 股四因子月度研究\\历史指数成员、RankIC 与含成本分层回测}
\date{}
\begin{document}
\begin{abstract}
本文以 BaoStock 历史沪深 300 与中证 500 成员并集构造月度股票池，使用动量、反转、低波动和换手率四个信号，按等权截面排名形成综合分数. 回测覆盖 @@START@@ 至 @@END@@，共 @@DAYS@@ 个交易日、@@MONTHS@@ 次月度调仓. 本次正式结果中，Q5 总收益为 @@Q5_RETURN@@，相同交易约束与成本下的等权基准为 @@BENCH_RETURN@@；综合分数平均 RankIC 为 @@SCORE_IC@@. 本报告给出数据范围、信号时点、成交规则和结果口径，数值由项目结果文件生成.
\end{abstract}
\maketitle

\section{研究范围与样本}
研究对象为按历史信号日期查询的沪深 300 和中证 500 成员并集. 每个信号月分别取得两个指数的历史成员，按证券代码合并去重；当期股票池随着成员调整变化. 研究入口采用月末收盘信号、下一个交易日开盘成交. 2018 年 1 月建仓使用 2017 年 12 月末信号，最后一次建仓使用 2025 年 11 月末信号，净值继续估值至 2025 年最后一个交易日.

交易日历覆盖因子回看与回测区间. 各股票至少保留首次入池信号前 252 个交易日的行情，并延续至研究结束；已经下载的完整历史缓存直接复用. 未复权价格用于成交约束，复权价格用于收益和持仓估值. 复权事件按生效日期向后匹配，不使用晚于当前交易日的事件进行向前填充. 日线中的 \texttt{isST} 与 \texttt{tradestatus} 分别用于识别 ST 和交易状态.

\paragraph{合格样本.} 股票在信号日须属于历史成员并集，非 ST、交易状态正常，并且四个因子均有有限数值. 缺少日线时在公共交易日网格上保留缺失值，不用前值补齐因子窗口. 本次每月成员数量为 @@MEMBER_MIN@@--@@MEMBER_MAX@@ 只，合格数量为 @@ELIGIBLE_MIN@@--@@ELIGIBLE_MAX@@ 只，平均 @@ELIGIBLE_MEAN@@ 只. 共形成 @@STOCK_MONTHS@@ 个合格股票月，每月平均排除 @@EXCLUDED_MEAN@@ 只. 排除数量汇总记录的是最终合格条件的并集，不按原因重复累计.

本报告的研究范围为四因子最小可运行版本：先固定定义与交易规则，再观察因子排序和分层表现. 参数选择以本节和下节的预先规定为准；研究未增加八因子扩展或组合优化结果. 项目使用的技术口径和研究边界集中记录于 \href{../README.md\#局限}{README 的“局限”一节}.

\section{四因子与未来收益标签}
设 $P_{i,t}$ 为股票 $i$ 在交易日 $t$ 的后复权收盘价，$r_{i,t}=P_{i,t}/P_{i,t-1}-1$ 为相邻交易日日收益率，$T_{i,t}$ 为当日换手率. 下标按交易日而非自然日计数. 四个因子统一定向为数值越大越有利：
\begin{align*}
 M_{i,t}&=\frac{P_{i,t-21}}{P_{i,t-252}}-1, &
 R_{i,t}&=-\left(\frac{P_{i,t}}{P_{i,t-21}}-1\right),\\
 V_{i,t}&=-\operatorname{Std}_{\mathrm{sample}}(r_{i,t-59},\ldots,r_{i,t}), &
 U_{i,t}&=-\frac1{20}\sum_{j=0}^{19}T_{i,t-j}.
\end{align*}
动量跳过最近 21 个交易日，以便与短期反转采用不同窗口. 低波动使用最近 60 个日收益率的样本标准差，换手率使用最近 20 个交易日的均值. 各窗口只使用信号日及此前数据. 信号计算保留供应商给出的停牌观察值；真正缺失的观察仍为缺失.

\paragraph{综合分数.} 对同一个信号日的合格截面，分别计算四个因子的平均并列秩，并除以当期股票数得到百分位排名 $q^{(k)}_{i,t}$. 综合分数定义为
\[
 S_{i,t}=\frac14\bigl(q^{(M)}_{i,t}+q^{(R)}_{i,t}+q^{(V)}_{i,t}+q^{(U)}_{i,t}\bigr).
\]
排名变换统一量纲，四因子等权. 本版本不依据回测收益拟合权重，也不在分组前对综合分数重新优化. 相同因子值取平均名次；综合分数相同时，分层由证券代码作确定性排序.

\paragraph{未来收益与 RankIC.} 记 $e(t)$ 为信号日后的首个交易日，$e^+(t)$ 为再下个月的首个交易日，$O^{\mathrm{adj}}$ 为复权开盘价. 收益标签和截面 RankIC 为
\[
 y_{i,t}=\frac{O^{\mathrm{adj}}_{i,e^+(t)}}{O^{\mathrm{adj}}_{i,e(t)}}-1,
 \qquad
 \operatorname{RankIC}_{k,t}=\operatorname{Corr}\bigl(\operatorname{rank}(F^{(k)}_{i,t}),
 \operatorname{rank}(y_{i,t})\bigr).
\]
相关系数只用当期因子与标签均有效的证券. 至少需要两个有效配对，且因子秩和收益秩都不是常数. 标签不计交易成本和成交阻挡，因此用于描述排序关系；实际可成交净值由下一节独立生成. 当未来开盘价缺失时不补值. 2025 年 11 月末信号需要 2026 年 1 月开盘价才能完成标签，本次数据范围内该月 IC 留空，2025 年 12 月净值仍正常记录.

\section{组合构造与执行}
每个月按综合分数从低到高将合格证券分成数量尽量相等的五组，记为 Q1 至 Q5. Q1 对应最低分，Q5 对应最高分，各组按目标权重等权持有. 同时构造当期全部合格证券的等权基准. 基准使用与五组相同的信号日期、成交约束和交易费用，便于比较同一股票池内的排序效果. 该基准是研究构建的组合，不是指数公司公布的指数净值.

各组合从现金 1 开始，月初开盘先卖出超出目标的头寸，再按剩余现金买入目标缺口. 卖出受阻时保留原有持仓；资金不足时同比例缩减可执行买单. 买入和卖出按各笔实际成交额收取 @@COST@@ bps. 持仓使用复权价格和对应数量记账，每个交易日收盘更新净值. 上述六个多头组合的现金与证券数量均不得为负.

\paragraph{成交判断.} 开盘价和昨收采用未复权价格，涨跌停价格按分位四舍五入. 开盘达到涨停价时不买入，达到跌停价时不卖出. 主板按通常的 10\% 及 ST 5\% 幅度判断；科创板采用 20\%，创业板自 2020-08-24 起采用 20\%. 停牌或关键状态和价格缺失均阻止成交. 各订单的实际成交额、成本和受阻原因写入交易表.

\paragraph{指标口径.} 总收益从初始净值 1 计算，年化收益率按实际观察交易日数和每年 252 日换算. 年化波动率为日收益率样本标准差乘以 $\sqrt{252}$，Sharpe 使用零无风险利率. 最大回撤包含初始现金净值及首个交易日的收益和费用. 下表中收益、波动及回撤均为百分数，Sharpe 无量纲.

\section{回测结果与因子检验}
\paragraph{累计与年化表现.} 结果表逐项读取本次运行的绩效文件. @@QUINTILE_INTERPRETATION@@ Q5 的最大回撤为 @@Q5_DD@@.
@@PERFORMANCE@@
@@FIGURE@@
图 1 按同一日度净值 CSV 绘制黑白矢量曲线；横轴依实际交易日顺序展开. 对应的项目原始图为 \href{../results/figures/mvp_nav.png}{\texttt{mvp\_nav.png}}. 六条曲线均已扣除上述成交费用. 图中等权基准同样受调仓成交条件约束，不以理想化指数代替实际执行.

\paragraph{逐年表现.} 年收益使用当年最后一日净值除以上一年的最后一日净值，再减一；第一年以前述初始现金净值为起点. 最后一列为两组合当年收益率的算术差，单位为百分点，不代表另行模拟的可交易多空账户.
@@ANNUAL@@

\paragraph{RankIC 汇总.} ICIR 为月度 RankIC 均值除以月度样本标准差，不年化. 正值比例只统计有效月份中 IC 大于零的比例. 综合分数本次包含 @@SCORE_MONTHS@@ 个有效月份，每个有效月份平均有 @@PAIR_MEAN@@ 个收益配对. 四个单因子和综合分数使用相同的合格样本规则；具体有效配对数按月保留在 IC 明细表中.
@@IC@@
RankIC 与组合净值回答不同问题：前者描述下一持有期收益的截面排序关系，后者还叠加等权组合构造、持仓路径、成交限制和成本. 因此 IC 正负及大小不应直接换算成净值收益率，IC 汇总和执行结果需结合解读.

\section{成交与成本}
本次六组合合计产生 @@FILLED@@ 笔实际成交和 @@BLOCKED@@ 笔受阻尝试. 一次调仓可能分别记录同一证券的买入或卖出事件，笔数不等于股票数. 受阻尝试的成交额和费用均为零. 下表的累计成交额与费用均以各组合初始资金 1 为单位，并分别沿该组合自身净值路径累积.
@@EXECUTION@@
@@TURNOVER@@
累计成本是各期成交额乘以费率后的现金费用之和. 它不是相对于期末净值的损失比例，也不能直接从有成本组合终值中加回以得到无成本净值，因为交易费用还会改变后续可投资资金. 成交额合计同时包括买入与卖出，两侧均使用本次运行的固定费率.
@@OPTIONAL@@
@@COVERAGE@@

\Needspace{19\baselineskip}
\section{关键决策与复现}
\paragraph{决策一：历史成员与必要回看.} 每月使用当时查询到的指数成员，价格窗口保留首次入池前的充分回看期. 这一安排将当期资格与历史因子窗口分开：证券不必在整个回看期内一直是指数成员，只需在信号日满足成员与交易状态条件.

\paragraph{决策二：统一排名后等权.} 四个因子按同一合格截面排名，再取等权平均. 这使成交比较对应一个事先固定、容易解释的组合规则，避免因单位差异改变权重，也便于分别检查单因子 IC 与综合分数 IC.

\paragraph{决策三：信号、标签和成交分离.} 月末收盘信号只在下一交易日开盘执行；收益标签按未来开盘价计算，实际净值另行施加涨跌停、停牌和成本. 将三者分别输出，可以直接核对时间对齐及执行差异.

数据缓存齐备后，在项目根目录运行 \texttt{python run\_all.py --offline} 生成正式结果，再运行 \texttt{python report/build\_report.py} 生成本报告及可编辑源文件. 报告程序读取净值、绩效、样本、IC 与成交 CSV，核对所需文件和关键维度后才写入输出；本次文中所有实证数值均由这些输入格式化或按明确公式计算.

代码入口依次调用 \texttt{src/data.py}、\texttt{src/factors.py}、\texttt{src/backtest.py} 与 \texttt{src/evaluate.py}. 对应的关键检查位于 \texttt{tests/test\_mvp.py}；运行方式为 \texttt{python -m pytest -q}. 报告不把测试框架存在视为测试已通过，具体通过情况以实际终端输出为准. 原始结果文件集中在 \texttt{results/tables/}，图形集中在 \texttt{results/figures/}. 方法的适用边界统一参见 \href{../README.md\#局限}{README 的“局限”一节}.
\end{document}
'''
    for key, value in replacements.items():
        source = source.replace(key, value)
    source = re.sub(r'(\\paragraph\{[^}]+)\.\}', r'\1}', source)
    if '@@' in source:
        raise ValueError('报告模板仍有未替换字段.')
    return source


def compile_tex(source_path, output_path, engine='xelatex'):
    binary = shutil.which(engine)
    if binary is None:
        raise ValueError('未找到 XeLaTeX. 请安装 TeX Live 或使用 --tex-only 导出源码.')
    with tempfile.TemporaryDirectory(prefix='a-share-report-') as temporary:
        build = Path(temporary)
        for _ in range(2):
            process = subprocess.run([binary, '-interaction=nonstopmode', '-halt-on-error',
                                      '-no-shell-escape', '-output-directory', str(build),
                                      str(source_path.resolve())], cwd=build, capture_output=True,
                                     text=True, encoding='utf-8', errors='replace', timeout=180)
            if process.returncode:
                raise RuntimeError('XeLaTeX 编译失败：\n' + process.stdout[-14000:] + process.stderr[-4000:])
        log = (build / (source_path.stem + '.log')).read_text(encoding='utf-8', errors='replace')
        if 'Missing character:' in log or 'Overfull \\hbox' in log or 'Overfull \\vbox' in log:
            details = [line for line in log.splitlines() if any(key in line for key in
                       ['Missing character:', 'Overfull \\hbox', 'Overfull \\vbox'])]
            raise RuntimeError('报告排版需修正，未替换正式 PDF：\n' + '\n'.join(details))
        shutil.copyfile(build / (source_path.stem + '.pdf'), output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tex-only', action='store_true')
    parser.add_argument('--xelatex', default='xelatex', help='XeLaTeX executable name or path.')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        data = read_inputs(root)
        source = render_tex(data)
        target = root / 'report' / 'report.tex'
        target.write_text(source, encoding='utf-8')
        if not args.tex_only:
            compile_tex(target, target.with_suffix('.pdf'), args.xelatex)
            print(f'PDF: {target.with_suffix(".pdf")}')
        print(f'TEX: {target}')
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
