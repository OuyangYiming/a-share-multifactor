"""MVP monthly average-rank IC and descriptive, nonannualized IC summaries."""

import numpy as np
import pandas as pd

from src.factors import FACTOR_COLUMNS, _keys

IC_FACTORS = FACTOR_COLUMNS + ['score']
IC_COLUMNS = ['date', 'factor', 'rank_ic', 'n']
SUMMARY_COLUMNS = ['factor', 'mean_rank_ic', 'ic_std', 'ic_ir', 'positive_fraction', 'months']


def _numeric(values: pd.Series, name: str) -> pd.Series:
    """Convert a numeric series while retaining NA and rejecting infinity."""
    data = pd.to_numeric(values, errors='raise').astype(float)
    if not np.isfinite(data.dropna()).all():
        raise ValueError(f'{name} must contain finite numbers or missing values.')
    return data


def calculate_rank_ic(signals: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.DataFrame:
    """Join on signal date/code and calculate Spearman correlations via average ranks.

    The caller supplies forward returns beginning strictly after signal close.
    This evaluator neither constructs nor fills labels. Every signal date gets
    one row per four factors plus score, including NA when labels are missing.
    n is the number of finite pairs for that factor, not the membership count.
    At least two pairs and nonconstant ranks on both sides are required.
    """
    features = _keys(signals, 'date', 'signals')
    labels = _keys(forward_returns, 'date', 'forward_returns')
    if not set(IC_FACTORS).issubset(features) or 'forward_return' not in labels:
        raise ValueError('Signals need four factors and score; labels need forward_return.')
    for factor in IC_FACTORS:
        features[factor] = _numeric(features[factor], factor)
    labels['forward_return'] = _numeric(labels['forward_return'], 'forward_return')
    pairs = features[['date', 'code'] + IC_FACTORS].merge(
        labels[['date', 'code', 'forward_return']], on=['date', 'code'],
        how='left', validate='one_to_one')
    rows = []
    for date, group in pairs.groupby('date', sort=True):
        for factor in IC_FACTORS:
            complete = group[[factor, 'forward_return']].dropna()
            n = len(complete)
            correlation = np.nan
            if n >= 2 and complete[factor].nunique() > 1 and complete['forward_return'].nunique() > 1:
                factor_rank = complete[factor].rank(method='average')
                return_rank = complete['forward_return'].rank(method='average')
                correlation = float(factor_rank.corr(return_rank))
            rows.append({'date': date, 'factor': factor, 'rank_ic': correlation, 'n': n})
    return pd.DataFrame(rows, columns=IC_COLUMNS)


def summarize_ic(ic: pd.DataFrame) -> pd.DataFrame:
    """Summarize valid months using sample std and nonannualized mean/std ICIR.

    months counts nonmissing monthly IC values. Positive_fraction uses strictly
    positive valid months. Fewer than two months or zero std gives an NA ICIR.
    """
    if not isinstance(ic, pd.DataFrame) or not ic.columns.is_unique or not {'date', 'factor', 'rank_ic'}.issubset(ic):
        raise ValueError('Expected date, factor and rank_ic columns.')
    data = ic.copy()
    data['date'] = pd.to_datetime(data['date'], errors='raise')
    if data['date'].isna().any() or data['date'].dt.tz is not None or not data['date'].eq(data['date'].dt.normalize()).all():
        raise ValueError('IC dates must be nonmissing, timezone-naive daily dates.')
    if not data['factor'].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError('Factor names must be nonempty strings.')
    if data.duplicated(['date', 'factor']).any():
        raise ValueError('Duplicate date/factor IC observations.')
    data['rank_ic'] = _numeric(data['rank_ic'], 'rank_ic')
    if data['rank_ic'].dropna().abs().gt(1 + 1e-12).any():
        raise ValueError('Rank IC outside [-1, 1].')
    rows = []
    for factor, group in data.groupby('factor', sort=False):
        values = group['rank_ic'].dropna()
        months = len(values)
        mean = float(values.mean()) if months else np.nan
        spread = float(values.std(ddof=1)) if months >= 2 else np.nan
        ratio = mean / spread if months >= 2 and spread > 0 else np.nan
        rows.append({'factor': factor, 'mean_rank_ic': mean, 'ic_std': spread,
                     'ic_ir': ratio, 'positive_fraction': float(values.gt(0).mean()) if months else np.nan,
                     'months': months})
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
