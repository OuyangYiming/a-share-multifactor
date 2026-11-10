import numpy as np
import pandas as pd
import pytest

from scripts.download_all_a import (Downloader, PRICE_COLUMNS, adjusted_daily,
                                    select_a_shares, uncovered_blocks, validate_raw)


def raw(code, days, close=10):
    result = pd.DataFrame({name: ['0'] * len(days) for name in PRICE_COLUMNS})
    result['date'], result['code'] = list(days), code
    for name in ['open', 'high', 'low', 'close', 'preclose']:
        result[name] = str(close)
    result['tradestatus'] = '1'
    return result


def test_historical_pool_keeps_delisted_st_and_suspended_but_excludes_other_securities():
    old = pd.DataFrame({
        'code': ['sh.600069', 'sz.000001', 'sh.688001', 'sz.302132',
                 'sh.000001', 'sz.399001', 'sh.510300', 'sz.159919',
                 'sh.900901', 'sz.200002', 'sh.689009', 'bj.920002'],
        'tradeStatus': ['1', '0'] + ['1'] * 10,
        'code_name': ['退市银鸽', '*ST历史名称'] + ['其他'] * 10})
    selected = select_a_shares(old)
    assert set(selected.code) == {'sh.600069', 'sz.000001', 'sh.688001', 'sz.302132'}
    assert selected.set_index('code').loc['sz.000001', 'tradeStatus'] == '0'
    later = select_a_shares(old.loc[~old.code.eq('sh.600069')])
    history = pd.concat([selected.assign(signal_date='2017-12-29'),
                         later.assign(signal_date='2025-11-28')])
    assert 'sh.600069' in set(history.code)
    assert 'sh.600069' not in set(later.code)


def test_late_starting_cache_needs_backfill_and_invalid_response_is_not_saved(tmp_path):
    calendar = pd.bdate_range('2020-01-01', periods=12)
    missing = uncovered_blocks(calendar, '2020-01-01', '2020-01-16',
                               [('2020-01-07', '2020-01-10'), ('2020-01-13', '2020-01-16')])
    assert missing == [('2020-01-01', '2020-01-06')]
    assert uncovered_blocks(calendar, '2020-01-01', '2020-01-16',
                            [('2020-01-01', '2020-01-16')]) == []
    downloader = Downloader(tmp_path)
    downloader.folder.mkdir(parents=True)
    class WrongCodeClient:
        def query(self, *args, **kwargs):
            return raw('sh.600004', ['2020-01-01'])
    downloader.client = WrongCodeClient()
    path = downloader.folder / 'raw/sh.600000_2020-01-01_2020-01-16.csv'
    validator = lambda x: validate_raw(x, 'sh.600000', '2020-01-01', '2020-01-16')
    with pytest.raises(ValueError, match='wrong code'):
        downloader.cached(path, 'query_history_k_data_plus', validator)
    assert not path.exists()
    assert len(pd.read_csv(downloader.folder / 'download_log.csv')) == 1


def test_adjustments_use_only_effective_events_and_reject_conflicting_overlap():
    code = 'sh.600000'
    prices = raw(code, ['2020-01-02', '2020-01-03', '2020-01-06'])
    factors = pd.DataFrame({'code': [code] * 3,
                            'dividOperateDate': ['2019-12-31', '2020-01-06', '2020-02-03'],
                            'backAdjustFactor': ['2', '3', '99']})
    result = adjusted_daily([prices], factors, code, '2020-01-01', '2020-01-31')
    np.testing.assert_array_equal(result.adj_close, [20, 20, 30])
    changed = factors.copy()
    changed.loc[2, 'backAdjustFactor'] = '999'
    pd.testing.assert_frame_equal(result, adjusted_daily(
        [prices, prices], changed, code, '2020-01-01', '2020-01-31'))
    with pytest.raises(ValueError, match='conflicting'):
        adjusted_daily([prices, raw(code, ['2020-01-03'], close=11)], factors,
                       code, '2020-01-01', '2020-01-31')
