# 交易日历、基准与上市日期

本模块补齐五股试点需要的交易日历、沪深 300 与中证 500 指数日线、五股上市日期. 它不提供历史 ST、停牌、行业、市值估值或完整退市证券池.

## 运行与数据位置

```bash
python -m src.reference_data
python -m src.reference_data --offline
python -m pytest tests/test_reference_data.py -q
```

调用入口为 `run_reference(root, offline=False)`，返回与 `results/reference_audit.json` 相同的审计字典. 请求串行执行，各 HTTP 连接/读取的超时为 15 秒；函数结束或异常时恢复 SDK 原超时设置. 该数值不是整批任务总时限.

原始 SDK 表与其来源记录位于 `data/reference/raw/`. 它们不是原始 HTTP 字节. 清洗后的数据位于 `data/reference/processed/`：`trading_calendar.csv` 只有日期列 `date`；`benchmark_daily.csv` 包含日期、指数代码和 OHLC；`pilot_listing_dates.csv` 包含证券代码、名称、上市日期与来源接口.

`results/reference_provenance.json` 记录每份快照的来源、请求参数、SDK 版本、下载时间和 SHA256. 本地 CSV、来源记录必须配对且校验一致；重新下载也须符合可提交的公共参考清单，不接受上游静默修订. 当前证券资料会随新上市和更名而变化；即使五股上市日期未变，整张来源表变化也可能触发校验失败. 应保留原始缓存，另行审查新快照，不能删除参考清单来掩盖变化.

## 来源与实际覆盖

| 数据 | 实际来源 | 本次结果 |
|---|---|---|
| 交易日历 | `tool_trade_date_hist_sina`，新浪日历 | 1990-12-19 至 2026-12-31，共 8797 个日期；2015–2025 年有 2674 个日期. |
| 沪深 300 | `stock_zh_index_daily_tx(symbol='sh000300', start_date='20150101', end_date='20251231')`，腾讯 | 2674 条，2015-01-05 至 2025-12-31. |
| 中证 500 | 同一接口，代码 `sh000905` | 2674 条，2015-01-05 至 2025-12-31. |
| 上海股票资料 | `stock_info_sh_name_code(symbol='主板A股')`，上交所 | 缓存当前表，提取浦发银行和贵州茅台的上市日期. |
| 深圳股票资料 | `stock_info_sz_name_code(symbol='A股列表')`，深交所 | 缓存当前表，提取平安银行、万科 A 和宁德时代的 A 股上市日期. |

覆盖数来自 `results/reference_audit.json` 和 `results/tables/reference_benchmark_summary.csv`. 指数每年行数、日历每年日期数分别见 `reference_benchmark_year_counts.csv`、`reference_calendar_year_counts.csv`.

两条指数在 2015–2025 年的日期集合与新浪日历完全一致. 这是一项跨供应商日期交叉检查，不是独立行情软件价格核对，也不是交易所对全部历史交易日的逐日认证. 日历中的 2026 年末部分包含预排日期，正式研究区间仍截至 2025-12-31.

新浪接口的锁定 SDK 明确补入了上游缺少的 1992-05-04；本项目没有再次补日期、按工作日猜测节假日或插值价格. 该日历上游文件名为 `klc_td_sh.txt`，早期沪深交易安排是否逐日一致尚未独立核查. 本模块记录这一限制，不把早期深圳上市满 60 日的推算日期当作已获交易所认证的日期. 对本试点中早已上市的股票，这一早期日期差异不影响其在 2015 年起已超过 60 个交易日的结论.

指数接口内部请求 `qfq`，返回供应商指数 OHLC 点位. 本模块按价格指数资料使用，不宣称为含分红再投资的总收益指数，不把指数点位当作股票成交价格，也不利用指数的缺失 ST、换手率等字段推断证券状态.

## 五股上市日期与 60 日约定

| 证券代码 | 上市日期 | 按本日历推算的第 60 个交易日 |
|---|---|---|
| sh.600000 | 1999-11-10 | 2000-02-18 |
| sh.600519 | 2001-08-27 | 2001-11-23 |
| sz.000001 | 1991-04-03 | 1991-06-26 |
| sz.000002 | 1991-01-29 | 1991-04-24 |
| sz.300750 | 2018-06-11 | 2018-09-03 |

上市日期证据见 `results/tables/reference_listing_dates.csv`，计算结果见 `reference_listing_60th_session.csv`. 上市日按第 1 个交易日计，因此第 60 个交易日起满足“上市至少 60 个交易日”的约定. 这是按交易所日历累计的上市年龄，不是股票实际有成交记录的 60 行；停牌不从市场日历中删掉. 日历必须覆盖上市日期，且上市日期必须出现在日历中，否则程序报错.

上海、深圳来源都是下载时的现行证券资料. 这里仅提取指定股票的名称和上市日期，当前名称用于识别展示，不能代表整个历史区间的证券名称. 当前行业、股本或状态没有进入输出，不能用这两张现行名单回溯成历史完整证券池.

## 校验与代码说明

`validate_calendar()` 要求日期非空、有效且唯一，保留来源日期并排序. `validate_benchmark()` 要求指数代码在指定范围内、日期唯一且位于请求区间，OHLC 为正有限数，并满足最低价与最高价约束.

`extract_listings()` 按五个目标代码筛选两张交易所资料. 任一股票缺失、重复、名称缺失或上市日期无效，都不能继续输出成功结果. 不以接口返回首日替代上市日期.

`load_reference_snapshot()` 对照请求与本地及公共 SHA256，保证当前运行使用的是已记录的输入. `check_reference_alignment()` 比较每条指数与日历的日期集合，并定位按约定计算的第 60 个交易日. 缺少某个日期时直接失败，不删除不一致记录后声称对齐.

当前测试使用构造表检查日期与价格错误、五股上市资料缺失和重复、缓存篡改、超时设置恢复及第 60 日边界. 构造表只存在于测试临时目录，不作历史证据. 测试记录见 `results/reference_test_summary.json`.

官方实现：[新浪日历](https://github.com/akfamily/akshare/blob/main/akshare/calendar/calendar_trade_date_hist.py)、[腾讯指数](https://github.com/akfamily/akshare/blob/main/akshare/index/index_stock_zh.py)、[沪深证券资料](https://github.com/akfamily/akshare/blob/main/akshare/stock/issue/stock_info.py).
