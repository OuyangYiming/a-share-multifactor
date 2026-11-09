# 数据源核查记录

核查日期：2026-11-09. 阶段 0 实际采用 AKShare / 腾讯；完整研究的主源选择仍以阶段 1 的字段与历史覆盖验收为准. 不要求用户购买数据或提供 Token 才能开展阶段 0.

## 本次实际数据来源

调用 `akshare.stock_zh_a_hist_tx(symbol='sh600000', start_date='20150101', end_date='20251231', adjust='hfq')`. SDK 通过腾讯 HTTPS 接口按窗口读取，合并后按用户请求日期筛选. 本模块保存日期和后复权收盘价，再补充请求对应的代码与复权标记. 实际行数、区间与 SHA256 见 `results/tables/example_summary.csv`，逐年行数见 `example_year_counts.csv`.

不同年份行数不同，不自动解释成停牌，也不自动补齐；需要阶段 1 用交易日历和停牌记录核查. AKShare 包装多个上游来源，不能由一个腾讯行情接口成功推断历史估值、历史行业与历史证券池都可得.

## BaoStock 备选源的核查记录

本机通过当前官方域名及其 DNS 地址尝试登录，均超时，未取得可用行情. 已停止请求；没有修改系统网络设置，也没有把失败的尝试写成数据成果.

官方 `query_history_k_data_plus()` 支持后复权日线，以及成交量、成交额、`tradestatus`、`isST`、`peTTM` 和 `pbMRQ`. 这些是文档层面的覆盖，未在本项目取得实际样本. 下表部分限制是 BaoStock 的已知接口限制，其余是所有候选源都需要验证的研究要求.

## 阶段 1 的必要核查

| 项目 | 核查要求 |
|---|---|
| 全 A 股范围 | 官方证券元信息描述沪深市场；北交所覆盖未确认. 覆盖不足时必须补充数据或征得范围调整，不能把沪深样本直接称作全 A 股. |
| 退市股票 | 使用历史日期股票列表合并证券集合，并抽查已退市证券. 不能以今天仍上市的名单回溯全部历史. |
| 总市值 | 日线没有直接总市值字段. 不能用后复权价格乘股本，也不能把流通市值当总市值. 季频股本推算日市值有近似误差与披露时点问题. |
| 行业分类 | 核对指定历史日期实际返回的 updateDate、分类体系与覆盖范围，不允许用今天的行业分类回填整个历史并声称无泄漏. |
| ST / 停牌 | 核查历史变更日期、停牌样本和缺失字段；当前名称不能替代历史 ST 标记. |
| 新股 | 用交易日历计算上市后的交易日数，不能直接用自然日差代替 60 个交易日. |
| 涨跌停 | 日线字段没有权威逐日涨跌停价及成交可行性. 需核对历史交易规则和证券状态；仅有收盘涨跌幅不足以证明可成交. |
| 估值 | 核查历史 PE / PB 的披露日和修订口径，不能由字段存在推断其完全符合时点可得信息. |
| 复权 | BaoStock 的涨跌幅复权口径可能与行情软件不同；抽查先比较不复权价，再核对复权算法. |
| 基准 | 沪深 300 为 sh.000300，中证 500 为 sh.000905. 指数不能请求其不存在的估值、ST 等字段. |

## 其他数据源

`yfinance` 是免费开源 Python 库，使用 Yahoo 的公开接口，库本身不是限时试用产品. 可用于行情练习，但其文档没有为本任务的全 A 历史证券池、历史 ST 与每日估值完整性提供验收依据，暂不选作主源.

Tushare 的数据接口存在积分或独立权限门槛. 如需切换，先列出所需接口与权限，并做少量调用验证，再决定是否需要付费；不因代码能导入就认定具备数据权限.

## 官方来源

- [AKShare 腾讯日线源码](https://github.com/akfamily/akshare/blob/main/akshare/stock/quote/stock_hist_tx.py)
- [AKShare 股票数据文档](https://akshare.akfamily.xyz/data/stock/stock.html)
- [BaoStock 日线](https://www.baostock.com/mainContent?file=stockKData.md)
- [BaoStock 证券列表](https://www.baostock.com/mainContent?file=StockBasicInfoAPI.md)
- [BaoStock 证券基本资料](https://www.baostock.com/mainContent?file=stockBasic.md)
- [BaoStock 行业分类](https://www.baostock.com/mainContent?file=stockIndustry.md)
- [BaoStock 数据口径](https://www.baostock.com/mainContent?file=dataExplain.md)
- [BaoStock 指数](https://www.baostock.com/mainContent?file=indexData.md)
- [BaoStock 访问规则](https://www.baostock.com/blacklist)
- [yfinance 官方仓库](https://github.com/ranaroussi/yfinance)
- [Tushare 权限说明](https://tushare.pro/document/1?doc_id=290)
