# 历史数据源核查与剩余条件

核查日期：2026-11-09. 证据摘要为 `results/source_discovery.json` 和 `results/tables/source_discovery.csv`. 实际响应与校验值保存在本地 `data/source_probes/`，不上传 GitHub. 下列接口仅做小规模探测，尚未纳入正式研究样本；返回成功不等于历史信息在当时可得.

| 数据项 | 实测发现 | 后续要求 |
|---|---|---|
| 历史市值、PE/PB | 东财 `stock_value_em('600000')` 从 2018-01-02 起有记录，原始 API 总数与 SDK 返回数一致. | 尚缺此股 2015–2017，其他股票和财务修订口径未验证. 不回填，不默默更改研究区间. |
| 沪深退市名单 | 两所的退市列表可取得上市、退出日期. | 过滤 B 股，核验上市事件、代码变更与重新上市，检查退市前行情是否仍可下载. 当前列表不能单独证明历史证券池完整. |
| 深圳历史 ST 候选 | 交易所简称变更表有日期、旧简称和新简称. | 可尝试构造事件区间，但须验证生效日、完整性以及沪市和北交所覆盖. 无事件不能推断“从未 ST”. |
| 历史停牌 | `stock_tfp_em(date='20151221')` 的响应包含 2026 年停牌记录. | 已否定其作为查询日精确快照的用法. 不按查询日期给全部返回记录贴标签，也不把未返回理解成未停牌. |
| 历史行业 | 巨潮试点返回多种分类体系各自的记录，没有确立 2015/2018 基准状态. | 不混合分类体系，也不向记录日期以前填充. 申万历史表的本次连接证书校验失败，未取得内容. |

沪市退市接口中，上游字段 `DELIST_DATE` 被当前 SDK 命名为“暂停上市日期”. 后续规范化必须核对原始字段含义，不能仅按中文列名处理为暂停上市事件.

当前免费来源足以继续做退市主表规范化、深圳简称事件区间与小样本估值接入，尚不足以确认完整的全 A 股历史样本可用于回测. 如后续使用已有的数据账号或导出文件，应先检查历史范围、权限和点时口径；本次没有购买数据、验证账号权限或收集 Token.

官方候选接口及权限页面记录在 `results/source_discovery.json` 的 `permission_documentation_only` 中. Tushare 的历史 ST、每日停复牌、每日指标与历史行业成员属于候选接口，不构成已取得数据的声明. RQData 的接口文档也不能证明试用或套餐已开放这些历史范围.

下一小任务是把交易所退市名单规范化为明确的上市区间，并以退市个股检验历史行情可用性. 若补充资料仍不能确定某日的 ST、停牌或行业状态，筛选器继续保留未知.

官方来源：[AKShare 股票资料实现](https://github.com/akfamily/akshare/blob/main/akshare/stock/issue/stock_info.py)、[历史估值实现](https://github.com/akfamily/akshare/blob/main/akshare/stock/valuation/stock_value_em.py)、[停复牌实现](https://github.com/akfamily/akshare/blob/main/akshare/stock/issue/stock_tfp_em.py)、[巨潮行业实现](https://github.com/akfamily/akshare/blob/main/akshare/stock/finance/stock_industry_cninfo.py).
