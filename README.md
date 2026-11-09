# A 股多因子研究与回测

仓库：[OuyangYiming/a-share-multifactor](https://github.com/OuyangYiming/a-share-multifactor)（私有）.

本项目按任务书分阶段实现，由本人学习和验收，Codex 协助编码与测试. 当前实现阶段 0 的单只股票日线示例及阶段 1 的五股数据试点，尚无因子有效性、策略收益或样本外研究结论.

## 环境与运行

锁定依赖在 Windows、Python 3.12 上验证. 任务书允许 Python 3.10 及以上，但本锁定环境不保证支持 3.10，复现请使用 Python 3.12. 首次下载需要联网，后续可使用本地缓存.

取得仓库后在项目根目录运行. 克隆私有仓库需要 GitHub 账号具备访问权限，并在使用 Git 的电脑上完成认证；应用中的 GitHub 连接与本机 Git 凭据分别管理. Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_all.py
.\.venv\Scripts\python.exe -m pytest -q
```

macOS / Linux：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_all.py
.venv/bin/python -m pytest -q
```

非 Windows 平台尚未实测. 安装依赖后，一条 `python run_all.py` 命令复现**阶段 0 和阶段 1 五股试点**，不是整个研究项目. 有全部缓存时可运行 `python run_all.py --offline`，确认不连接数据服务. 单独执行用 `--stage stage0` 或 `--stage stage1-pilot`.

## 示例数据和输出

- 来源：AKShare 的腾讯日线接口 `stock_zh_a_hist_tx`，证券代码 `sh600000`，请求区间为 2015-01-01 至 2025-12-31，`adjust="hfq"`（后复权）.
- 保存字段：交易日、证券代码、后复权收盘价、内部复权标记. 代码统一为 `sh.600000`，内部标记 `1` 表示本次请求后复权；二者由请求参数补充，不能视为对上游返回值的独立核验.
- 原始数据存入 `data/raw/`，该目录被 `.gitignore` 排除. 此示例只包含指定股票，并未完成全市场选样、ST / 停牌 / 新股剔除.
- 成功运行后生成 `results/figures/example_close.png`、`results/tables/example_summary.csv`、`results/tables/example_year_counts.csv` 与 `results/provenance.json`.
- `config/example_snapshot.json` 保存首次有效快照的请求参数及 SHA256. 后续缓存或重新下载数据若不匹配，会报错并停止，不会默默改变参考结果. 数据服务如修订历史，需要保留旧快照并单独审查，不能保证远端永不变化.
- 后复权价用于价格序列分析，不是历史上可以实际成交的报价. 此图也不是投资组合净值图.

## 目录

```text
config/       示例快照参数与校验值
data/         本地行情缓存，不提交 Git
docs/         阶段讲解、验收记录、后续任务与数据限制
notebooks/    交互练习目录，当前尚无 notebook
report/       最终 LaTeX 报告目录，尚未编写正式研究报告
results/      已实际运行的表、图、来源和验证记录
src/          example.py 单股示例；data.py 五股双复权口径试点与覆盖审计
tests/        使用明确标注的构造数据检验程序边界
requirements.txt
run_all.py
```

## 学习与验收

先阅读 `docs/stage0_walkthrough.md`，再按 `docs/acceptance.md` 记录运行与口头解释. 自动测试通过不能代替本人能讲清楚代码. 五股数据试点的说明和讲解见 `docs/stage1_data.md`；它不是全 A 股样本，也没有完成 ST、停牌及上市不足 60 个交易日的剔除.

试点固定浦发银行、平安银行、贵州茅台、万科 A、宁德时代，分别获取不复权与后复权数据，核对日期集合并保留两种价格. 每股逐年覆盖、字段缺失及阶段未完成项保存在 `results/tables/stage1_*` 和 `results/stage1_audit.json`. 这些是指定样本的下载记录，不是随机抽查、外部行情核对或完整交易日覆盖证明. 未取得的历史状态保持未知，未输出虚构的全市场剔除数量.

## 当前局限

1. 尚未完成全市场覆盖、历史退市股票和时点行业分类验收，不能排除幸存者偏差与分类泄漏.
2. 尚无信号、交易或回测模块；当前测试不构成无未来函数证明.
3. 数据下载依赖外部服务及其网络可达性. 本地缓存可固定输入，重新下载不保证服务方历史数据不变.
4. 待核查每日总市值、估值的时点口径、涨跌停可成交约束及北交所覆盖. 详见 `docs/data_source_review.md`.
5. 尚未由另一位同学从零复现；未进行简历成果描述或网站发布.

官方接口：[AKShare 文档](https://akshare.akfamily.xyz/data/stock/stock.html)，[腾讯日线实现](https://github.com/akfamily/akshare/blob/main/akshare/stock/quote/stock_hist_tx.py). BaoStock 在本机连接测试中超时，未用于本次行情输出.
