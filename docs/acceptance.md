# 阶段 0 验收记录

验证日期：2026-11-09. 阶段 0 的本地代码和真实数据示例已运行，已创建私有远程仓库；本人理解验收仍待完成. 阶段 1 目前为五股数据试点.

| 验收项 | 当前状态与依据 |
|---|---|
| 独立 Python 环境 | 已安装，`pip check` 通过，版本见 `results/environment.json`. |
| 目录与依赖 | 已建立任务书要求的目录，`requirements.txt` 锁定当前依赖. |
| 真实日线 | AKShare / 腾讯下载成功，区间与行数见 `results/tables/example_summary.csv`. |
| 价格图 | 已生成并目视检查 `results/figures/example_close.png`. |
| 缓存与来源 | 已写本地 data/ 缓存，记录请求参数、包版本、时间和 SHA256. |
| 数据结构 | 示例没有重复键、非正或缺失收盘价. 这不等于交易日覆盖完整或行情软件抽查通过. |
| 自动测试 | 通过，数量与失败数见 `results/test_summary.json`. 测试覆盖数据边界、接口错误、缓存与校验值. |
| 离线复现 | 已运行 `python run_all.py --offline`，读取同一缓存成功. |
| 独立副本 | 本机新克隆、新 Python 环境、无数据缓存时重新下载成功；原始数据、图和两张表逐字节一致. 测试通过. 详见 `results/reproduction_check.json`. |
| GitHub 仓库 | 已创建私有仓库 https://github.com/OuyangYiming/a-share-multifactor，并配置本地 origin. |
| 本人解释 DataFrame | 待本人验收，练习见 `docs/stage0_walkthrough.md`. |
| 同学从零复现 | 尚未执行，不以自动测试替代. |
| 阶段 1 | 五股双价格口径下载、缓存与覆盖审计已实现，详见 `docs/stage1_data.md` 与 `results/stage1_audit.json`；全市场数据和筛除仍未完成. |
| 阶段 2–5 | 尚未实现. |

源数据接入记录：先尝试 BaoStock，登录超时后停止；随后改用 AKShare / 腾讯取得真实数据. 本次结果中没有 BaoStock 行情，也没有合成行情. `tests/` 中的构造数据仅用于单元测试.

独立环境测试的首次运行继承了联网进程权限，无法访问 pytest 默认临时目录，出现测试初始化错误. 改为普通本地进程后全部测试通过，保留一项旧缓存目录不可写的警告. 初次失败与重跑记录均保存在复现记录中；不将其隐藏或解释成数据质量验证.

阶段结束检查：当前尚无信号与交易逻辑，不能宣称未来函数测试通过；本例有意选取一只现存股票，仅演示数据读取，不能用于全市场因子推断. 正式样本、历史信息可得性和幸存者偏差在阶段 1 开始单独验收.
