# 中国发债主体信用风险预警模型（Credit Risk Early Warning）

> 基于公开 A股上市公司财务数据构建的信用风险事件预警模型。
> 平台层基于开源 [MARVIS-Agent](https://github.com/eddyzzl/marvis-risk-agent)（MIT License, v2.3.2）二次开发；
> 数据管道、标签构建、特征工程、分析脚本与报告为本人自建。

一个对信用风险定价与债券市场研究的兴趣项目：用免费公开数据源（akshare / 同花顺 iFinD / 巨潮）构建
"发债主体信用风险事件预警评分"，定位为信用研究/风控流程中的**量化初筛工具**（"量化初筛＋基本面终审"）。

## 📄 项目报告

| 报告 | 内容 | 篇幅 |
|---|---|---|
| [📕 项目报告（详细教学版）](docs/项目报告-发债主体信用风险预警模型.pdf) | 从零讲起：数据/标签/特征逐个解释、模型原理（WOE/逻辑回归/评分卡/XGBoost）、评价指标含义、代码难点与bug修复 | 20页 |
| [📊 结果报告（图表版）](docs/结果报告-模型可视化.pdf) | ROC/KS/分数分布/特征重要性/校准趋势/行业画像/预警清单/平台验证 全套图表 | 10页 |

> 报告为 PDF 格式，GitHub 可直接在线预览。

---

## 功能特性

- **数据管道**：akshare 批量抓取 5,468 家 A股非金融公司 2019–2025 年报财务指标（37,956 行面板），断点续跑/限速/重试；
- **标签构建**：综合信用风险事件——首次年度亏损（财务困境学术代理，时点对齐无前视偏差）＋ 债券违约名单（公开可验证子集）；正样本 1,986（5.23%）；
- **特征工程**：49 个特征（盈利/偿债/现金流质量/营运/成长/规模＋同比趋势Δ＋缺失指示），IV 筛选后 25+ 入模；
- **双模型**：WOE 逻辑回归（可解释评分卡）＋ XGBoost（效果更优），时间序列切分（2019-22 训练 / 2023 验证 / 2024+25 样本外）；
- **平台验证**：Notebook 契约（RMC_SCORE_FN）+ PMML 一致性验证（37,956 行零失败）+ 独立效果/稳定性/压力测试；
- **分行业风险画像**：行业×年份风险矩阵、出险公司财务画像、所有制交叉分析；
- **预警清单**：最新一期财报打分 → 风险最高 TOP50（含主要风险特征标注）。

## 模型结果

| 模型 | 测试集（2024）AUC | KS |
|---|---|---|
| WOE 逻辑回归 | 0.864 | 0.596 |
| XGBoost | 0.885 | 0.639 |

平台独立验证（XGB / PMML）：

| 数据集 | AUC | KS |
|---|---|---|
| 训练（2019-2022） | 0.912 | 0.671 |
| 开发验证（2023） | 0.850 | 0.555 |
| 样本外（2024+2025） | 0.882 | 0.632 |

- 月度 PSI vs 训练：0.02（稳定）
- 压力测试：剔除盈利类特征后 KS 0.632→0.106（Δ-0.526）——模型核心依赖盈利质量，与信用分析先验一致

### 分行业风险画像（核心发现）

1. 高风险行业：建筑装饰（13.2%）、影视传媒（11.8%）、钢铁（10.6%）、房地产（10.5%）、商务服务/软件/环保/纺织（8-9%）——地产链＋周期；
2. **出险公司画像**：ROE / 净利率 / 每股收益较正常公司**腰斩**（3.4 vs 7.7 / 4.3 vs 8.1 / 0.18 vs 0.40），经营现金流/负债**减半**（0.068 vs 0.140），而**杠杆水平差异很小**（资产负债率 40.1% vs 39.8%）——盈利与现金流质量的恶化是比杠杆更早、更强的预警信号；
3. 所有制：民企 6.5% > 国企 5.8% > 无实控人 4.3%（与联合资信"2025 年新增违约主体 69% 为民营企业"方向一致）。

## 技术栈

- Python 3.12（conda env: marvis）；pandas / scikit-learn / xgboost
- 数据：akshare（财务）、同花顺 iFinD HTTP API（实控人批量）、巨潮（证监会行业）
- 平台：MARVIS-Agent v2.3.2（确定性计算内核：分箱/IV/KS/PSI/压力测试；PMML 验证工作流；Excel/Word 报告生成）
- 评分卡刻度手写实现（PDO=20, odds=1:1→600）

## 目录结构

```
├── README.md
├── docs/
│   └── 项目报告-发债主体信用风险预警模型.pdf   # 完整项目报告（数据/方法/结果/画像/预警/局限）
├── scripts/start_marvis.sh          # 启动平台工作台
├── tools/test_ifind.py              # iFinD HTTP API 测试脚本
└── zh_credit/                       # 自建增量层（数据/标签/特征/分析）
    ├── data_fetch.py                # akshare 财务面板抓取（断点续跑）
    ├── label_build.py               # 综合风险事件标签（时点对齐）
    ├── features.py                  # 特征工程（趋势Δ/缺失指示/缩尾）
    ├── ownership_fetch.py           # iFinD 实控人批量抓取＋所有制分类
    ├── fetch_industry_cninfo.py     # 巨潮证监会行业映射
    ├── industry_map.py              # 申万行业映射（备用）
    ├── audit_data.py                # 数据质量验收
    ├── build_modeling_notebook.py   # 建模 Notebook 生成器
    ├── industry_analysis.py         # 分行业风险画像
    ├── warning_list.py              # 预警清单 TOP50
    ├── notebooks/modeling.ipynb     # 建模 Notebook（RMC 契约）
    ├── pmml/model.pmml              # 平台验证用 PMML
    ├── data/ownership_map.csv       # 实控人映射（iFinD，注意数据条款）
    └── output/                      # 画像与预警产出（CSV/PNG）
```

## 复现

```bash
conda activate marvis   # 或任意 python 3.12 环境
pip install akshare xgboost pandas scikit-learn

cd zh_credit
python data_fetch.py --start-year 2019 --output data/financial_panel.csv
python label_build.py --input data/financial_panel.csv \
    --default-list data/bond_default_list.csv --output data/panel_labeled.csv
python features.py --input data/panel_labeled.csv --output data/modeling_dataset.csv
python build_modeling_notebook.py && jupyter execute notebooks/modeling.ipynb
python industry_analysis.py
python warning_list.py
```

平台验证（需另行部署 MARVIS-Agent）：

```bash
bash scripts/start_marvis.sh        # http://127.0.0.1:8899
# 创建验证任务（notebook + 样本 + PMML + 数据字典）→ 契约确认 → marvis validate <task_id>
```

## 数据与诚实声明

- 数据源：akshare（免费）、同花顺 iFinD（试用账号，实控人数据）、巨潮（行业分类）；
- 样本为上市公司（现存量主体），**城投/非上市发债主体不在范围**——模型结论不可外推至城投债定价；
- 主标签为首次亏损（困境早期信号），与 ST/违约存在时间差；违约名单为公开可验证的 A股子集（中国债市累计 309 家违约发行人的一部分）；
- 模型无法识别报表粉饰与表外负债——定位是初筛工具，需与舆情、审计意见、实地调研结合；
- `data/ownership_map.csv` 来自同花顺 iFinD 试用接口，请遵守同花顺数据服务条款。

## 致谢

- [MARVIS-Agent](https://github.com/eddyzzl/marvis-risk-agent)（MIT）：风控建模与验证平台（分箱/IV/KS/PSI/压力测试/PMML 验证/报告生成）；
- [AKShare](https://github.com/akfamily/akshare)：A股财务数据接口。

## License

本项目代码遵循 MIT License。上游 MARVIS-Agent 遵循其 MIT License。
