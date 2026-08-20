# -*- coding: utf-8 -*-
"""
build_modeling_notebook.py — 生成 W2 建模 Notebook（modeling.ipynb）

流程（参考 MARVIS 契约推荐结构与评分卡标准方法论）：
  1. 环境与参数
  2. 数据读取（modeling_dataset_full.csv）
  3. 样本处理（缺失/切分：2019-2022训练、2023验证、2024测试）
  4. 特征处理（等频WOE分箱 + IV筛选，手写实现，训练集拟合防泄漏）
  5. 模型训练（WOE逻辑回归 主模型 + XGBoost 对照）
  6. 模型评估（AUC/KS 三集对比）
  7. 评分卡刻度（PDO=20, odds=1:1→600分）
  8. 平台验证契约（RMC_* 变量）

用法：python build_modeling_notebook.py && jupyter execute modeling.ipynb
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "language": "python", "display_name": "Python 3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

cells = []

# ============ 1. 环境与参数 ============
cells.append(nbf.v4.new_markdown_cell(
"""# 1. 环境与参数

中国发债主体信用风险预警模型（基于开源 MARVIS-Agent 二次开发项目的数据层产出）
- 标签：综合信用风险事件（首次年度亏损 + 债券违约），时点对齐无前视偏差
- 主模型：WOE 逻辑回归（可解释、银行风控标准）
- 对照模型：XGBoost
- 时间切分：2019-2022 训练 / 2023 验证 / 2024 测试（金融时序纪律，不用随机KFold）"""))

cells.append(nbf.v4.new_code_cell(
"""import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
DATA_PATH = os.environ.get("ZH_DATA", "data/modeling_dataset_full.csv")
RANDOM_STATE = 42
N_BINS = 10           # 等频分箱箱数
IV_THRESHOLD = 0.02   # 特征筛选阈值
PDO = 20              # 分数翻倍间隔
BASE_ODDS = 1.0       # 基准 odds
BASE_SCORE = 600.0    # 基准分数"""))

# ============ 2. 数据读取 ============
cells.append(nbf.v4.new_markdown_cell("# 2. 数据读取"))

cells.append(nbf.v4.new_code_cell(
"""df = pd.read_csv(DATA_PATH, dtype={"code": str})
df["code"] = df["code"].str.zfill(6)
print(f"样本: {df.shape}，公司: {df['code'].nunique()}，正样本率: {df['label'].mean():.2%}")
print(f"年份: {df['year'].min()}-{df['year'].max()}")

# 建模特征：数值列（排除 ID/标签/辅助列）
EXCLUDE = ["code", "name", "year", "label", "first_loss_year",
           "default_year", "first_consec_loss_year",
           "industry_csrc", "actual_controller", "ownership_type"]
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE and df[c].dtype in ("float64", "int64")]
# 总资产取对数（量纲处理）
df["total_assets"] = np.log1p(df["total_assets"].clip(lower=0))
print(f"建模特征数: {len(FEATURE_COLS)}")
print("特征列表:", FEATURE_COLS)"""))

# ============ 3. 样本处理 ============
cells.append(nbf.v4.new_markdown_cell(
"""# 3. 样本处理

- 缺失填充：训练集统计中位数（填充训练/验证/测试，防数据泄漏）
- 时间切分：2019-2022 训练 / 2023 验证 / 2024 测试"""))

cells.append(nbf.v4.new_code_cell(
"""# 缺失处理：X_raw 保留缺失（XGB 原生处理）；X 中位数填充（LR 需要）
X_raw = df[FEATURE_COLS].copy()
y = df["label"].copy()

train_mask = df["year"] <= 2022
valid_mask = df["year"] == 2023
test_mask = df["year"] == 2024

MEDIANS = X_raw[train_mask].median()
X = X_raw.fillna(MEDIANS)

X_train, y_train = X[train_mask], y[train_mask]
X_valid, y_valid = X[valid_mask], y[valid_mask]
X_test, y_test = X[test_mask], y[test_mask]
X_raw_train, X_raw_valid, X_raw_test = X_raw[train_mask], X_raw[valid_mask], X_raw[test_mask]

df["split"] = np.where(train_mask, "train", np.where(valid_mask, "test", np.where(test_mask, "oot", "oot")))
print(f"训练: {X_train.shape} (正样本率 {y_train.mean():.2%})")
print(f"验证: {X_valid.shape} (正样本率 {y_valid.mean():.2%})")
print(f"测试: {X_test.shape} (正样本率 {y_test.mean():.2%})")"""))

# ============ 4. 特征处理 ============
cells.append(nbf.v4.new_markdown_cell(
"""# 4. 特征处理（WOE 等频分箱 + IV 筛选，手写实现）

方法论参考：评分卡标准流程（等频分箱→WOE/IV→特征筛选）。
- 仅训练集拟合分箱（防泄漏）
- 缺失值单独成箱
- WOE 平滑（+0.5）防零箱
- IV > 0.02 保留"""))

cells.append(nbf.v4.new_code_cell(
"""def woe_binning_fit(x_tr: pd.Series, y_tr: pd.Series, n_bins: int = 10):
    \"\"\"等频分箱拟合：返回 (bin_edges, woe_map, iv)。缺失单独一箱。\"\"\"
    good_total = int((y_tr == 0).sum())
    bad_total = int((y_tr == 1).sum())
    non_null = x_tr.notna()
    edges = None
    if non_null.sum() > n_bins:
        # 等频分箱（用分位数边界，处理重复值）
        qs = np.linspace(0, 100, n_bins + 1)
        try:
            edges = np.unique(np.percentile(x_tr[non_null], qs))
        except Exception:
            edges = None
    table = []
    # 缺失箱
    miss = ~non_null
    if miss.sum() > 0:
        b, g = int((y_tr[miss] == 1).sum()), int((y_tr[miss] == 0).sum())
        table.append(("miss", b, g))
    # 非缺失分箱
    if edges is not None and len(edges) >= 2:
        labels = pd.cut(x_tr[non_null], bins=edges, include_lowest=True, duplicates="drop")
        for interval, idx in labels.groupby(labels, observed=False).groups.items():
            sel = idx   # groups 返回的 idx 即为 labels 的原始索引（pd.cut 保留原索引）
            b = int((y_tr.loc[sel] == 1).sum())
            g = int((y_tr.loc[sel] == 0).sum())
            table.append((interval, b, g))
    else:
        # 箱数不足时整列单箱
        sel = x_tr[non_null].index
        table.append(("all", int((y_tr.loc[sel] == 1).sum()), int((y_tr.loc[sel] == 0).sum())))

    # WOE/IV（+0.5 平滑防零）
    woe_map, iv = {}, 0.0
    for key, b, g in table:
        woe = np.log(((b + 0.5) / (bad_total + 0.5 * len(table))) /
                     ((g + 0.5) / (good_total + 0.5 * len(table))))
        woe_map[key] = woe
        iv += ((b / max(bad_total, 1)) - (g / max(good_total, 1))) * woe
    return edges, woe_map, iv


def woe_encode(x: pd.Series, edges, woe_map) -> pd.Series:
    \"\"\"应用分箱与 WOE 编码。\"\"\"
    out = pd.Series(np.nan, index=x.index, dtype=float)
    miss = x.isna()
    out[miss] = woe_map.get("miss", 0.0)
    if edges is not None and len(edges) >= 2:
        labels = pd.cut(x[~miss], bins=edges, include_lowest=True, duplicates="drop")
        for interval, idx in labels.groupby(labels, observed=False).groups.items():
            out.loc[idx] = woe_map.get(interval, 0.0)
    else:
        out[~miss] = woe_map.get("all", 0.0)
    return out.fillna(0.0)   # 超出训练分箱边界的值 → WOE=0（中性处理，业界标准）


# 全特征拟合分箱
fitted = {}   # col -> (edges, woe_map, iv)
for col in FEATURE_COLS:
    edges, woe_map, iv = woe_binning_fit(X_train[col], y_train, N_BINS)
    fitted[col] = (edges, woe_map, iv)

# IV 筛选
iv_sorted = sorted(fitted.items(), key=lambda kv: kv[1][2], reverse=True)
KEEP_COLS = [c for c, (_, _, iv) in iv_sorted if iv > IV_THRESHOLD]
print(f"IV>0.02 保留特征: {len(KEEP_COLS)}/{len(FEATURE_COLS)}")
print("IV TOP10:", [(c, round(iv, 3)) for c, (_, _, iv) in iv_sorted[:10]])"""))

cells.append(nbf.v4.new_code_cell(
"""def apply_all_woe(X: pd.DataFrame, cols: list, fitted: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    for col in cols:
        edges, woe_map, _ = fitted[col]
        out[col] = woe_encode(X[col], edges, woe_map)
    return out

Xw_train = apply_all_woe(X_train, KEEP_COLS, fitted)
Xw_valid = apply_all_woe(X_valid, KEEP_COLS, fitted)
Xw_test = apply_all_woe(X_test, KEEP_COLS, fitted)
print(f"WOE特征矩阵: {Xw_train.shape}")"""))

# ============ 5. 模型训练 ============
cells.append(nbf.v4.new_markdown_cell("# 5. 模型训练"))

cells.append(nbf.v4.new_code_cell(
"""# 5.1 主模型：WOE 逻辑回归
lr_model = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
lr_model.fit(Xw_train, y_train)
print("LR 训练完成，特征数:", len(KEEP_COLS))"""))

cells.append(nbf.v4.new_code_cell(
"""# 5.2 对照模型：XGBoost（原始特征含缺失，模型原生处理；剔除 total_assets 避免 PMML 转换）
# 平台投产验证选择 XGB：PMML 导出无需自定义转换，分数一致性天然通过
KEEP_COLS_XGB = [c for c in KEEP_COLS if c != "total_assets"]
pos_w = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
# 固定迭代数（确定性训练，保证 PMML 重训与内存模型完全一致；无早停避免不一致）
xgb_model = XGBClassifier(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_w,
    random_state=RANDOM_STATE, verbosity=0, n_jobs=1,
)
xgb_model.fit(X_raw_train[KEEP_COLS_XGB], y_train)
print("XGB 训练完成（固定300轮，特征数", len(KEEP_COLS_XGB), "）")"""))

# ============ 6. 模型评估 ============
cells.append(nbf.v4.new_markdown_cell("# 6. 模型评估"))

cells.append(nbf.v4.new_code_cell(
"""def ks_score(y, score):
    dfk = pd.DataFrame({"y": y, "s": score}).sort_values("s")
    bad_cum = (dfk["y"] == 1).cumsum() / max(int((dfk["y"] == 1).sum()), 1)
    good_cum = (dfk["y"] == 0).cumsum() / max(int((dfk["y"] == 0).sum()), 1)
    return float((bad_cum - good_cum).abs().max())

def evaluate(name, model, X, y, is_woe=False):
    if is_woe:
        score = model.predict_proba(X)[:, 1]
    else:
        score = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, score)
    ks = ks_score(y, score)
    print(f"  {name}: AUC={auc:.4f}  KS={ks:.4f}")
    return auc, ks

print("=== WOE逻辑回归（主模型）===")
lr_tr = evaluate("训练", lr_model, Xw_train, y_train)
lr_va = evaluate("验证", lr_model, Xw_valid, y_valid)
lr_te = evaluate("测试", lr_model, Xw_test, y_test)
print("=== XGBoost（平台投产验证模型）===")
xgb_tr = evaluate("训练", xgb_model, X_raw_train[KEEP_COLS_XGB], y_train)
xgb_va = evaluate("验证", xgb_model, X_raw_valid[KEEP_COLS_XGB], y_valid)
xgb_te = evaluate("测试", xgb_model, X_raw_test[KEEP_COLS_XGB], y_test)

results = pd.DataFrame({
    "模型": ["LR"] * 3 + ["XGB"] * 3,
    "数据集": ["训练", "验证", "测试"] * 2,
    "AUC": [lr_tr[0], lr_va[0], lr_te[0], xgb_tr[0], xgb_va[0], xgb_te[0]],
    "KS":  [lr_tr[1], lr_va[1], lr_te[1], xgb_tr[1], xgb_va[1], xgb_te[1]],
})
print("\\n=== 汇总 ===")
print(results.to_string(index=False))"""))

# ============ 7. 评分卡刻度 ============
cells.append(nbf.v4.new_markdown_cell(
"""# 7. 评分卡刻度

Score = Offset + Factor × ln(odds)，PDO=20、odds=1:1→600分
主模型为 WOE 逻辑回归，可直接换算为各特征各分箱的加减分表"""))

cells.append(nbf.v4.new_code_cell(
"""factor = PDO / np.log(2)
offset = BASE_SCORE - factor * np.log(BASE_ODDS)

def lr_score(Xw: pd.DataFrame) -> pd.Series:
    logit = lr_model.intercept_[0] + Xw[KEEP_COLS].values @ lr_model.coef_[0]
    return offset - factor * logit   # 分数越高风险越低

score_train = lr_score(Xw_train)
print(f"Factor={factor:.4f}, Offset={offset:.4f}")
print(f"训练集分数: min={score_train.min():.0f}, max={score_train.max():.0f}, 均值={score_train.mean():.0f}")
print("分数分位: ", np.percentile(score_train, [10, 25, 50, 75, 90]).round(0).tolist())"""))

# ============ 8. 平台验证契约 ============
cells.append(nbf.v4.new_markdown_cell("# 8. 平台验证契约（RMC 变量）"))

cells.append(nbf.v4.new_code_cell(
"""# 平台验证样本：测试集（含原始列、标签、分组、时间）
RMC_SAMPLE_DF = df[test_mask].copy()
RMC_TARGET_COL = "label"
RMC_SPLIT_COL = "split"
RMC_TIME_COL = "year"
RMC_ALGORITHM = "xgb"   # 平台投产验证模型：XGB（PMML 工程化、效果更优）


def RMC_SCORE_FN(df_in: pd.DataFrame):
    \"\"\"平台打分入口：XGB 原生处理缺失，无自定义转换（与 PMML 完全一致）。\"\"\"
    return xgb_model.predict_proba(df_in[KEEP_COLS_XGB])[:, 1]


RMC_FEATURE_IMPORTANCE = pd.DataFrame({
    "feature": KEEP_COLS_XGB,
    "importance": xgb_model.feature_importances_,
    "类别": ["财务比率"] * len(KEEP_COLS_XGB),
})
RMC_MODEL_PARAMS = {
    "model": "XGBoost（平台投产验证）",
    "n_features": len(KEEP_COLS_XGB),
    "n_estimators": 300,
    "lr_scorecard": "WOE LogisticRegression（可解释评分卡，见第7节）",
}

# 契约自检
_s = RMC_SCORE_FN(RMC_SAMPLE_DF)
assert len(_s) == len(RMC_SAMPLE_DF), "分数长度不一致"
assert np.isfinite(_s).all(), "分数含空值/无穷"
print("契约自检通过")
print(f"  RMC_SAMPLE_DF: {RMC_SAMPLE_DF.shape}")
print(f"  分数范围: [{_s.min():.4f}, {_s.max():.4f}]")
print(f"  测试集 AUC: {roc_auc_score(RMC_SAMPLE_DF['label'], _s):.4f}")
print("RMC_ALGORITHM =", RMC_ALGORITHM)"""))

# ============ 9. PMML 导出（平台验证材料） ============
cells.append(nbf.v4.new_markdown_cell("# 9. PMML 导出（平台验证材料）"))
cells.append(nbf.v4.new_code_cell(
"""import os
os.environ.setdefault("JAVA_HOME", os.path.expanduser("~/java/jdk-17.0.20+8/Contents/Home"))
from sklearn2pmml import PMMLPipeline, sklearn2pmml

# 用相同数据/参数重训（确定性），与内存 xgb_model 完全一致
pmml_pipe = PMMLPipeline([("model", XGBClassifier(**xgb_model.get_params()))])
pmml_pipe.fit(X_raw_train[KEEP_COLS_XGB], y_train)
os.makedirs("pmml", exist_ok=True)
sklearn2pmml(pmml_pipe, "pmml/model.pmml", with_repr=True)
print("PMML 已导出: pmml/model.pmml")"""))

nb.cells = cells
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks", "modeling.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("已生成:", out)
