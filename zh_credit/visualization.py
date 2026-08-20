# -*- coding: utf-8 -*-
"""
zh_credit.visualization — 模型结果可视化（图表集）

生成图表（修复后数据/模型）：
  1. roc_curve.png           ROC 曲线（训练/验证/样本外三条）
  2. ks_curve.png            KS 曲线（好坏累计分布 + 最大间距）
  3. score_distribution.png  测试集分数分布（好坏叠加）
  4. feature_importance.png  特征重要性 TOP15
  5. calibration_trend.png   各年度实际坏率 vs 模型预测概率
  6. warning_list_dist.png   预警清单 TOP50 概率分布
输出：output/charts/

用法：python visualization.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Songti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "modeling_dataset_full.csv")
OUT = os.path.join(ROOT, "output", "charts")
RANDOM_STATE = 42

# —— 复用 warning_list.py 的训练逻辑 ——
def woe_binning_fit(x_tr, y_tr, n_bins=10):
    good_total = int((y_tr == 0).sum()); bad_total = int((y_tr == 1).sum())
    non_null = x_tr.notna(); edges = None
    if non_null.sum() > n_bins:
        try: edges = np.unique(np.percentile(x_tr[non_null], np.linspace(0, 100, n_bins + 1)))
        except Exception: edges = None
    table = []
    miss = ~non_null
    if miss.sum() > 0: table.append(("miss", int((y_tr[miss] == 1).sum()), int((y_tr[miss] == 0).sum())))
    if edges is not None and len(edges) >= 2:
        labels = pd.cut(x_tr[non_null], bins=edges, include_lowest=True, duplicates="drop")
        for interval, idx in labels.groupby(labels, observed=False).groups.items():
            sel = idx
            table.append((interval, int((y_tr.loc[sel] == 1).sum()), int((y_tr.loc[sel] == 0).sum())))
    else:
        sel = x_tr[non_null].index
        table.append(("all", int((y_tr.loc[sel] == 1).sum()), int((y_tr.loc[sel] == 0).sum())))
    woe_map, iv = {}, 0.0
    for key, b, g in table:
        woe = np.log(((b + 0.5) / (bad_total + 0.5 * len(table))) / ((g + 0.5) / (good_total + 0.5 * len(table))))
        woe_map[key] = woe
        iv += ((b / max(bad_total, 1)) - (g / max(good_total, 1))) * woe
    return edges, woe_map, iv


def main():
    os.makedirs(OUT, exist_ok=True)
    print("[1/6] 读取数据与训练模型（与生产一致）...")
    df = pd.read_csv(DATA, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    EXCLUDE = ["code", "name", "year", "label", "split", "first_loss_year",
               "default_year", "first_consec_loss_year",
               "industry_csrc", "actual_controller", "ownership_type"]
    FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE and df[c].dtype in ("float64", "int64")]
    df["total_assets"] = np.log1p(df["total_assets"].clip(lower=0))
    X_raw = df[FEATURE_COLS].copy(); y = df["label"].copy()
    train_mask = df["year"] <= 2022; valid_mask = df["year"] == 2023; test_mask = df["year"] == 2024
    fitted, ivs = {}, {}
    for col in FEATURE_COLS:
        e, w, iv = woe_binning_fit(X_raw.loc[train_mask, col], y[train_mask]); fitted[col] = (e, w, iv); ivs[col] = iv
    KEEP_COLS = [c for c, iv in sorted(ivs.items(), key=lambda kv: kv[1], reverse=True)
                 if iv > 0.02 and c != "total_assets"]
    pos_w = (y[train_mask] == 0).sum() / max(int((y[train_mask] == 1).sum()), 1)
    model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_w,
                          random_state=RANDOM_STATE, verbosity=0, n_jobs=1)
    model.fit(X_raw.loc[train_mask, KEEP_COLS], y[train_mask])
    print(f"  模型就绪（特征{len(KEEP_COLS)}个）")

    sets = {"训练": (train_mask, "#0d3b66"), "验证": (valid_mask, "#1d5a94"), "样本外": (test_mask, "#c77b1e")}
    probs = {k: model.predict_proba(X_raw.loc[m, KEEP_COLS])[:, 1] for k, (m, _) in sets.items()}
    ys = {k: y[m] for k, (m, _) in sets.items()}

    # ===== 1. ROC 曲线 =====
    print("[2/6] ROC 曲线 ...")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for k, (m, c) in sets.items():
        fpr, tpr, _ = roc_curve(ys[k], probs[k])
        auc = roc_auc_score(ys[k], probs[k])
        ax.plot(fpr, tpr, color=c, lw=1.8, label=f"{k} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="随机 (AUC=0.5)")
    ax.set_xlabel("假阳性率 (FPR)"); ax.set_ylabel("真阳性率 (TPR)")
    ax.set_title("ROC 曲线：模型区分度（三数据集独立验证）")
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "roc_curve.png"), dpi=150); plt.close()

    # ===== 2. KS 曲线 =====
    print("[3/6] KS 曲线 ...")
    k = "样本外"
    dfk = pd.DataFrame({"y": ys[k], "s": probs[k]}).sort_values("s").reset_index(drop=True)
    dfk["bad_cum"] = (dfk["y"] == 1).cumsum() / max(int((dfk["y"] == 1).sum()), 1)
    dfk["good_cum"] = (dfk["y"] == 0).cumsum() / max(int((dfk["y"] == 0).sum()), 1)
    ks = float((dfk["bad_cum"] - dfk["good_cum"]).abs().max())
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(dfk.index, dfk["bad_cum"], color="#c0392b", lw=1.8, label="坏样本累计占比")
    ax.plot(dfk.index, dfk["good_cum"], color="#2e7d32", lw=1.8, label="好样本累计占比")
    idx_ks = int((dfk["bad_cum"] - dfk["good_cum"]).abs().idxmax())
    ax.axvline(idx_ks, color="#333", ls="--", lw=1)
    ax.annotate(f"KS = {ks:.3f}", xy=(idx_ks, 0.5), xytext=(idx_ks * 0.5, 0.85),
                arrowprops=dict(arrowstyle="->", color="#333"), fontsize=11)
    ax.set_xlabel("按模型分数升序排列的样本（0=最低分/最高风险 → N=最高分/最低风险）")
    ax.set_ylabel("累计占比")
    ax.set_title(f"KS 曲线（样本外 2024）：KS={ks:.3f}")
    ax.legend(loc="center right")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "ks_curve.png"), dpi=150); plt.close()

    # ===== 3. 分数分布 =====
    print("[4/6] 分数分布 ...")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bins = np.linspace(0, 1, 40)
    ax.hist(probs[k][ys[k] == 1], bins=bins, alpha=0.6, color="#c0392b", label="坏样本（出险）")
    ax.hist(probs[k][ys[k] == 0], bins=bins, alpha=0.5, color="#2e7d32", label="好样本（正常）")
    ax.axvline(0.3, color="#333", ls="--", lw=1)
    ax.text(0.31, ax.get_ylim()[1] * 0.9, "示例预警线 0.30", fontsize=9)
    ax.set_xlabel("模型预测的风险事件概率")
    ax.set_ylabel("样本数")
    ax.set_title("样本外（2024）：好/坏样本的模型分数分布")
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "score_distribution.png"), dpi=150); plt.close()

    # ===== 4. 特征重要性 =====
    print("[5/6] 特征重要性 ...")
    imp = pd.DataFrame({"feature": KEEP_COLS, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#0d3b66")
    ax.set_xlabel("XGBoost 特征重要性（gain 归一化）")
    ax.set_title("特征重要性 TOP15：盈利类主导")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150); plt.close()

    # ===== 5. 校准趋势：年度实际坏率 vs 模型预测 =====
    print("[6/6] 校准趋势 ...")
    all_prob = model.predict_proba(X_raw[KEEP_COLS])[:, 1]
    cal = pd.DataFrame({"year": df["year"], "label": y, "prob": all_prob})
    trend = cal.groupby("year").agg(实际坏率=("label", "mean"), 平均预测概率=("prob", "mean"))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trend.index, trend["实际坏率"], "o-", color="#c0392b", label="实际坏率（标签）")
    ax.plot(trend.index, trend["平均预测概率"], "s--", color="#0d3b66", label="模型平均预测概率")
    ax.set_xlabel("财报年份"); ax.set_ylabel("比率")
    ax.set_title("各年度：实际坏率 vs 模型平均预测概率（校准趋势）")
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "calibration_trend.png"), dpi=150); plt.close()

    # ===== 6. 预警清单分布 =====
    latest = df[df["year"] == 2025].copy()
    latest["prob"] = model.predict_proba(latest[KEEP_COLS])[:, 1]
    top50 = latest.sort_values("prob", ascending=False).head(50)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(top50)), top50["prob"], color="#c77b1e")
    ax.axhline(0.5, color="#333", ls="--", lw=1)
    ax.text(1, 0.52, "0.50 参考线", fontsize=9)
    ax.set_xlabel("预警名单排序（1=风险最高）"); ax.set_ylabel("预警概率")
    ax.set_title("预警清单 TOP50：模型概率分布（2025年报 → 2026年风险）")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "warning_list_dist.png"), dpi=150); plt.close()

    print(f"\n全部图表 → {OUT}")
    for f in sorted(os.listdir(OUT)):
        print(" ", f)


if __name__ == "__main__":
    main()
