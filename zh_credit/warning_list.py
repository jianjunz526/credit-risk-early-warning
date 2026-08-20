# -*- coding: utf-8 -*-
"""
zh_credit.warning_list — 预警清单 TOP50（W3）

功能：用最新一期（2025年报）财务数据 + XGBoost 模型打分，
     输出风险最高的 50 家公司清单（含行业/所有制/违约概率/主要风险特征）。

实现：与建模 notebook 完全相同的训练逻辑（WOE分箱→IV筛选→XGB固定300轮），
     保证打分模型与平台验证模型一致。

输出：output/warning_list_top50.csv

用法：
  python warning_list.py
"""
import os

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "modeling_dataset_full.csv")
OUT_DIR = os.path.join(ROOT, "output")
RANDOM_STATE = 42
N_BINS = 10
IV_THRESHOLD = 0.02


def woe_binning_fit(x_tr, y_tr, n_bins=10):
    good_total = int((y_tr == 0).sum())
    bad_total = int((y_tr == 1).sum())
    non_null = x_tr.notna()
    edges = None
    if non_null.sum() > n_bins:
        qs = np.linspace(0, 100, n_bins + 1)
        try:
            edges = np.unique(np.percentile(x_tr[non_null], qs))
        except Exception:
            edges = None
    table = []
    miss = ~non_null
    if miss.sum() > 0:
        table.append(("miss", int((y_tr[miss] == 1).sum()), int((y_tr[miss] == 0).sum())))
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
        woe = np.log(((b + 0.5) / (bad_total + 0.5 * len(table))) /
                     ((g + 0.5) / (good_total + 0.5 * len(table))))
        woe_map[key] = woe
        iv += ((b / max(bad_total, 1)) - (g / max(good_total, 1))) * woe
    return edges, woe_map, iv


def woe_encode(x, edges, woe_map):
    out = pd.Series(np.nan, index=x.index, dtype=float)
    miss = x.isna()
    out[miss] = woe_map.get("miss", 0.0)
    if edges is not None and len(edges) >= 2:
        labels = pd.cut(x[~miss], bins=edges, include_lowest=True, duplicates="drop")
        for interval, idx in labels.groupby(labels, observed=False).groups.items():
            out.loc[idx] = woe_map.get(interval, 0.0)
    else:
        out[~miss] = woe_map.get("all", 0.0)
    return out.fillna(0.0)


def main():
    print(f"[1/5] 读取数据: {DATA}")
    df = pd.read_csv(DATA, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)

    EXCLUDE = ["code", "name", "year", "label", "split", "first_loss_year",
               "default_year", "first_consec_loss_year",
               "industry_csrc", "actual_controller", "ownership_type"]
    FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE and df[c].dtype in ("float64", "int64")]
    df["total_assets"] = np.log1p(df["total_assets"].clip(lower=0))
    print(f"  特征池: {len(FEATURE_COLS)}")

    print("[2/5] 训练集分箱 + IV 筛选（与 notebook 一致）...")
    X_raw = df[FEATURE_COLS].copy()
    y = df["label"].copy()
    train_mask = df["year"] <= 2022
    fitted, ivs = {}, {}
    for col in FEATURE_COLS:
        edges, woe_map, iv = woe_binning_fit(X_raw.loc[train_mask, col], y[train_mask], N_BINS)
        fitted[col] = (edges, woe_map, iv)
        ivs[col] = iv
    KEEP_COLS = [c for c, iv in sorted(ivs.items(), key=lambda kv: kv[1], reverse=True)
                 if iv > IV_THRESHOLD and c != "total_assets"]
    print(f"  入模特征: {len(KEEP_COLS)}")

    print("[3/5] 训练 XGBoost（固定300轮，确定性）...")
    pos_w = (y[train_mask] == 0).sum() / max(int((y[train_mask] == 1).sum()), 1)
    model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_w,
                          random_state=RANDOM_STATE, verbosity=0, n_jobs=1)
    model.fit(X_raw.loc[train_mask, KEEP_COLS], y[train_mask])

    print("[4/5] 对最新一期（2025年报）打分 ...")
    latest = df[df["year"] == 2025].copy()
    latest["pd"] = model.predict_proba(latest[KEEP_COLS])[:, 1]

    print("[5/5] 输出预警清单 TOP50 ...")
    top = latest.sort_values("pd", ascending=False).head(50)
    # 主要风险特征：出险画像中区分度最高的指标（低盈利/低现金流/高杠杆）
    risk_cols = ["roe", "net_margin", "ocf_to_debt", "debt_ratio"]
    desc = []
    for _, r in top.iterrows():
        flags = []
        if r["roe"] < 3.0:
            flags.append("ROE<3%")
        if r["net_margin"] < 4.0:
            flags.append("净利率<4%")
        if r["ocf_to_debt"] < 0.05:
            flags.append("经营现金流/负债<5%")
        if r["debt_ratio"] > 70:
            flags.append("负债率>70%")
        desc.append("；".join(flags) if flags else "关注")
    top["主要风险特征"] = desc
    out = top[["code", "name", "industry_csrc", "ownership_type", "year", "pd", "主要风险特征"]]
    out.columns = ["代码", "名称", "行业", "所有制", "财报年", "预警概率", "主要风险特征"]
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, "warning_list_top50.csv"), index=False, encoding="utf-8-sig")
    print(f"  输出: {os.path.join(OUT_DIR, 'warning_list_top50.csv')}")
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
