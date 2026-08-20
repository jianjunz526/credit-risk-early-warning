# -*- coding: utf-8 -*-
"""
zh_credit.features — 特征工程（V1）

输入：panel_labeled.csv（data_fetch + label_build 产出）
输出：modeling_dataset.csv（建模数据集）

特征体系：
  1. 水平特征：data_fetch 已抓取的 26 个财务比率/指标（直接使用）
  2. 趋势特征（Δ）：关键指标的同比变化（信评强调"趋势比水平更重要"）
  3. 缺失指示变量：缺失本身可能是信息（如某项指标未披露）
  4. 缩尾：1%/99% 分位（防极端值主导模型）

用法：
  python features.py --input data/panel_labeled.csv --output data/modeling_dataset.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

# 需要构造趋势特征的核心指标（同比变化）
DELTA_FEATURES = [
    "debt_ratio",      # 资产负债率 Δ
    "roe",             # ROE Δ
    "ocf_to_ni",       # 经营现金流/净利润 Δ
    "current_ratio",   # 流动比率 Δ
    "net_margin",      # 净利率 Δ
    "asset_turnover",  # 总资产周转率 Δ
]

WINSOR_QUANTILES = (0.01, 0.99)


def build_deltas(p: pd.DataFrame) -> pd.DataFrame:
    """按公司构造同比趋势特征。"""
    p = p.sort_values(["code", "year"]).reset_index(drop=True)
    for col in DELTA_FEATURES:
        if col in p.columns:
            p[f"{col}_delta"] = p.groupby("code")[col].diff()
    return p


def build_missing_indicators(p: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """缺失率>threshold 的列生成缺失指示变量。
    注意：事件/标签衍生列（default_year/first_consec_loss_year）绝不生成缺失指示——
    它们是标签信息的直接映射，作为特征会造成数据泄漏（审计发现，2026-08-20）。
    """
    EXCLUDE_NO_MISS = ("code", "name", "日期", "year", "label", "first_loss_year",
                       "default_year", "first_consec_loss_year")
    for col in p.columns:
        if col in EXCLUDE_NO_MISS:
            continue
        if p[col].dtype in ("float64", "int64", "float32"):
            miss_rate = p[col].isna().mean()
            if miss_rate > threshold:
                p[f"{col}_miss"] = p[col].isna().astype(int)
    return p


def winsorize(p: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """1%/99% 分位缩尾——分位数只在训练集（year<=2022）上计算，防验证/测试信息泄漏。
    注意：year/label/split 等非特征列绝不被缩尾（year 会被训练集分位 clip 破坏，2026-08-20 审计发现）。
    """
    NON_FEATURE = {"year", "label", "split", "code", "name"}
    train_rows = p["year"] <= 2022 if "year" in p.columns else pd.Series(True, index=p.index)
    for col in cols:
        if col in NON_FEATURE or col not in p.columns:
            continue
        train_vals = p.loc[train_rows, col].dropna()
        if train_vals.empty:
            continue
        lo, hi = train_vals.quantile(WINSOR_QUANTILES)
        p[col] = p[col].clip(lo, hi)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/panel_labeled.csv")
    ap.add_argument("--output", default="data/modeling_dataset.csv")
    args = ap.parse_args()

    print(f"[1/4] 读取标注面板: {args.input}")
    p = pd.read_csv(args.input)
    p["日期"] = pd.to_datetime(p["日期"])
    p["year"] = p["日期"].dt.year
    print(f"  形状: {p.shape}，公司数: {p['code'].nunique()}，正样本: {int(p['label'].sum())}")

    print("[2/4] 构造趋势特征（同比Δ）...")
    p = build_deltas(p)

    print("[3/4] 缺失指示变量 + 缩尾 ...")
    numeric_cols = p.select_dtypes(include=[np.number]).columns.tolist()
    p = build_missing_indicators(p)
    p = winsorize(p, numeric_cols)

    print("[4/4] 输出建模数据集 ...")
    # 时间序列切分标记（与建模 notebook 一致：train/valid/test/oot）
    # 平台规范三分：train（2019-2022）/ test（2023 开发验证）/ oot（2024 样本外）
    p["split"] = "oot"
    p.loc[p["year"] <= 2022, "split"] = "train"
    p.loc[p["year"] == 2023, "split"] = "test"
    p.loc[p["year"] == 2024, "split"] = "oot"
    # 只保留建模用列（剔除日期，保留 year 作时间序列切分）
    keep = ["code", "name", "year", "label", "split"] + [
        c for c in p.columns if c not in ("code", "name", "日期", "year", "label", "first_loss_year", "split")
    ]
    out = p[keep]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"  输出: {out.shape} → {args.output}")
    print(f"  特征列数: {len(out.columns) - 4}（含趋势与缺失指示）")
    print(f"  正样本率: {out['label'].mean():.2%}")


if __name__ == "__main__":
    main()
