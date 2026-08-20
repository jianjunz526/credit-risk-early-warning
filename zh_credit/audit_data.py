# -*- coding: utf-8 -*-
"""
zh_credit.audit_data — 全量数据验收（财务面板 + 标签 + 特征 + 行业映射）

在 W1 全量数据跑完后执行，输出数据质量报告：
  1. 面板规模：公司数/年份范围/行数
  2. 标签分布：正样本数/正样本率/按年份分布
  3. 特征缺失率TOP10
  4. 行业映射覆盖率（合并 industry_map.csv 后）
  5. 违约名单命中情况（label v2 合并后）

用法：
  python audit_data.py --panel data/financial_panel.csv --labeled data/panel_labeled.csv \
      --features data/modeling_dataset.csv --industry data/industry_map.csv
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/financial_panel.csv")
    ap.add_argument("--labeled", default="data/panel_labeled.csv")
    ap.add_argument("--features", default="data/modeling_dataset.csv")
    ap.add_argument("--industry", default="data/industry_map.csv")
    args = ap.parse_args()

    print("=" * 60)
    print("数据验收报告")
    print("=" * 60)

    # 1. 面板
    panel = pd.read_csv(args.panel, dtype={"code": str})
    panel["code"] = panel["code"].str.zfill(6)
    panel["日期"] = pd.to_datetime(panel["日期"])
    panel["year"] = panel["日期"].dt.year
    print(f"\n[1] 财务面板: {panel.shape}")
    print(f"    公司数: {panel['code'].nunique()} | 年份: {panel['year'].min()}-{panel['year'].max()}")
    print(f"    平均每公司年数: {panel.groupby('code').size().mean():.1f}")

    # 2. 标签
    labeled = pd.read_csv(args.labeled)
    labeled["日期"] = pd.to_datetime(labeled["日期"])
    labeled["year"] = labeled["日期"].dt.year
    pos = int(labeled["label"].sum())
    print(f"\n[2] 标签: 正样本 {pos} ({pos/len(labeled):.2%}) / 总样本 {len(labeled)}")
    dist = labeled[labeled["label"] == 1].groupby("year").size()
    print("    正样本按财报年份:")
    print(dist.to_string())
    if "default_year" in labeled.columns:
        n_def = int(labeled["default_year"].notna().sum())
        n_def_pos = int(((labeled["default_year"] == labeled["year"] + 1)).sum())
        print(f"    违约名单命中: {n_def_pos} 个样本行（名单覆盖 {n_def} 行）")

    # 3. 特征缺失率
    if os.path.exists(args.features):
        feat = pd.read_csv(args.features)
        miss = feat.isnull().mean().sort_values(ascending=False)
        miss = miss[miss > 0]
        print(f"\n[3] 特征缺失率TOP10（特征共{len(feat.columns)-4}个）:")
        print(miss.head(10).to_string())
        print(f"    正样本率: {feat['label'].mean():.2%}")

    # 4. 行业覆盖率
    if os.path.exists(args.industry):
        ind = pd.read_csv(args.industry, dtype={"code": str})
        ind["code"] = ind["code"].str.zfill(6)
        merged = panel[["code"]].drop_duplicates().merge(ind, on="code", how="left")
        cov = merged["industry_csrc"].notna().mean()
        print(f"\n[4] 行业映射: 覆盖率 {cov:.1%}（面板公司 {panel['code'].nunique()} 家，映射 {merged['industry_csrc'].notna().sum()} 家）")
        print("    行业数:", merged["industry_csrc"].nunique())

    print("\n" + "=" * 60)
    print("验收标准: 公司>=2000 | 正样本>=150 | 行业覆盖率>=85% | 缺失率TOP<50%")
    print("=" * 60)


if __name__ == "__main__":
    main()
