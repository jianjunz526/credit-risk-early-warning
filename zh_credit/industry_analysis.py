# -*- coding: utf-8 -*-
"""
zh_credit.industry_analysis — 分行业风险画像（W3，信评面试命门内容）

功能：
  1. 行业×年份正样本率矩阵（哪些行业风险高、何时恶化）
  2. 出险公司 vs 正常公司 的关键指标画像（出险前财务特征）
  3. 所有制×行业交叉分析（民企/国企风险差异）
  4. 输出图表（PNG）与画像表（CSV）

输入：modeling_dataset_full.csv（含 label/industry_csrc/ownership_type）
输出：output/industry_analysis/ 下的 CSV 与 PNG

用法：
  python industry_analysis.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Songti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "industry_analysis")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "modeling_dataset_full.csv")

# 出险画像关注的指标（含信用分析含义）
PROFILE_METRICS = {
    "debt_ratio": "资产负债率(%)",
    "current_ratio": "流动比率",
    "roe": "ROE(%)",
    "net_margin": "净利率(%)",
    "ocf_to_ni": "经营现金流/净利润(%)",
    "interest_coverage": "利息支付倍数",
    "ocf_to_debt": "经营现金流/负债(%)",
    "eps": "每股收益(元)",
}


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    # 排除 oot 中的 2025 行（无标签信息）
    df = df[~((df["split"] == "oot") & (df["year"] == 2025))].copy()
    print(f"样本: {len(df)}（已剔除2025无标签行）")

    # ===== 1. 行业×年份正样本率矩阵 =====
    print("\n[1] 行业×年份正样本率矩阵 ...")
    piv = df.pivot_table(index="industry_csrc", columns="year", values="label",
                         aggfunc="mean").sort_values(by=df["year"].max(), ascending=False)
    # 只保留样本量足够的行业
    counts = df.groupby("industry_csrc").size()
    piv = piv[counts[counts.index].ge(100)]
    piv["整体风险率"] = df.groupby("industry_csrc")["label"].mean()
    piv = piv.sort_values("整体风险率", ascending=False)
    piv.to_csv(os.path.join(OUT, "industry_risk_matrix.csv"), encoding="utf-8-sig")
    print(f"  行业数: {len(piv)}，TOP10 高风险行业（整体风险率）:")
    print(piv[["整体风险率"]].head(10).to_string())

    # 行业风险率柱状图（TOP15）
    fig, ax = plt.subplots(figsize=(12, 6))
    top = piv.head(15)
    ax.barh(top.index.str[:14], top["整体风险率"], color="#c77b1e")
    ax.set_xlabel("正样本率（信用风险事件发生率）")
    ax.set_title("分行业信用风险事件发生率 TOP15（2019-2024）")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "industry_risk_top15.png"), dpi=150)
    plt.close()

    # ===== 2. 出险公司 vs 正常公司 关键指标画像 =====
    print("\n[2] 出险 vs 正常公司关键指标画像 ...")
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    rows = []
    for col, label in PROFILE_METRICS.items():
        if col not in df.columns:
            continue
        rows.append({
            "指标": label,
            "出险公司中位数": pos[col].median(),
            "正常公司中位数": neg[col].median(),
            "差异方向": "出险更低" if pos[col].median() < neg[col].median() else "出险更高",
        })
    profile = pd.DataFrame(rows)
    profile.to_csv(os.path.join(OUT, "distress_profile.csv"), index=False, encoding="utf-8-sig")
    print(profile.to_string(index=False))

    # ===== 3. 所有制×行业交叉 =====
    print("\n[3] 所有制 × 行业交叉风险率 ...")
    cross = df.groupby(["ownership_type", "industry_csrc"]).agg(
        样本数=("label", "size"), 风险率=("label", "mean")
    ).reset_index()
    cross = cross[cross["样本数"] >= 50].sort_values("风险率", ascending=False)
    cross.to_csv(os.path.join(OUT, "ownership_industry_cross.csv", ), index=False, encoding="utf-8-sig")
    print("  TOP10 高风险的 所有制×行业 组合:")
    print(cross.head(10).to_string(index=False))

    # 所有制整体风险率
    own_risk = df.groupby("ownership_type")["label"].agg(["mean", "size"])
    own_risk.columns = ["风险率", "样本数"]
    print("\n  所有制整体风险率:")
    print(own_risk.to_string())

    # 所有制风险率柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = {"state": "国企", "private": "民企", "no_controller": "无实控人"}
    ax.bar([labels.get(i, i) for i in own_risk.index], own_risk["风险率"],
           color=["#0d3b66", "#c77b1e", "#7a8b99"])
    for i, v in enumerate(own_risk["风险率"]):
        ax.text(i, v + 0.002, f"{v:.2%}", ha="center")
    ax.set_ylabel("信用风险事件发生率")
    ax.set_title("所有制性质与信用风险事件发生率")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "ownership_risk.png"), dpi=150)
    plt.close()

    print(f"\n全部产出 → {OUT}")


if __name__ == "__main__":
    main()
