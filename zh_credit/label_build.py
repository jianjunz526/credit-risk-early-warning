# -*- coding: utf-8 -*-
"""
zh_credit.label_build — 信用风险事件标签构建（V1）

标签定义的学术依据（2026-08 GitHub/文献调研）：
  - 中国上市公司财务困境研究的通行做法：以 ST/*ST 作为困境标志
    （中国无美国式破产制度，ST≈困境信号；主要触发条件=连续两年净利润为负+其他）；
  - 学术文献（《基于机器学习的上市公司财务困境预警研究》等）常用
    "T-2/T-1 年财务 → 预测 T 年是否 ST" 的时点框架；
  - 本项目的可计算限制：历史 ST 生效日期无公开结构化数据（akshare 曾用名接口无日期），
    故 V1 用"首次年度亏损"（EPS 由正转负）作为困境早期信号的代理标签，
    其方向与 ST 触发条件（连续亏损）一致，且时点干净、无前视偏差；
  - V2 增强（已排期）：公开整理的债券违约名单 + 新增ST名单并入标签。

V1 标签设计：
  主标签 = "首次年度亏损"（EPS 由正转负）作为财务困境事件的代理
    - 事件事实年份：t（该年度 EPS < 0 且 t-1 年 EPS >= 0）
    - 信息可得性：t 年年报于 t+1 年 4 月 30 日前披露
    - 样本行（公司 × T 年财报）标签：y = 1 当且仅当 首次亏损年份 == T+1
      （T 年财报是 T+1 年 4 月 30 日前的全部可得信息 → 无泄漏）
  辅助标签 = 首次连续两年亏损起始年份（描述性，更接近 ST 触发条件）
  违约名单合并（--default-list）：债券违约主体名单中 default_year == T+1 的样本行同样标 y=1
    （名单为公开可验证的A股违约主体子集，见 data/bond_default_list.csv 口径说明）

用法：
  python label_build.py --input data/financial_panel.csv --output data/panel_labeled.csv
  python label_build.py --input data/financial_panel.csv --default-list data/bond_default_list.csv --output data/panel_labeled.csv
"""
import argparse
import os

import pandas as pd


def build_events(panel: pd.DataFrame) -> pd.DataFrame:
    """计算每公司首次亏损年份。"""
    p = panel.copy()
    p["日期"] = pd.to_datetime(p["日期"])
    p["year"] = p["日期"].dt.year
    p = p.sort_values(["code", "year"]).reset_index(drop=True)

    # 亏损标记
    p["loss"] = p["eps"] < 0

    # 首次亏损年份：EPS 由正转负（前一年非亏损）
    p["prev_loss"] = p.groupby("code")["loss"].shift(1)
    prev = p["prev_loss"].fillna(False).astype(bool)  # 显式 bool，防 object 取反 bug
    first_mask = p["loss"] & ~prev
    first_loss = p.loc[first_mask].groupby("code")["year"].first()  # 公司 → 首次亏损年份
    p["first_loss_year"] = p["code"].map(first_loss)

    # 首次连续两年亏损起始年份（辅助标签，更接近 ST 触发条件）
    p["loss2"] = p["loss"] & prev  # 本年和上年均亏损（本行是连续亏损的第二年）
    p["prev_loss2"] = p.groupby("code")["loss2"].shift(1).fillna(False).astype(bool)
    consec_mask = p["loss2"] & ~p["prev_loss2"]
    # 连续亏损起始年 = 连续亏损第二年的前一年
    consec_first = p.loc[consec_mask].groupby("code")["year"].first() - 1
    p["first_consec_loss_year"] = p["code"].map(consec_first)
    return p


def assign_label(p: pd.DataFrame, default_list: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    标签对齐：
      y = 1 当 首次亏损年份 == T+1（T = 该行财报年份）
      或 违约名单中 default_year == T+1（若提供 --default-list）
    即：用 T 年财报（T+1年4月30日可得）预测 T+1 年内发生首次亏损/债券违约。
    事件窗口按"财报年度"对齐，避免使用任何未来信息。
    """
    p["label"] = (p["first_loss_year"] == p["year"] + 1).astype(int)
    if default_list is not None and not default_list.empty:
        # 违约事件：default_year == T+1 的样本行标记为1
        default_map = dict(zip(default_list["code"], default_list["default_year"]))
        p["default_year"] = p["code"].map(default_map)
        default_mask = p["default_year"] == p["year"] + 1
        p.loc[default_mask, "label"] = 1
        n_default = int(default_mask.sum())
        print(f"  违约名单合并: {n_default} 个样本行因债券违约标记为1")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/financial_panel.csv")
    ap.add_argument("--output", default="data/panel_labeled.csv")
    ap.add_argument("--default-list", default=None, help="债券违约名单CSV（code,default_year），合并违约标签")
    args = ap.parse_args()

    print(f"[1/4] 读取面板: {args.input}")
    panel = pd.read_csv(args.input)
    print(f"  形状: {panel.shape}，公司数: {panel['code'].nunique()}")

    default_list = None
    if args.default_list and os.path.exists(args.default_list):
        default_list = pd.read_csv(args.default_list, encoding="utf-8-sig", comment="#")
        print(f"  违约名单: {len(default_list)} 家（{args.default_list}）")

    print("[2/4] 计算首次亏损事件 ...")
    p = build_events(panel)

    print("[3/4] 标签对齐（y=1 当 首次亏损年份==T+1）...")
    p = assign_label(p, default_list)

    # 汇总统计
    n_pos = int(p["label"].sum())
    n_total = len(p)
    print(f"[4/4] 标签分布: 正样本 {n_pos} ({n_pos/n_total:.1%}) / 总样本 {n_total}")
    # 按年份看正样本分布
    dist = p[p["label"] == 1].groupby(p.loc[p["label"] == 1, "year"]).size()
    print("正样本按财报年份分布:")
    print(dist.to_string())

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    p.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
