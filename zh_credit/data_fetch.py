# -*- coding: utf-8 -*-
"""
zh_credit.data_fetch — A股上市公司财务指标批量抓取（基于 akshare 官方接口）

功能：
  1. 获取全A股股票列表（ak.stock_info_a_code_name）
  2. 逐公司抓取历年财务指标（ak.stock_financial_analysis_indicator，含最新季度）
  3. 筛选年报行（12-31）、剔除金融行业、选择信用分析相关指标列
  4. 断点续跑（已抓取公司缓存）+ 限速 + 重试（防 RemoteDisconnected）
  5. 输出面板 CSV（公司×年度）

用法：
  python data_fetch.py --max-companies 30 --start-year 2023 --output data/financial_panel_test.csv
  python data_fetch.py --start-year 2019 --output data/financial_panel.csv   # 全量

依赖：conda marvis 环境（pip install akshare）
"""
import argparse
import os
import sys
import time

import akshare as ak
import pandas as pd

# ============ 配置 ============
# 财务指标列白名单（与开发计划特征体系对应；名称以 akshare 实际返回为准）
SELECT_COLS = {
    # 规模
    "总资产(元)": "total_assets",
    # 盈利
    "销售净利率(%)": "net_margin",
    # 注：akshare 该接口的"销售毛利率(%)"列普遍为空（数据源问题，2026-08实测），故以营业利润率/净利率替代
    "净资产收益率(%)": "roe",
    "总资产净利润率(%)": "roa",
    "营业利润率(%)": "op_margin",
    "成本费用利润率(%)": "cost_profit_ratio",
    # 偿债
    "资产负债率(%)": "debt_ratio",
    "流动比率": "current_ratio",
    "速动比率": "quick_ratio",
    "现金比率(%)": "cash_ratio",
    "利息支付倍数": "interest_coverage",
    "产权比率(%)": "equity_ratio",
    # 现金流质量
    "每股经营性现金流(元)": "ocf_per_share",
    "经营现金净流量对销售收入比率(%)": "ocf_to_sales",
    "经营现金净流量与净利润的比率(%)": "ocf_to_ni",
    "经营现金净流量对负债比率(%)": "ocf_to_debt",
    "现金流量比率(%)": "cashflow_ratio",
    # 营运
    "应收账款周转率(次)": "ar_turnover",
    "存货周转率(次)": "inv_turnover",
    "总资产周转率(次)": "asset_turnover",
    # 成长
    "主营业务收入增长率(%)": "rev_growth",
    "净利润增长率(%)": "ni_growth",
    "总资产增长率(%)": "asset_growth",
    # 每股
    "加权每股收益(元)": "eps",
    "每股净资产_调整前(元)": "bps",
}

# 金融行业名称关键词（剔除）
FIN_KEYWORDS = ["银行", "证券", "保险", "信托", "期货", "金融", "财务公司"]

REQUEST_INTERVAL = 0.35   # 每次请求间隔（秒），防限流
MAX_RETRY = 3             # 单公司重试次数
CACHE_FILE = "data/_fetched_cache.txt"  # 断点续跑缓存


def is_financial(name: str) -> bool:
    return any(k in name for k in FIN_KEYWORDS)


def fetch_one(code: str, name: str, start_year: int) -> pd.DataFrame | None:
    """抓取单公司财务指标，筛选年报行并选择列。失败返回 None。"""
    for attempt in range(MAX_RETRY):
        try:
            raw = ak.stock_financial_analysis_indicator(symbol=code, start_year=str(start_year))
            if raw is None or raw.empty:
                return None
            # 筛选年报行（12-31）
            raw = raw.copy()
            raw["日期"] = pd.to_datetime(raw["日期"])
            annual = raw[raw["日期"].dt.month == 12].copy()
            if annual.empty:
                return None
            # 选择列
            keep = {"日期"}
            rename = {}
            for zh, en in SELECT_COLS.items():
                if zh in annual.columns:
                    keep.add(zh)
                    rename[zh] = en
            annual = annual[list(keep)].rename(columns=rename)
            annual.insert(0, "code", code)
            annual.insert(1, "name", name)
            return annual
        except Exception as e:  # akshare 偶发断连/超时
            if attempt < MAX_RETRY - 1:
                time.sleep(REQUEST_INTERVAL * 3 * (attempt + 1))
            else:
                print(f"  [FAIL] {code} {name}: {type(e).__name__}")
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-companies", type=int, default=0, help="抓取公司数上限（0=全部）")
    ap.add_argument("--start-year", type=int, default=2019, help="起始财报年份")
    ap.add_argument("--output", default="data/financial_panel.csv")
    ap.add_argument("--cache", default=CACHE_FILE)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)

    # 已缓存（断点续跑）
    done = set()
    if os.path.exists(args.cache):
        done = set(open(args.cache, encoding="utf-8").read().splitlines())

    print(f"[1/3] 获取股票列表 ...")
    stocks = ak.stock_info_a_code_name()
    stocks = stocks[~stocks["name"].apply(is_financial)]
    print(f"  总股票 {len(stocks)}，剔除金融后 {len(stocks)}")

    if args.max_companies > 0:
        stocks = stocks.head(args.max_companies)
        print(f"  本次上限 {args.max_companies} 只")

    todo = stocks[~stocks["code"].isin(done)]
    print(f"  待抓取 {len(todo)}（已缓存 {len(done)}）")

    print(f"[2/3] 抓取财务指标（间隔{REQUEST_INTERVAL}s/只，年报起始 {args.start_year}）...")
    frames = []
    t0 = time.time()
    for i, (_, row) in enumerate(todo.iterrows()):
        code, name = row["code"], row["name"]
        df = fetch_one(code, name, args.start_year)
        if df is not None:
            frames.append(df)
        with open(args.cache, "a", encoding="utf-8") as f:
            f.write(code + "\n")
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(todo)} 完成，成功 {len(frames)}，用时 {el:.0f}s")
        time.sleep(REQUEST_INTERVAL)

    print(f"[3/3] 汇总输出 ...")
    if not frames:
        print("  无任何数据，退出")
        sys.exit(1)
    panel = pd.concat(frames, ignore_index=True)
    panel.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"  面板: {panel.shape}（公司×年度行）→ {args.output}")
    print(f"  公司数: {panel['code'].nunique()}，年份范围: {panel['日期'].min().date()} ~ {panel['日期'].max().date()}")
    print(f"  缺失率最高5列: \n{panel.isnull().mean().sort_values(ascending=False).head(5)}")


if __name__ == "__main__":
    main()
