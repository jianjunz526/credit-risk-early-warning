# -*- coding: utf-8 -*-
"""
zh_credit.fetch_industry_cninfo — 行业映射（巨潮证监会行业分类，替代申万官网）

背景（2026-08-19 实测）：
  - 申万官网接口 sw_index_third_info 页面解析失败（官网反爬/页面结构变化，连续重试失败）
  - iFinD 行业指标无试用权限（ths_industry 返回空）
  - 巨潮 stock_profile_cninfo 的"所属行业"字段可用（证监会行业分类，18门类+大类）
  - 结论：用证监会行业分类（官方标准，信评分行业足够），逐公司获取

用法：
  python fetch_industry_cninfo.py --codes data/all_a_codes.csv --output data/industry_map.csv
"""
import argparse
import os
import time

import akshare as ak
import pandas as pd

INTERVAL = 0.6
MAX_RETRY = 3


def fetch_one(code: str):
    for attempt in range(MAX_RETRY):
        try:
            df = ak.stock_profile_cninfo(symbol=code)
            if df is not None and not df.empty:
                return str(df.iloc[0].get("所属行业", ""))
            return ""
        except Exception:
            if attempt < MAX_RETRY - 1:
                time.sleep(INTERVAL * 3 * (attempt + 1))
            else:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="data/all_a_codes.csv")
    ap.add_argument("--output", default="data/industry_map.csv")
    ap.add_argument("--max-codes", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.codes, dtype={"code": str})
    codes = df["code"].str.zfill(6).unique().tolist()
    if args.max_codes > 0:
        codes = codes[:args.max_codes]
    print(f"[1/3] 股票数: {len(codes)}")

    # 断点续跑
    progress = args.output + ".progress"
    done = set()
    if os.path.exists(progress):
        done = set(open(progress, encoding="utf-8").read().splitlines())
        codes = [c for c in codes if c not in done]
        print(f"  断点续跑: 跳过 {len(done)} 家")

    print("[2/3] 逐公司获取行业（巨潮，限速+重试）...")
    rows = []
    t0 = time.time()
    for i, code in enumerate(codes):
        ind = fetch_one(code)
        if ind is not None:
            rows.append({"code": code, "industry_csrc": ind})
        with open(progress, "a", encoding="utf-8") as f:
            f.write(code + "\n")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(codes)} 完成，成功 {len(rows)}，用时 {time.time()-t0:.0f}s")
        time.sleep(INTERVAL)

    print("[3/3] 输出 ...")
    out = pd.DataFrame(rows).drop_duplicates(subset=["code"])
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"  覆盖: {len(out)} 家 → {args.output}")
    top = out["industry_csrc"].value_counts().head(8)
    print("  行业TOP8:")
    print(top.to_string())


if __name__ == "__main__":
    main()
