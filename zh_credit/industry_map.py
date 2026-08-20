# -*- coding: utf-8 -*-
"""
zh_credit.industry_map — 申万行业映射（股票 → 一级/二级/三级行业）

数据源：akshare 申万指数系列（已实测通过）
  - sw_index_third_info()：335个三级行业（含上级二级名）
  - sw_index_second_info()：131个二级行业（含上级一级名）→ 二级名→一级名映射
  - sw_index_third_cons(symbol)：三级行业成分股

输出：data/industry_map.csv（code, industry_l1, industry_l2, industry_l3）

用法：
  python industry_map.py --output data/industry_map.csv
"""
import argparse
import os
import socket
import time

socket.setdefaulttimeout(20)  # 全局超时，防接口挂起

import akshare as ak
import pandas as pd

REQUEST_INTERVAL = 0.5
MAX_RETRY = 3


def fetch_with_retry(fn, *args, **kwargs):
    for attempt in range(MAX_RETRY):
        try:
            df = fn(*args, **kwargs)
            if df is None or df.empty:
                return None
            return df
        except Exception:
            if attempt < MAX_RETRY - 1:
                time.sleep(REQUEST_INTERVAL * 3 * (attempt + 1))
            else:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/industry_map.csv")
    ap.add_argument("--max-industries", type=int, default=0, help="测试用：只处理前N个行业")
    args = ap.parse_args()

    print("[1/5] 获取三级/二级行业列表 ...")
    third = ak.sw_index_third_info()
    second = ak.sw_index_second_info()
    l2_to_l1 = dict(zip(second["行业名称"], second["上级行业"]))
    print(f"  三级行业 {len(third)} 个，二级行业 {len(second)} 个")

    if args.max_industries > 0:
        third = third.head(args.max_industries)

    # 断点续跑：跳过已完成行业
    progress_file = args.output + ".progress"
    done_codes = set()
    if os.path.exists(progress_file):
        done_codes = set(open(progress_file, encoding="utf-8").read().splitlines())
        third = third[~third["行业代码"].isin(done_codes)]
        print(f"  断点续跑: 跳过已完成 {len(done_codes)} 个行业")

    print("[2/5] 遍历三级行业取成分股（限速+重试）...")
    rows = []
    t0 = time.time()
    for i, (_, row) in enumerate(third.iterrows()):
        code3, name3, name2 = row["行业代码"], row["行业名称"], row["上级行业"]
        name1 = l2_to_l1.get(name2, "")
        cons = fetch_with_retry(ak.sw_index_third_cons, symbol=code3)
        if cons is not None and not cons.empty:
            for _, c in cons.iterrows():
                stock_code = str(c["股票代码"]).split(".")[0]  # 去后缀
                rows.append({"code": stock_code, "industry_l3": name3,
                             "industry_l2": name2, "industry_l1": name1})
        with open(progress_file, "a", encoding="utf-8") as pf:
            pf.write(code3 + "\n")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(third)} 行业，累计 {len(rows)} 行，用时 {time.time()-t0:.0f}s")
        time.sleep(REQUEST_INTERVAL)

    print("[3/5] 合并去重 ...")
    df = pd.DataFrame(rows).drop_duplicates(subset=["code"]).reset_index(drop=True)

    print("[4/5] 质量检查 ...")
    n_total = len(df)
    n_l1 = df["industry_l1"].notna().sum()
    print(f"  覆盖股票数: {n_total}，一级行业覆盖率: {n_l1/n_total:.1%}")
    # 重复检查：一只股票多个三级行业？
    dup = df["code"].duplicated().sum()
    print(f"  重复股票: {dup}")
    # 一级行业分布
    dist = df["industry_l1"].value_counts().head(8)
    print("  一级行业TOP8:")
    print(dist.to_string())

    print("[5/5] 输出 ...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"  → {args.output}")


if __name__ == "__main__":
    main()
