# -*- coding: utf-8 -*-
"""
zh_credit.ownership_fetch — 企业属性（国企/民企）获取（iFinD 实际控制人）

数据源：同花顺 iFinD HTTP API（token 见 tools/.ifind.env，2026-09-11 前有效）
  - 指标：ths_actual_controller（实际控制人，2026-08-19 实测可用，多公司批量支持）

分类规则（实控人文本 → 所有制）：
  - state：含 国有资产监督管理委员会/国资委/人民政府/国务院/财政局/国有资产/国有 等
  - no_controller：空（无实控人，如万科/平安；或查不到）
  - private：其他（个人/民营集团等）

已知局限（诚实声明）：
  - 已退市公司查不到实控人——但与 akshare 财务面板（仅现存公司）缺口一致，不影响合并
  - 实控人为境外实体/信托等复杂结构时分类可能不准，规则可迭代

用法：
  python ownership_fetch.py --codes data/financial_panel.csv --output data/ownership_map.csv
  # --codes 传财务面板CSV，自动取唯一 code 列表（code 无后缀，自动加 .SH/.SZ）
"""
import argparse
import json
import os
import ssl
import time
import urllib.request

import pandas as pd

BATCH = 20          # 每批请求股票数（实测20只正常）
INTERVAL = 1.0      # 批间间隔（秒）
MAX_RETRY = 3

STATE_KEYWORDS = [
    "国有资产监督管理委员会", "国资委", "人民政府", "国务院",
    "财政局", "国有资产", "国有", "中核", "中石油", "中石化",
    "中国航天", "中国航空", "中国船舶", "中国兵器", "中国电子",
    "国家电网", "南方电网", "中国移动", "中国联通", "中国电信",
]


def load_token(path):
    tok = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            tok[k.strip()] = v.strip()
    return tok.get("THS_ACCESS_TOKEN", "")


def classify(controller: str) -> str:
    if not controller:
        return "no_controller"
    if any(k in controller for k in STATE_KEYWORDS):
        return "state"
    return "private"


def fetch_batch(token, ctx, codes: list[str]):
    url = "https://ft.10jqka.com.cn/api/v1/basic_data_service"
    payload = {"codes": ",".join(codes), "indipara": [{"indicator": "ths_actual_controller", "indiparams": []}]}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "access_token": token})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="data/financial_panel.csv", help="含 code 列的CSV（取唯一值）")
    ap.add_argument("--output", default="data/ownership_map.csv")
    ap.add_argument("--max-codes", type=int, default=0, help="测试用：只处理前N个")
    args = ap.parse_args()

    token = load_token(os.path.expanduser("~/Desktop/credit-risk-project/tools/.ifind.env"))
    if not token:
        print("未找到 iFinD token（tools/.ifind.env）")
        return
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # 读取股票代码（强制 str 防前导零丢失，去后缀统一6位）
    df = pd.read_csv(args.codes, usecols=["code"], dtype={"code": str})
    codes = df["code"].str.split(".").str[0].str.zfill(6).unique().tolist()
    if args.max_codes > 0:
        codes = codes[:args.max_codes]
    print(f"[1/3] 股票数: {len(codes)}，分 {len(codes)//BATCH + 1} 批（每批{BATCH}只）")

    # 构造 THS 代码（600xxx→.SH, 000/002/300→.SZ, 8/4开头→.BJ）
    def ths_code(c):
        if c.startswith(("6", "9")):
            return f"{c}.SH"
        if c.startswith(("0", "3", "2")):
            return f"{c}.SZ"
        return f"{c}.BJ"

    print("[2/3] 批量抓取实际控制人（iFinD）...")
    results = {}
    t0 = time.time()
    for i in range(0, len(codes), BATCH):
        batch = codes[i:i + BATCH]
        ths = [ths_code(c) for c in batch]
        for attempt in range(MAX_RETRY):
            try:
                r = fetch_batch(token, ctx, ths)
                for t in r.get("tables", []):
                    ths_c = t.get("thscode", "")
                    code6 = ths_c.split(".")[0]
                    val = t.get("table", {}).get("ths_actual_controller")
                    controller = val[0] if val else ""
                    results[code6] = controller
                break
            except Exception as e:
                if attempt < MAX_RETRY - 1:
                    time.sleep(INTERVAL * 3 * (attempt + 1))
                else:
                    print(f"  批 {i//BATCH} 失败: {type(e).__name__}")
                    for c in batch:
                        results.setdefault(c, "")
        time.sleep(INTERVAL)

    print(f"[3/3] 分类与输出（用时 {time.time()-t0:.0f}s）...")
    out = pd.DataFrame([
        {"code": c, "actual_controller": v, "ownership_type": classify(v)}
        for c, v in results.items()
    ])
    dist = out["ownership_type"].value_counts()
    print("  所有制分布:")
    print(dist.to_string())
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"  → {args.output}")


if __name__ == "__main__":
    main()
