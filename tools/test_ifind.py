# -*- coding: utf-8 -*-
"""
iFinD (同花顺数据接口) HTTP API 测试脚本
用法：
  1) 把 refresh_token 和 access_token 填入 ~/Desktop/.ifind.env （一行一个，格式见下）
  2) python3 test_ifind.py
.env 格式：
  THS_REFRESH_TOKEN=eyJ...
  THS_ACCESS_TOKEN=eba2...
安全提示：.env 文件不要提交进 git 仓库（GitHub 有 secret 扫描，token 会被盗用）。
"""
import os
import json
import ssl
import sys
import urllib.request
import urllib.parse

BASE = "https://ft.10jqka.com.cn/api/v1"
ALT = "https://quantapi.51ifind.com/api/v1"

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def load_tokens():
    env_path = os.path.expanduser("~/Desktop/credit-risk-project/tools/.ifind.env")
    tokens = {}
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                tokens[k.strip()] = v.strip()
    tokens.setdefault("THS_ACCESS_TOKEN", os.environ.get("THS_ACCESS_TOKEN", ""))
    tokens.setdefault("THS_REFRESH_TOKEN", os.environ.get("THS_REFRESH_TOKEN", ""))
    return tokens


def post(url, payload, token):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "access_token": token},
    )
    with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def show(title, data, maxlen=600):
    s = json.dumps(data, ensure_ascii=False)
    print(f"\n===== {title} =====")
    print(s[:maxlen])


def main():
    t = load_tokens()
    if not t["THS_ACCESS_TOKEN"]:
        print("未找到 access_token，请先创建 ~/Desktop/.ifind.env")
        sys.exit(1)
    at = t["THS_ACCESS_TOKEN"]

    # 1. 实时行情（验证 token 有效性）
    show("1. 实时行情 000001.SZ", post(f"{BASE}/real_time_quotation", {
        "codes": "000001.SZ", "indicators": "latest,changeRatio,pe_ttm,pb"
    }, at))

    # 2. 财务数据：尝试多种参数组合（ROE）
    attempts = [
        {"indicator": "ths_roe_avg_index", "indiparams": []},
        {"indicator": "ths_roe_avg_index", "indiparams": ["2023", "1"]},
        {"indicator": "ths_roe_avg_index", "indiparams": ["20231231"]},
        {"indicator": "ths_sq_net_asset_yield_roe_index", "indiparams": ["2023", "1"]},
    ]
    for i, indi in enumerate(attempts):
        show(f"2.{i+1} 财务ROE尝试 {indi['indiparams']}",
             post(f"{BASE}/basic_data_service", {"codes": "600519.SH", "indipara": [indi]}, at), 400)

    # 3. 额度查询
    show("3. 数据量额度", post(f"{ALT}/get_data_volume", {}, at), 500)

    # 4. 交易日历
    show("4. 交易日查询 2026-08", post(f"{BASE}/get_trade_dates", {
        "marketcode": "212001",
        "functionpara": {"mode": "1", "dateType": "0", "period": "D", "dateFormat": "0"},
        "startdate": "2026-08-01", "enddate": "2026-08-31",
    }, at), 300)


if __name__ == "__main__":
    main()
