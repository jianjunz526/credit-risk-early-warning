#!/bin/bash
# 一键启动 MARVIS 信用风控工作台
# 用法: bash scripts/start_marvis.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 激活 conda marvis 环境
source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate marvis

echo "启动 MARVIS 工作台: http://127.0.0.1:8899"
echo "工作区: $PROJECT_DIR/workspace"
marvis serve --host 127.0.0.1 --port 8899 --workspace "$PROJECT_DIR/workspace"
