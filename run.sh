#!/usr/bin/env bash
# 脊椎骨折受伤时间智能预测系统 — 启动脚本
cd "$(dirname "$0")"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 使用 python3 -m，避免 pip 脚本目录未加入 PATH 时找不到 streamlit
if command -v python3 >/dev/null 2>&1; then
  exec python3 -m streamlit run app.py --server.port 8501 --server.headless true
else
  exec python -m streamlit run app.py --server.port 8501 --server.headless true
fi
