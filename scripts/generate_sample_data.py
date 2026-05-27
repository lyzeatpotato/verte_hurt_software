#!/usr/bin/env python3
"""命令行生成演示数据集。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.preprocessing import build_demo_dataset  # noqa: E402

if __name__ == "__main__":
    out = ROOT / "assets" / "demo_dataset"
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    path = build_demo_dataset(out, num_cases=n)
    print(f"Generated {n} cases at {path}")
