"""数据预处理模块：归一化、重采样、体数据构建与导出。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from config import PREPROCESSED_DIR, TARGET_DEPTH, TARGET_HEIGHT, TARGET_WIDTH
from core.image_analysis import load_slices_from_paths, load_volume_from_folder


def _normalize_slice(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99))
    if p99 > p1:
        img = np.clip((img - p1) / (p99 - p1), 0, 1)
    else:
        img = img / (img.max() + 1e-8)
    return img


def _resize_slice(img: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def resample_depth(volume: np.ndarray, target_depth: int) -> np.ndarray:
    """沿深度轴重采样到固定层数。"""
    d, h, w = volume.shape
    if d == target_depth:
        return volume
    indices = np.linspace(0, d - 1, target_depth).astype(int)
    return volume[indices]


def preprocess_volume(
    volume: np.ndarray,
    target_depth: int = TARGET_DEPTH,
    target_height: int = TARGET_HEIGHT,
    target_width: int = TARGET_WIDTH,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    预处理 3D 体数据：窗宽窗位归一化、空间重采样、深度统一。
    返回 (1, D, H, W) float32 张量格式数据及处理日志。
    """
    if volume.ndim == 4:
        volume = volume[..., 0]

    original_shape = list(volume.shape)
    processed = []
    for i in range(volume.shape[0]):
        sl = _normalize_slice(volume[i])
        sl = _resize_slice(sl, target_height, target_width)
        processed.append(sl)
    processed = np.stack(processed, axis=0)
    processed = resample_depth(processed, target_depth)

    # 单通道 3D -> (1, D, H, W) 供模型使用
    tensor = processed[np.newaxis, ...].astype(np.float32)

    log = {
        "original_shape": original_shape,
        "output_shape": list(tensor.shape),
        "target_depth": target_depth,
        "target_spatial": [target_height, target_width],
        "normalization": "percentile_1_99",
    }
    return tensor, log


def preprocess_folder(
    folder: str | Path,
    target_depth: int = TARGET_DEPTH,
    target_height: int = TARGET_HEIGHT,
    target_width: int = TARGET_WIDTH,
) -> tuple[np.ndarray, dict[str, Any]]:
    volume = load_volume_from_folder(folder)
    return preprocess_volume(volume, target_depth, target_height, target_width)


def export_preprocessed_npy(
    volume: np.ndarray,
    case_id: str,
    output_dir: Path | None = None,
) -> Path:
    output_dir = output_dir or PREPROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{case_id}_preprocessed.npy"
    np.save(out_path, volume)
    return out_path


def build_demo_dataset(output_dir: Path, num_cases: int = 8) -> Path:
    """生成演示用合成数据集，便于无真实数据时完成训练流程演示。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(num_cases):
        days = int(rng.integers(1, 91))
        label = _days_to_label(days)
        case_dir = output_dir / f"case_{i:03d}_days{days}_cls{label}"
        case_dir.mkdir(exist_ok=True)
        depth = rng.integers(12, 24)
        for z in range(depth):
            base = rng.normal(120, 30, (256, 256))
            yy, xx = np.mgrid[0:256, 0:256]
            blob = np.exp(-((yy - 128) ** 2 + (xx - 128) ** 2) / (2 * (40 + days) ** 2))
            sl = np.clip(base + 80 * blob, 0, 255).astype(np.uint8)
            np.save(case_dir / f"slice_{z:03d}.npy", sl)
    return output_dir


def _days_to_label(days: int) -> int:
    if days <= 14:
        return 0
    if days <= 30:
        return 1
    if days <= 45:
        return 2
    if days <= 60:
        return 3
    return 4


def parse_case_metadata(folder_name: str) -> dict[str, int | None]:
    """从文件夹名解析 days 与 cls 标签（演示数据命名规范）。"""
    import re

    days_m = re.search(r"days(\d+)", folder_name)
    cls_m = re.search(r"cls(\d+)", folder_name)
    return {
        "days_injured": int(days_m.group(1)) if days_m else None,
        "healing_class": int(cls_m.group(1)) if cls_m else None,
    }
