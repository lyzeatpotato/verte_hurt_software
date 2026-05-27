"""图像分析模块：质量评估、统计特征与切片可视化。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def analyze_single_image(image: np.ndarray) -> dict[str, Any]:
    """分析单张 CT 切片。"""
    gray = _to_gray_uint8(image)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    contrast = float(gray.max() - gray.min())
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    entropy = float(-np.sum((hist / hist.sum() + 1e-12) * np.log(hist / hist.sum() + 1e-12)))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    quality = "优" if laplacian_var > 100 and contrast > 40 else ("良" if laplacian_var > 50 else "待复核")
    return {
        "mean_intensity": round(mean_val, 2),
        "std_intensity": round(std_val, 2),
        "contrast": round(contrast, 2),
        "entropy": round(entropy, 4),
        "sharpness_laplacian": round(laplacian_var, 2),
        "quality_grade": quality,
        "shape": list(gray.shape),
    }


def load_slices_from_paths(paths: list[Path]) -> np.ndarray:
    """从文件路径列表加载切片序列，返回 (D, H, W)。"""
    slices = []
    for p in sorted(paths):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            img = np.array(Image.open(p).convert("L"))
        elif p.suffix.lower() == ".npy":
            arr = np.load(p)
            img = arr if arr.ndim == 2 else arr[..., 0]
        else:
            continue
        slices.append(img)
    if not slices:
        raise ValueError("未找到有效图像文件")
    return np.stack(slices, axis=0)


def analyze_image_volume(volume: np.ndarray) -> dict[str, Any]:
    """
    分析 3D 体数据（D, H, W）或 (D, H, W, C)。
    返回整体统计、逐层摘要及推荐中间层索引。
    """
    if volume.ndim == 4:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"期望 3D 体数据，实际维度: {volume.ndim}")

    depth, height, width = volume.shape
    per_slice = [analyze_single_image(volume[i]) for i in range(depth)]
    sharpness = [s["sharpness_laplacian"] for s in per_slice]
    best_idx = int(np.argmax(sharpness))
    mean_intensities = [s["mean_intensity"] for s in per_slice]

    return {
        "volume_shape": [depth, height, width],
        "slice_count": depth,
        "global_mean_intensity": round(float(np.mean(volume)), 2),
        "global_std_intensity": round(float(np.std(volume)), 2),
        "recommended_center_slice": best_idx,
        "slice_summaries": per_slice,
        "intensity_trend": mean_intensities,
        "overall_quality": _aggregate_quality(per_slice),
    }


def _aggregate_quality(summaries: list[dict]) -> str:
    grades = [s["quality_grade"] for s in summaries]
    if grades.count("待复核") > len(grades) // 2:
        return "待复核"
    if grades.count("优") >= len(grades) // 2:
        return "优"
    return "良"


def load_volume_from_folder(folder: str | Path) -> np.ndarray:
    folder = Path(folder)
    paths = [
        p
        for p in folder.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
    ]
    return load_slices_from_paths(paths)
