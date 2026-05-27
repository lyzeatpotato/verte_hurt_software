"""Matplotlib 中文字体配置（macOS / Windows / Linux 通用回退）。"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager

# 按平台常见字体优先级排列
FONT_CANDIDATES = [
    "Hiragino Sans GB",
    "PingFang SC",
    "PingFang HK",
    "Heiti SC",
    "STHeiti",
    "Songti SC",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]

_font_initialized = False
_selected_font: str | None = None


def _collect_available_font_names() -> set[str]:
    return {f.name for f in font_manager.fontManager.ttflist}


def setup_chinese_matplotlib() -> str | None:
    """
    配置 Matplotlib 使用系统中文字体，避免中文标题/标签显示为方框。
    在应用启动时调用一次即可。
    """
    global _font_initialized, _selected_font
    if _font_initialized:
        return _selected_font

    available = _collect_available_font_names()
    chosen = None
    for name in FONT_CANDIDATES:
        if name in available:
            chosen = name
            break

    if chosen is None:
        # 模糊匹配含 SC / CJK / Hei 的字体
        for f in font_manager.fontManager.ttflist:
            n = f.name
            if any(k in n for k in ("PingFang", "Heiti", "Songti", "CJK", "YaHei", "SimHei", "SC")):
                chosen = n
                break

    if chosen:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        _selected_font = chosen
    else:
        warnings.warn(
            "未检测到中文字体，图表中文可能无法正常显示。"
            " macOS 一般自带 PingFang SC，请确认系统字体完整。",
            UserWarning,
            stacklevel=2,
        )
        _selected_font = None

    plt.rcParams["axes.unicode_minus"] = False
    _font_initialized = True
    return _selected_font


def plot_intensity_trend(intensity_trend: list[float]):
    """逐层平均灰度趋势折线图。"""
    setup_chinese_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(intensity_trend, marker="o", markersize=4, color="#2874a6")
    ax.set_xlabel("切片索引")
    ax.set_ylabel("平均灰度")
    ax.set_title("逐层平均灰度趋势")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_training_loss(history: list[dict]) -> plt.Figure:
    """训练损失曲线。"""
    setup_chinese_matplotlib()
    epochs_x = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs_x, [h["loss_cls"] for h in history], marker="o", label="分类损失")
    ax.plot(epochs_x, [h["loss_reg"] for h in history], marker="s", label="回归损失")
    ax.plot(epochs_x, [h["loss_total"] for h in history], marker="^", label="总损失")
    ax.legend(loc="best")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("损失值")
    ax.set_title("训练损失曲线")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_class_probabilities(labels: list[str], values: list[float]) -> plt.Figure:
    """愈合期分类概率柱状图。"""
    setup_chinese_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(labels))
    bars = ax.bar(x, values, color="#1a5276", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("概率")
    ax.set_title("愈合期分类概率分布")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    return fig
