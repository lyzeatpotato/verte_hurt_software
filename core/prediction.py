"""模型预测模块。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import CHECKPOINTS_DIR, HEALING_PERIOD_LABELS
from core.preprocessing import preprocess_volume
from models.demo_model import DemoInjuryNet


def _latest_checkpoint() -> Path | None:
    if not CHECKPOINTS_DIR.exists():
        return None
    ckpts = sorted(CHECKPOINTS_DIR.glob("demo_model_*.pt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def load_model(checkpoint: Path | None = None, device: str | None = None) -> tuple[DemoInjuryNet, str, Path | None]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint) if checkpoint else _latest_checkpoint()
    model = DemoInjuryNet()
    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        model.to(device)
        model.eval()
        return model, device, ckpt_path
    model.to(device)
    model.eval()
    return model, device, None


def _interpret_prediction(cls_idx: int, days: float) -> dict[str, Any]:
    return {
        "healing_class_index": cls_idx,
        "healing_period": HEALING_PERIOD_LABELS.get(cls_idx, "未知"),
        "predicted_days_injured": round(max(0.0, days), 1),
        "confidence_note": "模型输出经 Softmax/回归头解码，仅供临床辅助参考",
    }


@torch.no_grad()
def predict_from_volume(
    volume: np.ndarray,
    checkpoint: Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    tensor, preprocess_log = preprocess_volume(volume)
    model, device, ckpt = load_model(checkpoint, device)
    x = torch.from_numpy(tensor).unsqueeze(0).to(device)
    out = model(x)
    probs = torch.softmax(out["classification"], dim=1)[0].cpu().numpy()
    cls_idx = int(torch.argmax(out["classification"], dim=1).item())
    days = float(out["regression"].squeeze().cpu().item())

    result = _interpret_prediction(cls_idx, days)
    result["class_probabilities"] = {
        HEALING_PERIOD_LABELS[i]: round(float(probs[i]), 4) for i in range(len(probs))
    }
    result["preprocess_log"] = preprocess_log
    result["checkpoint_used"] = str(ckpt) if ckpt else "未加载权重（随机初始化，仅演示流程）"
    return result


def predict_from_npy(
    npy_path: Path,
    checkpoint: Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    arr = np.load(npy_path)
    if arr.ndim == 4 and arr.shape[0] == 1:
        volume = arr[0]
    elif arr.ndim == 3:
        volume = arr
    else:
        volume = arr
    return predict_from_volume(volume, checkpoint, device)
