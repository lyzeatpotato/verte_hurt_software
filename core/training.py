"""模型训练模块。"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import CHECKPOINTS_DIR, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR, LOGS_DIR
from core.preprocessing import parse_case_metadata, preprocess_volume
from models.demo_model import DemoInjuryNet


class DemoVolumeDataset(Dataset):
    def __init__(self, data_root: Path):
        self.samples: list[tuple[Path, int, float]] = []
        data_root = Path(data_root)
        for case_dir in sorted(data_root.iterdir()):
            if not case_dir.is_dir():
                continue
            meta = parse_case_metadata(case_dir.name)
            if meta["healing_class"] is None or meta["days_injured"] is None:
                continue
            self.samples.append((case_dir, meta["healing_class"], float(meta["days_injured"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from core.image_analysis import load_volume_from_folder

        case_dir, label, days = self.samples[idx]
        vol = load_volume_from_folder(case_dir)
        tensor, _ = preprocess_volume(vol)
        return {
            "image": torch.from_numpy(tensor),
            "label": torch.tensor(label, dtype=torch.long),
            "days_injured": torch.tensor(days, dtype=torch.float32),
        }


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "days_injured": torch.stack([b["days_injured"] for b in batch]),
    }


def train_demo_model(
    data_root: Path,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device: str | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """
    在演示数据集上训练多任务模型，保存检查点与训练日志。
    progress_callback(message, progress_0_to_1)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DemoVolumeDataset(data_root)
    if len(dataset) < 2:
        raise ValueError("训练样本不足，请先生成或导入至少 2 个病例数据")

    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, collate_fn=_collate)
    model = DemoInjuryNet().to(device)
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict[str, float]] = []
    total_steps = epochs * len(loader)

    model.train()
    step = 0
    for epoch in range(epochs):
        epoch_cls, epoch_reg, epoch_total, n = 0.0, 0.0, 0.0, 0
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            days = batch["days_injured"].to(device)

            optimizer.zero_grad()
            out = model(images)
            loss_cls = cls_loss_fn(out["classification"], labels)
            loss_reg = reg_loss_fn(out["regression"].squeeze(-1), days)
            loss = loss_cls + 0.5 * loss_reg
            loss.backward()
            optimizer.step()

            epoch_cls += loss_cls.item()
            epoch_reg += loss_reg.item()
            epoch_total += loss.item()
            n += 1
            step += 1
            if progress_callback:
                progress_callback(
                    f"Epoch {epoch + 1}/{epochs} — 分类损失 {loss_cls.item():.4f}, 回归损失 {loss_reg.item():.4f}",
                    step / total_steps,
                )

        history.append(
            {
                "epoch": epoch + 1,
                "loss_cls": round(epoch_cls / n, 4),
                "loss_reg": round(epoch_reg / n, 4),
                "loss_total": round(epoch_total / n, 4),
            }
        )

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = CHECKPOINTS_DIR / f"demo_model_{ts}.pt"
    torch.save({"model_state": model.state_dict(), "epochs": epochs, "device": device}, ckpt_path)

    log_path = LOGS_DIR / f"train_history_{ts}.json"
    summary = {
        "checkpoint": str(ckpt_path),
        "epochs": epochs,
        "samples": len(dataset),
        "device": device,
        "history": history,
        "finished_at": datetime.now().isoformat(),
    }
    log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def load_training_history(log_path: Path) -> dict[str, Any]:
    return json.loads(Path(log_path).read_text(encoding="utf-8"))
