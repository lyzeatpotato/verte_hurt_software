"""演示用多任务 3D CNN：愈合期分类 + 受伤天数回归。"""
import torch
import torch.nn as nn


class DemoInjuryNet(nn.Module):
    """
    轻量级 3D 卷积网络，结构对齐主项目的多任务学习思路。
    输入: (B, 1, D, H, W)
    输出: classification (B, 5), regression (B, 1)
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((2, 4, 4)),
        )
        flat = 64 * 2 * 4 * 4
        self.attention = nn.Sequential(
            nn.Linear(flat, flat // 8),
            nn.ReLU(inplace=True),
            nn.Linear(flat // 8, flat),
            nn.Sigmoid(),
        )
        self.cls_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(flat, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        self.reg_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(flat, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.features(x).flatten(1)
        feat = feat * self.attention(feat)
        return {
            "classification": self.cls_head(feat),
            "regression": self.reg_head(feat),
        }
