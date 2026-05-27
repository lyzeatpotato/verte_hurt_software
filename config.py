"""系统全局配置。"""
from pathlib import Path

APP_NAME = "脊椎骨折受伤时间智能预测系统"
APP_VERSION = "1.0.0"
APP_AUTHOR = "SFJD Research Team"

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
SAMPLE_DIR = ASSETS_DIR / "sample_slices"
REPORTS_DIR = BASE_DIR / "reports"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
PREPROCESSED_DIR = BASE_DIR / "assets" / "preprocessed"

# 与主项目一致的五分类愈合期标签
HEALING_PERIOD_LABELS = {
    0: "损伤当天～14天",
    1: "15～30天（1个月内）",
    2: "31～45天（1～1.5月）",
    3: "46～60天（1.5～2月）",
    4: "61～90天（2～3月）",
}

VERTEBRA_LOCATION_LABELS = {0: "颈椎 (C)", 1: "胸椎 (T)", 2: "腰椎 (L)"}

# 3D 体数据默认尺寸（与主项目预处理对齐）
TARGET_DEPTH = 16
TARGET_HEIGHT = 224
TARGET_WIDTH = 224

# 训练默认超参数
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 1e-4
DEFAULT_NUM_WORKERS = 0
