# 脊椎骨折受伤时间智能预测系统（软件著作权演示版）

本目录为独立于主研究仓库 `vertehurtdateprediction` 的**软件著作权申请演示系统**，提供完整的图形化操作界面与五大功能模块，便于录制操作视频与提交鉴别材料。

## 软件名称

**脊椎骨折受伤时间智能预测系统** v1.0.0

## 功能模块

| 序号 | 模块 | 说明 |
|------|------|------|
| 1 | 图像分析 | CT 切片灰度统计、清晰度（拉普拉斯方差）、熵、整体质量评级、推荐展示层 |
| 2 | 数据预处理 | 百分位归一化、空间重采样、深度统一至 16 层，导出 `.npy` 体数据 |
| 3 | 模型训练 | 多任务 3D CNN（愈合期五分类 + 受伤天数回归），损失曲线与检查点保存 |
| 4 | 模型预测 | 加载检查点推理，输出愈合期类别、概率分布与回归天数 |
| 5 | 报告生成 | 自动生成 HTML 预测分析报告 / 训练过程报告，支持下载 |

## 目录结构

```
verte_hurt_software_demo/
├── app.py                 # Streamlit 主界面（软著演示入口）
├── config.py              # 系统配置与标签定义
├── run.sh                 # 一键启动脚本
├── requirements.txt
├── core/                  # 核心业务逻辑
│   ├── image_analysis.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── prediction.py
│   └── report.py
├── models/
│   └── demo_model.py      # 演示用多任务 3D CNN
├── assets/
│   └── demo_dataset/      # 运行后生成的演示数据
├── checkpoints/           # 训练得到的模型权重
├── reports/               # 生成的 HTML 报告
└── logs/                  # 训练日志 JSON
```

## 环境安装

```bash
cd verte_hurt_software_demo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 启动方式

```bash
# 推荐：使用启动脚本（内部为 python3 -m streamlit，不依赖 PATH）
./run.sh

# 或直接运行（若 streamlit 命令找不到，请用下面这行）
python3 -m streamlit run app.py
```

浏览器访问 **http://localhost:8501**

> 若 `pip install` 后提示脚本装在 `~/Library/Python/3.9/bin` 且未在 PATH 中，使用 `python3 -m streamlit` 即可，无需单独配置 PATH。

## 发布到公网（供他人访问）

详见 **[DEPLOY.md](./DEPLOY.md)**，推荐两种方式：

1. **Streamlit Community Cloud（免费）**：代码推 GitHub → [share.streamlit.io](https://share.streamlit.io) 部署，Main file 填 `verte_hurt_software_demo/app.py`
2. **云服务器 + Docker**：`docker compose up -d --build`，放行 8501 端口

## 软著演示推荐流程（约 5～8 分钟）

1. **系统首页**：展示软件名称、版本与五大模块说明  
2. **数据预处理** →「生成演示数据集」：生成 8 例合成 CT 序列  
3. **图像分析**：选择一例，查看质量评估与灰度趋势图  
4. **数据预处理** →「执行预处理」：导出标准化体数据  
5. **模型训练**：设置轮次后点击训练，展示损失下降曲线  
6. **模型预测**：选择最新检查点，执行推理并展示愈合期与天数  
7. **报告生成**：下载 HTML 预测报告作为附件材料  

## 与主研究项目的关系

- 标签定义、五分类愈合期划分与主项目 `training_data/dataset.py` 保持一致  
- 演示版采用轻量级 `DemoInjuryNet`；生产环境可替换为主项目的 R3D-18 多任务模型权重  
- 主项目 DICOM 预处理流水线见上级目录 `data_preprocessing/README.md`  

## 技术栈

Python 3.9+ · PyTorch · Streamlit · OpenCV · NumPy · Matplotlib

## 免责声明

本软件演示版输出结果仅供科研与软件功能展示，不作为临床单独诊断依据。
