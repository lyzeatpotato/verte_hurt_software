"""
脊椎骨折受伤时间智能预测系统 — 软件著作权演示主程序
运行: streamlit run app.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.plot_style import (  # noqa: E402
    plot_class_probabilities,
    plot_intensity_trend,
    plot_training_loss,
    setup_chinese_matplotlib,
)

setup_chinese_matplotlib()

from config import (  # noqa: E402
    APP_AUTHOR,
    APP_NAME,
    APP_VERSION,
    CHECKPOINTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    REPORTS_DIR,
)
from core.image_analysis import analyze_image_volume, analyze_single_image, load_volume_from_folder  # noqa: E402
from core.prediction import predict_from_npy, predict_from_volume  # noqa: E402
from core.preprocessing import build_demo_dataset, export_preprocessed_npy, preprocess_folder, preprocess_volume  # noqa: E402
from core.report import generate_prediction_report, generate_training_report  # noqa: E402
from core.training import train_demo_model  # noqa: E402

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEMO_DATA_DIR = ROOT / "assets" / "demo_dataset"


def _ensure_demo_data():
    if not DEMO_DATA_DIR.exists() or not any(DEMO_DATA_DIR.iterdir()):
        build_demo_dataset(DEMO_DATA_DIR, num_cases=8)


def sidebar():
    st.sidebar.title(APP_NAME)
    st.sidebar.caption(f"v{APP_VERSION}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**功能模块**")
    module = st.sidebar.radio(
        "选择功能",
        [
            "系统首页",
            "图像分析",
            "数据预处理",
            "模型训练",
            "模型预测",
            "报告生成",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"本系统用于脊椎 CT 序列的受伤时间预测演示。\n\n"
        f"开发单位：{APP_AUTHOR}\n\n"
        f"含图像分析、预处理、训练、预测、报告五大模块。"
    )
    return module


def page_home():
    st.title(f"🦴 {APP_NAME}")
    st.markdown(
        f"""
### 软件简介

本软件面向**脊椎骨折受伤时间预测**场景，基于深度学习多任务学习框架，
对 CT 影像序列进行智能分析，输出**愈合期五分类**与**受伤天数回归**结果。

| 模块 | 功能说明 |
|------|----------|
| 图像分析 | CT 切片质量评估、灰度统计、清晰度与推荐展示层 |
| 数据预处理 | 窗宽窗位归一化、空间重采样、3D 体数据构建与导出 |
| 模型训练 | 多任务 3D CNN 训练，分类+回归联合优化 |
| 模型预测 | 单病例推理，愈合期与受伤天数输出 |
| 报告生成 | 自动生成 HTML 格式分析/训练/预测报告 |

### 快速开始

1. 侧边栏进入 **数据预处理** → 点击「生成演示数据集」
2. 进入 **模型训练** → 启动训练并保存检查点
3. 进入 **模型预测** → 上传切片或选择演示病例进行推理
4. 进入 **报告生成** → 导出 PDF 可打印的 HTML 报告

> 演示版使用轻量级网络；可与主研究项目 `vertehurtdateprediction` 的 R3D-18 权重对接扩展。
        """
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("愈合期分类", "5 类")
    with col2:
        st.metric("回归目标", "受伤天数")
    with col3:
        ckpts = list(CHECKPOINTS_DIR.glob("demo_model_*.pt")) if CHECKPOINTS_DIR.exists() else []
        st.metric("已训练模型", f"{len(ckpts)} 个")


def page_image_analysis():
    st.header("📊 图像分析")
    st.markdown("对 CT 切片序列进行质量与统计特征分析。")

    source = st.radio("数据来源", ["演示数据集病例", "上传切片文件夹（多图）", "上传单个 NPY 体数据"], horizontal=True)

    volume = None
    case_name = "upload"

    if source == "演示数据集病例":
        _ensure_demo_data()
        cases = sorted([d.name for d in DEMO_DATA_DIR.iterdir() if d.is_dir()])
        case_name = st.selectbox("选择病例", cases)
        volume = load_volume_from_folder(DEMO_DATA_DIR / case_name)
    elif source == "上传单个 NPY 体数据":
        f = st.file_uploader("上传 .npy 体数据", type=["npy"])
        if f:
            volume = np.load(f)
            if volume.ndim == 4:
                volume = volume[0]
    else:
        files = st.file_uploader("上传多张 CT 切片", type=["png", "jpg", "jpeg", "bmp", "npy"], accept_multiple_files=True)
        if files:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                for uf in files:
                    (tmp_path / uf.name).write_bytes(uf.getvalue())
                volume = load_volume_from_folder(tmp_path)

    if volume is None:
        st.warning("请选择或上传数据后进行分析。")
        return

    if st.button("开始分析", type="primary"):
        with st.spinner("分析中..."):
            result = analyze_image_volume(volume)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("层数", result["slice_count"])
        c2.metric("体数据尺寸", str(result["volume_shape"]))
        c3.metric("整体质量", result["overall_quality"])
        c4.metric("推荐层", result["recommended_center_slice"])

        fig = plot_intensity_trend(result["intensity_trend"])
        st.pyplot(fig)
        plt.close(fig)

        mid = result["recommended_center_slice"]
        st.subheader(f"推荐展示层（第 {mid} 层）")
        sl = volume[mid]
        if sl.dtype != np.uint8:
            sl = np.clip(sl, 0, 255).astype(np.uint8)
        st.image(sl, caption=f"第 {mid} 层（推荐展示层）", use_container_width=True)

        st.subheader("逐层分析明细")
        import pandas as pd

        col_map = {
            "mean_intensity": "平均灰度",
            "std_intensity": "灰度标准差",
            "contrast": "对比度",
            "entropy": "熵",
            "sharpness_laplacian": "清晰度",
            "quality_grade": "质量等级",
            "shape": "尺寸",
        }
        df = pd.DataFrame(result["slice_summaries"]).rename(columns=col_map)
        df.index.name = "切片序号"
        st.dataframe(df, use_container_width=True)

        st.session_state["last_analysis"] = result
        st.session_state["last_volume"] = volume
        st.session_state["last_case"] = case_name


def page_preprocessing():
    st.header("⚙️ 数据预处理")
    st.markdown("将原始 CT 切片序列转换为模型可用的 3D 归一化体数据。")

    tab1, tab2 = st.tabs(["病例预处理", "演示数据生成"])

    with tab2:
        st.subheader("生成演示数据集")
        st.caption("无真实 DICOM 数据时，可生成合成演示集以完成训练与预测流程。")
        n_cases = st.slider("病例数量", 4, 20, 8)
        if st.button("生成演示数据集", type="primary"):
            path = build_demo_dataset(DEMO_DATA_DIR, num_cases=n_cases)
            st.success(f"已生成 {n_cases} 例演示数据：{path}")
            _ensure_demo_data()

    with tab1:
        _ensure_demo_data()
        cases = sorted([d.name for d in DEMO_DATA_DIR.iterdir() if d.is_dir()])
        case_name = st.selectbox("选择待预处理病例", cases, key="pre_case")
        if st.button("执行预处理", type="primary"):
            tensor, log = preprocess_folder(DEMO_DATA_DIR / case_name)
            out = export_preprocessed_npy(tensor, case_name)
            st.success(f"预处理完成，已保存至 {out}")
            st.json(log)
            st.session_state["last_preprocessed"] = str(out)
            st.session_state["last_tensor"] = tensor

        if "last_tensor" in st.session_state:
            t = st.session_state["last_tensor"]
            st.info(f"最近预处理输出形状: {t.shape}")


def page_training():
    st.header("🧠 模型训练")
    st.markdown("多任务学习：愈合期五分类 + 受伤天数回归。")

    _ensure_demo_data()

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("训练轮次", 1, 50, DEFAULT_EPOCHS)
        batch_size = st.number_input("批大小", 1, 16, DEFAULT_BATCH_SIZE)
    with col2:
        lr = st.number_input("学习率", format="%.1e", value=DEFAULT_LR, step=1e-5)

    progress = st.progress(0)
    status = st.empty()

    def on_progress(msg: str, p: float):
        status.text(msg)
        progress.progress(min(1.0, p))

    if st.button("开始训练", type="primary"):
        try:
            with st.spinner("训练中，请稍候..."):
                summary = train_demo_model(
                    DEMO_DATA_DIR,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    lr=float(lr),
                    progress_callback=on_progress,
                )
            st.session_state["train_summary"] = summary
            st.success(f"训练完成！检查点: {summary['checkpoint']}")

            fig = plot_training_loss(summary["history"])
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"训练失败: {e}")


def page_prediction():
    st.header("🔮 模型预测")
    st.markdown("对预处理后的 3D 体数据进行推理，输出愈合期与受伤天数。")

    ckpts = sorted(CHECKPOINTS_DIR.glob("demo_model_*.pt")) if CHECKPOINTS_DIR.exists() else []
    ckpt_choice = st.selectbox(
        "模型检查点",
        ["自动使用最新"] + [str(p) for p in ckpts],
    )
    checkpoint = None if ckpt_choice == "自动使用最新" else Path(ckpt_choice)

    mode = st.radio("输入方式", ["演示病例", "上传 NPY", "使用最近预处理结果"], horizontal=True)

    uploaded_npy = None
    demo_case = None
    if mode == "演示病例":
        _ensure_demo_data()
        cases = sorted([d.name for d in DEMO_DATA_DIR.iterdir() if d.is_dir()])
        demo_case = st.selectbox("演示病例", cases, key="pred_case_sel")
    elif mode == "上传 NPY":
        uploaded_npy = st.file_uploader("上传 .npy", type=["npy"], key="pred_npy")
    elif "last_preprocessed" not in st.session_state:
        st.warning("请先在「数据预处理」中执行预处理。")

    if st.button("执行预测", type="primary"):
        try:
            if mode == "演示病例" and demo_case:
                vol = load_volume_from_folder(DEMO_DATA_DIR / demo_case)
                result = predict_from_volume(vol, checkpoint)
                st.session_state["last_case"] = demo_case
            elif mode == "使用最近预处理结果" and "last_preprocessed" in st.session_state:
                result = predict_from_npy(Path(st.session_state["last_preprocessed"]), checkpoint)
                st.session_state["last_case"] = "preprocessed"
            elif mode == "上传 NPY" and uploaded_npy:
                with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
                    tmp.write(uploaded_npy.getvalue())
                    result = predict_from_npy(Path(tmp.name), checkpoint)
                st.session_state["last_case"] = "upload"
            else:
                st.warning("请先选择或上传有效输入。")
                return

            st.session_state["last_prediction"] = result

            c1, c2 = st.columns(2)
            c1.metric("愈合期预测", result["healing_period"])
            c2.metric("受伤天数（回归）", f"{result['predicted_days_injured']} 天")

            st.subheader("各类别概率")
            import pandas as pd

            labels = list(result["class_probabilities"].keys())
            values = list(result["class_probabilities"].values())
            fig = plot_class_probabilities(labels, values)
            st.pyplot(fig)
            plt.close(fig)
            st.json(result)
        except Exception as e:
            st.error(f"预测失败: {e}")


def page_report():
    st.header("📄 报告生成")
    st.markdown("导出 HTML 格式的预测分析报告或训练报告，可用于软著材料附件。")

    report_type = st.radio("报告类型", ["预测分析报告", "训练过程报告"], horizontal=True)

    if report_type == "预测分析报告":
        if "last_prediction" not in st.session_state:
            st.warning("请先在「模型预测」模块执行一次预测。")
            return
        case_id = st.session_state.get("last_case", "unknown")
        analysis = st.session_state.get("last_analysis")
        prediction = st.session_state["last_prediction"]
        if st.button("生成预测报告", type="primary"):
            path = generate_prediction_report(case_id, analysis, prediction)
            st.success(f"报告已生成: {path}")
            with open(path, encoding="utf-8") as f:
                st.download_button("下载 HTML 报告", f.read(), file_name=path.name, mime="text/html")
    else:
        if "train_summary" not in st.session_state:
            st.warning("请先在「模型训练」模块完成一次训练。")
            return
        if st.button("生成训练报告", type="primary"):
            path = generate_training_report(st.session_state["train_summary"])
            st.success(f"报告已生成: {path}")
            with open(path, encoding="utf-8") as f:
                st.download_button("下载 HTML 报告", f.read(), file_name=path.name, mime="text/html")

    st.markdown(f"历史报告目录: `{REPORTS_DIR}`")
    if REPORTS_DIR.exists():
        reports = sorted(REPORTS_DIR.glob("*.html"), reverse=True)
        if reports:
            st.subheader("最近报告")
            for r in reports[:5]:
                st.text(r.name)


def main():
    module = sidebar()
    pages = {
        "系统首页": page_home,
        "图像分析": page_image_analysis,
        "数据预处理": page_preprocessing,
        "模型训练": page_training,
        "模型预测": page_prediction,
        "报告生成": page_report,
    }
    pages[module]()


if __name__ == "__main__":
    main()
