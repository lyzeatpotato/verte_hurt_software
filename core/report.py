"""报告生成模块：HTML / 文本格式预测与训练报告。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from config import APP_NAME, APP_VERSION, REPORTS_DIR


def _html_header(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
body {{ font-family: "PingFang SC", "Microsoft YaHei", sans-serif; margin: 40px; color: #222; }}
h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 8px; }}
h2 {{ color: #2874a6; margin-top: 28px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #bdc3c7; padding: 10px 14px; text-align: left; }}
th {{ background: #ebf5fb; }}
.footer {{ margin-top: 40px; font-size: 12px; color: #7f8c8d; }}
.badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; background: #d5f5e3; }}
</style>
</head>
<body>
"""


def generate_prediction_report(
    case_id: str,
    analysis: dict[str, Any] | None,
    prediction: dict[str, Any],
    output_dir: Path | None = None,
) -> Path:
    """生成病例预测分析报告（HTML）。"""
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"prediction_report_{case_id}_{ts}.html"

    rows_analysis = ""
    if analysis:
        rows_analysis = f"""
        <tr><th>体数据尺寸</th><td>{analysis.get('volume_shape')}</td></tr>
        <tr><th>推荐展示层</th><td>第 {analysis.get('recommended_center_slice')} 层</td></tr>
        <tr><th>整体图像质量</th><td><span class="badge">{analysis.get('overall_quality')}</span></td></tr>
        <tr><th>全局平均灰度</th><td>{analysis.get('global_mean_intensity')}</td></tr>
        """

    probs = prediction.get("class_probabilities", {})
    prob_rows = "".join(
        f"<tr><td>{k}</td><td>{v:.2%}" if isinstance(v, float) and v <= 1 else f"<tr><td>{k}</td><td>{v}"
        for k, v in probs.items()
    )
    # fix percent display
    prob_rows = ""
    for k, v in probs.items():
        pct = f"{v * 100:.2f}%" if isinstance(v, float) and v <= 1 else str(v)
        prob_rows += f"<tr><td>{k}</td><td>{pct}</td></tr>"

    html = _html_header(f"{APP_NAME} - 预测报告")
    html += f"""
<h1>{APP_NAME}</h1>
<p>版本 {APP_VERSION} · 报告类型：病例预测分析 · 生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<h2>一、病例信息</h2>
<table>
<tr><th>病例编号</th><td>{case_id}</td></tr>
</table>

<h2>二、图像分析摘要</h2>
<table>
{rows_analysis if rows_analysis else '<tr><td colspan="2">未执行图像分析或未提供分析结果</td></tr>'}
</table>

<h2>三、AI 预测结果</h2>
<table>
<tr><th>愈合期分类</th><td><strong>{prediction.get('healing_period')}</strong>（类别索引 {prediction.get('healing_class_index')}）</td></tr>
<tr><th>预测受伤天数</th><td><strong>{prediction.get('predicted_days_injured')}</strong> 天</td></tr>
<tr><th>模型检查点</th><td>{prediction.get('checkpoint_used', '—')}</td></tr>
</table>

<h2>四、愈合期分类概率</h2>
<table>
<tr><th>类别</th><th>概率</th></tr>
{prob_rows}
</table>

<h2>五、说明与免责声明</h2>
<p>{prediction.get('confidence_note', '')}</p>
<p>本报告由软件系统自动生成，预测结果需结合临床病史、实验室检查及影像科医师判读综合使用，不作为单独诊断依据。</p>

<div class="footer">© {APP_NAME} · 自动生成报告</div>
</body></html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_training_report(train_summary: dict[str, Any], output_dir: Path | None = None) -> Path:
    """生成训练过程报告（HTML）。"""
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"training_report_{ts}.html"

    history = train_summary.get("history", [])
    hist_rows = "".join(
        f"<tr><td>{h['epoch']}</td><td>{h['loss_cls']}</td><td>{h['loss_reg']}</td><td>{h['loss_total']}</td></tr>"
        for h in history
    )

    html = _html_header(f"{APP_NAME} - 训练报告")
    html += f"""
<h1>{APP_NAME} — 模型训练报告</h1>
<p>生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<h2>训练配置</h2>
<table>
<tr><th>训练样本数</th><td>{train_summary.get('samples')}</td></tr>
<tr><th>训练轮次</th><td>{train_summary.get('epochs')}</td></tr>
<tr><th>计算设备</th><td>{train_summary.get('device')}</td></tr>
<tr><th>模型检查点</th><td>{train_summary.get('checkpoint')}</td></tr>
<tr><th>完成时间</th><td>{train_summary.get('finished_at')}</td></tr>
</table>

<h2>逐轮损失</h2>
<table>
<tr><th>Epoch</th><th>分类损失</th><th>回归损失</th><th>总损失</th></tr>
{hist_rows}
</table>

<div class="footer">© {APP_NAME} · 训练报告</div>
</body></html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
