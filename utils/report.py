"""
报告生成工具 — Markdown / HTML 格式化与导出。
"""
import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import datetime
import io
import base64

import pandas as pd

from completeV6_patched import (
    DIM_NAMES,
    CAUSAL_ORDER,
    generate_report_template,
)


def build_markdown_report(diag_result: dict) -> str:
    """从诊断结果构建 Markdown 报告"""
    r = diag_result
    student = r['student']
    df_results = r['df_results']
    base_pred = r['base_pred']
    shap_threshold = r['shap_threshold']
    ce_threshold = r['ce_threshold']

    lines = []
    lines.append(f"# 学生 {r['student_id']} 计算思维诊断报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**预测作业质量**: {base_pred:.1f}分 / 100分")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 一、优先干预
    priority_dims = df_results[df_results['分类'].str.startswith('✅')]
    if len(priority_dims) > 0:
        lines.append("## 一、优先干预维度 ✅")
        lines.append("")
        for _, row in priority_dims.iterrows():
            lines.append(f"### {DIM_NAMES[row['维度']]}")
            lines.append(f"- **当前得分**: {row['得分']:.1f} / 5")
            lines.append(f"- **因果效应**: +{row['因果效应']:.1f}分（专项训练预计提分）")
            lines.append(f"- **SHAP重要性**: {abs(row['SHAP']):.3f}（高于阈值 {shap_threshold:.3f}）")
            lines.append(f"- **建议**: {row['建议']}")
            lines.append("")

    # 二、仅观察
    observe_dims = df_results[df_results['分类'].str.startswith('⚠️')]
    if len(observe_dims) > 0:
        lines.append("## 二、需关注但暂不干预 ⚠️")
        lines.append("")
        for _, row in observe_dims.iterrows():
            lines.append(f"### {DIM_NAMES[row['维度']]}")
            lines.append(f"- **当前得分**: {row['得分']:.1f} / 5")
            lines.append(f"- **SHAP重要性**: {abs(row['SHAP']):.3f}（模型认为重要）")
            lines.append(f"- **因果效应**: +{row['因果效应']:.1f}分（单独干预效果有限）")
            lines.append(f"- **原因**: {row['建议']}")
            lines.append("")

    # 三、潜在有效
    potential_dims = df_results[df_results['分类'].str.startswith('💡')]
    if len(potential_dims) > 0:
        lines.append("## 三、备选干预维度 💡")
        lines.append("")
        for _, row in potential_dims.iterrows():
            lines.append(f"### {DIM_NAMES[row['维度']]}")
            lines.append(f"- **当前得分**: {row['得分']:.1f} / 5")
            lines.append(f"- **因果效应**: +{row['因果效应']:.1f}分")
            lines.append(f"- **说明**: {row['建议']}")
            lines.append("")

    # 四、正常
    normal_dims = df_results[df_results['分类'].str.startswith('➖')]
    if len(normal_dims) > 0:
        names = [DIM_NAMES[row['维度']] for _, row in normal_dims.iterrows()]
        lines.append("## 四、当前正常 ➖")
        lines.append(f"维度: {', '.join(names)}，当前表现正常，无需特别干预。")
        lines.append("")

    # 反事实参考
    lines.append("---")
    lines.append("")
    lines.append("## 附：反事实分析参考（模型模拟，仅供参考）")
    cf_best_dim = r.get('cf_best_dim', '')
    cf_best_gain = r.get('cf_best_gain', 0)
    lines.append(f"- 若将「{DIM_NAMES.get(cf_best_dim, '无')}」提升至满分5分")
    lines.append(f"- 模型预测变化: +{cf_best_gain:.1f}分")
    lines.append(f"- **注意**: 以上为模型假设模拟，真实效果请以Layer 2因果效应为准")
    lines.append("")

    # 免责声明
    lines.append("---")
    lines.append("")
    lines.append("## 三层约束保真机制说明")
    lines.append("")
    lines.append("本报告采用三层约束保真机制生成：")
    lines.append("1. **程序模板强制填充**（100%保真）：所有数据、分类、优先级直接来自算法原始输出")
    lines.append("2. **大模型仅做语言润色**：Ollama仅在可用时优化文字表达，不改变数据")
    lines.append("3. **自动保真度校验**：程序自动核对润色后报告是否100%忠实于算法原始数据，否则自动回退")
    lines.append("")
    lines.append("*本报告基于可解释AI分析（SHAP + 因果效应），仅供参考，不构成教育决策的唯一依据*")

    return '\n'.join(lines)


def build_batch_summary(results: list) -> pd.DataFrame:
    """构建批量诊断汇总表"""
    rows = []
    for r in results:
        row = {
            '学生ID': r['student_id'],
            '预测作业质量': f"{r['base_pred']:.1f}",
        }
        for dim in CAUSAL_ORDER:
            row[f'{dim}得分'] = f"{r['student'][dim]:.1f}"
        priority = r['df_results'][r['df_results']['分类'].str.startswith('✅')]
        row['优先干预维度'] = '、'.join(priority['维度'].tolist()) if len(priority) > 0 else '无'
        rows.append(row)
    return pd.DataFrame(rows)


def to_excel_download(df: pd.DataFrame) -> str:
    """将DataFrame转换为Excel下载链接的数据"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='诊断汇总')
    return base64.b64encode(output.getvalue()).decode()


def markdown_download_link(text: str, filename: str, label: str = "📥 下载Markdown报告") -> str:
    """生成Markdown文件下载链接"""
    b64 = base64.b64encode(text.encode('utf-8')).decode()
    return f'<a href="data:text/markdown;base64,{b64}" download="{filename}">{label}</a>'


def format_counterfactual_table(diag_result: dict) -> str:
    """格式化反事实分析结果为HTML表格"""
    r = diag_result
    student = r['student']
    features = [c for c in CAUSAL_ORDER if c in student.index]
    base_pred = r['base_pred']

    rows_html = ""
    for feat in features:
        current = student[feat]
        gains = []
        for target in range(1, 6):
            modified = student.copy()
            modified[feat] = float(target)
            pred = base_pred  # placeholder
            gains.append((target, pred))
        rows_html += f"""
        <tr>
            <td>{DIM_NAMES.get(feat, feat)}</td>
            <td>{current:.1f}</td>
            <td>{'⭐' if current < 3 else '✓'}</td>
        </tr>"""

    return f"""
    <table style="width:100%; border-collapse:collapse;">
        <thead>
            <tr style="background:#f0f0f0;">
                <th>维度</th>
                <th>当前得分</th>
                <th>状态</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """
