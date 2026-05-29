"""
计算思维诊断系统 — Web 应用
双通道（量表 / 源代码）→ 三层因果可解释诊断 → 报告生成
"""
import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import io
import base64
import tempfile
import zipfile
import os

from utils.diagnostics import (
    build_reference_model,
    train_on_data,
    run_single_diagnosis,
    run_code_diagnosis,
    CAUSAL_ORDER,
    DIM_NAMES,
    DAG_EDGES,
    CTCodeAnalyzer,
    evaluate_code,
    prepare_pipeline_input,
)

from utils.report import (
    build_markdown_report,
    build_batch_summary,
    to_excel_download,
    markdown_download_link,
)

from completeV6_patched import (
    plot_dual_diagnosis,
    generate_report_template,
    generate_teacher_report_v6,
    ollama_generate,
    layer1_counterfactual,
    layer2_dual_diagnosis,
    save_report,
)

# ── 页面配置 ──────────────────────────────────────────────

st.set_page_config(
    page_title="计算思维诊断系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 中文字体（继承completeV6_patched的跨平台配置）──────
from completeV6_patched import _CJK_FONTS
import matplotlib
matplotlib.rcParams['font.sans-serif'] = _CJK_FONTS
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 会话状态初始化 ─────────────────────────────────────────

DEFAULT_STATE = {
    'model': None,
    'X_train': None,
    'df_all': None,
    'shap_threshold': None,
    'ce_threshold': None,
    'model_ready': False,
    'diag_result': None,
    'batch_results': None,
    'ollama_available': False,
}

for key, val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── 辅助函数 ──────────────────────────────────────────────


def ensure_model():
    """确保参考模型已加载"""
    if not st.session_state.model_ready:
        with st.spinner("正在构建参考诊断模型（基于模拟数据训练XGBoost + SHAP）..."):
            model, X_train, df_all, shap_th, ce_th = build_reference_model(n_samples=200)
            st.session_state.model = model
            st.session_state.X_train = X_train
            st.session_state.df_all = df_all
            st.session_state.shap_threshold = shap_th
            st.session_state.ce_threshold = ce_th
            st.session_state.model_ready = True


def check_ollama():
    """检测Ollama是否可用"""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            st.session_state.ollama_available = True
            return True
    except Exception:
        pass
    st.session_state.ollama_available = False
    return False


def render_dimension_scores(scores: dict):
    """渲染五维度得分条"""
    cols = st.columns(5)
    for i, dim in enumerate(CAUSAL_ORDER):
        score = scores.get(dim, 0)
        color = (
            '#2ECC71' if score >= 4.0 else
            '#F39C12' if score >= 3.0 else
            '#E74C3C'
        )
        with cols[i]:
            st.markdown(f"**{dim}**")
            st.markdown(
                f"<h2 style='text-align:center; color:{color};'>{score:.1f}</h2>",
                unsafe_allow_html=True,
            )
            st.progress(min(float(score) / 5.0, 1.0))


def render_classification_badge(tag: str) -> str:
    """渲染分类标签HTML"""
    colors = {
        '✅': ('#27AE60', '#D5F5E3'),
        '⚠️': ('#E67E22', '#FDEBD0'),
        '💡': ('#2980B9', '#D6EAF8'),
        '➖': ('#7F8C8D', '#EAECEE'),
    }
    emoji = tag.split()[0] if tag else '➖'
    fg, bg = colors.get(emoji, ('#7F8C8D', '#EAECEE'))
    return f'<span style="background:{bg};color:{fg};padding:2px 10px;border-radius:12px;font-weight:bold;">{tag}</span>'


# ═══════════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🧠 计算思维诊断系统")
    st.markdown("双通道 · 三层因果可解释AI")
    st.markdown("---")

    mode = st.radio(
        "## 📌 选择输入模式",
        ["📊 量表输入（CSV/Excel）", "💻 代码输入（Python）"],
        help="量表模式：上传CSV/Excel进行诊断；代码模式：上传.py文件自动评估五维度得分后诊断",
    )

    st.markdown("---")

    # Ollama 配置
    st.markdown("### 🤖 大模型润色（可选）")
    use_ollama = st.checkbox("启用Ollama报告润色", value=False)
    if use_ollama:
        ollama_model = st.text_input("模型名称", value="qwen:7b")
        if st.button("检测Ollama连接"):
            if check_ollama():
                st.success("✅ Ollama已连接")
            else:
                st.warning("⚠️ Ollama未启动，将使用模板报告")

    st.markdown("---")

    # 批量模式
    batch_mode = st.checkbox("📦 批量诊断模式", help="上传包含多个学生的数据进行批量诊断")

    st.markdown("---")
    st.caption("v6.0 — 三层约束保真机制")
    st.caption("模板填充 + 大模型润色 + 自动校验")


# ═══════════════════════════════════════════════════════════
# 主页内容
# ═══════════════════════════════════════════════════════════

st.title("计算思维诊断系统")
st.markdown(
    "基于可解释AI（SHAP + 因果效应）的教育诊断工具，"
    "支持量表数据与源代码双通道输入，自动生成三层可解释诊断报告。"
)

# ── 模式 A：量表输入 ─────────────────────────────────────

if mode.startswith("📊"):
    st.header("📊 量表输入模式")

    if batch_mode:
        st.subheader("批量诊断 — 上传学生数据文件")

        uploaded_file = st.file_uploader(
            "上传CSV或Excel文件（须包含学生ID、五维度得分列）",
            type=["csv", "xlsx", "xls"],
            help="必含列：student_id（或 学生ID）、抽象、分解、算法设计、建模、评估。可选：作业质量",
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    raw_df = pd.read_csv(uploaded_file)
                else:
                    raw_df = pd.read_excel(uploaded_file)

                st.success(f"已加载 {len(raw_df)} 条记录")
                st.dataframe(raw_df.head(10), use_container_width=True)

                # 列名映射
                id_col = next(
                    (c for c in raw_df.columns if c.lower() in ('student_id', '学生id', '学生编号', 'id', '姓名')),
                    raw_df.columns[0],
                )

                has_homework = '作业质量' in raw_df.columns

                if st.button("🚀 开始批量诊断", type="primary", use_container_width=True):
                    ensure_model()

                    # 在用户数据上重训练模型（如果有作业质量列）
                    if has_homework:
                        with st.spinner("在用户数据上重训练诊断模型..."):
                            model, X_train, shap_th, ce_th = train_on_data(raw_df)
                    else:
                        model = st.session_state.model
                        X_train = st.session_state.X_train
                        shap_th = st.session_state.shap_threshold
                        ce_th = st.session_state.ce_threshold

                    results = []
                    progress = st.progress(0)
                    status = st.empty()

                    for idx, (_, row) in enumerate(raw_df.iterrows()):
                        sid = str(row.get(id_col, f"S{idx:03d}"))
                        status.text(f"正在诊断: {sid} ({idx + 1}/{len(raw_df)})")

                        student_row = row[CAUSAL_ORDER].astype(float)
                        student_row['作业质量'] = row.get('作业质量', 75.0)

                        diag = run_single_diagnosis(
                            student_row, model, X_train,
                            st.session_state.df_all, sid,
                        )
                        results.append(diag)
                        progress.progress((idx + 1) / len(raw_df))

                    st.session_state.batch_results = results
                    status.text("✅ 批量诊断完成！")
                    st.success(f"已完成 {len(results)} 名学生的诊断")

            except Exception as e:
                st.error(f"文件处理失败: {e}")

        # 批量结果展示
        if st.session_state.batch_results:
            st.markdown("---")
            st.subheader("📋 批量诊断汇总")

            summary_df = build_batch_summary(st.session_state.batch_results)
            st.dataframe(summary_df, use_container_width=True)

            # 导出
            col1, col2 = st.columns(2)
            with col1:
                csv = summary_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 下载CSV汇总",
                    csv,
                    f"诊断汇总_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                )
            with col2:
                excel_data = io.BytesIO()
                with pd.ExcelWriter(excel_data, engine='openpyxl') as writer:
                    summary_df.to_excel(writer, index=False, sheet_name='诊断汇总')
                st.download_button(
                    "📥 下载Excel汇总",
                    excel_data.getvalue(),
                    f"诊断汇总_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                )

            # 个体报告查看
            st.markdown("---")
            st.subheader("📝 查看个体报告")
            student_ids = [r['student_id'] for r in st.session_state.batch_results]
            selected_id = st.selectbox("选择学生", student_ids)
            selected = next(
                r for r in st.session_state.batch_results if r['student_id'] == selected_id
            )
            st.session_state.diag_result = selected

    else:
        # 单学生量表输入
        st.subheader("单个学生诊断")

        input_method = st.radio("数据输入方式", ["手动输入五维度得分", "上传CSV（取第一行）"], horizontal=True)

        if input_method == "手动输入五维度得分":
            cols = st.columns(5)
            scores = {}
            for i, dim in enumerate(CAUSAL_ORDER):
                with cols[i]:
                    scores[dim] = st.slider(
                        f"{dim}得分",
                        1.0, 5.0, 3.0, 0.1,
                        help=DIM_NAMES.get(dim, ''),
                    )
            student_id = st.text_input("学生ID", value="S001")

            if st.button("🔍 开始诊断", type="primary", use_container_width=True):
                ensure_model()
                with st.spinner("正在运行三层诊断分析..."):
                    student_row = pd.Series(scores)
                    diag = run_single_diagnosis(
                        student_row,
                        st.session_state.model,
                        st.session_state.X_train,
                        st.session_state.df_all,
                        student_id,
                    )
                    st.session_state.diag_result = diag
                st.success("诊断完成！")

        else:
            uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])
            if uploaded_file:
                raw_df = pd.read_csv(uploaded_file)
                st.dataframe(raw_df.head(), use_container_width=True)

                id_col = st.selectbox("选择学生ID列", raw_df.columns.tolist())
                student_idx = st.number_input("选择第几行数据", 0, len(raw_df) - 1, 0)

                if st.button("🔍 开始诊断", type="primary", use_container_width=True):
                    ensure_model()
                    row = raw_df.iloc[student_idx]
                    sid = str(row[id_col])

                    scores = {}
                    for dim in CAUSAL_ORDER:
                        if dim in raw_df.columns:
                            scores[dim] = float(row[dim])
                        else:
                            st.error(f"缺少列: {dim}")
                            st.stop()

                    student_row = pd.Series(scores)
                    diag = run_single_diagnosis(
                        student_row,
                        st.session_state.model,
                        st.session_state.X_train,
                        st.session_state.df_all,
                        sid,
                    )
                    st.session_state.diag_result = diag
                    st.success(f"诊断完成！学生: {sid}")

# ── 模式 B：代码输入 ─────────────────────────────────────

else:
    st.header("💻 代码输入模式")

    if batch_mode:
        st.subheader("批量代码评估 — 上传ZIP文件")

        zip_file = st.file_uploader("上传包含多个.py文件的ZIP压缩包", type=["zip"])
        homework_score = st.slider("默认作业质量分", 0, 100, 75, help="当无法从代码推断作业质量时使用")

        if zip_file:
            try:
                with zipfile.ZipFile(zip_file) as zf:
                    py_files = [f for f in zf.namelist() if f.endswith('.py') and not f.startswith('__')]
                    st.info(f"检测到 {len(py_files)} 个Python文件")

                if st.button("🚀 开始批量评估与诊断", type="primary", use_container_width=True):
                    ensure_model()
                    analyzer = CTCodeAnalyzer()

                    results = []
                    progress = st.progress(0)
                    status = st.empty()

                    with zipfile.ZipFile(zip_file) as zf:
                        for i, py_file in enumerate(py_files):
                            sid = Path(py_file).stem
                            status.text(f"正在评估: {sid} ({i + 1}/{len(py_files)})")

                            code = zf.read(py_file).decode('utf-8', errors='replace')
                            diag = run_code_diagnosis(
                                code, sid,
                                st.session_state.model,
                                st.session_state.X_train,
                                st.session_state.df_all,
                                st.session_state.shap_threshold,
                                st.session_state.ce_threshold,
                                homework_score,
                            )
                            results.append(diag)
                            progress.progress((i + 1) / len(py_files))

                    st.session_state.batch_results = results
                    status.text("✅ 批量评估完成！")
                    st.success(f"已完成 {len(results)} 名学生的评估与诊断")

            except Exception as e:
                st.error(f"处理失败: {e}")

        # 批量结果
        if st.session_state.batch_results:
            st.markdown("---")
            st.subheader("📋 批量诊断汇总")
            summary_df = build_batch_summary(st.session_state.batch_results)
            st.dataframe(summary_df, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                csv = summary_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 下载CSV", csv, "批量诊断.csv", "text/csv")
            with col2:
                excel_data = io.BytesIO()
                with pd.ExcelWriter(excel_data, engine='openpyxl') as writer:
                    summary_df.to_excel(writer, index=False, sheet_name='诊断汇总')
                st.download_button("📥 下载Excel", excel_data.getvalue(), "批量诊断.xlsx")

            student_ids = [r['student_id'] for r in st.session_state.batch_results]
            selected_id = st.selectbox("查看个体报告", student_ids)
            selected = next(r for r in st.session_state.batch_results if r['student_id'] == selected_id)
            st.session_state.diag_result = selected

    else:
        st.subheader("单个学生代码评估")

        code_input_method = st.radio("代码输入方式", ["📁 上传.py文件", "✏️ 粘贴代码"], horizontal=True)

        if code_input_method == "📁 上传.py文件":
            py_file = st.file_uploader("上传Python源代码文件", type=["py"])
            if py_file:
                code = py_file.read().decode('utf-8', errors='replace')
                student_id = st.text_input("学生ID", value=Path(py_file.name).stem)
                with st.expander("📄 代码预览"):
                    st.code(code, language="python")
        else:
            code = st.text_area(
                "粘贴Python源代码",
                height=300,
                placeholder="在此粘贴学生的Python代码...",
            )
            student_id = st.text_input("学生ID", value="S001")

        homework_score = st.slider("历史作业质量分（如无可使用默认值）", 0, 100, 75)

        if st.button("🔍 开始评估与诊断", type="primary", use_container_width=True):
            if not code or not code.strip():
                st.error("请先输入代码")
            else:
                ensure_model()

                with st.spinner("正在AST静态分析 + 三层因果诊断..."):
                    diag = run_code_diagnosis(
                        code, student_id,
                        st.session_state.model,
                        st.session_state.X_train,
                        st.session_state.df_all,
                        st.session_state.shap_threshold,
                        st.session_state.ce_threshold,
                        homework_score,
                    )
                    st.session_state.diag_result = diag
                st.success("评估与诊断完成！")


# ═══════════════════════════════════════════════════════════
# 诊断结果展示（共用）
# ═══════════════════════════════════════════════════════════

if st.session_state.diag_result:
    r = st.session_state.diag_result

    st.markdown("---")
    st.header(f"📋 诊断报告 — {r['student_id']}")

    # 概览卡片
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("预测作业质量", f"{r['base_pred']:.1f} 分")
    with col2:
        priority_count = len(r['df_results'][r['df_results']['分类'].str.startswith('✅')])
        st.metric("优先干预维度数", f"{priority_count} 个")
    with col3:
        avg_score = r['student'].mean()
        st.metric("五维度均分", f"{avg_score:.2f} / 5")

    # 五维度得分
    st.markdown("### 📊 五维度得分概览")
    render_dimension_scores(r['student'].to_dict())

    # ── Tab切换三层结果 ──

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Layer 1 反事实分析",
        "🎯 Layer 2 双维度诊断",
        "🔬 Layer 3 ITE 个体处理效应",
        "📝 综合报告",
        "📄 代码评估详情",
    ])

    # ── Layer 1 ──
    with tab1:
        st.markdown("### 反事实分析（Counterfactual Estimation）")
        st.warning(
            "⚠️ **免责声明**：以下结果为模型层面的假设模拟，仅展示"
            "「若该维度独立变化，模型预测如何改变」。"
            "实际干预效果请以Layer 2因果效应为准。"
        )

        student = r['student']
        features = [c for c in CAUSAL_ORDER if c in student.index]
        model = st.session_state.model or st.session_state.get('model')

        if model is not None:
            base_pred = r['base_pred']

            for feat in features:
                from completeV6_patched import counterfactual_simulation
                df_cf, _, current_val = counterfactual_simulation(model, student, feat)
                target_row = df_cf[df_cf[f'{feat}假设值'] == 5]
                gain = target_row['相比当前变化'].values[0] if not target_row.empty else 0

                with st.expander(
                    f"**{DIM_NAMES.get(feat, feat)}** — 当前 {current_val:.1f}分 → 5分预计变化 +{gain:.1f}分"
                ):
                    st.dataframe(df_cf, use_container_width=True)

            # 最佳维度
            st.markdown("---")
            st.markdown(f"**🎯 反事实结论**：优先提升维度 **{DIM_NAMES.get(r.get('cf_best_dim', ''), r.get('cf_best_dim', ''))}**")
            st.caption("以上为模型假设模拟，真实效果以Layer 2为准")

    # ── Layer 2 ──
    with tab2:
        st.markdown("### 双维度诊断矩阵（SHAP × 因果效应）")

        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            st.markdown("#### 📈 诊断散点图")
            fig = plot_dual_diagnosis(
                r['student'], r['sv'], r['ce_dict'],
                r['student_id'], r['shap_threshold'], r['ce_threshold'],
            )
            st.pyplot(fig)
            plt.close('all')

        with col_table:
            st.markdown("#### 📋 分类详情")
            for _, row in r['df_results'].iterrows():
                tag = row['分类']
                emoji = tag.split()[0]
                st.markdown(
                    f"**{emoji} {DIM_NAMES.get(row['维度'], row['维度'])}** — 得分 {row['得分']:.1f}/5",
                )
                st.markdown(
                    f"SHAP: {row['SHAP']:+.3f} | 因果效应: +{row['因果效应']:.1f}分",
                )
                st.caption(row['建议'])
                st.markdown("---")

        st.markdown("---")
        st.caption(f"自适应阈值 — SHAP: {r['shap_threshold']:.3f} | 因果效应: {r['ce_threshold']:.1f}分（基于训练集分布自动计算）")

    # ── Layer 3 ITE ──
    with tab3:
        st.markdown("### 🔬 Layer 3: ITE 个体处理效应估计")
        st.info(
            "Layer 3 需要教师提供前测/后测实验数据（实验组+对照组）。"
            "如需使用此功能，请准备包含 `抽象、分解、算法设计、建模、评估、T（干预标记）、Y（成绩变化）` 列的CSV文件。"
        )
        with st.expander("📋 查看数据收集模板"):
            st.markdown("""
            | 阶段 | 操作 |
            |------|------|
            | **前测（第0周）** | 计算思维量表 → 5维度分数 X |
            | **随机分组** | 实验组接受干预 / 对照组正常上课 |
            | **干预（第1-4周）** | 针对性训练，记录出勤 |
            | **后测（第5周）** | 再次测量 → post_Y |
            | **数据列** | T=1(实验)/0(对照), Y=post-baseline |
            """)

    # ── 综合报告 ──
    with tab4:
        st.markdown("### 📝 综合诊断报告")

        # 生成报告
        md_report = build_markdown_report(r)
        template_report = generate_report_template(
            r['student_id'], r['student'], r['df_results'], r['base_pred'],
            r.get('cf_best_dim', ''), r.get('cf_best_gain', 0),
            r['shap_threshold'], r['ce_threshold'],
        )

        if use_ollama and st.session_state.ollama_available:
            with st.spinner("正在用大模型润色报告语言..."):
                from completeV6_patched import polish_report_with_llm, verify_report_fidelity
                polished = polish_report_with_llm(template_report, r['student_id'])
                if not polished.startswith('['):
                    is_ok, checks = verify_report_fidelity(polished, r['df_results'], r['student_id'])
                    if is_ok:
                        st.success("🤖 大模型润色完成 | 保真度校验通过 ✅")
                        md_report = polished
                    else:
                        st.warning("⚠️ 保真度校验未通过，使用模板报告（100%保真）")
                        for c in checks:
                            if c.startswith('❌'):
                                st.error(c)
                else:
                    st.warning(f"⚠️ Ollama不可用（{polished}），使用模板报告")

        # 渲染报告
        st.markdown(md_report)

        # 下载按钮
        col_down1, col_down2 = st.columns(2)
        with col_down1:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "📥 下载 Markdown 报告",
                md_report.encode('utf-8'),
                f"诊断报告_{r['student_id']}_{timestamp}.md",
                "text/markdown",
            )
        with col_down2:
            st.download_button(
                "📥 下载模板报告（未润色）",
                template_report.encode('utf-8'),
                f"诊断报告_{r['student_id']}_模板_{timestamp}.md",
                "text/markdown",
            )

    # ── 代码评估详情 ──
    with tab5:
        if 'eval_result' in r and r['eval_result'] is not None:
            eval_r = r['eval_result']
            st.markdown("### 🔬 源代码AST分析详情")
            st.markdown(eval_r.overall_summary)

            for dim_name in CAUSAL_ORDER:
                dim_result = eval_r.dimensions.get(dim_name)
                if dim_result:
                    with st.expander(f"**{dim_name}** — {dim_result.score}/5.0"):
                        for feat in dim_result.features:
                            st.markdown(f"- **{feat.name}**: {feat.explanation}")
                        if dim_result.code_evidence:
                            st.markdown("**代码证据**:")
                            for ev in dim_result.code_evidence[:5]:
                                st.caption(f"  • {ev}")
        else:
            st.info("当前为量表输入模式，无代码评估详情。切换到代码输入模式可查看AST分析。")


# ═══════════════════════════════════════════════════════════
# 页脚
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    "计算思维诊断系统 v6.0 | "
    "基于可解释AI（XGBoost + SHAP + 后门调整因果效应） | "
    "三层约束保真机制确保报告可靠性 | "
    "仅供教育参考，不构成教育决策的唯一依据"
)
