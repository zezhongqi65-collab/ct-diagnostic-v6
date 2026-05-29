"""
诊断管道封装模块 — 桥接 completeV6 和 source_code_evaluator。
为 Streamlit 前端提供干净的 API。
"""
import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import pandas as pd
import numpy as np

from completeV6_patched import (
    CAUSAL_ORDER,
    DIM_NAMES,
    DAG_EDGES,
    prepare_data,
    train_model,
    compute_adaptive_thresholds,
    layer1_counterfactual,
    layer2_dual_diagnosis,
    plot_dual_diagnosis,
    generate_report_template,
    generate_teacher_report_v6,
    save_report,
    estimate_all_causal_effects,
    backdoor_causal_effect,
    ollama_generate,
)

from source_code_evaluator_patched import (
    CTCodeAnalyzer,
    evaluate_code,
    prepare_pipeline_input,
)


def build_reference_model(n_samples: int = 200):
    """构建参考模型（基于模拟数据训练）"""
    df = prepare_data(n=n_samples)
    model, X_train, _ = train_model(df)
    shap_threshold, ce_threshold = compute_adaptive_thresholds(model, X_train)
    return model, X_train, df, shap_threshold, ce_threshold


def train_on_data(df: pd.DataFrame):
    """在用户上传的数据上训练模型

    Args:
        df: 包含五维度列和'作业质量'列的DataFrame

    Returns:
        model, X_train, shap_threshold, ce_threshold
    """
    features = [c for c in CAUSAL_ORDER if c in df.columns]
    model, X_train, _ = train_model(df[features + ['作业质量']])
    shap_threshold, ce_threshold = compute_adaptive_thresholds(model, X_train[features])
    return model, X_train[features], shap_threshold, ce_threshold


def run_single_diagnosis(
    student_row: pd.Series,
    model,
    X_train: pd.DataFrame,
    df_all: pd.DataFrame,
    student_id: str,
) -> dict:
    """对单个学生运行完整的三层诊断（不含Layer 3 ITE）

    Returns:
        dict with keys: cf_report, cf_best_dim, cf_best_gain,
                        dd_report, df_results, sv, ce_dict,
                        shap_threshold, ce_threshold, base_pred
    """
    features = [c for c in CAUSAL_ORDER if c in student_row.index]
    student = student_row[features].astype(float)
    base_pred = model.predict(student.values.reshape(1, -1))[0]

    # Layer 1
    cf_report, cf_best_dim, cf_best_gain = layer1_counterfactual(
        model, student, student_id
    )

    # Layer 2
    dd_report, df_results, sv, ce_dict, shap_threshold, ce_threshold = (
        layer2_dual_diagnosis(model, student, df_all, student_id, X_train)
    )

    return {
        'student_id': student_id,
        'student': student,
        'base_pred': base_pred,
        'cf_report': cf_report,
        'cf_best_dim': cf_best_dim,
        'cf_best_gain': cf_best_gain,
        'dd_report': dd_report,
        'df_results': df_results,
        'sv': sv,
        'ce_dict': ce_dict,
        'shap_threshold': shap_threshold,
        'ce_threshold': ce_threshold,
    }


def run_code_diagnosis(
    code: str,
    student_id: str,
    model,
    X_train: pd.DataFrame,
    df_all: pd.DataFrame,
    shap_threshold: float,
    ce_threshold: float,
    homework_score: float = 75.0,
) -> dict:
    """评估代码并运行诊断

    Returns:
        dict with evaluation_result + diagnosis_result
    """
    analyzer = CTCodeAnalyzer()
    eval_result = analyzer.evaluate(code, student_id=student_id)
    scores = eval_result.to_pipeline_dict()

    student = pd.Series(scores)
    base_pred = model.predict(student.values.reshape(1, -1))[0]

    cf_report, cf_best_dim, cf_best_gain = layer1_counterfactual(
        model, student, student_id
    )

    dd_report, df_results, sv, ce_dict, _, _ = layer2_dual_diagnosis(
        model, student, df_all, student_id, X_train
    )

    return {
        'student_id': student_id,
        'student': student,
        'base_pred': base_pred,
        'eval_result': eval_result,
        'cf_report': cf_report,
        'cf_best_dim': cf_best_dim,
        'cf_best_gain': cf_best_gain,
        'dd_report': dd_report,
        'df_results': df_results,
        'sv': sv,
        'ce_dict': ce_dict,
        'shap_threshold': shap_threshold,
        'ce_threshold': ce_threshold,
    }
