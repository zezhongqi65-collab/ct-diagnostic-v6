"""
计算思维诊断系统 — 后端 API（FastAPI）

封装真实诊断引擎（XGBoost + TreeSHAP + 后门调整因果推断），
为精致 HTML 前端提供 JSON 接口。启动时构建参考模型并缓存。

运行: uvicorn backend:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from completeV6_patched import (
    CAUSAL_ORDER,
    DIM_NAMES,
    counterfactual_simulation,
)
from utils.diagnostics import build_reference_model, run_single_diagnosis

app = FastAPI(title="计算思维诊断系统")

# ── 启动时构建参考模型（缓存 SHAP explainer 与自适应阈值）────
print("正在构建参考诊断模型（模拟数据训练 XGBoost + SHAP）...")
_model, _X_train, _df_all, _shap_th, _ce_th = build_reference_model(n_samples=200)
print("参考模型就绪。")


class DiagnoseRequest(BaseModel):
    student_id: str = "S001"
    scores: dict[str, float]


class BatchRequest(BaseModel):
    students: list[DiagnoseRequest]


def _run_diagnosis(student_id: str, scores: dict[str, float]) -> dict:
    """对单个学生运行三层诊断，返回 JSON 友好的结果。"""
    student = pd.Series({dim: float(scores.get(dim, 3.0)) for dim in CAUSAL_ORDER})
    r = run_single_diagnosis(student, _model, _X_train, _df_all, student_id)

    # Layer 1 反事实表
    layer1 = []
    for feat in CAUSAL_ORDER:
        df_cf, _, current_val = counterfactual_simulation(_model, student, feat)
        layer1.append({
            "dim": feat,
            "current": round(float(current_val), 1),
            "series": df_cf[[f'{feat}假设值', '预测成绩', '相比当前变化']].to_dict(orient='records'),
        })

    return {
        "student_id": str(r['student_id']),
        "base_pred": round(float(r['base_pred']), 1),
        "scores": {dim: round(float(r['student'][dim]), 1) for dim in CAUSAL_ORDER},
        "causal_order": CAUSAL_ORDER,
        "dim_names": DIM_NAMES,
        "cf_best_dim": r['cf_best_dim'],
        "cf_best_gain": round(float(r['cf_best_gain']), 1),
        "shap_threshold": round(float(r['shap_threshold']), 3),
        "ce_threshold": round(float(r['ce_threshold']), 1),
        "layer1": layer1,
        "layer2": r['df_results'].to_dict(orient='records'),
    }


@app.post("/api/diagnose")
def diagnose(req: DiagnoseRequest):
    return _run_diagnosis(req.student_id, req.scores)


@app.post("/api/batch")
def batch(req: BatchRequest):
    results = [_run_diagnosis(s.student_id, s.scores) for s in req.students]
    return {"results": results}


# ── 静态前端（放在最后挂载，避免覆盖 /api 路由）────
_STATIC_DIR = _PARENT / 'static'
_STATIC_DIR.mkdir(exist_ok=True)


@app.get("/")
def index():
    index_file = _STATIC_DIR / 'index.html'
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "后端已就绪，请访问 /static/ 或部署前端 index.html"}


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
