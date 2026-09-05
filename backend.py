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


# ── 干预出题（跨周去重：持久化已用题目，抽题自动跳过）────

import json as _json


def _resource_path(relative: str) -> Path:
    """资源读取路径：兼容源码运行与 PyInstaller 打包（sys._MEIPASS）。"""
    base = Path(getattr(sys, '_MEIPASS', _PARENT))
    return base / relative


def _writable_data_dir() -> Path:
    """可写数据目录：打包后用 exe 所在目录，源码运行用项目根目录。"""
    if getattr(sys, 'frozen', False):
        base = Path(sys.executable).resolve().parent
    else:
        base = _PARENT
    return base / 'data'


_BANK: dict | None = None


def _get_bank() -> dict:
    """加载题库（data/question_bank.json），缓存。"""
    global _BANK
    if _BANK is None:
        p = _resource_path('data/question_bank.json')
        with open(p, encoding='utf-8') as f:
            _BANK = _json.load(f)
    return _BANK


def _history_path() -> Path:
    return _writable_data_dir() / 'intervention_history.json'


def _load_history() -> dict:
    p = _history_path()
    if p.exists():
        try:
            return _json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def _save_history(history: dict) -> None:
    p = _history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')


def _used_ids(history: dict, dim: str) -> set[str]:
    """某维度在所有周次里已经用过的题目 id 集合。"""
    used: set[str] = set()
    for week_dims in history.values():
        for qid in week_dims.get(dim, []):
            used.add(qid)
    return used


class InterventionRequest(BaseModel):
    dims: list[str]
    count: int = 10
    week: int = 1
    max_diff: int = 3


class StudentIntervention(BaseModel):
    id: str
    dims: list[str]
    name: str = ""
    cls: str = ""


class StudentsInterventionRequest(BaseModel):
    students: list[StudentIntervention]
    count: int = 10
    week: int = 1
    max_diff: int = 3
    max_dims: int = 2


def _serialize(q: dict) -> dict:
    return {'id': q['id'], 'type': q['type'], 'stem': q['stem'],
            'answer': q['answer'], 'hint': q.get('hint', ''), 'difficulty': q['difficulty']}


def _pick_dim(bank: dict, history: dict, dim: str, count: int, max_diff: int,
              ) -> tuple[list[dict], list[str]]:
    """为某维度选出 count 道题（跨周去重），返回 (题目列表, 告警列表)。"""
    used = _used_ids(history, dim)
    remaining_unused = [q for q in bank[dim] if q['id'] not in used]
    pool_filtered = [q for q in bank[dim] if q['difficulty'] <= max_diff]
    avail = [q for q in pool_filtered if q['id'] not in used]
    difficulty_relaxed = False
    if len(avail) < count:  # 难度内不够，放宽难度（但仍不重复）
        avail = remaining_unused
        difficulty_relaxed = True

    selected = avail[:count]
    shortage = count - len(selected)
    warnings: list[str] = []
    if shortage > 0:  # 未用题目彻底用完，才允许重复，并强告警
        selected += pool_filtered[:shortage]
        warnings.append(
            f'维度「{dim}」未使用题目已用完，本次开始重复使用旧题（建议扩充题库或减少每周题数）。'
        )
    else:
        remaining_after = len(remaining_unused) - len(selected)
        if difficulty_relaxed:
            warnings.append(
                f'维度「{dim}」在所选难度内的未用题不足，已放宽难度补齐（仍保证不重复）。'
            )
        if remaining_after < count:
            warnings.append(
                f'维度「{dim}」未使用题目仅剩 {remaining_after} 道，下周可能不够，请注意。'
            )
    return selected, warnings


@app.post("/api/intervention/generate")
def intervention_generate(req: InterventionRequest):
    """按维度生成题单，跨周不重复；题库不足时明确告警。"""
    bank = _get_bank()
    history = _load_history()
    week_key = str(req.week)
    used_now = history.setdefault(week_key, {})

    warnings: list[str] = []
    groups: list[dict] = []
    total = 0

    for dim in req.dims:
        if dim not in bank:
            warnings.append(f'未知维度「{dim}」，已跳过。')
            continue
        selected, w = _pick_dim(bank, history, dim, req.count, req.max_diff)
        warnings += w
        used_now[dim] = list(dict.fromkeys(used_now.get(dim, []) + [q['id'] for q in selected]))
        total += len(selected)
        groups.append({
            'dim': dim,
            'full': DIM_NAMES.get(dim, dim),
            'questions': [_serialize(q) for q in selected],
        })

    _save_history(history)
    return {'week': req.week, 'groups': groups, 'warnings': warnings, 'total': total}


@app.post("/api/intervention/students")
def intervention_students(req: StudentsInterventionRequest):
    """按学生生成「一人一单」：每名学生一张个人题单（含其薄弱维度的题）+ 分发清单。"""
    bank = _get_bank()
    history = _load_history()
    week_key = str(req.week)
    used_now = history.setdefault(week_key, {})

    warnings: list[str] = []
    dim_sets: dict[str, list[dict]] = {}

    # 所有学生涉及的维度各选一次题，同维度学生共享本周题目
    for dim in sorted({d for s in req.students for d in s.dims}):
        if dim not in bank:
            warnings.append(f'未知维度「{dim}」，已跳过。')
            continue
        selected, w = _pick_dim(bank, history, dim, req.count, req.max_diff)
        warnings += w
        used_now[dim] = list(dict.fromkeys(used_now.get(dim, []) + [q['id'] for q in selected]))
        dim_sets[dim] = selected

    _save_history(history)

    packets: list[dict] = []
    for s in req.students:
        dims = [d for d in s.dims if d in dim_sets][:req.max_dims]
        questions = []
        for d in dims:
            for q in dim_sets[d]:
                questions.append({**_serialize(q), 'dim': d})
        packets.append({'id': s.id, 'name': s.name, 'cls': s.cls, 'dims': dims, 'questions': questions})

    distribution = [
        {'cls': p['cls'], 'id': p['id'], 'name': p['name'], 'dims': p['dims'], 'count': len(p['questions'])}
        for p in packets
    ]
    return {'week': req.week, 'packets': packets, 'distribution': distribution, 'warnings': warnings}


@app.get("/api/intervention/history")
def intervention_history():
    return _load_history()


@app.post("/api/intervention/reset")
def intervention_reset():
    _save_history({})
    return {"status": "已清空出题历史"}


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
