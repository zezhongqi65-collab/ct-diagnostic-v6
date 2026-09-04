"""
量表数据共享工具 — 维度题项映射与五维度计算。

这是「量表数据收集模板」和「预处理脚本」共用的唯一数据源，
保证两处对量表结构的理解完全一致。

⚠️ 重要：维度 → 题项 的映射当前与 completeV6_patched.prepare_data
严格对齐（分解 A1-A3、抽象 A4-A6、建模 A7-A8、算法设计 A9-A10、评估 A11-A15）。
若你实际使用的「张屹计算思维量表」题号顺序或维度归属与此不同，
只需修改下方 DIMENSION_ITEMS 字典，模板与预处理脚本会同步生效。
"""
from __future__ import annotations

import pandas as pd

# ── 维度 → 题项 映射（唯一需核对的地方）──────────────────
DIMENSION_ITEMS: dict[str, list[str]] = {
    '分解': ['A1', 'A2', 'A3'],
    '抽象': ['A4', 'A5', 'A6'],
    '建模': ['A7', 'A8'],
    '算法设计': ['A9', 'A10'],
    '评估': ['A11', 'A12', 'A13', 'A14', 'A15'],
}

# 按题号顺序展开的完整题项列表（A1..A15）
ALL_ITEMS: list[str] = [
    item for items in DIMENSION_ITEMS.values() for item in items
]

# 已有成绩满分（用于归一化到系统 0-100 分制）
# 若你的成绩满分不是 100（例如 120 分制），请改为对应满分，或运行时用 --score-max 指定。
SCORE_MAX: float = 100.0

# 五维度输出顺序（与 completeV6_patched.CAUSAL_ORDER 严格一致，勿改动）
# 此处本地定义，避免本模块被 Web 应用之外的工具脚本引用时加载 xgboost/shap 等重依赖。
CAUSAL_ORDER: list[str] = ['抽象', '分解', '算法设计', '建模', '评估']


# 纸质量表选项 A-E → 1-5 分 的映射
# A.完全不符合=1  B.比较不符合=2  C.一般=3  D.比较符合=4  E.完全符合=5
ANSWER_MAP: dict[str, int] = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5,
}


def map_answer_to_score(value) -> float:
    """把纸质量表作答（A-E 字母）或已录入的 1-5 数字统一转成 1-5 分。

    无法识别的值返回 NaN，交由校验环节提示。
    """
    if pd.isna(value):
        return float('nan')
    if isinstance(value, (int, float)):
        return float(value) if 1 <= value <= 5 else float('nan')
    v = str(value).strip().upper()
    if v in ANSWER_MAP:
        return float(ANSWER_MAP[v])
    try:
        num = int(v)
        return float(num) if 1 <= num <= 5 else float('nan')
    except ValueError:
        return float('nan')


def compute_dimension_scores(df: pd.DataFrame) -> pd.DataFrame:
    """从原始题项列（A1..A15）计算五维度平均分。

    题项可填 A-E 字母（纸质量表作答）或 1-5 数字，统一转换为 1-5 后取维度平均。

    Args:
        df: 包含 A1..A15 列的 DataFrame。

    Returns:
        列名为五个维度名、取值范围 1-5 的 DataFrame。
    """
    out = pd.DataFrame(index=df.index)
    for dim, items in DIMENSION_ITEMS.items():
        scores = df[items].apply(lambda col: col.map(map_answer_to_score))
        out[dim] = scores.mean(axis=1)
    return out


def normalize_score(score: pd.Series, score_max: float = SCORE_MAX) -> pd.Series:
    """把已有成绩归一化到系统 0-100 分制。

    Args:
        score: 原始成绩序列。
        score_max: 原始成绩的满分值。
    """
    return (score.astype(float) / score_max * 100.0).clip(0, 100).round(1)
