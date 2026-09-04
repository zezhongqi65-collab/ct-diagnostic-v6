"""
量表前测数据预处理脚本。

把收集到的原始量表数据（学生ID + 班级 + A1..A15 + 已有成绩）
转换为诊断系统可直接批量导入的格式：

1. 计算五维度平均分（抽象/分解/算法设计/建模/评估）
2. 已有成绩归一化到 0-100（对应系统的「作业质量」）
3. 输出两个 CSV：
   - 前测_系统诊断输入.csv  → 直接上传到 Web 应用「批量诊断」
   - 前测_完整数据.csv      → 保留原始题项与班级，供后续分组、信度、ANCOVA 分析

用法:
    python preprocess_scale_data.py 前测数据.xlsx
    python preprocess_scale_data.py 前测数据.csv --score-max 100

输出目录默认为 data/，可用 --out-dir 指定。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台默认 GBK，强制 stdout 用 UTF-8，避免中文/符号乱码（Python 3.7+）
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_PARENT = Path(__file__).resolve().parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import pandas as pd

from utils.scale_data import (
    ALL_ITEMS,
    CAUSAL_ORDER,
    SCORE_MAX,
    compute_dimension_scores,
    map_answer_to_score,
    normalize_score,
)


def _load_data(path: Path) -> pd.DataFrame:
    """按扩展名读取 Excel 或 CSV。"""
    suffix = path.suffix.lower()
    if suffix in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    if suffix == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'不支持的文件类型: {suffix}（仅支持 .xlsx/.xls/.csv）')


def _validate(raw: pd.DataFrame) -> None:
    """校验必需列与题项取值，异常时给出明确提示。"""
    required = ['学生ID', '班级'] + ALL_ITEMS + ['已有成绩']
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f'缺少必需列: {missing}\n请使用模板生成的表头格式。')

    # 题项可识别性检查（应填 A-E 或 1-5）
    scores = raw[ALL_ITEMS].apply(lambda col: col.map(map_answer_to_score))
    invalid = scores.isnull().any(axis=1)
    if invalid.any():
        bad_ids = raw.loc[invalid, '学生ID'].astype(str).tolist()
        print(f'[!] 以下学生存在无法识别的答案（应填 A-E 或 1-5），将按缺失处理: {bad_ids}')

    # 缺失值检查
    null_rows = raw[required].isnull().any(axis=1)
    if null_rows.any():
        bad_ids = raw.loc[null_rows, '学生ID'].astype(str).tolist()
        print(f'[!] 以下学生存在缺失值，将按缺失处理（维度计算会跳过该题）: {bad_ids}')


def preprocess(raw: pd.DataFrame, score_max: float) -> pd.DataFrame:
    """从原始数据生成含五维度与归一化作业质量的完整表。

    返回列: 学生ID, 班级, A1..A15, 五维度(CAUSAL_ORDER), 作业质量
    """
    result = raw.copy()

    # 五维度平均分（1-5）
    dim_scores = compute_dimension_scores(raw)
    for dim in CAUSAL_ORDER:
        result[dim] = dim_scores[dim]

    # 已有成绩 → 作业质量（0-100）
    result['作业质量'] = normalize_score(raw['已有成绩'], score_max)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='预处理量表前测数据')
    parser.add_argument('input', type=str, help='原始数据文件（.xlsx/.xls/.csv）')
    parser.add_argument('--score-max', type=float, default=SCORE_MAX,
                        help=f'已有成绩满分（默认 {SCORE_MAX}）')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='输出目录（默认 data/）')
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f'[X] 找不到文件: {in_path}')

    raw = _load_data(in_path)
    print(f'已读取 {len(raw)} 条记录，列: {list(raw.columns)}')

    _validate(raw)

    full = preprocess(raw, args.score_max)

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent / 'data'
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 诊断系统输入（只含系统需要的列）
    diag_cols = ['学生ID'] + CAUSAL_ORDER + ['作业质量']
    diag_path = out_dir / '前测_系统诊断输入.csv'
    full[diag_cols].to_csv(diag_path, index=False, encoding='utf-8-sig')

    # 2) 完整数据（保留班级与原始题项，供统计分析）
    full_path = out_dir / '前测_完整数据.csv'
    full.to_csv(full_path, index=False, encoding='utf-8-sig')

    print(f'[OK] 预处理完成:')
    print(f'   诊断输入 -> {diag_path}')
    print(f'   完整数据 -> {full_path}')
    print(f'   五维度均分示例（前 3 名）:')
    print(full[['学生ID'] + CAUSAL_ORDER + ['作业质量']].head(3).to_string(index=False))


if __name__ == '__main__':
    main()
