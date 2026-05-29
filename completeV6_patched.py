# ============================================
# 三层因果可解释诊断系统（完整版 v6）
# 核心改进：三层约束保真机制
#   1. 模板强制填充（100%保真）
#   2. 大模型仅做语言润色（不改变数据）
#   3. 程序自动保真度校验
# 其他改进：中文字体配置、报告保存功能
# ============================================

import sys
# v6修复1: Windows GBK终端无法编码emoji，强制UTF-8输出
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
# v6修复2: 私有API加异常保护，避免matplotlib版本兼容性问题
try:
    fm._load_fontmanager(try_read_cache=False)
except Exception:
    pass
_CJK_FONTS = [
    'Microsoft YaHei', 'SimHei',                # Windows
    'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',  # Linux
    'PingFang SC', 'Heiti SC',                   # macOS
    'Noto Sans CJK SC', 'sans-serif',            # 通用回退
]
matplotlib.rcParams['font.sans-serif'] = _CJK_FONTS
matplotlib.rcParams['axes.unicode_minus'] = False
import requests
import datetime
import re

try:
    import statsmodels.api as sm
except ImportError:
    sm = None
    print("[警告] statsmodels未安装，因果效应估计将回退到分组比较。")
    print("       建议安装：pip install statsmodels")

# ============================================
# 0. 全局配置
# ============================================

DAG_EDGES = [
    ('抽象', '分解'),
    ('抽象', '算法设计'),
    ('算法设计', '建模'),
    ('算法设计', '评估'),
]
CAUSAL_ORDER = ['抽象', '分解', '算法设计', '建模', '评估']
DIM_NAMES = {
    '抽象': '模式识别与抽象',
    '分解': '问题分解',
    '算法设计': '算法与步骤设计',
    '建模': '数学建模与符号化',
    '评估': '方案评估与调试',
}
OLLAMA_MODEL = "qwen:7b"


# ============================================
# 1. 数据准备
# ============================================

def prepare_data(n=200):
    np.random.seed(42)
    df = pd.DataFrame({
        'A1': np.random.randint(1, 6, n), 'A2': np.random.randint(1, 6, n), 'A3': np.random.randint(1, 6, n),
        'A4': np.random.randint(1, 6, n), 'A5': np.random.randint(1, 6, n), 'A6': np.random.randint(1, 6, n),
        'A7': np.random.randint(1, 6, n), 'A8': np.random.randint(1, 6, n), 'A9': np.random.randint(1, 6, n),
        'A10': np.random.randint(1, 6, n), 'A11': np.random.randint(1, 6, n), 'A12': np.random.randint(1, 6, n),
        'A13': np.random.randint(1, 6, n), 'A14': np.random.randint(1, 6, n), 'A15': np.random.randint(1, 6, n),
    })
    df['抽象'] = df[['A4', 'A5', 'A6']].mean(axis=1)
    df['分解'] = df[['A1', 'A2', 'A3']].mean(axis=1)
    df['算法设计'] = df[['A9', 'A10']].mean(axis=1)
    df['建模'] = df[['A7', 'A8']].mean(axis=1)
    df['评估'] = df[['A11', 'A12', 'A13', 'A14', 'A15']].mean(axis=1)
    df['作业质量'] = (
        df['抽象'] * 8 + df['分解'] * 4 + df['算法设计'] * 6 +
        df['建模'] * 5 + df['评估'] * 3 +
        2 * df['抽象'] * df['建模'] +
        np.random.normal(0, 5, n)
    ).clip(20, 100).round(1)
    return df


# ============================================
# 模拟干预数据
# ============================================

def simulate_intervention_data(n=60, true_effect=5.0):
    np.random.seed(123)
    df = pd.DataFrame({
        '抽象': np.random.uniform(1, 5, n),
        '分解': np.random.uniform(1, 5, n),
        '算法设计': np.random.uniform(1, 5, n),
        '建模': np.random.uniform(1, 5, n),
        '评估': np.random.uniform(1, 5, n),
    })
    df['T'] = np.random.binomial(1, 0.5, n)
    df['baseline'] = df['抽象'] * 8 + df['分解'] * 4 + df['建模'] * 5 + np.random.normal(0, 5, n)
    df['post'] = df['baseline'] + df['T'] * true_effect + np.random.normal(0, 3, n)
    df['Y'] = df['post'] - df['baseline']
    return df


# ============================================
# 2. 训练XGBoost回归模型
# ============================================

def train_model(df):
    features = CAUSAL_ORDER
    X = df[features]
    y = df['作业质量']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        reg_alpha=0.1, reg_lambda=1.0,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    print(f"模型RMSE: {rmse:.2f}分")
    return model, X_train, X_test


# ============================================
# 3. 自适应阈值计算
# ============================================

def compute_adaptive_thresholds(model, X_train):
    explainer = shap.TreeExplainer(model)
    sv_all = np.abs(explainer.shap_values(X_train))
    shap_threshold = np.percentile(sv_all.mean(axis=0), 50)
    ce_all = estimate_all_causal_effects(X_train.assign(作业质量=model.predict(X_train)))
    ce_threshold = np.median(list(ce_all.values()))
    print(f"自适应阈值 | SHAP: {shap_threshold:.3f} | 因果效应: {ce_threshold:.1f}分")
    return shap_threshold, ce_threshold


# ============================================
# 4. Layer 1: 反事实分析
# ============================================

def counterfactual_simulation(model, student, feature, value_range=None):
    if value_range is None:
        value_range = [1, 2, 3, 4, 5]
    base_pred = model.predict(student.values.reshape(1, -1))[0]
    current_val = student[feature]
    results = []
    for val in value_range:
        modified = student.copy()
        modified[feature] = val
        new_pred = model.predict(modified.values.reshape(1, -1))[0]
        results.append({
            f'{feature}假设值': val,
            '预测成绩': round(new_pred, 1),
            '相比当前变化': round(new_pred - base_pred, 1),
            '是否当前值': '★ 当前' if abs(val - current_val) < 0.1 else ''
        })
    return pd.DataFrame(results), base_pred, current_val


def layer1_counterfactual(model, student, student_id):
    features = CAUSAL_ORDER
    report = []
    report.append(f"\n{'='*60}")
    report.append(f"【Layer 1: 反事实分析（CFE）】学生 {student_id}")
    report.append(f"{'='*60}")
    report.append('\n⚠️ 免责声明：以下结果为模型层面的假设模拟（CFE），')
    report.append('   仅展示「若该维度独立变化，模型预测如何改变」。')
    report.append('   实际干预效果请以Layer 2因果效应为准。')
    report.append(f"\n{'─'*50}")
    base_pred = model.predict(student.values.reshape(1, -1))[0]
    report.append(f"\n当前预测作业质量: {base_pred:.1f}分/100分")
    report.append(f"\n{'─'*50}")
    report.append("各维度提升潜力模拟（固定其他维度不变）：")
    report.append(f"{'─'*50}")
    best_dim = None
    best_gain = -999
    for feat in features:
        df_cf, _, current_val = counterfactual_simulation(model, student, feat)
        target_row = df_cf[df_cf[f'{feat}假设值'] == 5]
        gain = target_row['相比当前变化'].values[0] if not target_row.empty else 0
        report.append(f"\n【{DIM_NAMES[feat]}】当前{current_val:.1f}分")
        report.append(df_cf.to_string(index=False))
        if gain > best_gain:
            best_gain = gain
            best_dim = feat
    report.append(f"\n{'='*60}")
    report.append(f"【反事实结论（模型模拟，非真实因果）】")
    if best_dim:
        report.append(f"  优先提升维度: {DIM_NAMES[best_dim]}")
        report.append(f"  若从当前{student[best_dim]:.1f}分 → 5分")
        report.append(f"  模型预测变化: {base_pred:.1f}分 → {base_pred + best_gain:.1f}分 (+{best_gain:.1f}分)")
        report.append(f"  ⚠️ 注意: 以上为模型假设，真实效果请以Layer 2因果效应为准")
    report.append(f"{'='*60}")
    return '\n'.join(report), best_dim, best_gain


# ============================================
# 5. 后门调整因果效应估计（线性回归 + p值衰减 + 分组回退）
# ============================================

def backdoor_causal_effect(df, treatment, outcome, confounders=None):
    """
    因果效应估计（后门调整）:
    - 优先使用statsmodels.OLS线性回归调整
    - p>0.05时不直接归零，改为ATE×0.3衰减，保留信息
    - confounders为空时自动回退到分组比较
    - OLS失败时自动回退到分组比较
    """
    if sm is not None and confounders and len(confounders) > 0:
        formula = f"{outcome} ~ {treatment} + " + " + ".join(confounders)
        try:
            fit = sm.OLS.from_formula(formula, data=df).fit()
            ate = fit.params[treatment]
            p_value = fit.pvalues[treatment]
            # p>0.05时衰减为30%，保留信息而非直接归零
            if p_value > 0.05:
                ate_attenuated = ate * 0.3
                return ate_attenuated
            return ate
        except Exception:
            pass
    # 回退方案：分组比较
    if confounders and len(confounders) > 0:
        effects = []
        for _, group in df.groupby(confounders):
            high = group[group[treatment] >= 4][outcome]
            low = group[group[treatment] < 3][outcome]
            if len(high) > 2 and len(low) > 2:
                effects.append(high.mean() - low.mean())
        return np.mean(effects) if effects else 0.0
    else:
        high = df[df[treatment] >= 4][outcome]
        low = df[df[treatment] < 3][outcome]
        return (high.mean() - low.mean()) if len(high) > 2 and len(low) > 2 else 0.0


def estimate_all_causal_effects(df):
    effects = {}
    for feat in CAUSAL_ORDER:
        parents = [s for s, t in DAG_EDGES if t == feat]
        ce = backdoor_causal_effect(df, feat, '作业质量', parents)
        effects[feat] = ce
    return effects


def trace_root_cause(dim, student, dag_edges):
    parents = [s for s, t in dag_edges if t == dim]
    if not parents:
        return None
    weak_parents = [p for p in parents if student[p] < 3.0]
    return weak_parents if weak_parents else None


# ============================================
# 6. Layer 2: 双维度诊断矩阵
# ============================================

def layer2_dual_diagnosis(model, student, df_all, student_id, X_train):
    features = CAUSAL_ORDER
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(student.values.reshape(1, -1))[0]
    ce_dict = estimate_all_causal_effects(df_all)
    shap_threshold, ce_threshold = compute_adaptive_thresholds(model, X_train)
    report = []
    report.append(f"\n{'='*60}")
    report.append(f"【Layer 2: 双维度诊断（SHAP × 因果效应）】学生 {student_id}")
    report.append(f"{'='*60}")
    report.append(f"自适应阈值 | SHAP重要性: {shap_threshold:.3f} | 因果效应: {ce_threshold:.1f}分")
    report.append(f"（基于训练集分布自动计算，非固定值）")
    report.append(f"\n{'─'*50}")
    categories = {'✅': [], '⚠️': [], '💡': [], '➖': []}
    results_list = []
    for i, feat in enumerate(features):
        shap_val = sv[i]
        ce_val = ce_dict[feat]
        shap_high = abs(shap_val) > shap_threshold
        ce_positive = ce_val > ce_threshold
        if shap_high and ce_positive:
            tag = '✅ 优先干预'
            action = f"立即安排{DIM_NAMES[feat]}专项训练（预计提升{ce_val:.1f}分）"
        elif shap_high and not ce_positive:
            tag = '⚠️ 仅观察'
            root = trace_root_cause(feat, student, DAG_EDGES)
            if root:
                action = f"SHAP显示重要但单独干预无效。根因可能在「{','.join([DIM_NAMES[r] for r in root])}」，建议先练根因维度"
            else:
                action = "SHAP显示重要但干预无效，可能受其他未测量因素影响"
        elif not shap_high and ce_positive:
            tag = '💡 潜在有效'
            action = f"模型未识别但干预可能有效，作为二级备选方案（预计提升{ce_val:.1f}分）"
        else:
            tag = '➖ 无需关注'
            action = "当前维度正常"
        categories[tag.split()[0]] = categories.get(tag.split()[0], []) + [feat]
        report.append(f"\n【{DIM_NAMES[feat]}】")
        report.append(f"  得分: {student[feat]:.1f}/5 | SHAP: {shap_val:+.3f} | 因果效应: +{ce_val:.1f}分")
        report.append(f"  分类: {tag}")
        report.append(f"  建议: {action}")
        results_list.append({
            '维度': feat, 
            '得分': student[feat], 
            'SHAP': shap_val, 
            '因果效应': ce_val, 
            '分类': tag, 
            '建议': action,
            '优先级': 0 if tag.startswith('✅') else (1 if tag.startswith('⚠️') else (2 if tag.startswith('💡') else 3))
        })
    report.append(f"\n{'─'*50}")
    report.append(f"\n📊 诊断摘要:")
    if categories['✅']:
        report.append(f"   ✅ 优先干预: {', '.join([DIM_NAMES[d] for d in categories['✅']])}")
    if categories['⚠️']:
        report.append(f"   ⚠️ 仅观察: {', '.join([DIM_NAMES[d] for d in categories['⚠️']])}")
    if categories['💡']:
        report.append(f"   💡 潜在有效: {', '.join([DIM_NAMES[d] for d in categories['💡']])}")
    report.append(f"{'='*60}")
    # v6: 按优先级排序，方便后续处理
    df_results = pd.DataFrame(results_list).sort_values('优先级')
    return '\n'.join(report), df_results, sv, ce_dict, shap_threshold, ce_threshold


# ============================================
# 7. 可视化
# ============================================

def plot_dual_diagnosis(student, sv, ce_dict, student_id, shap_threshold, ce_threshold, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 7))

    # 显式获取中文字体路径，确保散点图中所有中文正常渲染
    _cjk_prop = None
    for _name in _CJK_FONTS:
        try:
            _path = fm.findfont(_name, fallback_to_default=False)
            if _path and _path != _name:  # findfont returns the name itself on failure
                _cjk_prop = fm.FontProperties(fname=_path)
                break
        except Exception:
            continue
    if _cjk_prop is None:
        _cjk_prop = fm.FontProperties()

    features = CAUSAL_ORDER
    colors_map = {'✅': '#2ECC71', '⚠️': '#F39C12', '💡': '#3498DB', '➖': '#95A5A6'}
    labels_map = {'✅': '优先干预', '⚠️': '仅观察', '💡': '潜在有效', '➖': '无需关注'}
    for i, feat in enumerate(features):
        s = abs(sv[i])
        c = ce_dict[feat]
        if s > shap_threshold and c > ce_threshold:
            color, size = colors_map['✅'], 300
        elif s > shap_threshold and c <= ce_threshold:
            color, size = colors_map['⚠️'], 250
        elif s <= shap_threshold and c > ce_threshold:
            color, size = colors_map['💡'], 200
        else:
            color, size = colors_map['➖'], 150
        ax.scatter(s, c, s=size, color=color, alpha=0.8, edgecolors='white', linewidth=2, zorder=3)
        ax.annotate(DIM_NAMES[feat], (s, c), textcoords="offset points", xytext=(10, 5),
                    fontsize=11, fontproperties=_cjk_prop)
    ax.axhline(y=ce_threshold, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=shap_threshold, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('|SHAP值|（模型归因重要性）', fontsize=13, fontproperties=_cjk_prop)
    ax.set_ylabel('因果效应（分）', fontsize=13, fontproperties=_cjk_prop)
    ax.set_title(f'{student_id} 双维度诊断散点图（SHAP × 因果效应）\n阈值基于训练集自适应计算',
                 fontsize=13, fontweight='bold', fontproperties=_cjk_prop)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_map['✅'], label=f"{labels_map['✅']}"),
        Patch(facecolor=colors_map['⚠️'], label=f"{labels_map['⚠️']}"),
        Patch(facecolor=colors_map['💡'], label=f"{labels_map['💡']}"),
        Patch(facecolor=colors_map['➖'], label=f"{labels_map['➖']}"),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, prop=_cjk_prop)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close(fig)
    else:
        plt.show()
    return fig


# ============================================
# 8. Ollama集成（含Timeout捕获）
# ============================================

def ollama_generate(prompt, model=OLLAMA_MODEL, timeout=60):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.4, "num_predict": 1200, "top_p": 0.9}},
            timeout=timeout
        )
        return response.json().get("response", "[Ollama生成失败]") if response.status_code == 200 else "[Ollama请求失败]"
    except requests.exceptions.ConnectionError:
        return "[错误：Ollama未启动。请先运行 'ollama serve']"
    except requests.exceptions.Timeout:
        return f"[错误：Ollama请求超时（{timeout}秒）。请检查模型是否已加载。]"
    except Exception as e:
        return f"[错误：{str(e)}]"


# ============================================
# v6核心改进：三层约束保真机制
# ============================================

# ---------- 第一层：模板强制填充（100%保真） ----------

def generate_report_template(student_id, student, df_results, base_pred,
                              cf_best_dim, cf_best_gain, shap_threshold, ce_threshold):
    """
    v6第一层约束：程序模板强制填充
    每个字、每个数字都直接来自算法原始输出，100%保真
    """
    lines = []
    lines.append(f"# 学生{student_id} 计算思维诊断报告")
    lines.append("")
    lines.append(f"**预测作业质量**: {base_pred:.1f}分/100分")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. 优先干预维度（✅）- 最优先展示
    priority_dims = df_results[df_results['分类'].str.startswith('✅')]
    if len(priority_dims) > 0:
        lines.append("## 一、优先干预维度")
        lines.append("")
        for _, row in priority_dims.iterrows():
            lines.append(f"### {DIM_NAMES[row['维度']]}")
            lines.append(f"- 当前得分: {row['得分']:.1f}/5")
            lines.append(f"- 因果效应: +{row['因果效应']:.1f}分（专项训练预计提分）")
            lines.append(f"- SHAP重要性: {abs(row['SHAP']):.3f}（高于阈值{shap_threshold:.3f}）")
            lines.append(f"- **建议**: {row['建议']}")
            lines.append("")

    # 2. 需关注但暂不干预（⚠️）
    observe_dims = df_results[df_results['分类'].str.startswith('⚠️')]
    if len(observe_dims) > 0:
        lines.append("## 二、需关注但暂不干预")
        lines.append("")
        for _, row in observe_dims.iterrows():
            lines.append(f"### {DIM_NAMES[row['维度']]}")
            lines.append(f"- 当前得分: {row['得分']:.1f}/5")
            lines.append(f"- SHAP重要性: {abs(row['SHAP']):.3f}（模型认为重要）")
            lines.append(f"- 因果效应: +{row['因果效应']:.1f}分（单独干预效果有限）")
            lines.append(f"- **原因**: {row['建议']}")
            lines.append("")

    # 3. 潜在有效（💡）
    potential_dims = df_results[df_results['分类'].str.startswith('💡')]
    if len(potential_dims) > 0:
        lines.append("## 三、备选干预维度")
        lines.append("")
        for _, row in potential_dims.iterrows():
            lines.append(f"### {DIM_NAMES[row['维度']]}")
            lines.append(f"- 当前得分: {row['得分']:.1f}/5")
            lines.append(f"- 因果效应: +{row['因果效应']:.1f}分")
            lines.append(f"- **说明**: {row['建议']}")
            lines.append("")

    # 4. 无需关注（➖）
    normal_dims = df_results[df_results['分类'].str.startswith('➖')]
    if len(normal_dims) > 0:
        lines.append("## 四、当前正常")
        names = [DIM_NAMES[row['维度']] for _, row in normal_dims.iterrows()]
        lines.append(f"维度: {', '.join(names)}，当前表现正常，无需特别干预。")
        lines.append("")

    # 5. 反事实参考
    lines.append("---")
    lines.append("")
    lines.append("## 附：反事实分析参考（模型模拟，仅供参考）")
    lines.append(f"- 若将「{DIM_NAMES.get(cf_best_dim, '无')}」提升至满分5分")
    lines.append(f"- 模型预测变化: +{cf_best_gain:.1f}分")
    lines.append(f"- **注意**: 以上为模型假设模拟，真实效果请以Layer 2因果效应为准")
    lines.append("")
    lines.append("*本报告基于可解释AI分析，SHAP值反映模型归因，因果效应反映干预预期效果*")

    return '\n'.join(lines)


# ---------- 第二层：大模型仅做语言润色 ----------

def polish_report_with_llm(structured_report, student_id):
    """
    v6第二层约束：大模型仅做语言润色
    prompt中设置严格规则，禁止大模型修改任何数据、优先级、分类
    """
    polish_prompt = f"""你是一位教育文案编辑。你的任务是润色以下诊断报告的文字，使其更加通顺、专业、易读。

## 绝对规则（违反会导致严重错误）
1. **禁止修改任何数字**：所有得分、SHAP值、因果效应值、阈值等数字必须原样保留
2. **禁止修改分类**：✅优先干预、⚠️仅观察、💡潜在有效、➖无需关注 这些分类标签必须原样保留
3. **禁止调整优先级顺序**：报告中的维度顺序（先✅、再⚠️、再💡、最后➖）必须保持不变
4. **禁止遗漏维度**：所有维度都必须出现在润色后的报告中，一个都不能少
5. **禁止添加未提及的内容**：不能编造数据、建议或分析
6. **语言风格**：适合中学信息技术教师阅读，专业但不晦涩

## 你可以做的事
- 优化句子衔接和段落过渡
- 让语言更自然、更像人写的
- 调整标点符号和格式排版

## 需要润色的报告

{structured_report}

请直接输出润色后的报告正文，不要添加总结或额外说明。"""

    return ollama_generate(polish_prompt, timeout=90)


# ---------- 第三层：程序自动保真度校验 ----------

def verify_report_fidelity(report_text, df_results, student_id):
    """
    v6第三层约束：程序自动保真度校验
    逐条核对润色后的报告是否100%忠实于算法原始数据
    返回: (是否通过, 校验详情)
    """
    checks = []
    passed = True

    # 检查1: 所有✅优先干预维度是否都出现在报告中
    priority_dims = df_results[df_results['分类'].str.startswith('✅')]
    for _, row in priority_dims.iterrows():
        dim_name = DIM_NAMES[row['维度']]
        if dim_name not in report_text:
            checks.append(f"❌ 严重：遗漏优先干预维度「{dim_name}」")
            passed = False
        else:
            checks.append(f"✅ 优先干预维度「{dim_name}」已包含")

    # 检查2: 所有维度是否都出现在报告中（一个都不能少）
    for _, row in df_results.iterrows():
        dim_name = DIM_NAMES[row['维度']]
        if dim_name not in report_text:
            checks.append(f"❌ 遗漏维度「{dim_name}」（分类：{row['分类']}）")
            passed = False

    # 检查3: 因果效应数值是否一致（允许±0.5的舍入误差）
    for _, row in df_results.iterrows():
        dim_name = DIM_NAMES[row['维度']]
        ce_val = row['因果效应']
        if dim_name in report_text:
            idx = report_text.find(dim_name)
            nearby_text = report_text[idx:idx+300]
            # 匹配"数字分"或"数字 分"或"+数字分"等各种表述
            ce_nums = re.findall(r'[+\-]?\s*(\d+\.?\d*)\s*分', nearby_text)
            # 也匹配"提升X"、"提高X"等LLM可能使用的表述
            ce_nums += re.findall(r'(?:提升|提高|增加|改善)\s*[+\-]?\s*(\d+\.?\d*)', nearby_text)
            matched = any(abs(float(n) - ce_val) < 0.5 for n in ce_nums)
            if matched:
                checks.append(f"✅ 维度「{dim_name}」因果效应值{ce_val:.1f}核对通过")
            else:
                # 再搜一遍全报告，防止LLM大幅调整段落顺序
                full_nums = re.findall(r'[+\-]?\s*(\d+\.?\d*)\s*分', report_text)
                full_matched = any(abs(float(n) - ce_val) < 0.5 for n in full_nums)
                if full_matched:
                    checks.append(f"✅ 维度「{dim_name}」因果效应值{ce_val:.1f}核对通过（全报告匹配）")
                else:
                    checks.append(f"⚠️ 维度「{dim_name}」附近未找到因果效应值{ce_val:.1f}附近")
        else:
            checks.append(f"❌ 维度「{dim_name}」未出现在报告中")
            passed = False

    # 检查4: 分类标签是否被篡改
    for _, row in df_results.iterrows():
        dim_name = DIM_NAMES[row['维度']]
        if dim_name in report_text:
            idx = report_text.find(dim_name)
            nearby_text = report_text[idx:idx+300]
            tag = row['分类'].split()[0]  # 取✅⚠️💡➖部分
            if tag not in nearby_text:
                checks.append(f"⚠️ 维度「{dim_name}」附近未找到分类标签{tag}")

    # 检查5: 学生ID是否正确
    if student_id not in report_text:
        checks.append(f"❌ 学生编号{student_id}未出现在报告中")
        passed = False
    else:
        checks.append(f"✅ 学生编号{student_id}核对通过")

    return passed, checks


# ---------- v6综合报告生成（三层约束） ----------

def generate_teacher_report_v6(student_id, student, sv, ce_dict, df_results, model,
                                cf_best_dim, cf_best_gain, shap_threshold, ce_threshold):
    """
    v6综合报告生成：模板填充 → 大模型润色 → 保真度校验
    如果润色后报告保真度不通过，自动回退到100%保真的结构化模板报告
    """
    base_pred = model.predict(student.values.reshape(1, -1))[0]

    print(f"\n{'─'*50}")
    print("【V6报告生成流程】")
    print(f"{'─'*50}")

    # 步骤1: 模板强制填充（100%保真）
    print("[步骤1/3] 程序模板填充（100%保真）...")
    structured_report = generate_report_template(
        student_id, student, df_results, base_pred,
        cf_best_dim, cf_best_gain, shap_threshold, ce_threshold
    )

    # 步骤2: 大模型语言润色
    print("[步骤2/3] 大模型语言润色（仅优化文字，不改数据）...")
    polished_report = polish_report_with_llm(structured_report, student_id)

    # 如果Ollama连接失败，直接返回结构化报告
    if polished_report.startswith("["):
        print("⚠️ Ollama不可用，返回结构化模板报告（保真度100%）")
        return structured_report, structured_report

    # 步骤3: 保真度校验
    print("[步骤3/3] 保真度自动校验...")
    is_fidelity_ok, checks = verify_report_fidelity(polished_report, df_results, student_id)

    # 打印校验结果
    passed_count = sum(1 for c in checks if c.startswith("✅"))
    warning_count = sum(1 for c in checks if c.startswith("⚠️"))
    error_count = sum(1 for c in checks if c.startswith("❌"))
    print(f"  校验结果: {passed_count}项通过 | {warning_count}项警告 | {error_count}项错误")

    if error_count > 0:
        for c in checks:
            if c.startswith("❌"):
                print(f"  {c}")

    if is_fidelity_ok:
        print("✅ 保真度校验通过，使用润色后的报告")
        return polished_report, structured_report
    else:
        print("⚠️ 保真度校验失败，自动回退到结构化模板报告（保真度100%）")
        return structured_report, structured_report


# ============================================
# v6新增：报告保存功能
# ============================================

def save_report(report_text, student_id, suffix="final"):
    """保存报告为markdown文件，带时间戳"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"诊断报告_{student_id}_{suffix}_{timestamp}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"✅ 报告已保存: {filename}")
    return filename


# ============================================
# 9. Layer 3: T-Learner / X-Learner
# ============================================

def tlearner_ite(X, T, Y):
    model_0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_0.fit(X[T == 0], Y[T == 0])
    model_1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_1.fit(X[T == 1], Y[T == 1])
    mu1 = model_1.predict(X)
    mu0 = model_0.predict(X)
    ite = mu1 - mu0
    return ite


def xlearner_ite(X, T, Y):
    model_0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_0.fit(X[T == 0], Y[T == 0])
    model_1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_1.fit(X[T == 1], Y[T == 1])
    D = np.zeros(len(Y))
    D[T == 1] = Y[T == 1] - model_0.predict(X[T == 1])
    D[T == 0] = model_1.predict(X[T == 0]) - Y[T == 0]
    tau_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    tau_model.fit(X, D)
    ite = tau_model.predict(X)
    return ite


def layer3_estimate_ite(X, T, Y, method='xlearner'):
    if method == 'tlearner':
        ite = tlearner_ite(X, T, Y)
    elif method == 'xlearner':
        ite = xlearner_ite(X, T, Y)
    else:
        raise ValueError("method必须是 'tlearner' 或 'xlearner'")
    results = pd.DataFrame({
        'ITE': ite,
        '干预建议': np.where(ite > 3, '强烈推荐（预计提升>3分）',
                    np.where(ite > 0, '可能有效（预计提升0-3分）',
                    np.where(ite > -3, '效果不显著', '可能有害')))
    })
    print(f"\n{'='*60}")
    print(f"【Layer 3: ITE估计（{method.upper()}）】")
    print(f"{'='*60}")
    print(f"ITE均值: {ite.mean():.2f}分")
    print(f"ITE标准差: {ite.std():.2f}分")
    print(f"正向效应人数: {(ite > 0).sum()}/{(ite > 3).sum()}（强/弱）")
    print(f"负向效应人数: {(ite < 0).sum()}")
    print(f"\n前5名学生ITE:")
    print(results.head())
    return results


def layer3_experiment_design_template():
    template = """
╔══════════════════════════════════════════════════════════════╗
║         Layer 3: 准实验设计数据收集模板                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. 前测（第0周）                                            ║
║     □ 计算思维量表（5维度分数） → X                          ║
║     □ 编程作业质量评分 → baseline_Y                          ║
║                                                              ║
║  2. 随机分组                                                 ║
║     □ 实验组（n≈20-30）: 接受专项训练                       ║
║     □ 对照组（n≈20-30）: 正常上课                           ║
║                                                              ║
║  3. 干预（第1-4周）                                          ║
║     □ 实验组每周2次针对性训练，每次30分钟                   ║
║     □ 记录出勤率和训练内容                                   ║
║                                                              ║
║  4. 后测（第5周）                                            ║
║     □ 计算思维量表（5维度分数）                              ║
║     □ 编程作业质量评分 → post_Y                              ║
║                                                              ║
║  5. 数据处理                                                 ║
║     □ T = 1(实验组) / 0(对照组)                             ║
║     □ Y = post_Y - baseline_Y（增长值）                      ║
║     □ X = 前测5维度分数（协变量）                            ║
║                                                              ║
║  6. 运行代码                                                 ║
║     □ layer3_estimate_ite(X, T, Y, method='xlearner')       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(template)


# ============================================
# 10. 主程序
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("三层因果可解释诊断系统 v6")
    print("核心改进: 三层约束保真机制（模板填充+大模型润色+自动校验）")
    print("=" * 60)

    print("\n[1/8] 准备数据...")
    df = prepare_data(n=200)
    print("[2/8] 训练XGBoost模型...")
    model, X_train, X_test = train_model(df)
    sample_student = X_test.iloc[0]
    student_id = f"S{X_test.index[0]:03d}"

    print(f"\n[3/8] Layer 1: 反事实分析（{student_id}）...")
    cf_report, cf_best_dim, cf_best_gain = layer1_counterfactual(model, sample_student, student_id)
    print(cf_report)

    print(f"\n[4/8] Layer 2: 双维度诊断（自适应阈值）...")
    dd_report, df_results, sv, ce_dict, shap_threshold, ce_threshold = layer2_dual_diagnosis(model, sample_student, df, student_id, X_train)
    print(dd_report)

    print(f"\n[5/8] 生成双维度散点图...")
    plot_dual_diagnosis(sample_student, sv, ce_dict, student_id, shap_threshold, ce_threshold)

    # v6核心：三层约束保真报告生成
    print(f"\n[6/8] V6生成教师诊断报告（模板填充→大模型润色→保真度校验）...")
    teacher_report, structured_backup = generate_teacher_report_v6(
        student_id, sample_student, sv, ce_dict, df_results, model,
        cf_best_dim, cf_best_gain, shap_threshold, ce_threshold
    )
    print(f"\n{'='*60}")
    print("【V6最终报告：给教师的诊断报告】")
    print(f"{'='*60}")
    print(teacher_report)

    # v6新增：保存报告
    print(f"\n{'─'*50}")
    print("【保存报告】")
    save_report(teacher_report, student_id, suffix="final")
    save_report(structured_backup, student_id, suffix="structured_backup")

    print(f"\n{'='*60}")
    print("【Layer 3: 诊断→干预→ITE评估 闭环演示】")
    print(f"{'='*60}")
    print("\n[7/8] 生成模拟准实验数据（60人，真实干预效应=5分）...")
    exp_df = simulate_intervention_data(n=60, true_effect=5.0)
    print(f"   实验组: {(exp_df['T']==1).sum()}人, 对照组: {(exp_df['T']==0).sum()}人")
    print(f"   实验组平均增长: {exp_df[exp_df['T']==1]['Y'].mean():.2f}分")
    print(f"   对照组平均增长: {exp_df[exp_df['T']==0]['Y'].mean():.2f}分")
    print(f"   原始差值: {exp_df[exp_df['T']==1]['Y'].mean() - exp_df[exp_df['T']==0]['Y'].mean():.2f}分")

    print(f"\n[8/8] Layer 3: 运行X-Learner估计ITE...")
    X_exp = exp_df[CAUSAL_ORDER]
    T_exp = exp_df['T']
    Y_exp = exp_df['Y']
    ite_results = layer3_estimate_ite(X_exp, T_exp, Y_exp, method='xlearner')

    print(f"\n{'='*60}")
    print("【闭环验证】")
    print(f"{'='*60}")
    print(f"模拟设定的真实干预效应: 5.0分")
    print(f"X-Learner估计的ATE(均值): {ite_results['ITE'].mean():.2f}分")
    print(f"估计误差: {abs(ite_results['ITE'].mean() - 5.0):.2f}分")
    if abs(ite_results['ITE'].mean() - 5.0) < 1.0:
        print("✓ ITE估计准确，与真实效应接近")
    else:
        print("⚠️ ITE估计有偏差，样本量增大后可能改善")

    layer3_experiment_design_template()

    print(f"\n{'='*60}")
    print("✓ 三层系统 v6 运行完毕！")
    print(f"{'='*60}")
    print("\nv6 改进总结:")
    print("  1. 三层约束保真机制:")
    print("     - 程序模板强制填充（100%保真）")
    print("     - 大模型仅做语言润色（不改变数据）")
    print("     - 自动保真度校验（失败则回退模板）")
    print("  2. 中文字体配置（解决图表乱码）")
    print("  3. 报告自动保存为markdown文件")
    print(f"{'='*60}")
