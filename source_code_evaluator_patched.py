# ============================================
# 源代码自动评估模块（Source Code Evaluator）
# 基于AST静态分析，自动从Python源代码中提取计算思维五维度特征
# 输出1-5分制评分 + 可解释的详细特征分解
#
# 五维度映射：
#   抽象 (Abstraction)      — 函数/类定义、接口抽象、模式识别
#   分解 (Decomposition)    — 问题拆解、函数粒度、模块化
#   算法设计 (Algorithm)    — 数据结构选择、算法效率、内置函数使用
#   建模 (Modeling)         — 领域建模、类型注解、数据类使用
#   评估 (Evaluation)       — 测试、断言、异常处理、边界检查
#
# 设计原则：每一条评分规则都可追溯到具体的代码行/结构
#          教师可以直接引用规则向学生解释得分依据
# ============================================

import sys
# Windows GBK终端UTF-8兼容
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import ast
import math
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

# ============================================
# 数据类
# ============================================


@dataclass
class FeatureDetail:
    """单个特征的详细分解"""
    name: str
    raw_value: Any
    normalized: float          # 0-1 归一化值
    contribution: float        # 对维度得分的贡献
    explanation: str           # 人类可读的解释


@dataclass
class DimensionResult:
    """单个维度的评估结果"""
    dimension: str             # 维度名称
    score: float               # 1-5 得分
    features: List[FeatureDetail] = field(default_factory=list)
    code_evidence: List[str] = field(default_factory=list)  # 代码证据（行引用）


@dataclass
class EvaluationResult:
    """完整的评估结果"""
    student_id: str
    dimensions: Dict[str, DimensionResult] = field(default_factory=dict)
    overall_summary: str = ""
    syntax_errors: List[str] = field(default_factory=list)

    def to_pipeline_dict(self) -> Dict[str, float]:
        """转换为v6系统需要的格式 {'抽象': 3.2, '分解': 4.1, ...}"""
        return {dim: r.score for dim, r in self.dimensions.items()}


# ============================================
# 配置：可调节的评分阈值
# ============================================

# 以下阈值基于学生编程作业场景（代码量50-300行）校准
# 可根据实际教学要求调整
CONFIG = {
    # ---- 抽象 ----
    "abs_func_min": 1,           # 至少1个函数定义
    "abs_func_target": 4,        # 4个及以上为优秀
    "abs_class_min": 0,
    "abs_class_target": 2,
    "abs_interface_weight": 0.3, # 抽象方法/ABC权重
    "abs_typing_weight": 0.2,    # 类型注解权重
    "abs_const_weight": 0.15,    # 命名常量使用权重
    "abs_decorator_weight": 0.15, # property/staticmethod等权重
    "abs_inheritance_weight": 0.2,
    # ---- 分解 ----
    "dec_func_ratio_min": 0.01,  # 每行代码对应的函数数下限
    "dec_func_ratio_target": 0.05, # 优秀阈值
    "dec_max_func_len_ok": 20,   # 函数长度<=20行为优秀
    "dec_max_func_len_bad": 50,  # 函数长度>50行为差
    "dec_modules_target": 2,     # 多文件鼓励
    # ---- 算法设计 ----
    "alg_set_usage_weight": 0.2,
    "alg_comprehension_weight": 0.2,
    "alg_builtin_weight": 0.2,
    "alg_nesting_penalty": -0.3, # 深层嵌套惩罚
    "alg_max_nesting_ok": 2,
    "alg_max_nesting_bad": 4,
    "alg_generator_weight": 0.15,
    # ---- 建模 ----
    "mod_class_modeling_weight": 0.3,
    "mod_dataclass_weight": 0.25,
    "mod_enum_weight": 0.15,
    "mod_typehint_weight": 0.2,
    "mod_namedtuple_weight": 0.1,
    # ---- 评估 ----
    "eval_test_weight": 0.25,
    "eval_assert_weight": 0.15,
    "eval_try_weight": 0.2,
    "eval_validation_weight": 0.2,
    "eval_docstring_weight": 0.1,
    "eval_logging_weight": 0.1,
}


# ============================================
# AST访问器基础设施
# ============================================


class _MetricsVisitor(ast.NodeVisitor):
    """通用的AST指标收集器"""

    def __init__(self):
        self.reset()

    def reset(self):
        # 基本结构
        self.function_defs: List[ast.FunctionDef] = []
        self.async_function_defs: List[ast.AsyncFunctionDef] = []
        self.class_defs: List[ast.ClassDef] = []
        # 抽象相关
        self.abstract_methods: List[ast.FunctionDef] = []
        self.classes_with_inheritance: List[ast.ClassDef] = []
        self.decorator_counts: Counter = Counter()
        self.const_assignments: List[ast.Assign] = []
        # 分解相关
        self.func_lines: Dict[str, int] = {}  # 函数名→行数
        # 算法相关
        self.comprehensions: int = 0
        self.generators: int = 0
        self.set_usage: int = 0
        self.dict_usage: int = 0
        self.builtin_calls: set = set()
        self.max_nesting_depth: int = 0
        self.nested_loops: int = 0
        self.recursions: int = 0
        self.sorted_usage: bool = False
        self.any_all_usage: bool = False
        # 建模相关
        self.dataclass_classes: List[ast.ClassDef] = []
        self.enum_classes: List[ast.ClassDef] = []
        self.namedtuple_assignments: List[Any] = []
        self.type_annotations: int = 0
        self.total_annotatable: int = 0  # 可以加注解的位置总数
        # 评估相关
        self.try_blocks: int = 0
        self.assert_statements: int = 0
        self.test_functions: List[ast.FunctionDef] = []
        self.input_validations: int = 0
        self.docstrings: int = 0
        self.logging_calls: int = 0
        self.print_calls: int = 0
        self.is_not_none_checks: int = 0
        self.len_checks: int = 0
        # 常量
        self.magic_numbers: List[int] = []
        # 代码证据
        self.evidence: Dict[str, List[str]] = {
            "abstract": [], "decomposition": [], "algorithm": [],
            "modeling": [], "evaluation": [],
        }

    def generic_visit(self, node):
        self._check_nesting(node)
        super().generic_visit(node)

    def _check_nesting(self, node):
        """追踪嵌套深度"""
        depth = 1
        parent = getattr(node, 'parent', None)
        while parent is not None:
            if isinstance(parent, (ast.For, ast.While, ast.If, ast.With,
                                   ast.Try, ast.FunctionDef, ast.AsyncFunctionDef)):
                depth += 1
            parent = getattr(parent, 'parent', None)
        if depth > self.max_nesting_depth:
            self.max_nesting_depth = depth
        if depth >= 3 and isinstance(node, (ast.For, ast.While)):
            self.nested_loops += 1

    def _set_parents(self, node, parent=None):
        """递归设置父节点引用"""
        node.parent = parent
        for child in ast.iter_child_nodes(node):
            self._set_parents(child, node)

    # ---- 各节点类型的visit方法 ----

    def visit_FunctionDef(self, node):
        self.function_defs.append(node)
        if node.name.startswith('test_') or node.name.endswith('_test'):
            self.test_functions.append(node)
        if any(self._is_abstract_decorator(d) for d in node.decorator_list):
            self.abstract_methods.append(node)
        if ast.get_docstring(node):
            self.docstrings += 1
        # 递归检测
        self._check_recursion(node)
        # 函数行数
        if node.end_lineno and node.lineno:
            self.func_lines[node.name] = node.end_lineno - node.lineno + 1
        self._count_decorators(node.decorator_list)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.async_function_defs.append(node)
        if ast.get_docstring(node):
            self.docstrings += 1
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.class_defs.append(node)
        if node.bases:
            self.classes_with_inheritance.append(node)
        if ast.get_docstring(node):
            self.docstrings += 1
        # 检测dataclass
        for d in node.decorator_list:
            if self._is_dataclass_decorator(d):
                self.dataclass_classes.append(node)
                break
        # 检测Enum
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in ('Enum', 'IntEnum', 'StrEnum'):
                self.enum_classes.append(node)
                break
            elif isinstance(base, ast.Attribute) and base.attr in ('Enum', 'IntEnum', 'StrEnum'):
                self.enum_classes.append(node)
                break
        self.generic_visit(node)

    def visit_Assign(self, node):
        # 检测常量定义（全大写变量名）
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper() and '_' in target.id:
                self.const_assignments.append(node)
                break
            # 检测NamedTuple赋值
            if isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute) and call.func.attr == 'NamedTuple':
                    self.namedtuple_assignments.append(node)
        self.generic_visit(node)
        # 普通赋值（无类型注解），仅计入可注解总数
        self.total_annotatable += len(node.targets)

    def visit_AnnAssign(self, node):
        self.type_annotations += 1
        self.total_annotatable += 1
        self.generic_visit(node)

    def visit_arg(self, node):
        if node.annotation:
            self.type_annotations += 1
        self.total_annotatable += 1

    def visit_ListComp(self, node):
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self.generators += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        # 收集内置函数调用
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in _BUILTIN_FUNCTIONS:
                self.builtin_calls.add(name)
            if name == 'sorted':
                self.sorted_usage = True
            if name in ('any', 'all'):
                self.any_all_usage = True
            if name == 'print':
                self.print_calls += 1
        # 检测set/dict字面量使用
        if isinstance(node.func, ast.Name):
            if node.func.id == 'set':
                self.set_usage += 1
            elif node.func.id == 'dict':
                self.dict_usage += 1
        # 检测namedtuple调用
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'namedtuple':
            self.namedtuple_assignments.append(node)
        # 检测logging调用
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'logging':
                self.logging_calls += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.try_blocks += 1
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.assert_statements += 1
        self.generic_visit(node)

    def visit_If(self, node):
        if self._is_input_validation(node):
            self.input_validations += 1
        self.generic_visit(node)

    def visit_Compare(self, node):
        # None检查
        for op in node.ops:
            if isinstance(op, ast.Is) or isinstance(op, ast.IsNot):
                for comp in node.comparators:
                    if isinstance(comp, ast.Constant) and comp.value is None:
                        self.is_not_none_checks += 1
        # len()检查（边界检查）
        if isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Name):
            if node.left.func.id == 'len':
                self.len_checks += 1
        self.generic_visit(node)

    # 白名单函数：这些函数中的数字参数不被视为魔法数字
    _WHITELIST_CALLS = {'range', 'slice', 'round', 'int', 'float', 'str', 'bool',
                        'len', 'ord', 'chr', 'hex', 'bin', 'oct', 'format',
                        'max', 'min', 'sum', 'abs', 'pow', 'divmod'}

    def visit_Constant(self, node):
        # 魔法数字检测
        if isinstance(node.value, (int, float)) and node.value not in (0, 1, -1, 2):
            parent = getattr(node, 'parent', None)
            # 排除定义常量的赋值右侧
            if isinstance(parent, ast.Assign) and                     any(isinstance(t, ast.Name) and t.id.isupper() for t in parent.targets):
                self.generic_visit(node)
                return
            # v6修复: 排除白名单函数参数中的数字（如range(3), slice(1,5)）
            if self._is_whitelist_call_arg(node, parent):
                self.generic_visit(node)
                return
            self.magic_numbers.append(node.value)
        self.generic_visit(node)

    @classmethod
    def _is_whitelist_call_arg(cls, node, parent):
        """检查节点是否是白名单函数的参数"""
        if parent is None:
            return False
        # 向上查找最近的Call节点
        current = parent
        while current is not None:
            if isinstance(current, ast.Call):
                func = current.func
                if isinstance(func, ast.Name) and func.id in cls._WHITELIST_CALLS:
                    return True
                if isinstance(func, ast.Attribute) and func.attr in cls._WHITELIST_CALLS:
                    return True
                # 不是白名单函数，停止向上查找（避免误判嵌套）
                return False
            # 如果遇到了其他独立语句边界，停止查找
            if isinstance(current, (ast.Assign, ast.FunctionDef, ast.ClassDef,
                                    ast.Module, ast.Expr, ast.Return)):
                return False
            current = getattr(current, 'parent', None)
        return False

    # ---- 辅助方法 ----

    @staticmethod
    def _is_abstract_decorator(decorator):
        if isinstance(decorator, ast.Name):
            return decorator.id == 'abstractmethod'
        if isinstance(decorator, ast.Attribute):
            return decorator.attr == 'abstractmethod'
        return False

    @staticmethod
    def _is_dataclass_decorator(decorator):
        if isinstance(decorator, ast.Name):
            return decorator.id == 'dataclass'
        if isinstance(decorator, ast.Attribute):
            return decorator.attr == 'dataclass'
        if isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name):
                return func.id == 'dataclass'
            if isinstance(func, ast.Attribute):
                return func.attr == 'dataclass'
        return False

    def _count_decorators(self, decorator_list):
        for d in decorator_list:
            if isinstance(d, ast.Name):
                self.decorator_counts[d.id] += 1
            elif isinstance(d, ast.Attribute):
                self.decorator_counts[d.attr] += 1
            elif isinstance(d, ast.Call):
                if isinstance(d.func, ast.Name):
                    self.decorator_counts[d.func.id] += 1
                elif isinstance(d.func, ast.Attribute):
                    self.decorator_counts[d.func.attr] += 1

    def _check_recursion(self, node):
        """检测函数是否包含自递归调用"""
        func_name = node.name
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id == func_name:
                    self.recursions += 1
                    return

    def _is_input_validation(self, if_node):
        """检测if语句是否为输入验证模式"""
        test = if_node.test
        if isinstance(test, ast.Compare):
            if isinstance(test.left, ast.Name):
                return True
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            if isinstance(test.operand, ast.Call) and isinstance(test.operand.func, ast.Name):
                return True
        if isinstance(test, ast.BoolOp):
            return True
        return False


# 内置函数集合（用于算法设计维度）
_BUILTIN_FUNCTIONS = {
    'sorted', 'reversed', 'enumerate', 'zip', 'map', 'filter',
    'min', 'max', 'sum', 'any', 'all', 'len', 'range',
    'abs', 'round', 'pow', 'divmod', 'isinstance', 'issubclass',
    'iter', 'next', 'open', 'hash', 'id', 'type', 'chr', 'ord',
    'bin', 'hex', 'oct', 'format', 'repr', 'str', 'int', 'float',
    'bool', 'list', 'tuple', 'set', 'dict', 'frozenset', 'slice',
}


# ============================================
# 核心评估器
# ============================================


class CTCodeAnalyzer:
    """计算思维源代码分析器

    用法:
        analyzer = CTCodeAnalyzer()
        result = analyzer.evaluate(source_code, student_id="S001")
        print(result.to_pipeline_dict())  # {'抽象': 3.5, '分解': 4.0, ...}

        # 查看详细的特征分解
        for dim, detail in result.dimensions.items():
            print(f"{dim}: {detail.score:.1f}")
            for feat in detail.features:
                print(f"  - {feat.name}: {feat.explanation}")
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = CONFIG.copy()
        if config:
            self.config.update(config)
        self.visitor = _MetricsVisitor()

    # ================================================================
    # 公开API
    # ================================================================

    def evaluate(self, code: str, student_id: str = "unknown",
                 source_name: str = "") -> EvaluationResult:
        """对一段Python源代码进行五维度评估

        Args:
            code: Python源代码字符串
            student_id: 学生标识符
            source_name: 来源文件名（可选，用于展示）

        Returns:
            EvaluationResult，包含五维度得分和详细特征分解
        """
        result = EvaluationResult(student_id=student_id)

        # 1. 解析AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result.syntax_errors.append(f"语法错误: {e}")
            # 语法错误时给最低分
            for dim in ('抽象', '分解', '算法设计', '建模', '评估'):
                result.dimensions[dim] = DimensionResult(
                    dimension=dim, score=1.0,
                    features=[FeatureDetail("语法错误", 0, 0, f"代码存在语法错误: {e}")],
                    code_evidence=[f"第{e.lineno}行附近"]
                )
            result.overall_summary = "代码存在语法错误，无法进行五维度评估。请先修正语法问题。"
            return result

        # 2. 收集所有AST指标
        self.visitor.reset()
        self.visitor._set_parents(tree)
        self.visitor.visit(tree)

        # 3. 行数统计
        total_lines = len(code.split('\n'))

        # 4. 逐维度评估
        result.dimensions['抽象'] = self._evaluate_abstraction(total_lines)
        result.dimensions['分解'] = self._evaluate_decomposition(total_lines)
        result.dimensions['算法设计'] = self._evaluate_algorithm(total_lines)
        result.dimensions['建模'] = self._evaluate_modeling(total_lines)
        result.dimensions['评估'] = self._evaluate_evaluation(total_lines)

        # 5. 生成摘要
        result.overall_summary = self._generate_summary(result)

        return result

    def evaluate_with_tests(self, code: str, test_code: str = None,
                            student_id: str = "unknown", run_tests: bool = True) -> EvaluationResult:
        """
        v6新增: 评估学生代码，可选运行教师提供的单元测试。
        动态测试通过率会整合到"评估"维度得分中(占40%权重)。
        """
        # 步骤1: 常规AST静态分析
        result = self.evaluate(code, student_id=student_id)

        if not test_code or not run_tests:
            result.overall_summary += "\n[动态测试: 未运行]"
            return result

        # 步骤2: 运行动态测试
        import tempfile, subprocess, os, sys

        test_meta = {'total': 0, 'passed': 0, 'failed': 0, 'errors': 0,
                     'pass_rate': 0.0, 'details': []}

        with tempfile.TemporaryDirectory() as tmpdir:
            student_path = os.path.join(tmpdir, 'student_code.py')
            with open(student_path, 'w', encoding='utf-8') as f:
                f.write(code)

            # 测试代码包装（自动导入学生代码）
            test_path = os.path.join(tmpdir, 'test_student.py')
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write('import sys\n')
                f.write('sys.path.insert(0, ' + repr(tmpdir) + ')\n')
                f.write('from student_code import *\n\n')
                f.write(test_code)

            try:
                proc = subprocess.run(
                    [sys.executable, '-m', 'pytest', test_path, '-v', '--tb=short'],
                    capture_output=True, text=True, timeout=30, cwd=tmpdir
                )
                output = proc.stdout + proc.stderr

                # 解析pytest输出
                for line in output.split('\n'):
                    line = line.strip()
                    if ' passed' in line:
                        # "X passed, Y failed, Z error" 或 "X passed in Ns"
                        parts = line.split(',')
                        for part in parts:
                            p = part.strip()
                            if ' passed' in p and 'failed' not in p:
                                try: test_meta['passed'] = int(p.split()[0])
                                except: pass
                            elif ' failed' in p:
                                try: test_meta['failed'] = int(p.split()[0])
                                except: pass
                            elif ' error' in p:
                                try: test_meta['errors'] = int(p.split()[0])
                                except: pass

                test_meta['total'] = test_meta['passed'] + test_meta['failed'] + test_meta['errors']
                if test_meta['total'] > 0:
                    test_meta['pass_rate'] = test_meta['passed'] / test_meta['total']
                test_meta['details'] = [l for l in output.split('\n') 
                                        if 'passed' in l or 'failed' in l or 'error' in l][:3]

            except subprocess.TimeoutExpired:
                test_meta['details'] = ['[测试超时]']
            except Exception as e:
                test_meta['details'] = [f'[测试运行失败: {e}]']

        # 步骤3: 整合到"评估"维度（动态测试占40%权重）
        eval_dim = result.dimensions.get('评估')
        if eval_dim and test_meta['total'] > 0:
            old_score = eval_dim.score
            new_score = 0.6 * old_score + 0.4 * (1.0 + 4.0 * test_meta['pass_rate'])
            eval_dim.score = round(max(1.0, min(5.0, new_score)), 1)

            eval_dim.features.append(CodeFeatureDetail(
                "动态测试通过率",
                f"{test_meta['passed']}/{test_meta['total']}",
                test_meta['pass_rate'],
                test_meta['pass_rate'] * 0.2,
                f"运行{test_meta['total']}个测试，通过{test_meta['passed']}个({test_meta['pass_rate']:.0%})"
                + ("，功能实现正确" if test_meta['pass_rate'] == 1.0 else "，部分功能需修正")
            ))
            if test_meta['failed'] > 0:
                eval_dim.code_evidence.append(f"测试失败: {test_meta['failed']}个")

        result.overall_summary += f"\n[动态测试: {test_meta['passed']}/{test_meta['total']}通过({test_meta['pass_rate']:.0%})]"
        result._test_info = test_meta
        return result

    def evaluate_file(self, filepath: str, student_id: Optional[str] = None) -> EvaluationResult:
        """评估单个Python文件"""
        path = Path(filepath)
        if student_id is None:
            student_id = path.stem
        code = path.read_text(encoding='utf-8')
        return self.evaluate(code, student_id=student_id, source_name=path.name)

    def evaluate_files(self, filepaths: List[str],
                       student_id: Optional[str] = None) -> EvaluationResult:
        """评估多个Python文件（合并分析）"""
        all_code = []
        for fp in filepaths:
            try:
                all_code.append(Path(fp).read_text(encoding='utf-8'))
            except Exception as e:
                all_code.append(f"# [读取失败: {fp}]  # {e}")
        combined = "\n\n".join(all_code)
        return self.evaluate(combined, student_id=student_id or "multi_file")

    def batch_evaluate(self, submissions: List[Tuple[str, str]]) -> "pd.DataFrame":
        """批量评估，返回DataFrame可直接接入v6管道

        Args:
            submissions: [(student_id, code), ...]

        Returns:
            DataFrame with columns: student_id, 抽象, 分解, 算法设计, 建模, 评估
        """
        import pandas as pd
        rows = []
        for sid, code in submissions:
            result = self.evaluate(code, student_id=sid)
            row = {'student_id': sid, **result.to_pipeline_dict()}
            rows.append(row)
        return pd.DataFrame(rows)

    # ================================================================
    # 维度1: 抽象 (Abstraction)
    # 测量：函数/类定义的层次结构、接口抽象、模式识别与泛化
    # ================================================================

    def _evaluate_abstraction(self, total_lines: int) -> DimensionResult:
        v = self.visitor
        features = []
        evidence = []

        # 1.1 函数定义数量
        func_count = len(v.function_defs) + len(v.async_function_defs)
        f_func = self._normalize_count(func_count, self.config["abs_func_min"],
                                       self.config["abs_func_target"])
        if func_count > 0:
            evidence.append(f"定义了{func_count}个函数: " +
                            ", ".join(f.name for f in v.function_defs[:5]))
        features.append(FeatureDetail(
            "函数定义数", func_count, f_func, f_func * 0.25,
            f"定义了{func_count}个函数" + ("，达到抽象封装的基本要求" if func_count >= 2 else "，建议将重复逻辑封装为函数")
        ))

        # 1.2 类定义数量
        class_count = len(v.class_defs)
        f_class = self._normalize_count(class_count, self.config["abs_class_min"],
                                        self.config["abs_class_target"])
        if class_count > 0:
            evidence.append(f"定义了{class_count}个类: " +
                            ", ".join(c.name for c in v.class_defs[:3]))
        features.append(FeatureDetail(
            "类定义数", class_count, f_class, f_class * 0.20,
            f"定义了{class_count}个类" + ("，体现了面向对象的抽象能力" if class_count >= 1 else "")
        ))

        # 1.3 继承使用
        inheritance_count = len(v.classes_with_inheritance)
        f_inherit = min(inheritance_count / max(class_count, 1), 1.0)
        if inheritance_count > 0:
            evidence.append(f"使用了继承的类: " +
                            ", ".join(c.name for c in v.classes_with_inheritance[:3]))
        features.append(FeatureDetail(
            "继承与多态", inheritance_count, f_inherit,
            f_inherit * self.config["abs_inheritance_weight"],
            f"{inheritance_count}个类使用了继承，体现了泛化/特化思维"
            if inheritance_count > 0 else "未使用继承，建议对相似概念使用基类抽象"
        ))

        # 1.4 抽象方法使用
        abs_count = len(v.abstract_methods)
        f_abs = min(abs_count / 2.0, 1.0) if abs_count > 0 else 0.0
        if abs_count > 0:
            evidence.append(f"定义了{abs_count}个抽象方法")
        features.append(FeatureDetail(
            "抽象方法/接口", abs_count, f_abs,
            f_abs * self.config["abs_interface_weight"],
            f"定义了{abs_count}个抽象方法，明确了接口契约" if abs_count > 0
            else "未使用抽象方法，对于复杂系统建议定义接口"
        ))

        # 1.5 装饰器使用（property, staticmethod, classmethod体现封装抽象）
        deco_score = 0.0
        for deco in ('property', 'staticmethod', 'classmethod'):
            if v.decorator_counts.get(deco, 0) > 0:
                deco_score += 0.33
                evidence.append(f"使用了@{deco}装饰器")
        deco_score = min(deco_score, 1.0)
        features.append(FeatureDetail(
            "封装装饰器", dict(v.decorator_counts), deco_score,
            deco_score * self.config["abs_decorator_weight"],
            "合理使用了@property/@staticmethod等封装装饰器" if deco_score > 0
            else "未使用封装装饰器，数据访问可以更规范"
        ))

        # 1.6 命名常量（用常量替代魔法数字，体现模式识别）
        const_count = len(v.const_assignments)
        magic_count = len(v.magic_numbers)
        f_const = 1.0 - min(magic_count / max(total_lines * 0.1, 1), 1.0) if magic_count > 0 else 0.7
        f_const = max(0.0, f_const + min(const_count / 3.0, 0.3))
        f_const = min(f_const, 1.0)
        if const_count > 0:
            evidence.append(f"定义了{const_count}个命名常量，{magic_count}个魔法数字")
        features.append(FeatureDetail(
            "常量vs魔法数字", f"常量{const_count}/魔法{magic_count}", f_const,
            f_const * self.config["abs_const_weight"],
            f"命名常量{const_count}个，魔法数字{magic_count}个" +
            ("，常量使用良好" if f_const > 0.6 else "，建议用命名常量替代魔法数字")
        ))

        # 计算最终得分
        score = self._features_to_score(features)
        score = self._clamp_score(score)

        return DimensionResult(
            dimension='抽象', score=score,
            features=features, code_evidence=evidence
        )

    # ================================================================
    # 维度2: 分解 (Decomposition)
    # 测量：问题拆解为子问题的能力，函数粒度，模块化
    # ================================================================

    def _evaluate_decomposition(self, total_lines: int) -> DimensionResult:
        v = self.visitor
        features = []
        evidence = []

        # 2.1 函数密度（每行代码对应的函数数）
        func_count = len(v.function_defs) + len(v.async_function_defs)
        func_ratio = func_count / max(total_lines, 1)
        f_density = self._normalize_ratio(
            func_ratio,
            self.config["dec_func_ratio_min"],
            self.config["dec_func_ratio_target"]
        )
        features.append(FeatureDetail(
            "函数密度", f"{func_ratio:.3f}函数/行", f_density, f_density * 0.30,
            f"每行代码对应{func_ratio:.3f}个函数" +
            ("，拆分充分" if f_density > 0.6 else "，建议将长代码块拆分为更多小函数")
        ))

        # 2.2 函数长度分布
        func_lens = list(v.func_lines.values())
        if func_lens:
            avg_len = sum(func_lens) / len(func_lens)
            max_len = max(func_lens)
            long_funcs = [name for name, l in v.func_lines.items()
                          if l > self.config["dec_max_func_len_bad"]]
            ok_funcs = [name for name, l in v.func_lines.items()
                        if l <= self.config["dec_max_func_len_ok"]]

            f_avg_len = 1.0 - min(
                max(avg_len - self.config["dec_max_func_len_ok"], 0) /
                (self.config["dec_max_func_len_bad"] - self.config["dec_max_func_len_ok"]),
                1.0
            )
            if long_funcs:
                evidence.append(f"过长函数({'>'+str(self.config['dec_max_func_len_bad'])+'行'}): {', '.join(long_funcs[:3])}")
            if ok_funcs:
                evidence.append(f"良好粒度函数: {', '.join(ok_funcs[:5])}")

            features.append(FeatureDetail(
                "平均函数长度", f"{avg_len:.0f}行", f_avg_len, f_avg_len * 0.25,
                f"平均函数长度{avg_len:.0f}行" +
                ("，粒度合适" if avg_len <= self.config["dec_max_func_len_ok"] else "，建议拆分长函数")
            ))

            f_max = 1.0 if max_len <= self.config["dec_max_func_len_ok"] else (
                0.0 if max_len >= self.config["dec_max_func_len_bad"]
                else 0.5
            )
            features.append(FeatureDetail(
                "最大函数长度", f"{max_len}行", f_max, f_max * 0.20,
                f"最长函数{max_len}行" +
                ("，所有函数粒度合理" if f_max == 1.0 else "，存在过长函数需要拆分")
            ))
        else:
            features.append(FeatureDetail(
                "平均函数长度", "N/A", 0.0, 0.0, "未定义函数，代码全在全局作用域中"
            ))
            features.append(FeatureDetail(
                "最大函数长度", "N/A", 0.0, 0.0, "无函数定义，建议将全局代码封装为函数"
            ))

        # 2.3 类方法分解
        class_count = len(v.class_defs)
        if class_count > 0:
            cls_method_counts = []
            for cls in v.class_defs:
                methods = [n for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
                cls_method_counts.append(len(methods))
            avg_methods = sum(cls_method_counts) / len(cls_method_counts) if cls_method_counts else 0
            f_cls = min(avg_methods / 4.0, 1.0)  # 每个类平均4个方法为佳
            features.append(FeatureDetail(
                "类方法分解", f"平均{avg_methods:.1f}方法/类", f_cls, f_cls * 0.15,
                f"每个类平均{avg_methods:.1f}个方法" +
                ("，职责拆分合理" if f_cls > 0.5 else "")
            ))
        else:
            features.append(FeatureDetail(
                "类方法分解", "N/A", 0.5, 0.075, "无类定义（对于简单程序可接受）"
            ))

        # 2.4 代码重复度近似（用helper/util函数名模式）
        helper_count = sum(1 for f in v.function_defs
                           if any(kw in f.name.lower()
                                  for kw in ('helper', 'util', 'aux', 'common', '_')))
        f_helper = min(helper_count / 3.0, 1.0)
        features.append(FeatureDetail(
            "辅助函数/工具化", helper_count, f_helper, f_helper * 0.10,
            f"检测到{helper_count}个辅助/工具型函数" +
            ("，代码复用意识好" if helper_count >= 2 else "")
        ))

        score = self._features_to_score(features)
        score = self._clamp_score(score)
        return DimensionResult(
            dimension='分解', score=score,
            features=features, code_evidence=evidence
        )

    # ================================================================
    # 维度3: 算法设计 (Algorithm Design)
    # 测量：数据结构选择、算法效率、Python惯用法
    # ================================================================

    def _evaluate_algorithm(self, total_lines: int) -> DimensionResult:
        v = self.visitor
        features = []
        evidence = []

        # 3.1 Set使用（O(1)查找 vs O(n)列表查找）
        f_set = min(v.set_usage / 2.0, 1.0)
        if v.set_usage > 0:
            evidence.append(f"使用了set进行O(1)成员检查（{v.set_usage}处）")
        features.append(FeatureDetail(
            "Set/Hash使用", v.set_usage, f_set,
            f_set * self.config["alg_set_usage_weight"],
            f"使用set数据结构{v.set_usage}次" +
            ("，体现了对查找效率的理解" if v.set_usage > 0
             else "，建议在需要成员检查时使用set替代list")
        ))

        # 3.2 推导式使用（Pythonic效率）
        comp_count = v.comprehensions
        f_comp = min(comp_count / 3.0, 1.0)
        if comp_count > 0:
            evidence.append(f"使用了{comp_count}个推导式（list/set/dict comprehension）")
        features.append(FeatureDetail(
            "推导式使用", comp_count, f_comp,
            f_comp * self.config["alg_comprehension_weight"],
            f"使用了{comp_count}个推导式" +
            ("，代码Pythonic且高效" if comp_count >= 2 else
             "，适当使用推导式可使代码更简洁高效")
        ))

        # 3.3 内置函数使用
        builtin_count = len(v.builtin_calls)
        f_builtin = min(builtin_count / 5.0, 1.0)
        key_funcs = [f for f in ['sorted', 'min', 'max', 'sum', 'any', 'all', 'enumerate', 'zip']
                     if f in v.builtin_calls]
        if key_funcs:
            evidence.append(f"高效内置函数: {', '.join(key_funcs)}")
        features.append(FeatureDetail(
            "内置函数使用", f"{builtin_count}种",
            f_builtin, f_builtin * self.config["alg_builtin_weight"],
            f"使用了{builtin_count}种内置高效函数" +
            (f"（如{', '.join(key_funcs[:3])}）" if key_funcs else "，建议多利用Python内置函数")
        ))

        # 3.4 嵌套深度
        max_depth = v.max_nesting_depth
        f_depth = 1.0 if max_depth <= self.config["alg_max_nesting_ok"] else (
            0.0 if max_depth >= self.config["alg_max_nesting_bad"]
            else 0.5
        )
        if max_depth >= 3:
            evidence.append(f"最大嵌套深度{max_depth}层，建议使用early return或提取函数来展平")
        features.append(FeatureDetail(
            "最大嵌套深度", f"{max_depth}层", f_depth,
            f_depth * 0.15 + (self.config["alg_nesting_penalty"] if max_depth >= 4 else 0),
            f"最大嵌套深度{max_depth}层" +
            ("，代码扁平" if max_depth <= 2 else
             "，嵌套较深，建议用early return或提取函数展平")
        ))

        # 3.5 递归使用
        f_recur = min(v.recursions / 2.0, 1.0) if v.recursions > 0 else 0.5
        if v.recursions > 0:
            evidence.append(f"使用了递归（{v.recursions}处）")
        features.append(FeatureDetail(
            "递归应用", v.recursions, f_recur, f_recur * 0.05,
            f"{'合理使用了递归' if v.recursions > 0 else '未使用递归（对大多数问题这不是必须的）'}"
        ))

        # 3.6 生成器表达式（内存效率）
        gen_count = v.generators
        f_gen = min(gen_count / 2.0, 1.0)
        if gen_count > 0:
            evidence.append(f"使用了{gen_count}个生成器表达式（内存友好）")
        features.append(FeatureDetail(
            "生成器/惰性求值", gen_count, f_gen,
            f_gen * self.config["alg_generator_weight"],
            f"使用了{gen_count}个生成器表达式" +
            ("，体现了对内存效率的考虑" if gen_count > 0 else "")
        ))

        # 3.7 sorted/any/all 使用
        extra = 0.0
        if v.sorted_usage:
            extra += 0.5
            evidence.append("使用了sorted()进行高效排序")
        if v.any_all_usage:
            extra += 0.5
            evidence.append("使用了any()/all()进行批量条件判断")
        f_extra = min(extra, 1.0)
        features.append(FeatureDetail(
            "高效算法函数", f"sorted={v.sorted_usage}, any/all={v.any_all_usage}",
            f_extra, f_extra * 0.05,
            "使用了sorted/any/all等高效内置算法函数" if f_extra > 0
            else "建议使用sorted/any/all等内置函数简化代码"
        ))

        score = self._features_to_score(features)
        score = self._clamp_score(score)
        return DimensionResult(
            dimension='算法设计', score=score,
            features=features, code_evidence=evidence
        )

    # ================================================================
    # 维度4: 建模 (Modeling)
    # 测量：用代码结构表达现实世界概念的能力
    # ================================================================

    def _evaluate_modeling(self, total_lines: int) -> DimensionResult:
        v = self.visitor
        features = []
        evidence = []

        # 4.1 领域类建模
        class_count = len(v.class_defs)
        # 排除明显的工具/Helper类
        domain_classes = [c for c in v.class_defs
                          if not any(kw in c.name.lower()
                                     for kw in ('helper', 'util', 'manager', 'factory', 'builder'))]
        domain_count = len(domain_classes)
        f_domain = min(domain_count / 2.0, 1.0)
        if domain_count > 0:
            evidence.append(f"领域模型类: {', '.join(c.name for c in domain_classes[:5])}")
        features.append(FeatureDetail(
            "领域类建模", domain_count, f_domain,
            f_domain * self.config["mod_class_modeling_weight"],
            f"{domain_count}个领域模型类" +
            ("，体现了现实世界概念→代码结构的转化" if domain_count >= 1
             else "，建议用类来建模问题中的实体和概念")
        ))

        # 4.2 Dataclass使用
        dc_count = len(v.dataclass_classes)
        f_dc = min(dc_count / 2.0, 1.0)
        if dc_count > 0:
            evidence.append(f"使用了dataclass: {', '.join(c.name for c in v.dataclass_classes)}")
        features.append(FeatureDetail(
            "Dataclass建模", dc_count, f_dc,
            f_dc * self.config["mod_dataclass_weight"],
            f"{dc_count}个dataclass" +
            ("，数据建模规范" if dc_count > 0 else "，对于纯数据结构建议使用@dataclass")
        ))

        # 4.3 Enum使用
        enum_count = len(v.enum_classes)
        f_enum = min(enum_count / 1.0, 1.0)
        if enum_count > 0:
            evidence.append(f"使用了枚举: {', '.join(c.name for c in v.enum_classes)}")
        features.append(FeatureDetail(
            "枚举建模", enum_count, f_enum,
            f_enum * self.config["mod_enum_weight"],
            f"{enum_count}个枚举类" +
            ("，对分类/状态建模正确" if enum_count > 0 else
             "，建议对有限的分类/状态值使用Enum代替字符串常量")
        ))

        # 4.4 类型注解覆盖率
        annotatable = v.total_annotatable
        annotated = v.type_annotations
        type_coverage = annotated / max(annotatable, 1)
        f_type = min(type_coverage / 0.5, 1.0)  # 50%覆盖率为满分
        features.append(FeatureDetail(
            "类型注解覆盖率", f"{type_coverage:.0%} ({annotated}/{annotatable})",
            f_type, f_type * self.config["mod_typehint_weight"],
            f"类型注解覆盖率{type_coverage:.0%}" +
            ("，类型建模规范" if type_coverage > 0.3 else "，建议增加类型注解以明确数据模型")
        ))

        # 4.5 NamedTuple
        nt_count = len(v.namedtuple_assignments)
        f_nt = min(nt_count / 2.0, 1.0)
        if nt_count > 0:
            evidence.append(f"使用了NamedTuple进行轻量数据建模")
        features.append(FeatureDetail(
            "NamedTuple建模", nt_count, f_nt,
            f_nt * self.config["mod_namedtuple_weight"],
            f"{nt_count}个NamedTuple，适合不可变数据建模" if nt_count > 0
            else "对于不可变的简单数据结构，可考虑NamedTuple"
        ))

        score = self._features_to_score(features)
        score = self._clamp_score(score)
        return DimensionResult(
            dimension='建模', score=score,
            features=features, code_evidence=evidence
        )

    # ================================================================
    # 维度5: 评估 (Evaluation)
    # 测量：测试、调试、边界条件、防御性编程意识
    # ================================================================

    def _evaluate_evaluation(self, total_lines: int) -> DimensionResult:
        v = self.visitor
        features = []
        evidence = []

        # 5.1 测试函数
        test_count = len(v.test_functions)
        f_test = min(test_count / 3.0, 1.0)
        if test_count > 0:
            evidence.append(f"测试函数: {', '.join(f.name for f in v.test_functions[:5])}")
        features.append(FeatureDetail(
            "测试函数", test_count, f_test,
            f_test * self.config["eval_test_weight"],
            f"{test_count}个测试函数" +
            ("，有系统测试意识" if test_count >= 2 else
             "，建议添加测试函数验证代码正确性")
        ))

        # 5.2 Assert语句
        assert_count = v.assert_statements
        f_assert = min(assert_count / 3.0, 1.0)
        if assert_count > 0:
            evidence.append(f"使用了{assert_count}个assert进行不变量检查")
        features.append(FeatureDetail(
            "Assert断言", assert_count, f_assert,
            f_assert * self.config["eval_assert_weight"],
            f"{assert_count}个assert" +
            ("，善于使用断言确保程序正确性" if assert_count >= 2 else "")
        ))

        # 5.3 异常处理
        try_count = v.try_blocks
        f_try = min(try_count / 2.0, 1.0)
        if try_count > 0:
            evidence.append(f"{try_count}个try/except异常处理块")
        features.append(FeatureDetail(
            "异常处理", try_count, f_try,
            f_try * self.config["eval_try_weight"],
            f"{try_count}个try块" +
            ("，有异常处理意识" if try_count > 0 else
             "，建议对可能失败的操作（IO、网络、类型转换）添加异常处理")
        ))

        # 5.4 输入验证
        validation_count = v.input_validations
        f_val = min(validation_count / 2.0, 1.0)
        if validation_count > 0:
            evidence.append(f"检测到{validation_count}处输入验证/边界检查")
        features.append(FeatureDetail(
            "输入验证", validation_count, f_val,
            f_val * self.config["eval_validation_weight"],
            f"{validation_count}处输入验证" +
            ("，防御性编程意识强" if validation_count >= 2 else
             "，建议在函数入口处验证参数合法性")
        ))

        # 5.5 Docstring
        doc_count = v.docstrings
        f_doc = min(doc_count / max(len(v.function_defs) + len(v.class_defs), 1), 1.0)
        features.append(FeatureDetail(
            "文档字符串", doc_count, f_doc,
            f_doc * self.config["eval_docstring_weight"],
            f"{doc_count}个docstring" +
            ("，文档化习惯好" if f_doc > 0.5 else
             "，建议为函数和类添加docstring说明用途和参数")
        ))

        # 5.6 Logging/调试输出
        log_count = v.logging_calls
        f_log = min(log_count / 2.0, 1.0)
        if log_count > 0:
            evidence.append(f"使用了logging模块进行运行状态记录")
        features.append(FeatureDetail(
            "Logging使用", log_count, f_log,
            f_log * self.config["eval_logging_weight"],
            f"{log_count}处logging调用" +
            ("，便于运行时诊断" if log_count > 0 else "")
        ))

        # 5.7 None检查和len检查（边界意识）
        check_total = v.is_not_none_checks + v.len_checks
        f_check = min(check_total / 3.0, 1.0)
        if check_total > 0:
            evidence.append(f"None/空值检查{check_total}处，边界条件意识好")
        features.append(FeatureDetail(
            "边界条件检查", f"None:{v.is_not_none_checks} Len:{v.len_checks}", f_check,
            f_check * 0.05,
            f"None检查{v.is_not_none_checks}处，len检查{v.len_checks}处"
        ))

        score = self._features_to_score(features)
        score = self._clamp_score(score)
        return DimensionResult(
            dimension='评估', score=score,
            features=features, code_evidence=evidence
        )

    # ================================================================
    # 辅助方法
    # ================================================================

    def _normalize_count(self, value: int, min_val: int, target: int) -> float:
        """将计数归一化到[0, 1]，target及以上为1.0"""
        if value >= target:
            return 1.0
        if value <= min_val:
            return 0.1  # 不为0，给学生基本分
        return 0.1 + 0.9 * (value - min_val) / (target - min_val)

    def _normalize_ratio(self, value: float, min_val: float, target: float) -> float:
        """将比率归一化到[0, 1]"""
        if value >= target:
            return 1.0
        if value <= min_val:
            return 0.1
        return 0.1 + 0.9 * (value - min_val) / (target - min_val)

    def _features_to_score(self, features: List[FeatureDetail]) -> float:
        """将所有特征的贡献加总，映射到1-5分"""
        total = sum(f.contribution for f in features)
        # total的理论范围约为0-1，映射到1-5
        return 1.0 + 4.0 * min(total, 1.0)

    @staticmethod
    def _clamp_score(score: float) -> float:
        """钳制到[1, 5]并保留一位小数"""
        return round(max(1.0, min(5.0, score)), 1)

    def _generate_summary(self, result: EvaluationResult) -> str:
        """生成评估摘要"""
        scores = result.to_pipeline_dict()
        avg = sum(scores.values()) / len(scores)
        strongest = max(scores, key=scores.get)
        weakest = min(scores, key=scores.get)

        if avg >= 4.0:
            level = "优秀"
        elif avg >= 3.0:
            level = "良好"
        elif avg >= 2.0:
            level = "一般"
        else:
            level = "需要提升"

        return (
            f"计算思维综合评定: {level}（均分{avg:.1f}/5.0）\n"
            f"最强维度: {strongest}（{scores[strongest]:.1f}分）\n"
            f"最需提升: {weakest}（{scores[weakest]:.1f}分）"
        )


# ============================================
# 便捷函数
# ============================================


def evaluate_code(code: str, student_id: str = "unknown") -> EvaluationResult:
    """快捷评估函数"""
    analyzer = CTCodeAnalyzer()
    return analyzer.evaluate(code, student_id=student_id)


def evaluate_file(filepath: str) -> EvaluationResult:
    """快捷文件评估函数"""
    analyzer = CTCodeAnalyzer()
    return analyzer.evaluate_file(filepath)


# ============================================
# 集成适配器：连接source_code_evaluator和completeV6
# ============================================


def prepare_pipeline_input(code: str, student_id: str,
                           homework_score: float = 75.0) -> "pd.DataFrame":
    """从源代码生成v6管道所需的DataFrame

    这是source_code_evaluator和completeV6之间的桥接函数。
    单次调用即可完成：源代码 → CT五维度分数 → v6管道输入DataFrame

    Args:
        code: Python源代码
        student_id: 学生ID
        homework_score: 历史作业质量（百分制，默认75）

    Returns:
        DataFrame with columns: 抽象, 分解, 算法设计, 建模, 评估, 作业质量
        可直接传入completeV6的train_model()和后续管道
    """
    import pandas as pd
    analyzer = CTCodeAnalyzer()
    result = analyzer.evaluate(code, student_id=student_id)

    row = result.to_pipeline_dict()
    row['作业质量'] = homework_score

    df = pd.DataFrame([row])
    df.index = [student_id]
    return df


def batch_prepare_pipeline_input(submissions: List[Tuple[str, str, float]]
                                 ) -> "pd.DataFrame":
    """批量生成v6管道输入

    Args:
        submissions: [(student_id, code, homework_score), ...]

    Returns:
        DataFrame ready for v6 pipeline
    """
    import pandas as pd
    analyzer = CTCodeAnalyzer()
    rows = []
    for sid, code, hw in submissions:
        result = analyzer.evaluate(code, student_id=sid)
        row = result.to_pipeline_dict()
        row['作业质量'] = hw
        row['student_id'] = sid
        rows.append(row)
    df = pd.DataFrame(rows)
    df.set_index('student_id', inplace=True)
    return df


# ============================================
# 可选：LLM辅助评语生成（需配置Ollama/OpenAI API）
# ============================================


def generate_llm_feedback(result: EvaluationResult,
                          api_base: str = "http://localhost:11434",
                          model: str = "qwen:7b",
                          timeout: int = 60) -> str:
    """调用LLM对评估结果生成教师评语（仅做语言润色，不改变评分）

    Args:
        result: CTCodeAnalyzer的评估结果
        api_base: Ollama API地址
        model: 模型名称
        timeout: 超时秒数

    Returns:
        教师评语文本
    """
    scores = result.to_pipeline_dict()

    # 构建详细的特征信息
    feature_details = []
    for dim_name, dim_result in result.dimensions.items():
        details = []
        for feat in dim_result.features:
            details.append(f"  - {feat.name}: {feat.explanation}")
        feature_details.append(f"【{dim_name}】（{dim_result.score}分）\n" + "\n".join(details))

    prompt = f"""你是一位计算思维教育专家。请根据以下代码分析结果，为一位编程初学者写一段鼓励性的反馈评语。

学生: {result.student_id}
综合评语框架: {result.overall_summary}

各维度详细分析:
{chr(10).join(feature_details)}

要求:
1. 先肯定学生的优点（从得分最高的维度入手）
2. 指出1-2个最需要改进的维度，给出具体建议
3. 语气温暖、鼓励，面向学生
4. 不要虚构任何数据，严格基于上述分析
5. 控制在200字以内"""

    try:
        import requests
        resp = requests.post(
            f"{api_base}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return f"[LLM调用失败: HTTP {resp.status_code}]"
    except Exception as e:
        return f"[LLM调用失败: {e}]"


# ============================================
# 自测代码
# ============================================

if __name__ == "__main__":
    # 示例：一个简单但结构较好的学生代码
    SAMPLE_CODE = '''
"""学生成绩管理系统 - 计算平均分和等级"""

from dataclasses import dataclass
from typing import List

PASS_THRESHOLD = 60
GRADE_A_THRESHOLD = 85


@dataclass
class Student:
    """学生数据模型"""
    name: str
    scores: List[float]

    def average(self) -> float:
        """计算平均分"""
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def grade(self) -> str:
        """根据平均分评定等级"""
        avg = self.average()
        if avg >= GRADE_A_THRESHOLD:
            return "A"
        elif avg >= PASS_THRESHOLD:
            return "B"
        else:
            return "C"


def calculate_class_average(students: List[Student]) -> float:
    """计算班级平均分"""
    if not students:
        return 0.0
    total = sum(s.average() for s in students)
    return total / len(students)


def find_top_students(students: List[Student], top_n: int = 3) -> List[Student]:
    """找出成绩最好的N个学生"""
    return sorted(students, key=lambda s: s.average(), reverse=True)[:top_n]


def test_student_average():
    """测试平均分计算"""
    s = Student("测试", [80, 90, 85])
    assert s.average() == 85.0, f"期望85，得到{s.average()}"
    print("测试通过!")


if __name__ == "__main__":
    # 创建样本数据
    students = [
        Student("张三", [85, 90, 78]),
        Student("李四", [92, 88, 95]),
        Student("王五", [60, 65, 70]),
    ]
    avg = calculate_class_average(students)
    print(f"班级平均分: {avg:.1f}")

    top = find_top_students(students, top_n=2)
    print(f"前2名: {[s.name for s in top]}")

    test_student_average()
'''

    print("=" * 60)
    print("源代码自动评估模块 - 自测")
    print("=" * 60)

    analyzer = CTCodeAnalyzer()
    result = analyzer.evaluate(SAMPLE_CODE, student_id="S001")

    print(f"\n学生: {result.student_id}")
    print(result.overall_summary)
    print()

    for dim_name in ('抽象', '分解', '算法设计', '建模', '评估'):
        dim = result.dimensions[dim_name]
        print(f"{'─' * 40}")
        print(f"【{dim_name}】得分: {dim.score}/5.0")
        for feat in dim.features:
            bar = "#" * int(feat.contribution * 20) + "-" * (20 - int(feat.contribution * 20))
            print(f"  {feat.name:16s} [{bar}] {feat.explanation}")
        if dim.code_evidence:
            for ev in dim.code_evidence[:3]:
                print(f"    [*] {ev}")

    print(f"\n{'=' * 60}")
    print("管道输入格式:")
    print(result.to_pipeline_dict())
