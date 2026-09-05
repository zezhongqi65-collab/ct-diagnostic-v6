# 论文大纲：可解释AI驱动的计算思维个性化诊断与干预系统

> 遵循 `scientific-writing` 技能 IMRAD 结构 + `venue-templates` 期刊模板标准。
> 标注: ✅ 暑假可完成 | 🔬 需下学期实验数据 | 📝 已有初稿

---

## 拟投期刊

| 优先级 | 期刊 | 类型 | 特点 |
|--------|------|------|------|
| 1 | *Computers & Education* (IF ~12) | 英文，教育技术顶刊 | 要求实证研究，需实验数据 |
| 2 | *British Journal of Educational Technology* (IF ~5) | 英文，教育技术 | 接受系统设计+实证验证 |
| 3 | *Learning and Instruction* (IF ~4) | 英文，学习科学 | 侧重认知机制与学习效果 |
| 4 | *数据分析与知识发现* (中文核心) | 中文，CS+数据 | 王振宇等(2026)发表期刊，接受XAI教育应用 |
| 5 | *电化教育研究* (CSSCI) | 中文，教育技术 | 国内教育技术顶刊 |

**暑假建议**：先按英文期刊准备（Introduction + Related Work + Methodology），实验后确定目标期刊再调格式。

---

## 论文标题（草案）

**中文**: 可解释人工智能驱动的计算思维诊断与个性化干预：融合SHAP归因与因果推断的双维度方法

**English**: Explainable AI-Driven Computational Thinking Diagnosis and Personalized Intervention: A Dual-Dimensional Approach Integrating SHAP Attribution with Causal Inference

---

## 完整大纲

### Abstract (200-250 words) 🔬

```
[Background] 计算思维教育面临"诊断不精准、干预一刀切"的困境
[Purpose] 提出融合SHAP归因与因果推断的可解释AI诊断系统
[Methods] 双通道输入（量表+代码AST）→ XGBoost预测 → 三层因果诊断
          → 四分类干预建议 → 三层保真报告
[Key innovation] SHAP×因果效应双维度诊断矩阵 + 根因追溯
[Results] 🔬 待实验数据
[Conclusion] 🔬
```

**状态**: 🔬 实验结果出后填写

---

### 1. Introduction ✅📝

#### 1.1 研究背景
- 计算思维是21世纪核心素养（Wing, 2006）
- 信息技术课程中教师面临的核心困境：如何精准诊断个体薄弱点
- 传统评估的三局限：总分掩盖多维结构、相关性≠因果性、诊断难以转化为教学行动

#### 1.2 问题陈述
- 现有CT评估工具输出单一总分，无法指导差异化教学
- 机器学习预测模型缺乏可解释性
- 缺乏从"诊断"到"干预"的因果闭环

#### 1.3 研究目标与贡献
1. 构建双通道CT五维度诊断系统（量表 + 代码AST）
2. 提出SHAP×因果效应双维度干预矩阵
3. 设计三层保真报告机制确保AI输出的可靠性
4. 🔬 验证XAI个性化干预的增量效果

#### 1.4 论文结构
Section 2 文献综述 → Section 3 系统设计 → Section 4 实验 → Section 5 讨论

**状态**: ✅ 引言框架已有（研究报告第一章），需改写为正式学术风格
**参考**: `scientific-writing` 技能 Section 2: Introduction Development

---

### 2. Related Work ✅

#### 2.1 计算思维评估方法
- 2.1.1 量表评估（Korkmaz CTS, 中文CT量表）
- 2.1.2 代码自动评估（Dr. Scratch, AST分析）
- 2.1.3 研究空白：双通道统一框架的缺失

#### 2.2 可解释AI在教育中的应用
- 2.2.1 模型归因方法（SHAP, LIME及教育应用）
- 2.2.2 反事实解释
- 2.2.3 研究空白：归因≠因果，缺乏可操作的诊断框架

#### 2.3 因果推断在教育中的应用
- 2.3.1 从相关到因果（Pearl DAG框架）
- 2.3.2 后门调整与教育准实验
- 2.3.3 个体处理效应估计（T/X-Learner）
- 2.3.4 研究空白：因果推断在CT教育中的系统应用

**状态**: ✅ 详细内容见 `paper/literature_review.md`

---

### 3. Methodology ✅📝

#### 3.1 系统总览
**内容**: 双通道→诊断管道→报告生成的完整架构图
**状态**: 📝 研究报告第三章已有ASCII图，需升级为规范图
**参考**: `scientific-schematics` 技能 — 生成 Fig.1 系统架构图

#### 3.2 五维度CT评估模型
**内容**: 量表15题→五维度聚合 | AST → 52类指标 → 五维度得分
**状态**: 📝 研究报告第二章
**需要的公式**:
- 量表维度聚合公式（(A1+A2+A3)/3 型）
- AST指标归一化公式

#### 3.3 预测模型：XGBoost
**内容**: 目标函数、正则化、训练配置
**状态**: 📝 研究报告第四章，需补充公式
**需要的公式**:
- XGBoost目标函数: $\mathcal{L}(\theta) = \sum_{i=1}^N \ell(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$
- 正则化项: $\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$

#### 3.4 三层诊断架构

##### 3.4.1 Layer 1: 反事实分析
**公式**: $CFE(x, j, v) = f(x_{[j \leftarrow v]}) - f(x)$
其中 $x_{[j \leftarrow v]}$ 表示将第j维特征替换为值v
**状态**: 📝

##### 3.4.2 Layer 2: SHAP × 因果效应双维度诊断（个体化调制）
**SHAP公式**: $\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f_x(S \cup \{j\}) - f_x(S)]$
**后门调整公式**: $ATE_j = \mathbb{E}[Y | do(T_j = t + 1)] - \mathbb{E}[Y | do(T_j = t)]$（统一为「每 1 分」的边际效应）
**DAG结构**: 抽象→{分解, 算法设计}→{建模, 评估}
**个体化因果效应**: $CE^{ind}_j = CE_j \times \max(0,\ 5 - s_j)$（引入个体当前得分 $s_j$ 与提升空间）
**四象限分类规则**: $(|SHAP_j| > \tau_s,\ CE^{ind}_j > \tau_c)$
**提升空间前置过滤**: $\max(0,\ 5 - s_j) < 0.5 \Rightarrow$ 「无需关注」（接近满分，提升空间有限）
**自适应阈值**: $\tau_s = P_{50}(\text{mean\_abs\_SHAP}), \tau_c = \text{median}(CE_{all})$
**状态**: 📝

##### 3.4.3 Layer 3: ITE个体处理效应
**T-Learner**: $\hat{\tau}(x_i) = \hat{\mu}_1(x_i) - \hat{\mu}_0(x_i)$
**X-Learner**: 两阶段效应模型
**状态**: 📝

#### 3.5 AST源代码静态分析
**内容**: AST解析→52类指标→五维度评分
**关键公式**: 评分 = $\sum$ 特征贡献，映射到[1,5]
**状态**: 📝 source_code_evaluator_patched.py

#### 3.6 三层保真报告生成
**内容**: 模板填充(100%保真)→LLM润色(数据不动)→程序校验(失败回退)
**状态**: 📝

#### 3.7 润色引擎自动切换
**内容**: Ollama → DeepSeek API → 模板回退
**状态**: 📝

**状态**: ✅ 核心方法已有代码实现，暑假任务是将代码逻辑转为正式论文方法章节
**参考**: `scientific-writing` 技能 Section 5: Methods Documentation

---

### 4. Experiments 🔬

#### 4.1 实验设计
- 研究问题与假设 (H1, H2, H3)
- 参与者（人数、年级、背景）
- 前测-后测对照组设计（A: XAI个性化 / B: 统一干预 / C: 对照）
- 随机化方案（分层区组）

#### 4.2 数据集
- 样本量、课程范围、数据收集过程
- 描述性统计（前测得分分布、各组基线平衡性检验）

#### 4.3 评估指标
- 主指标：CT五维度得分变化
- 次指标：教师可用性问卷、预测准确率

#### 4.4 基线模型对比
- 9个基线模型的RMSE/MAE/R²对比表
- 10折交叉验证 + 配对t检验
- SHAP重要性排序一致性（Kendall W）

#### 4.5 消融实验
- 移除单维度的性能衰减
- 量表 vs 代码通道独立贡献
- Layer 1/2/3 逐层消融

#### 4.6 诊断有效性
- A组薄弱维度提升 vs B组同维度提升
- 四分类干预建议的命中率（"优先干预"维度是否确实提升最大）

#### 4.7 教师可用性评估
- 问卷得分分布
- 教师对报告各维度的评分

#### 4.8 保真度校验结果
- 大模型润色通过率
- 润色前后数据一致性

#### 4.9 系统性能
- 诊断管道运行时间
- Ollama vs DeepSeek 润色耗时对比

**状态**: 🔬 全部依赖下学期实验数据
**暑假可做**: 4.4 基线对比（模拟数据预跑）、4.5 消融实验框架

---

### 5. Discussion 🔬

#### 5.1 主要发现
- XAI个性化干预是否显著优于统一干预？
- 薄弱维度的精准诊断是否比笼统训练更有效？

#### 5.2 SHAP×因果融合的价值
- 对比纯SHAP和SHAP+因果的诊断一致性
- 根因追溯的命中率

#### 5.3 与现有研究的对比
- 与王振宇等(2026)的慕课行为检测对比：互补而非竞争
- 与Tomasevic等(2020)的成绩预测对比
- 与Künzel等(2019)的ITE方法对比

#### 5.4 局限与未来
- 样本量与泛化性
- DAG依赖专家知识
- AST仅支持Python
- 干预材料未充分验证
- 未来：纵向追踪、多语言支持、自动因果发现

**状态**: 🔬 依赖实验数据，但框架和局限分析可提前写

---

### 6. Conclusion 🔬

- 总结贡献
- 实践意义（对信息技术教师的价值）
- 未来工作

**状态**: 🔬 实验后完成

---

## 暑期任务优先级

| 优先级 | 任务 | 产出 | 状态 |
|--------|------|------|------|
| P0 | 文献综述定稿 | `paper/literature_review.md` | ✅ 已有框架，需补检索 |
| P0 | 实验方案定稿 | `paper/experimental_design.md` | ✅ 已完成 |
| P0 | 基线对比预跑 | `benchmark.py` | ✅ 代码已写，待运行 |
| P1 | Introduction 改写 | 正式学术风格 | 📝 研究报告第一章 |
| P1 | Methodology 形式化 | 补充公式 + 规范架构图 | 见研究报告第四-五章 |
| P1 | 画 Fig.1-4 | `paper/figures/` | 需 scientific-schematics |
| P2 | 消融实验代码框架 | `benchmark.py` 已含 | ✅ |
| P2 | 设计干预训练材料 | 5维度×3套=15套 | 未开始 |
| P2 | 伦理审批材料 | IRB申请表+知情同意书 | 未开始 |
| P2 | 完整论文 LaTeX 模板 | `paper/manuscript.tex` | 未开始 |
| P3 | 英文翻译/润色 | — | 实验完成后 |

---

## 参考技能映射

| 论文部分 | 使用的 scientific-agent-skills |
|----------|-------------------------------|
| Related Work | `literature-review` + `paper-lookup` + `bgpt-paper-search` |
| Methodology | `scientific-writing` + `scientific-schematics` |
| SHAP分析 | `shap` (Workflow 4: Model Comparison) |
| 因果推断 | `statsmodels` + `pymc` |
| 统计检验 | `statistical-analysis` + `statsmodels` |
| 实验设计 | `experimental-design` + `statistical-power` |
| 图表 | `scientific-visualization` + `matplotlib` + `seaborn` |
| 投稿准备 | `venue-templates` + `peer-review` + `scientific-critical-thinking` |
