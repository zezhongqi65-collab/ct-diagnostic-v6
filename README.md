# 计算思维诊断系统 (CT Diagnostic System v6)

双通道、三层因果可解释AI教育诊断系统。基于 XGBoost + SHAP + 后门调整因果效应，自动诊断学生计算思维薄弱维度，生成可视化诊断报告。

## 核心能力

- **双输入通道**：量表数据（CSV/Excel） / Python源代码（AST静态分析）
- **三层因果诊断**：反事实分析 → SHAP×因果效应双维度矩阵 → ITE个体处理效应
- **四分类干预建议**：✅优先干预 / ⚠️仅观察 / 💡潜在有效 / ➖无需关注
- **批量诊断**：多学生同时诊断，汇总导出 CSV/Excel
- **量表数据预处理**：原始 15 题（A–E 作答）→ 五维度得分，一键转换导入
- **个性化干预出题**：按薄弱维度分组，生成可打印的思维训练题单（Word）
- **三层保真机制**：模板强制填充 → 润色引擎自动切换（Ollama / DeepSeek）→ 自动保真度校验

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 DeepSeek API Key（用于报告润色，可选）
export DEEPSEEK_API_KEY="sk-你的密钥"

# 3. 启动应用
streamlit run app.py

# 4. 可选：启动 Ollama 作为优先润色后端
ollama serve
ollama pull qwen:7b
```

打开浏览器访问 `http://localhost:8501`。

## 项目结构

```
├── app.py                              # Streamlit Web 应用主入口
├── completeV6_patched.py               # 三层因果诊断核心引擎
├── source_code_evaluator_patched.py    # 源代码 AST 静态评估模块
├── make_scale_template.py              # 生成量表数据收集模板（Excel）
├── preprocess_scale_data.py            # 量表数据预处理脚本
├── utils/
│   ├── diagnostics.py                  # 诊断管道封装 API
│   ├── report.py                       # Markdown/Excel 报告生成与导出
│   ├── scale_data.py                   # 量表维度映射与 A-E/分数转换
│   └── intervention.py                 # 个性化干预出题模块
├── data/
│   └── question_bank.json              # 思维训练母题题库（50 道）
├── requirements.txt                    # Python 依赖
├── packages.txt                        # Streamlit Cloud Linux 系统依赖
└── .gitignore
```

## 三层诊断架构

| 层级 | 名称 | 方法 | 输出 |
|------|------|------|------|
| Layer 1 | 反事实分析 | 单维度 counterfactual simulation | 每维度假设提至满分时的预测变化 |
| Layer 2 | 双维度诊断 | SHAP 归因 × 后门调整因果效应 | 四分类干预矩阵 + 散点图 |
| Layer 3 | ITE 估计 | X-Learner / T-Learner | 个体处理效应（需前后测实验数据） |

## 使用方式

### 模式 A：量表输入

上传包含学生五维度得分（抽象、分解、算法设计、建模、评估）的 CSV/Excel 文件，或手动输入得分。系统使用参考模型或用户数据重训练的 XGBoost 模型进行诊断。

### 模式 B：代码输入

上传 `.py` 文件或粘贴代码。系统通过 AST 静态分析自动评估五维度得分（识别模式抽象能力、问题分解结构、算法设计模式、建模抽象层次、代码质量评估），然后接入诊断管道。

### 批量模式

上传包含多个学生的数据文件或代码 ZIP 包，系统依次诊断并生成汇总表，支持导出为 CSV/Excel。

### 量表数据预处理

1. 运行 `python make_scale_template.py` 生成量表数据收集模板（Excel，含数据校验与维度配色）。
2. 学生纸质作答 A–E 后，录入模板（填 A–E 或 1–5 均可）。
3. 运行 `python preprocess_scale_data.py 你的数据.xlsx`，生成：
   - `前测_系统诊断输入.csv`：直接上传到「批量诊断」
   - `前测_完整数据.csv`：保留原始题项与班级，供后测统计分析

### 个性化干预出题

1. 完成批量诊断后，页面底部出现「生成干预题单」。
2. 选择周次、是否用 DeepSeek 生成变式、每份题数，点击生成。
3. 下载 Word 题单（按薄弱维度分组），打印发给对应学生。
   - 题目为不插电思维训练题，纸笔可做、无需电脑。
   - 题库预置 50 道母题（每维度 10 道）；开启变式后 DeepSeek 会换数字/情境扩充，失败自动回退母题。

## 云部署

### Streamlit Cloud

1. Fork 本仓库
2. 在 [share.streamlit.io](https://share.streamlit.io) 连接 GitHub 并选择本仓库
3. 设置 Main file path 为 `app.py`
4. 在 Settings → Secrets 中添加 DeepSeek API Key：
   ```toml
   DEEPSEEK_API_KEY = "sk-你的密钥"
   ```
5. 部署 — `packages.txt` 会自动安装中文字体

### 本地 Docker（可选）

```bash
docker run -p 8501:8501 -v $(pwd):/app python:3.11-slim bash -c "
  apt-get update && apt-get install -y fonts-wqy-zenhei &&
  pip install -r /app/requirements.txt &&
  streamlit run /app/app.py
"
```

## 依赖

- Python ≥ 3.10
- streamlit, pandas, numpy
- xgboost, shap, scikit-learn
- matplotlib, statsmodels
- openpyxl, requests, openai, python-docx

## 报告保真机制

1. **程序模板强制填充**（100%保真）：所有数据、分类、优先级直接来自算法输出
2. **润色引擎自动切换**：Ollama → DeepSeek API → 模板回退，任一可用即自动选用
   - 优先尝试本地 Ollama（低延迟、无外网依赖）
   - 不可用时自动切换 DeepSeek API（云端大模型）
   - 均不可用时直接使用模板报告（保证服务不中断）
3. **自动保真度校验**：程序核对润色后报告是否忠实于原始数据，否则自动回退

*本报告基于可解释AI分析，仅供参考，不构成教育决策的唯一依据。*
