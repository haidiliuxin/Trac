# TracLLM 项目代码结构深度解析

本文档详细解析了 TracLLM 项目的代码结构与各个模块的功能，旨在帮助开发者快速理解项目架构并进行二次开发。

## 📁 目录结构概览

```
TracLLM/
├── main.py                  # [核心入口] 实验启动主程序
├── run_deepseek_reproduce.py # [复现脚本] 适配 DeepSeek 的一键运行脚本
├── analyze_metrics.py       # [工具脚本] 离线分析实验结果指标
├── src/                     # [核心源码] 包含模型、算法、评估等核心逻辑
│   ├── models/              # 模型层：封装不同后端 LLM
│   ├── attribution/         # 算法层：实现 TracLLM 及其他归因算法
│   ├── evaluate.py          # 评估层：计算精确率、ASR 等指标
│   ├── load_dataset.py      # 数据层：加载 LongBench, PoisonedRAG 等数据集
│   └── utils.py             # 工具层：通用辅助函数
├── model_configs/           # [配置] 模型参数配置文件 (.json)
├── datasets/                # [数据] 存放数据集文件
└── results/                 # [输出] 实验结果保存目录
```

---

## 🧩 核心模块详细说明

### 1. 根目录核心文件

*   **`main.py`**
    *   **作用**：整个框架的入口。负责解析命令行参数、初始化模型、加载数据集、执行归因算法、并保存结果。
    *   **关键逻辑**：它串联了 `load_dataset` -> `create_model` -> `create_attr` -> `attr.attribute` -> `evaluate` 的完整工作流。

*   **`run_deepseek_reproduce.py`** (新增)
    *   **作用**：专为复现实验编写的启动脚本。
    *   **功能**：自动设置环境变量（解决代理/镜像问题）、覆盖 `main.py` 的参数（如 `sh_N`, `K`），适配 DeepSeek API 调用。

### 2. 模型层 (`src/models/`)

负责统一不同 LLM 的调用接口，屏蔽底层差异。

*   **`__init__.py`**: 工厂模式入口，根据配置中的 `provider` (如 huggingface, deepseek) 实例化对应的模型类。
*   **`GPT.py`**:
    *   **作用**：适配 OpenAI 格式的 API（包括 DeepSeek, GPT-4）。
    *   **关键点**：处理 API Key、Base URL，并封装 `query` 方法用于发送请求。
*   **`Llama.py` / `HF_model.py`**:
    *   **作用**：适配本地 Hugging Face 模型。
    *   **关键点**：处理本地显存加载、Tokenizer 编解码、以及 `bfloat16` 等精度设置。

### 3. 算法层 (`src/attribution/`)

这是 TracLLM 的核心，实现了归因算法逻辑。

*   **`attribute.py`**:
    *   **作用**：定义了归因的基类 `Attribution`。
    *   **功能**：规定了输入（Question, Contexts, Answer）和输出（Important Scores）的标准接口。
*   **`perturbation_based.py`**:
    *   **作用**：实现了 **TracLLM** 及其变体（STC, LOO, Shapley）。
    *   **核心逻辑**：
        *   **STC (Similarity-To-Context)**: 计算上下文与答案的相似度。
        *   **LOO (Leave-One-Out)**: 依次移除某段上下文，观察答案变化（困惑度/Token概率变化）来计算重要性。
        *   **Shapley**: 通过多次采样排列组合，计算 Shapley 值，是 TracLLM 归因精度的核心保障。
*   **`self_citation.py`**:
    *   **作用**：实现基于自引用的归因基线方法（让模型自己标注引用）。

### 4. 数据与评估层

*   **`src/load_dataset.py`**:
    *   **作用**：统一加载不同格式的数据集。
    *   **支持数据**：
        *   `LongBench` (MuSiQue, NarrativeQA): 长文本理解。
        *   `PoisonedRAG` (NQ-Poison): 知识投毒防御。
        *   `NeedleInHaystack`: 大海捞针测试。
*   **`src/evaluate.py`**:
    *   **作用**：计算实验指标。
    *   **关键指标**：
        *   `Precision/Recall`: 归因是否准确找到了由于注入或投毒的片段。
        *   `ASR (Attack Success Rate)`: 移除归因片段后，模型是否还能防御攻击。

### 5. 配置文件 (`model_configs/`)

存放模型的具体参数，例如 `deepseek_config.json`：
```json
{
    "model_info": { "provider": "deepseek", "base_url": "..." },
    "params": { "temperature": 0.001, "max_output_tokens": 100 }
}
```

## 🚀 代码执行流向

1.  **启动**: 用户运行 `run_deepseek_reproduce.py`。
2.  **配置**: 脚本加载 `model_configs/deepseek_config.json`。
3.  **加载**: `src/load_dataset.py` 读取 `datasets/` 下的数据。
4.  **推理**: `src/models/GPT.py` 调用 DeepSeek API 生成初始答案。
5.  **归因**: `src/attribution/perturbation_based.py` 对上下文进行多次扰动（Mask/Remove），计算每个片段的贡献分（Shapley Value）。
6.  **评估**: `src/evaluate.py` 移除高分片段，再次推理，检查攻击是否成功。
7.  **结果**: 数据存入 `results/`，并通过 `analyze_metrics.py` 打印最终报表。

---
*文档生成时间: 2025-12-27*
