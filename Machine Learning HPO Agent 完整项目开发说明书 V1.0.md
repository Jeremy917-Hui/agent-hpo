# Machine Learning HPO Agent
## 完整项目开发说明书（Codex Handoff）

**版本：V1.0**

**日期：2026-11-17**

**开发阶段：MVP 开始实施**

**项目部署：Windows Local Mode 优先**

**项目根目录：`E:\agent-hpo\`**

---

# 0. 给 Codex 的总体开发指令

本项目是一个面向传统机器学习模型的智能超参数优化与实验管理 Agent。

目前已经完成了初步的需求分析和技术架构讨论，现在正式进入实际开发阶段。

请不要只提供理论方案，也不要一次性生成一个规模庞大但无法运行的系统。应该采用分阶段、模块化、可测试的开发方式，优先实现真正可以运行的端到端 MVP。

**最重要的开发原则：先跑通完整工作流，再逐步完善分布式能力。**

第一版必须能够完成以下过程：

1. 用户通过 Web Dashboard 上传 CSV 数据集。
2. 用户通过自然语言描述机器学习任务。
3. Agent 调用 Qwen 商用 API 理解任务。
4. Agent 通过工具检查数据集和查询模型。
5. Agent 生成机器学习实验方案。
6. 用户在 Dashboard 确认实验方案。
7. 系统调用 Optuna 执行超参数优化。
8. Worker 在后台训练模型。
9. Dashboard 展示实验进度。
10. 实验完成后 Agent 分析结果并生成报告。

不要让 Agent 自己编写任意 Python 代码执行训练。所有模型和实验操作必须通过系统预先定义的工具及模型注册表完成。

---

# 1. 项目背景与目标

## 1.1 项目定位

Machine Learning HPO Agent 是一个利用大语言模型进行实验规划、利用 Optuna 进行超参数优化、利用 Orchestrator 和 Worker 管理后台实验任务的智能机器学习实验平台。

它不只是一个调用 LLM 的聊天机器人，而是一个能够完成实际机器学习任务的 Agent 系统。

用户上传数据集并提出需求，Agent 自动完成数据分析、模型选择、超参数空间设计和实验方案生成。用户确认后，系统执行模型训练和超参数搜索，最后输出实验结果。

## 1.2 项目目标

分为两类。

**第一类：机器学习实验能力。**

- 支持分类、回归、聚类和降维任务。
- 自动分析数据集。
- 自动选择合适的机器学习模型。
- 自动构建超参数搜索空间。
- 自动运行 Optuna。
- 比较不同模型和参数组合。
- 输出模型指标与实验报告。

**第二类：分布式系统能力。**

- 支持多个 Worker 并发执行 Trial。
- 通过共享存储维护实验状态。
- 管理任务生命周期。
- 实现超时、重试、幂等和故障检测。
- 测量 Worker 数量对实验效率的影响。
- 分析计算资源利用率和并行加速比。

---

# 2. 开发环境与硬件约束

## 2.1 用户设备

当前 Windows 64 位笔记本配置：

| 项目 | 配置 |
|---|---|
| CPU | AMD Ryzen 7 5825U |
| CPU 核心 | 8 核，16 线程（以系统实际检测为准） |
| RAM | 16 GB |
| GPU | AMD Radeon 集成显卡 |
| 显存 | 系统显示约 496 MB |
| 系统盘 | C 盘 |
| 项目盘 | E 盘，用户提供的可用空间超过 70 GB |

## 2.2 资源使用原则

1. 不在本地部署 Qwen3-8B。
2. 使用商用 Qwen API。
3. Scikit-learn 第一版采用 CPU 训练。
4. 不要求本地 CUDA 或 NVIDIA GPU。
5. 默认只启动一个 Worker。
6. 经过内存和 CPU 测试后，再尝试两个 Worker。
7. 4 个 Worker 仅在小数据集性能测试或未来远程服务器环境中考虑。
8. 防止多个 Worker 和 Scikit-learn 内部并行共同造成 CPU 过度订阅。
9. 数据集上传设置大小限制，并在训练前估算内存需求。
10. 系统启动时检查 E 盘可用空间，避免实验输出耗尽硬盘。

## 2.3 路径要求

所有项目文件统一放置于：

`E:\agent-hpo\`

建议通过配置项：

`PROJECT_ROOT`

管理项目根目录。

默认数据目录：

- `E:\agent-hpo\data\datasets\`
- `E:\agent-hpo\data\runs\`
- `E:\agent-hpo\data\db\`

默认日志目录：

`E:\agent-hpo\logs\`

Python 虚拟环境：

`E:\agent-hpo\.venv\`

依赖缓存：

`E:\agent-hpo\.cache\`

临时文件：

`E:\agent-hpo\tmp\`

不要把项目文件、用户上传的数据集、实验结果和数据库默认放到 C 盘。

所有路径必须通过统一配置管理，不允许在各模块中散落硬编码路径。

---

# 3. 核心技术栈

以下为第一版推荐实现方案。

| 模块 | 技术 |
|---|---|
| 后端 | Python + FastAPI |
| 前端 | FastAPI + Jinja2 + HTML/CSS/JavaScript |
| 数据验证 | Pydantic |
| LLM | Qwen 商用 API |
| Agent | 自定义 ReAct Loop |
| 工具协议 | MCP Python SDK |
| 数据处理 | Pandas / NumPy |
| 机器学习 | Scikit-learn |
| 超参数优化 | Optuna |
| 应用数据库 | SQLite |
| Optuna 多 Worker Storage | PostgreSQL |
| 日志 | Python logging，结构化输出 |
| 测试 | Pytest |

前端第一版优先使用 Jinja2 和原生 JavaScript，减少 Node.js 和前端工程配置成本。未来如果 Dashboard 复杂度明显增加，再考虑迁移 React。

PostgreSQL 不要求在第一天安装。第一阶段单 Worker 使用 SQLite 即可；进入多 Worker 并行阶段后再引入 PostgreSQL。

Docker 是否可用由 Codex 检查，不要假设用户已经安装 Docker。

Qwen API 的模型名称、接口地址和 Tool Calling 格式应通过配置适配，不能随意硬编码未经验证的接口参数。

---

# 4. 系统总体架构

## 4.1 架构图

```text
                   User
                    |
                    v
              Web Dashboard
                    |
                    v
              FastAPI Backend
                    |
       +------------+------------+
       |            |            |
       v            v            v
  Agent Runtime  Dataset      Model
       |         Manager      Registry
       |
       +---- Qwen API
       |
       +---- MCP Client
       |       |
       |       +---- HPO MCP Server
       |       |
       |       +---- MatPool MCP Server
       |
       v
  Experiment Orchestrator
       |
       v
    Optuna Study
       |
       v
  Worker Process(es)
       |
       v
  Model Training / Evaluation
       |
       v
  Experiment Storage
       |
       v
  Dashboard + Agent Report
```

## 4.2 模块职责

### Dashboard

负责：

- 数据集上传。
- 数据集列表和详情。
- 用户自然语言输入。
- 实验计划展示。
- 用户审批。
- 实验状态展示。
- Trial 结果展示。
- 模型对比。
- 最终实验报告。

### Backend

负责：

- API 路由。
- 请求校验。
- 用户和实验状态管理。
- 数据库操作。
- 审批状态校验。
- Agent 与 Orchestrator 的协调。

### Agent Runtime

负责：

- 理解用户需求。
- 调用 Qwen API。
- 维护 ReAct Loop。
- 发现和选择工具。
- 分析工具返回结果。
- 生成实验计划。
- 根据实验结果生成报告。

Agent 不直接执行模型训练。

### Dataset Manager

负责：

- 文件接收。
- 文件存储。
- CSV 解析。
- 数据集元信息提取。
- 数据质量分析。
- 数据集 ID 管理。
- 数据访问权限。

### Model Registry

负责：

- 注册允许使用的 Scikit-learn 模型。
- 定义模型参数空间。
- 返回模型元信息。
- 检查参数合法性。
- 创建模型实例。

### Plan Validator

负责：

- 检查模型是否属于允许范围。
- 检查参数空间是否合法。
- 检查任务类型和模型是否匹配。
- 检查数据集和目标变量。
- 检查实验预算。
- 检查指标设置。
- 检查用户审批版本。

### Orchestrator

负责 Run 级别的管理：

- 创建实验任务。
- 启动 Worker。
- 监控 Worker。
- 控制并发数。
- 管理总 Trial 预算。
- 处理取消请求。
- 监控实验状态。
- 处理 Worker 异常。
- 实验完成后触发 Agent 分析。

### Optuna

负责 Trial 级别的优化：

- 创建 Study。
- 生成超参数。
- 记录 Trial 状态。
- 保存参数和指标。
- 根据历史 Trial 调整后续搜索。

### Worker

负责实际执行：

- 加载数据集。
- 进行数据预处理。
- 创建模型。
- 训练模型。
- 预测。
- 计算指标。
- 将结果写入 Optuna Storage。

---

# 5. 数据集管理

## 5.1 上传流程

用户通过 Dashboard 上传 CSV。

系统：

1. 检查文件类型。
2. 检查文件大小。
3. 检查文件名和路径安全性。
4. 生成唯一 Dataset ID。
5. 保存原始 CSV。
6. 提取数据集元信息。
7. 保存 metadata.json。

目录示例：

```text
data/
└── datasets/
    └── dataset_001/
        ├── original.csv
        └── metadata.json
```

## 5.2 数据集摘要

Agent 通过 `inspect_dataset()` 获取：

- 行数。
- 列数。
- 列名。
- 数据类型。
- 缺失值数量和比例。
- 数值型特征统计。
- 类别型特征数量。
- 重复记录。
- 目标变量分布。
- 初步任务类型判断。

不默认将完整 CSV 或大量原始样本传给 LLM。

## 5.3 目标变量

分类和回归任务需要目标变量。

用户可以在上传数据集后选择目标列，也可以在任务输入中说明。

例如：

`is_bad_customer`

但不能把这个变量写死在代码中。

聚类不需要目标变量。

降维默认不需要目标变量；如果用户选择降维后的下游预测任务，则可以指定目标变量。

## 5.4 数据泄漏防范

- 先划分训练集和测试集，再进行需要学习数据分布的预处理。
- 使用 Scikit-learn Pipeline。
- 预处理器只能在训练数据上 fit。
- 测试集不能参与超参数优化。
- 分类任务注意类别不平衡。
- 根据任务选择合理的交叉验证方法。

---

# 6. Model Registry

第一版只允许使用注册表中的模型。

不允许 LLM 自行生成任意 Python 训练代码。

## 6.1 Classification

第一阶段模型：

- LogisticRegression
- RandomForestClassifier

后续扩展：

- DecisionTreeClassifier
- KNeighborsClassifier
- SVC
- HistGradientBoostingClassifier

指标：

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

## 6.2 Regression

第一阶段模型：

- LinearRegression
- RandomForestRegressor

后续扩展：

- Ridge
- DecisionTreeRegressor
- SVR
- HistGradientBoostingRegressor

指标：

- MAE
- MSE
- RMSE
- R²

## 6.3 Clustering

候选模型：

- KMeans
- DBSCAN
- AgglomerativeClustering
- GaussianMixture

指标：

- Silhouette Score
- Calinski-Harabasz
- Davies-Bouldin

注意处理只有一个簇、全部噪声点等无法正常计算指标的情况。

## 6.4 Dimensionality Reduction

候选模型：

- PCA
- TruncatedSVD

指标：

- Explained Variance Ratio
- Reconstruction Error
- 可选下游任务性能

降维任务的优化目标必须结合用户需求定义，不能简单地无限增大维度以追求解释方差。

## 6.5 Registry 接口

至少提供：

```python
list_models(task_type)

get_model_spec(model_name)

validate_model_params(model_name, params)

create_estimator(model_name, params)
```

每个模型需要定义：

- 模型名称。
- 任务类型。
- Estimator。
- 参数空间。
- 参数类型。
- 参数边界。
- 默认值。
- 评价指标支持情况。

---

# 7. Agent 与 ReAct Loop

## 7.1 Agent 定位

第一版使用单 Agent，不开发多 Agent 系统。

LLM 负责：

- 用户需求理解。
- 工具选择。
- 数据集分析。
- 模型选择。
- 实验计划生成。
- 结果分析。

LLM 不负责：

- 直接读取任意本地文件。
- 执行任意 Python。
- 自己训练模型。
- 自己执行超参数搜索。
- 绕过用户审批。
- 直接管理数据库。

## 7.2 ReAct 流程

```text
User Input
    |
    v
LLM Reasoning
    |
    v
Tool Selection
    |
    v
Tool Dispatcher
    |
    v
MCP Tool Execution
    |
    v
Structured Result
    |
    v
LLM Observation
    |
    +---- Continue
    |
    +---- Ask User
    |
    +---- Generate Plan
```

最大 Agent 步数第一版设置为可配置参数，默认 10。

每个 Tool 设置超时和有限重试。

如果需要用户补充信息，Agent 应停止自动执行并请求用户输入。

## 7.3 两阶段 Agent

第一轮：

理解需求 → 检查数据 → 查询模型 → 生成计划 → 用户审批 → 提交实验。

后台：

Orchestrator + Optuna + Worker 执行实验。

第二轮：

实验完成 → 触发 Agent → 分析指标 → 生成报告。

不要让 Agent 每隔几秒调用一次 Qwen API 等待实验结束。

---

# 8. 实验方案与用户审批

Agent 生成的实验计划应包含：

```json
{
  "dataset_id": "dataset_001",
  "task_type": "classification",
  "target_column": "is_bad_customer",
  "models": [
    "LogisticRegression",
    "RandomForestClassifier"
  ],
  "optimization_metric": "roc_auc",
  "direction": "maximize",
  "total_trials": 20,
  "sampler": "TPESampler",
  "validation_strategy": "StratifiedKFold",
  "random_seed": 2021
}
```

上述仅为示例，不能将所有任务都固定为 classification。

用户可以：

- 确认。
- 修改。
- 拒绝。

审批记录必须绑定：

- Plan ID。
- Plan Version。
- 审批时间。
- 审批状态。

计划修改后必须生成新版本，旧版本的审批不能自动生效。

实验提交前，Backend 必须重新验证审批状态。

---

# 9. Optuna 与实验执行

## 9.1 Study 与 Trial

Study 表示一次完整的超参数优化实验。

Trial 表示一个具体的参数组合和对应的训练评估过程。

Optuna 负责：

- 参数生成。
- Trial 管理。
- 结果保存。
- 搜索策略更新。

## 9.2 Worker Pull 架构

第一版已经确定采用：

**Worker Pull + Optuna Shared Storage。**

多个独立 Worker 连接同一个 Optuna Study，各自调用：

```python
study.optimize(objective, n_trials=...)
```

Worker 完成 Trial 后，将结果写入共享 Storage，然后继续申请下一组参数。

不采用第一版自行开发的 ask/tell + Task Queue 架构。

## 9.3 Worker 数量

开发阶段：

- MVP：1 Worker。
- 后续本机测试：2 Workers。
- 4 Workers：仅在资源允许时测试。
- 远程分布式：未来通过矩池云扩展。

每个 Worker 同时执行一个 Trial。

Scikit-learn Estimator 内部 `n_jobs` 默认设置为 1，避免过度订阅 CPU。

## 9.4 全局 Trial 预算

必须特别注意：

Optuna 的 `n_trials` 是每个 Worker 本次调用的 Trial 数量，不是整个 Study 的全局数量。

例如：

如果总预算为 20，不能让 4 个 Worker 各自运行 `n_trials=20`，否则可能执行 80 个 Trial。

多 Worker 阶段必须由 Orchestrator 分配总预算，确保所有 Worker 的配额之和不超过用户批准的总 Trial 数。

第一版单 Worker 直接使用批准的总预算。

多 Worker 阶段可以先采用明确的 Worker 配额分配，并记录已分配、已完成和未完成的 Trial。发生 Worker 故障时，要先核对已执行和正在执行的 Trial，再决定是否补充配额。

## 9.5 Storage

第一阶段：

- 应用元数据库：SQLite。
- Optuna 单 Worker：可以使用本地 SQLite RDBStorage。

多 Worker 阶段：

- Optuna Storage 改为 PostgreSQL。
- 所有 Worker 连接同一数据库。
- 不使用 SQLite 作为多进程 Optuna 并行优化的共享存储。

建议分离：

```text
data/db/app.db
data/db/optuna.db
```

未来 PostgreSQL 独立部署。

## 9.6 Worker 生命周期

Run 状态：

- CREATED
- QUEUED
- RUNNING
- CANCELLING
- COMPLETED
- FAILED
- CANCELLED

Worker 状态：

- STARTING
- IDLE
- BUSY
- STOPPED
- FAILED

Trial 状态由 Optuna 管理。

Orchestrator 需要能够查询 Worker 进程状态，记录日志，并响应取消请求。

第一版可以使用独立 Python 子进程启动 Worker，避免在 Windows 上引入过于复杂的进程管理。

Worker 入口必须兼容 Windows multiprocessing/spawn 环境。

---

# 10. 异步任务与幂等机制

## 10.1 异步提交

`create_optimization_run()` 不等待模型训练结束。

提交后立即返回：

```json
{
  "run_id": "run_001",
  "status": "QUEUED"
}
```

Dashboard 通过 Backend 查询任务状态。

第一版使用 2–5 秒轮询即可，不必过早引入复杂 WebSocket。

## 10.2 幂等

Backend 为每个实验创建请求生成或接收幂等键。

数据库建立唯一约束。

如果相同请求重复提交：

- 不创建新 Run。
- 返回原来的 Run ID。
- 返回当前真实状态。

必须区分：

- Idempotency Key
- Run ID
- Optuna Study Name
- Trial Number

## 10.3 Retry

只对可恢复错误重试。

例如：

- 临时网络错误。
- API 限流。
- 暂时性数据库连接失败。

对于数据集不存在、参数不合法、用户未审批等错误，不应盲目重试。

使用有限重试和指数退避。

## 10.4 Heartbeat

多 Worker 阶段增加 Worker 心跳和失效检测。

不能将一次心跳丢失直接认定为 Worker 崩溃。

需要容忍时间和进一步状态检查。

Optuna RDBStorage 的 Heartbeat 机制需要根据当前使用的 Optuna 版本验证具体能力和适用范围，不能假定它自动解决所有故障恢复问题。

---

# 11. MCP 架构

## 11.1 两个独立 MCP Server

系统有两个 MCP Server：

1. HPO MCP Server：自行开发。
2. MatPool MCP Server：使用矩池云提供的服务。

Agent Runtime 作为 MCP Client，分别连接两个 Server。

```text
Agent Runtime
      |
      +---- HPO MCP Server
      |
      +---- MatPool MCP Server
```

第一版不开发 MCP 聚合代理。

## 11.2 HPO MCP Tools

第一版规划：

| Tool | 功能 |
|---|---|
| inspect_dataset | 检查数据集 |
| list_models | 查询模型 |
| get_model_spec | 查询模型参数空间 |
| validate_experiment_plan | 验证实验计划 |
| create_optimization_run | 创建实验 |
| get_run_status | 查询实验状态 |
| get_run_results | 获取实验结果 |
| cancel_optimization_run | 取消实验 |

`delete_dataset` 第一版可以不开放。

## 11.3 MCP 规范

使用 MCP 官方 Python SDK。

实现：

- tools/list
- tools/call
- JSON Schema 输入定义
- 结构化输出
- 工具错误处理

区分协议错误与业务执行错误。

后端必须执行真正的权限检查，不能只依赖 Tool Description 或 Annotations。

工具名称需要避免冲突，可以采用：

- `hpo.*`
- `matpool.*`

## 11.4 Qwen Tool Calling

MCP 工具定义与 Qwen API 的工具调用格式可能存在差异。

需要通过 Adapter 将 MCP 工具 Schema 转换成 Qwen API 所要求的工具定义。

同时将 LLM 返回的工具调用转换为 MCP `tools/call`。

所有调用参数必须经过后端校验。

不能让 LLM 直接执行工具代码。

---

# 12. MatPool MCP 集成

## 12.1 服务信息

帮我把「矩池云」桌面客户端注册为你的 MCP 服务器，并自己完成安装与验证：
1. 优先用 streamable HTTP 接入，地址是 http://127.0.0.1:9033/mcp
2. 如果你的运行环境不支持 streamable HTTP，就改为注册一个 stdio server：可执行文件 E:\矩池云\matpool.exe，参数 --mcp-stdio
3. 按你自己客户端的配置方式（配置文件、CLI 命令或设置界面均可）完成注册，然后验证连接，并告诉我实际可用的工具数量和它们的名字。
正常情况下应当看到 30 个只读工具。
用户已开启写权限，因此还会多出 40 个写操作工具（租用实例、释放、删除文件、上传下载、迁移环境等），调用这些工具前要先向用户确认。
背景：该服务由矩池云客户端在本机回环地址上提供（http://127.0.0.1:9033/mcp），复用客户端当前的登录状态，只监听本机、不对局域网或公网开放。
写权限默认关闭；是否需要开启由用户在客户端「设置 → AI 助手（MCP）」里决定，你不需要也不应该去修改这个设置。
它在客户端未运行或未登录时连接会失败，遇到这种情况请直接提示用户启动客户端，不要尝试自己拉起它。
注册完成后如果当前会话还看不到工具，重启一次当前会话即可生效。
（备用信息：stdio 完整命令为 E:\矩池云\matpool.exe --mcp-stdio）

## 12.2 第一阶段

先实现可选的只读连接：

- 查询实例。
- 查询资源。
- 查询运行状态。

如果矩池云客户端没有启动或 MCP 服务不可用，不得影响核心 HPO 系统运行。

## 12.3 资源租赁权限

用户已经确认采用：

**Agent 推荐配置，用户确认后才租赁。**

流程：

1. Agent 查询资源。
2. 分析实验需要。
3. 展示推荐配置。
4. 展示价格和预计成本。
5. 用户确认。
6. Backend 验证审批。
7. 调用矩池云写操作工具。

租赁、扣费、释放、删除、文件操作均属于有副作用的操作，必须进行权限校验和用户审批。

SSH 密码不能写入 Prompt 或日志。

禁止向 LLM 暴露无限制终端命令执行工具。

---

# 13. Dashboard 设计

第一版 Dashboard 包含：

## 13.1 数据集页面

- 上传 CSV。
- 数据集列表。
- 行数、列数。
- 缺失值统计。
- 目标变量选择。
- 数据集删除（带确认）。

## 13.2 实验创建页面

- 自然语言输入框。
- 任务类型。
- 数据集选择。
- 目标变量。
- Agent 生成实验计划。
- 模型列表。
- 参数搜索空间。
- Trial 数量。
- 优化指标。
- 用户审批按钮。

## 13.3 实验监控页面

展示：

- Run ID。
- 当前状态。
- Trial 完成数量。
- 总 Trial 数。
- 最优指标。
- 最优参数。
- 运行时间。
- 失败 Trial 数量。

## 13.4 实验结果页面

展示：

- 模型性能对比。
- Trial 参数表。
- 指标变化。
- 最优模型。
- 最优超参数。
- 结果分析。
- 报告导出。

Dashboard 设计应简洁、专业，优先保证交互可用性。

---

# 14. 安全要求

1. CSV 文件上传限制类型和大小。
2. 防止路径穿越。
3. 不允许 LLM 任意访问本地文件。
4. 不允许 LLM 执行任意 Python。
5. CSV 单元格内容一律视为数据，而不是系统指令。
6. Qwen API Key 只放在后端 `.env`。
7. `.env` 必须加入 `.gitignore`。
8. 日志不得记录 API Key、SSH 密码等秘密信息。
9. 所有写操作必须经过后端权限检查。
10. 记录工具调用和实验创建审计日志。
11. 任务取消必须检查 Run 所有权和状态。
12. 实验计划版本必须与用户审批版本一致。

---

# 15. 分阶段开发路线

不要一次开发所有模块。

## Milestone 0：环境检查和项目初始化

检查：

- Windows 版本。
- Python 版本。
- Node.js 是否存在（第一版不强制）。
- Git 是否存在。
- Docker 是否存在。
- PostgreSQL 是否存在。
- E 盘空间。
- CPU 和内存情况。

创建：

`E:\agent-hpo\`

配置 Python 虚拟环境。

创建项目基础结构、README、`.env.example`、`.gitignore`。

验收：

- 项目可以正常启动。
- `/health` 接口正常。
- Dashboard 可以打开。
- 日志写入 E 盘。

## Milestone 1：数据集上传和检查

实现：

- CSV 上传。
- Dataset ID。
- 本地存储。
- 元信息提取。
- 缺失值统计。
- 数据集预览。
- 目标列选择。

验收：

能够上传一个 CSV，并在 Dashboard 查看其基本信息。

## Milestone 2：Model Registry

首先实现：

- LogisticRegression。
- RandomForestClassifier。

完成一个真正的分类任务。

然后扩展：

- 回归。
- 聚类。
- 降维。

验收：

能够从 Model Registry 获取合法模型和参数空间，完成基线训练。

## Milestone 3：Qwen API + ReAct

实现：

- Qwen API Client。
- Tool Calling Adapter。
- Agent Context。
- ReAct Loop。
- MCP Client。
- HPO MCP Server。
- 只读工具。

验收：

用户提出机器学习需求，Agent 能够检查数据集、查询模型并生成实验方案。

## Milestone 4：用户审批 + Optuna

实现：

- Plan ID。
- Plan Version。
- Approval Record。
- Plan Validator。
- Optuna Study。
- Objective Function。
- 单 Worker。

验收：

用户确认实验方案后，系统能够执行真实 Optuna Trial。

## Milestone 5：异步 Orchestrator

实现：

- Run ID。
- 异步提交。
- Worker 子进程。
- Run 状态管理。
- 任务取消。
- 结果持久化。
- Dashboard 状态轮询。

验收：

用户提交实验后不需要等待页面阻塞，可以在 Dashboard 查看后台训练进度。

## Milestone 6：结果分析

实现：

- Trial 历史。
- 最优参数。
- 模型比较。
- 结果报告。
- 实验完成事件。
- Agent 第二轮结果分析。

验收：

实验结束后，Agent 能够解释实验结果，并生成结构化报告。

## Milestone 7：多 Worker

实现：

- PostgreSQL。
- Optuna RDBStorage。
- 多进程 Worker Pull。
- 全局 Trial 预算控制。
- Worker 状态。
- 心跳。
- 有限故障恢复。

验收：

两个 Worker 能够连接同一个 Study，并行执行不同 Trial，最终不超过批准的全局 Trial 数量。

## Milestone 8：MatPool MCP

实现：

- 连接本地矩池云 MCP。
- 动态工具发现。
- 只读资源查询。
- 工具命名空间。
- 用户审批流程。

租赁操作可以放到后续小版本。

验收：

能够查询矩池云资源，但未经用户批准不能租赁或释放实例。

## Milestone 9：测试、文档和打包

完成：

- 单元测试。
- 集成测试。
- 异常测试。
- README。
- 环境安装说明。
- 数据集获取说明。
- 实验复现说明。
- 项目启动脚本。

---

# 16. 分布式性能实验

这是课程期末项目的重要部分。

在多 Worker 阶段设计对照实验：

- 1 Worker。
- 2 Workers。
- 4 Workers（资源允许时）。
- 未来可以使用远程 Worker 扩展。

记录：

- 总耗时。
- Trial Throughput。
- CPU Utilization。
- Memory Usage。
- Average Trial Time。
- Scheduling Overhead。
- Failure Rate。
- Recovery Time。

加速比：

S(p) = T(1) / T(p)

并行效率：

E(p) = S(p) / p

实验必须控制：

- 数据集。
- 模型。
- 搜索空间。
- Trial 数量。
- 数据划分。
- 随机种子。
- 训练配置。

需要特别说明：并行 TPE 搜索可能因为 Trial 完成顺序不同而产生不同搜索轨迹，因此性能实验应重点比较系统执行效率，而不是单纯比较最终模型指标。

考虑到用户笔记本只有 16 GB RAM，本机性能测试以 1、2 Worker 为主。4 Worker 仅在小规模测试中尝试。需要更大规模的并行性能实验时，可以使用矩池云远程计算资源。

---

# 17. 测试要求

每个阶段都必须提供测试。

最低要求：

### Dataset Manager

- 正常 CSV。
- 空 CSV。
- 不支持的文件。
- 缺失值。
- 大文件。
- 路径安全。

### Model Registry

- 正确模型。
- 不存在的模型。
- 非法超参数。
- 任务类型不匹配。

### Agent

- 正常 Tool Calling。
- 无效工具名称。
- 非法参数。
- 最大步数限制。
- API 超时。

### Approval

- 未审批不能提交。
- 修改计划后旧审批失效。
- 重复审批请求不产生异常状态。

### Optuna

- Trial 正常完成。
- Trial 失败。
- Trial 总数不超预算。
- Study 恢复。

### Worker

- 正常执行。
- 进程退出。
- 超时。
- 取消。
- 重复提交。

### Idempotency

相同幂等键重复提交，只能产生一个 Run。

---

# 18. Codex 的实际开发行为要求

1. 先检查现有环境和代码，不要覆盖用户已有文件。
2. 以 `E:\agent-hpo\` 为项目根目录。
3. 先输出环境检查结果。
4. 输出实际项目目录结构。
5. 说明第一阶段技术方案。
6. 然后开始实现 Milestone 0 和 Milestone 1。
7. 每完成一个 Milestone，都要运行测试。
8. 发现错误时主动排查和修复。
9. 提供准确的启动命令。
10. 提供当前阶段的功能清单。
11. 对于无法实现的功能说明具体原因。
12. 不要只创建大量空文件和接口而没有实际功能。
13. 不要一次性安装不必要的大量依赖。
14. 不要在没有验证的情况下宣称功能完成。
15. 不要擅自修改已确认的核心架构决策。
16. 如果遇到真正阻塞开发的问题，说明原因和最小必要处理方式，不要反复询问无关的技术偏好。

开发优先级：

**可运行性 > 核心工作流完整性 > 模块扩展性 > UI 美观度。**

最终目标是让用户能够亲自启动系统、上传数据、提出需求、审批方案、运行模型实验并查看结果。

---

# 19. 最终交付标准

第一版 MVP 至少应具备：

- 可启动的本地 Dashboard。
- CSV 上传。
- 数据集检查。
- Qwen API 调用。
- Agent ReAct Loop。
- MCP 工具调用。
- 用户审批。
- Scikit-learn 模型注册表。
- Optuna 超参数优化。
- 异步 Orchestrator。
- 至少一个本地 Worker。
- 实验进度查询。
- 最终结果报告。
- 完整 README。
- 可复现实验。

后续再完善：

- 四类任务全部支持。
- 多 Worker。
- PostgreSQL。
- 故障恢复。
- MatPool MCP。
- 远程 Worker。
- Cloud Mode。

**本项目的最终原则：先让 Agent 真正完成一次机器学习实验，再让它逐步成为可靠的分布式机器学习实验管理系统。**