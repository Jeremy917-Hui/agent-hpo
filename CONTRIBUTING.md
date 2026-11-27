# 四人协作与开发交接

当前基线为 M3 单 Worker HPO。仓库所有者 `Jeremy917-Hui` 担任总工。B、C、D 为角色代称，由总工与队友确认人选后在 GitHub 上分配。开发进度见 [GitHub Issues](https://github.com/Jeremy917-Hui/agent-hpo/issues)。

## 当前基线

已经实现：FastAPI、本地 Dashboard、CSV 上传检查、SQLite 数据集索引、框架适配契约、两个 sklearn 分类模型、离线基线、Optuna 单 Worker 异步 HPO、开发 CLI、Run/Trial 只读查询、基础幂等与协作取消/显式恢复。完整测试共 50 项通过；Iris 的 8 Trial 后台验收已通过。

仍未实现：Qwen、ReAct、MCP、实验计划审批、网页端实验闭环、多 Worker、PostgreSQL 共享存储、跨节点心跳及完整故障恢复、正式性能/可靠性实验、另外三类机器学习任务。PyTorch 只有抽象扩展位置，没有训练能力。

当前多个 Run 是由单 Worker 顺序处理，不是同一 Study 上多个 Worker 并行；请勿把开发验收结果写成分布式实验结论。

## 四人分工与验收

| 负责人 | 开发职责 | 交付和验收 | 任务 |
|---|---|---|---|
| **A 总工** | 决定接口和技术取舍，负责计划版本、审批、Run 编排、结果接入、PR 评审与合并 | 接口样例、审批测试、端到端验收。未审批不得执行，修改计划使旧审批失效，重复提交只产生一个 Run。统筹报告和答辩 | [#1](https://github.com/Jeremy917-Hui/agent-hpo/issues/1) |
| **B Agent 与 MCP** | Qwen 客户端、自定义 ReAct、MCP Server/Client、工具校验、超时和重试 | 用真实 Qwen + MCP 生成实验计划，产出调用轨迹、状态机或调用序列图、错误测试以及报告的方法章节 | [#2](https://github.com/Jeremy917-Hui/agent-hpo/issues/2)、[#3](https://github.com/Jeremy917-Hui/agent-hpo/issues/3) |
| **C 分布式执行** | PostgreSQL 共享 Study、Worker Pull、全局 Trial 配额、心跳与恢复 | 多 Worker 并发完成同一 Study，预算可核对。提交部署手册、失联和恢复测试、Worker 事件及报告的系统设计章节 | [#4](https://github.com/Jeremy917-Hui/agent-hpo/issues/4)、[#5](https://github.com/Jeremy917-Hui/agent-hpo/issues/5) |
| **D 应用验证与实验** | 公开数据、默认参数与 HPO 对比、最小演示、可复现实验 | 先验证分类闭环，逐步补齐回归、聚类、降维。组织 Agent/MCP 对照、Worker 扩展和故障注入实验，提交复现步骤、数据、图表与分析 | [#6](https://github.com/Jeremy917-Hui/agent-hpo/issues/6)、[#7](https://github.com/Jeremy917-Hui/agent-hpo/issues/7) |

四人各自提交代码或实验、可复现记录和对应报告素材。总工确定优先级和合并顺序，统一评审 PR、决定是否合并，并验收整条链路。演示界面优先保证用户能看懂计划、完成审批和查看结果。

## 文件边界与先约定的接口

A 负责方案、审批及 Run 服务入口。B 负责 Agent 与工具实现。C 维护 `app/worker.py`、`app/hpo.py` 和 `app/runs.py` 的执行与存储部分。D 负责应用验证脚本、数据与指标配置、最小演示和实验材料。`app/main.py`、`app/config.py`、依赖文件和 Adapter 契约由总工协调。D 扩展模型或任务类型时，由 C 一起检查对 HPO 引擎的影响。

第一轮应先冻结以下契约，再并行开发：

1. 实验方案字段：Dataset ID/数据 SHA256、任务类型、目标列、模型、参数空间、指标/方向、全局 Trial 预算、种子/切分策略、资源限制。
2. Plan ID、Version、审批记录和所有权：审批由后端验证，不接受 LLM 自称“已批准”。
3. Run 提交/查询/取消/结果的服务入口；保留现有 Run ID 与幂等键的区分。
4. `run_id`、`worker_id`、`trial_number`、`tool_call_id` 及 UTC 时间/单调耗时的事件格式。
5. 结构化错误的 code、可重试性和副作用状态；超时不代表一定未执行。

B、C、D 先用接口样例和现有 M3 CLI 并行开发。联调按以下顺序推进。

1. **接口和单 Worker 闭环。** A 确认接口，B 交付规划与只读工具，D 用公开分类数据验收从需求、工具调用、审批到运行和结果的完整流程。
2. **共享存储和可靠性。** C 交付同一 Study 的共享存储、配额与恢复机制，B 交付可测量的超时和重试。D 与 B、C 核对数据记录并开展实验。
3. **课程证据与交付。** 先复现 1/2/4 Worker、工具对照与失败案例，再按资源测 8 Worker 和跨节点。四类应用、报告图表和答辩材料按实际运行结果汇总，16 Worker 由预实验和预算决定。PyTorch 和 MatPool 写操作列为扩展。

多机器不能通过 GitHub 或共享 SQLite 文件交换运行状态。应使用 PostgreSQL 共享 Optuna 状态，并明确 Run/Worker 元数据的服务访问或数据库方案。当前本地单槽不能直接放宽成多 Worker；需要原子预算分配及故障后的配额核对。

## GitHub 工作方式

- `main` 保持可运行。按 Issue 开分支，例如 `feat/plan-approval`、`feat/agent-mcp`、`feat/distributed-worker`、`feat/application-validation`，分小批提交。
- PR 作者关联对应 Issue，写明改动、复现命令和测试结果，并解释自己提交的实现。总工评审并合并，Codex 可协助检查差异、运行测试和整理评审意见。合并前由总工确认验收条件是否满足。
- 每个 PR 运行现有测试和对应新增测试，并写清结果与未完成项。将来 CI 不应默认调用收费 Qwen API 或租赁云资源；真实服务测试采用显式配置和受控预算。
- 大型接口变动先约定，再改实现，避免两个人分别定义一份 Plan、Run 状态或模型规范。
- 报告分工应依据真实提交、实验执行和分析记录，不把全部团队成果记为某一个人的工作。

仓库同步内容包括：`app/`、`scripts/`、`tests/`、`examples/`、README、本文、依赖文件、`.gitignore`、`.gitattributes`、`.env.example` 以及两份项目需求说明。每次提交前检查暂存内容；Git 提交记录用于保留开发历史，本地运行数据不参与同步。

不上传：`.venv/`、`.cache/`、`.pytest_cache/`、`tmp/`、`data/`、`logs/`、`.env`、密钥、数据库、模型产物和原始用户数据。忽略 Git 不等于删除本地文件。需要共享正式实验依据时，另行整理脱敏的汇总 CSV、图和配置清单；不要直接同步运行中的数据库。当前全部 `data/`、`logs/` 均被忽略，因此克隆者不会拥有本机已有 Dataset ID/Run ID。

## 队友首次启动（Windows）

在自己的克隆目录创建独立环境，使用 Python 3.11。下面的操作仅适用于尚未创建 `.venv` 的新克隆目录，不覆盖已有环境。无需全局安装 sklearn/Optuna，也不需要 Node 构建、PyTorch、Docker 或 PostgreSQL 来复现当前 M3。

```powershell
# 先进入自己的 agent-hpo 克隆目录。
New-Item -ItemType Directory -Force -Path tmp,.cache/pip | Out-Null
$projectPath=(Get-Location).Path
$env:TEMP=Join-Path $projectPath 'tmp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR=Join-Path $projectPath '.cache/pip'
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

不用创建 `.env` 也能运行，默认根目录由源码位置推导。若创建 `.env`，应调整 `.env.example` 中的 `PROJECT_ROOT` 为自己的目录；不要保留别人的盘符和 API key。当前 Windows PowerShell 脚本可直接使用；未来在 Linux 云端用 `.venv/bin/python` 执行同一 Python 模块，Linux 全链路尚未实测。

再开一个终端，生成本机自己的 Iris 数据：

```powershell
.\.venv\Scripts\python.exe scripts/03_baseline.py --demo iris
# 从上一步输出复制自己的 Dataset ID，不使用别人机器的 ID。
.\.venv\Scripts\python.exe scripts/04_hpo.py submit --dataset-id YOUR_DATASET_ID --trials 8 --metric f1_macro
.\.venv\Scripts\python.exe scripts/04_hpo.py status YOUR_RUN_ID
```

## 实验与提交物

共同维护三条证据线：真实机器学习任务完成、分布式性能、可靠性与失败语义。正式 Worker 规模计划 1/2/4/8，按预实验及预算决定是否增加 16；区分固定工作量、随资源增加工作量、固定总 Worker 数的跨节点对照，控制模型/数据/Trial 配置和硬件资源。

开发测试、模拟故障和真实云端实验分别标注，保留重复运行和原始测量记录。API/工具耗时、业务处理耗时、训练耗时分别统计。Agent 结果分析不反复轮询收费 LLM；租赁、释放、扣费必须经用户明确确认。

报告 PDF（不超过 5 页）、代码 ZIP、项目海报 PDF 以课程平台最新要求为准。四人从各自任务开始积累章节、实验图表和贡献记录，交由总工统一整合，每人复核自己的实验和结论。
