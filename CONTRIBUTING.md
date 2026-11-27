# 四人协作与开发交接

当前基线为 M3 单 Worker HPO。仓库所有者 `Jeremy917-Hui` 担任总工。B、C、D 为角色代称，由总工与队友确认人选后在 GitHub 上分配。开发进度见 [GitHub Issues](https://github.com/Jeremy917-Hui/agent-hpo/issues)。

## 课程目标与首轮范围

W14-A 关注 Agent Loop、ReAct、MCP 和 Tool Runtime。首轮使用 Iris、逻辑回归、随机森林和一个后台 Worker，跑通 Qwen 规划、MCP 查询、用户审批、异步提交与结果展示。报告准备一张调用序列图，重点验证跨进程调用的 schema、服务和工具发现、身份与权限、幂等、超时及重试。模型和任务类型扩展、多轮自动调参放在首轮验收之后。

## 当前基线

已经实现：FastAPI、本地 Dashboard、CSV 上传检查、SQLite 数据集索引、框架适配契约、两个 sklearn 分类模型、离线基线、Optuna 单 Worker 异步 HPO、开发 CLI、Run/Trial 只读查询、基础幂等与协作取消/显式恢复。完整测试共 50 项通过；Iris 的 8 Trial 后台验收已通过。

仍未实现：Qwen、ReAct、MCP、实验计划审批、网页端实验闭环、多 Worker、PostgreSQL 共享存储、跨节点心跳及完整故障恢复、正式性能/可靠性实验、另外三类机器学习任务。PyTorch 只有抽象扩展位置，没有训练能力。

当前多个 Run 是由单 Worker 顺序处理，不是同一 Study 上多个 Worker 并行；请勿把开发验收结果写成分布式实验结论。

## 四人分工与验收

| 负责人 | 开发职责 | 交付和验收 | 任务 |
|---|---|---|---|
| **A 总工** | 接口、鉴权与审批、端到端联调、演示、正式对照实验和报告 | 统一数据、配置与验收标准，集中运行正式实验并解释结果。B、C、D 提供脚本、事件和各自的模块测试 | [#1](https://github.com/Jeremy917-Hui/agent-hpo/issues/1)、[#6](https://github.com/Jeremy917-Hui/agent-hpo/issues/6)、[#7](https://github.com/Jeremy917-Hui/agent-hpo/issues/7) |
| **B Agent 与 MCP** | Qwen API、自定义 ReAct、MCP Server/Client、工具发现、schema 与结构化结果 | 用公开 Iris 交付一次真实规划调用轨迹、调用序列图和对应测试，给 D 提供可独立调用的工具入口 | [#2](https://github.com/Jeremy917-Hui/agent-hpo/issues/2) |
| **C Worker 与执行** | Worker、Trial 状态与执行记录，后续负责共享 Study、预算、心跳和恢复 | 先复现单 Worker，验证状态、结果和耗时能供联调。闭环验收后再验证多个 Worker 并发执行同一 Study | [#4](https://github.com/Jeremy917-Hui/agent-hpo/issues/4)、[#5](https://github.com/Jeremy917-Hui/agent-hpo/issues/5) |
| **D Tool Runtime 与可靠性** | 调用超时、错误和重试策略、调用事件、故障注入 | 与 A 对齐幂等和审批语义、与 B 对接 MCP 调用入口。交付可复现的故障注入和调用测量脚本，供 A 统一跑正式实验 | [#3](https://github.com/Jeremy917-Hui/agent-hpo/issues/3) |

四人各自提交代码、可复现测试和对应报告素材。总工确定接口、优先级和集成验收标准，并统一复现、整理正式实验结果。仓库维护方负责 PR 检查和合并。演示界面优先保证用户能看懂计划、完成审批和查看结果。

## 文件边界与先约定的接口

A 负责方案、鉴权审批、Run 服务入口及统一验收脚本。B 负责 ReAct 和 MCP 协议层，定义工具 schema 和结构化返回。C 维护 `app/worker.py`、`app/hpo.py` 和 `app/runs.py` 的执行与存储部分。D 负责调用层的超时、重试、结构化错误、事件与故障注入。A、B、D 先约定调用入口、请求关联 ID 及错误字段。C 与 A 对齐 Run 状态和 Worker 事件。

第一轮应先冻结以下契约，再并行开发：

1. 实验方案字段：Dataset ID/数据 SHA256、任务类型、目标列、模型、参数空间、指标/方向、全局 Trial 预算、种子/切分策略、资源限制。
2. Plan ID、版本、实验所有权和审批记录。后端先校验调用者身份与权限，再校验具体方案的审批。
3. Run 提交/查询/取消/结果的服务入口；保留现有 Run ID 与幂等键的区分。
4. `run_id`、`worker_id`、`trial_number`、`tool_call_id` 及 UTC 时间/单调耗时的事件格式。
5. MCP 服务地址配置和工具发现方式，以及结构化错误的 code、可重试性和执行结果是否已确定。响应超时后通过查询或同键重试确认状态。

B、C、D 先用接口样例和现有 M3 CLI 并行开发。联调按以下顺序推进。

1. **单 Worker 闭环。** A 定接口与审批，B 交付 Qwen 规划和只读 MCP 工具，C 独立复现 Worker 及结果记录，D 交付基础调用事件。A 用 Iris 验收全流程并保存 ReAct 调用轨迹。
2. **工具可靠性。** D 交付故障注入、有限重试和调用测量，B 提供真实工具入口，A 统一跑 schema、发现、鉴权、审批和“Run 写入后响应超时”的验收。保留调用序列图、相同幂等键的请求和对应 Run 记录。
3. **并行性能。** 首轮验收后，C 先让两个 Worker 共享 Study 并守住全局预算。A 用 C 提供的入口固定工作量测 1/2/4 Worker，按预实验和资源评估 8 Worker 及跨节点。

多机器不能通过 GitHub 或共享 SQLite 文件交换运行状态。应使用 PostgreSQL 共享 Optuna 状态，并明确 Run/Worker 元数据的服务访问或数据库方案。当前本地单槽不能直接放宽成多 Worker；需要原子预算分配及故障后的配额核对。

## GitHub 工作方式

- `main` 保持可运行。按 Issue 开分支，例如 `feat/plan-approval`、`feat/agent-mcp`、`feat/distributed-worker`、`feat/application-validation`，分小批提交。
- PR 作者关联对应 Issue，写明改动、复现命令和测试结果，并解释自己提交的实现。仓库维护方检查差异、测试结果和验收条件后管理、合并 PR。总工协调接口变更和端到端验收。
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

报告保留三组证据。第一组用 Iris 证明真实任务跑通并呈现调用轨迹。第二组记录 schema、发现、鉴权、审批和故障注入的预期与实际结果，重点核对重试前后的 Run ID、数据库创建次数以及 Worker 事件。第三组比较同一只读工具的直接调用与 MCP 调用，并用中等规模公开分类数据和固定的 Trial 配置测 Worker 吞吐与加速比。

开发测试、模拟故障和真实云端实验分别标注，保留重复运行和原始测量记录。API/工具耗时、业务处理耗时、训练耗时分别统计。Agent 结果分析不反复轮询收费 LLM；租赁、释放、扣费必须经用户明确确认。

报告 PDF（不超过 5 页）、代码 ZIP、项目海报 PDF 以课程平台最新要求为准。四人从各自任务开始积累章节、实验图表和贡献记录，交由总工统一整合，每人复核自己的实验和结论。
