# 两人协作与开发交接

本文基于当前 M3 开发验收状态。角色分配为建议，尚未代表队友已接受分工。项目按用户要求托管于公开仓库 [Jeremy917-Hui/agent-hpo](https://github.com/Jeremy917-Hui/agent-hpo)。协作者由仓库所有者邀请。

## 当前基线

已经实现：FastAPI、本地 Dashboard、CSV 上传检查、SQLite 数据集索引、框架适配契约、两个 sklearn 分类模型、离线基线、Optuna 单 Worker 异步 HPO、开发 CLI、Run/Trial 只读查询、基础幂等与协作取消/显式恢复。完整测试共 50 项通过；Iris 的 8 Trial 后台验收已通过。

仍未实现：Qwen、ReAct、MCP、实验计划审批、网页端实验闭环、多 Worker、PostgreSQL 共享存储、跨节点心跳及完整故障恢复、正式性能/可靠性实验、另外三类机器学习任务。PyTorch 只有抽象扩展位置，没有训练能力。

当前多个 Run 是由单 Worker 顺序处理，不是同一 Study 上多个 Worker 并行；请勿把开发验收结果写成分布式实验结论。

## 建议分工

| 负责人 | 主模块 | 验收输出 |
|---|---|---|
| 你（A） | Qwen 客户端、自定义 ReAct、MCP Client/HPO MCP Server、工具 Schema 与结构化错误 | 通过真实 Qwen + MCP 检查数据/查询模型/生成方案；有步骤、工具调用和错误记录，最大步数与超时有效 |
| 你（A） | Plan ID/Version、用户审批、审批后的后端提交、Agent 结果解释 | 未审批不可执行；修改方案使旧审批失效；审批只授权对应配置；实验完成后可生成结果解读 |
| 队友（B） | PostgreSQL/Optuna Shared Storage、多 Worker Pull、全局预算和 Worker 生命周期 | 同一 Study 的多个 Worker 并行执行，失败/重试不绕过预算，保留 Worker/Trial 事件 |
| 队友（B） | 心跳、故障识别、安全重试/恢复、性能导出和故障注入 | 区分超时与失败；未知完成状态先查询/核对；能测成功率、重复 Run、恢复时间和资源利用率 |
| 你（A）牵头，B 评审 | 分类/回归/聚类/降维应用验证、代表性公开数据与默认/优化对比 | 按任务定义评价流程和指标，避免测试集泄漏；不为模型数量延迟单 Worker 主流程或系统实验 |
| 两人共同 | 最小网页闭环、实验设计、报告、海报与答辩 | A 整合论述、Agent/应用实验；B 写系统执行/性能/可靠性部分；两人都复核关键结果 |

不假定队友已经擅长某一技术。如能力/兴趣不同可以交换 A、B，但应保留模块边界和共同验收。前端后续优先让普通用户理解和完成操作：模型名称、适用任务和参数用表格/卡片表达，原始 JSON 放入开发者详情；美化排在核心闭环和课程实验之后。

## 文件边界与先约定的接口

A 主要新增 Agent、MCP、方案和审批模块；目录在真正实现时创建，不生成空壳。B 主要维护 `app/worker.py`、`app/hpo.py`、`app/runs.py` 的执行/存储部分。`app/main.py`、`app/config.py`、依赖文件和现有 Adapter 契约属于共同接口，修改前在对应 Issue/PR 中说明。

第一轮应先冻结以下契约，再并行开发：

1. 实验方案字段：Dataset ID/数据 SHA256、任务类型、目标列、模型、参数空间、指标/方向、全局 Trial 预算、种子/切分策略、资源限制。
2. Plan ID、Version、审批记录和所有权：审批由后端验证，不接受 LLM 自称“已批准”。
3. Run 提交/查询/取消/结果的服务入口；保留现有 Run ID 与幂等键的区分。
4. `run_id`、`worker_id`、`trial_number`、`tool_call_id` 及 UTC 时间/单调耗时的事件格式。
5. 结构化错误的 code、可重试性和副作用状态；超时不代表一定未执行。

B 可以通过现有开发 CLI 独立验证系统侧改动，不必等待真实 LLM。A 可以用可控的执行服务替身验证 Agent/审批，再联合真实后端联调。主分支按里程碑验收；两条功能分支可以并行推进，未完成的能力不得在主分支标为已实现。

多机器不能通过 GitHub 或共享 SQLite 文件交换运行状态。应使用 PostgreSQL 共享 Optuna 状态，并明确 Run/Worker 元数据的服务访问或数据库方案。当前本地单槽不能直接放宽成多 Worker；需要原子预算分配及故障后的配额核对。

## GitHub 工作方式

- `main` 保持可运行；A、B 分别使用 `feat/agent-mcp`、`feat/distributed-worker` 等功能分支，小范围提交。
- 每个任务写清负责人、输入输出和验收条件；通过 Pull Request 合并，由另一位成员评审。
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

报告 PDF（不超过 5 页）、代码 ZIP、项目海报 PDF 以课程平台最新要求为准。两人各自维护本模块章节、实验图表和贡献记录，再统一整合；不等全部开发结束才收集实验依据。
