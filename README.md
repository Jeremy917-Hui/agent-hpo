# ReAct 与 MCP 工具运行时 · 以机器学习调参为例

公开仓库：[Jeremy917-Hui/agent-hpo](https://github.com/Jeremy917-Hui/agent-hpo)。

Windows Local Mode。采用 Python + FastAPI + Jinja2 + 原生 JavaScript，逐阶段实现 Qwen API、自定义 ReAct、MCP、Optuna 与独立 Worker。当前阶段进度以本文件和实际测试为准。

**课程选题 W14-A** 研究 Agent 跨进程调用工具时，如何发现接口、校验参数、验证身份和权限，并在响应超时后安全重试。机器学习调参提供一个会创建后台任务的真实场景。

**当前进度** M0–M3 已验收。公开 Iris 数据已跑通两个 sklearn 分类模型的单 Worker 调参，当前从开发 CLI 提交。Qwen ReAct、MCP 和网页审批是接下来的联调工作。四人分工见 [协作指南](CONTRIBUTING.md)，具体任务见 [GitHub Issues](https://github.com/Jeremy917-Hui/agent-hpo/issues)。

**首轮目标** 用 Iris 跑通上传、自然语言需求、MCP 查数据与模型、计划审批、异步返回 Run ID、后台训练和结果展示。模型沿用逻辑回归与随机森林。更多模型、其他任务类型和多轮自动调参放在闭环验收之后。

## 环境与启动

Python 3.11.4，虚拟环境位于 `E:\agent-hpo\.venv`。现有 Python 无需重装。Node.js、Docker、PostgreSQL 不是 M0/M1 的运行条件。

```powershell
cd E:\agent-hpo
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000>，健康检查 <http://127.0.0.1:8000/health>，接口文档 <http://127.0.0.1:8000/docs>。Ctrl+C 停止服务。也可运行 `powershell -ExecutionPolicy Bypass -File .\scripts\01_start.ps1`。这是仅监听回环地址的单用户本地应用，不作为公网服务部署。

依赖安装与测试（无需激活环境）：

```powershell
cd E:\agent-hpo
$env:TEMP='E:\agent-hpo\tmp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR='E:\agent-hpo\.cache\pip'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

重建环境用 `python -m venv E:\agent-hpo\.venv`，仅在该目录尚不存在时执行。`requirements.txt` 固定直接依赖，`requirements.lock.txt` 保存完成验收时全部依赖版本；精确复现可安装 lock 文件。不要覆盖已有环境。测试产物保存在 `tmp/tests/`，不会自动删除。

当前电脑的旧版 pip 通过既有代理访问 PyPI 时出现 TLS 中断，直连又出现下载超时。首次安装已通过官方 HTTPS 下载并核对 PyPI SHA256 后缓存 wheel，由 pip 离线校验依赖关系；没有关闭证书校验或修改全局代理。如在线安装受阻，当前机器可直接复用缓存：

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --find-links .cache\wheels -r requirements.lock.txt
```

环境实测：Windows 11 家庭版 64 位（10.0.26200），Ryzen 7 5825U / 8 核 16 线程，RAM 15.31 GiB，检查时可用约 3.50 GiB。E 盘初始剩余 77.98 GiB。Git 2.56.0.windows.4、Node.js 24.22.1 可调用（来自 Codex 附带运行时）；npm 不在 PATH。Docker CLI 不在 PATH，存在已停止的 Docker 服务且原服务路径文件未找到，因此不视为可用。PostgreSQL CLI、服务、常见路径及注册表中未发现可用安装。本阶段无需修复或安装它们。SQLite 3.42.0 来自 Python 标准库。

## 模块与路径

- `app/config.py`：从项目 `.env` 和进程环境读取配置，进程环境优先；集中解析路径和资源限制。
- `app/main.py`：应用工厂、生命周期、HTTP 路由、同源写入检查、JSON 日志。
- `app/dataset_api.py`：数据集 HTTP 接口、流式接收及上传大小校验。
- `app/datasets.py`：CSV 检查、不可变入库元信息、SQLite 索引与目标列持久化；不依赖 HTTP，后续 MCP 可复用。
- `app/templates/`、`app/static/`：同源 Dashboard，无 Node 构建流程和外部 CDN。
- `tests/`：里程碑验收；`scripts/01_start.ps1`、`scripts/02_test.ps1`：运行入口。
- `data/datasets/`：原始数据及元信息；`data/db/app.db`：应用元数据库；`data/runs/`：后续实验结果。
- `logs/backend.jsonl`：追加写入的结构化日志；`.cache/pip/`、`tmp/`：依赖缓存和项目临时文件。

默认根目录由源码位置推导，可通过 `.env.example` 中的 `PROJECT_ROOT`、`DATA_DIR`、`LOG_DIR` 调整。没有 `.env` 也可运行；需要配置时自行从示例创建。Qwen key 只应保存在后端 `.env`，该文件和所有运行数据均被 Git 忽略。启动时检查至少 2 GiB 空间，不够则拒绝启动。

## M1 数据集使用与边界

1. 启动后在 Dashboard 选择 `examples/01_分类演示.csv`，点击“上传并检查”。该文件是小规模合成样例，仅用于验证交互，不代表真实业务效果。
2. 查看行列数、原始重复行数量、字段类型、缺失值及前 20 行预览。
3. 选择 `target` 并保存；刷新或重启后选择仍保留。聚类、降维可以清空目标列。
4. 展开“数值摘要与文件信息”查看均值、样本标准差、范围、SHA256 和目标列分布。

默认限制 20 MiB、100000 行、256 列、单字段 1 MiB；列名需非空且唯一、最长 256 字符。支持逗号分隔的 UTF-8（可带 BOM）及 GB18030；不自动猜测分隔符。数据类型是检查阶段的启发式推断，训练时仍需根据任务验证 schema。空白、NA、N/A、NaN、null、none（忽略大小写及两端空白）作为缺失值；数值统计跳过缺失值，非有限数字视为文本。唯一值计数超过 1024 时仅报告下界，避免占用过多内存。原始字节完全保留，预览单元格截取前 500 字符。重复行按原始解析单元格值的 SHA256 比较，非缺失值语义归一化后的去重。

每份数据保存到 `data/datasets/<dataset_id>/original.csv` 和 `metadata.json`。JSON 是上传时快照；可修改的目标列以 `data/db/app.db` 为准，API 查询合并二者。目标列唯一值不超过 50 时扫描并返回前 20 个类别计数，较多时仅返回已有数值摘要。数据集列表显示最近 100 条。

上传采用原始 CSV 请求体，不使用 multipart，避免额外解析依赖；Dashboard 已处理该格式：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/datasets?filename=demo.csv" -H "Content-Type: text/csv" --data-binary "@E:\agent-hpo\examples\01_分类演示.csv"
```

接口：`POST /api/datasets?filename=...`、`GET /api/datasets`、`GET /api/datasets/{id}`、`PATCH /api/datasets/{id}/target`（JSON `{"target_column":"target"}`，清空用 null）。未知 ID 返回 404、超限返回 413、无效 CSV 返回 422。文件名不能包含目录或控制字符，存储路径只使用后端生成的 ID。写入检查浏览器 Origin，未提供公网认证或多用户权限；保持仅监听回环地址。

不提供数据集删除接口。不自动删除上传失败的暂存文件、未完成的入库目录、测试数据或日志；如需清理，应先逐项说明并获得用户批准后移动到回收站。失败数据不进入正常数据集列表；检查 `tmp/upload_*.part` 和日志可定位暂存记录。磁盘阈值也在每次上传前检查。大规模数据、训练时内存预算、结果文件恢复事务将在相应阶段完善。

验收：M0 独立运行 3 项测试通过；M1 合并后 20 项测试通过，覆盖正常上传、缺失值、目标列与重启持久化、GB18030、引号换行、空文件/表头、非法扩展名/路径、行列/字节限制（包括无 Content-Length）、安全预览与数值溢出。HTTPX 0.28.1 在当前 Starlette TestClient 中产生一条迁移弃用提示，测试可运行；当前不为消除此提示增加 httpx2 依赖。

## M2 模型注册表与离线分类基线

当前完成 M2 的分类核心：`LogisticRegression`、`RandomForestClassifier`，可从 Dashboard 查询模型，或用离线命令实际训练。回归、聚类、降维只定义任务类型和能力接口，尚未注册模型；后续逐步补充，优先衔接单 Worker 主流程及分布式/可靠性实验。

```powershell
cd E:\agent-hpo
# 使用 sklearn 自带的公开 Iris 数据，不联网下载，不需要 API key。
.\.venv\Scripts\python.exe scripts\03_baseline.py --demo iris
# 使用已经上传的数据（把 ID 替换为 Dashboard 显示的值）。
.\.venv\Scripts\python.exe scripts\03_baseline.py --dataset-id dataset_046172e94c17407bae99bdd63066e854 --target target
# 指定模型和随机种子，参数必须属于注册表允许范围。
.\.venv\Scripts\python.exe scripts\03_baseline.py --dataset-id dataset_046172e94c17407bae99bdd63066e854 --target target --models LogisticRegression --seed 2021
```

`--demo iris` 每次生成新的 Dataset ID，不覆盖已有数据。`--params-json` 接受按模型名组织的 JSON，例如 `{"LogisticRegression":{"C":2.0}}`。参数值不作为代码执行。命令支持 `--help`。上述第二条 ID 是 M1 验收时保留的合成样例，本机若不存在该 ID，请选择自己的 ID 或使用 `--demo iris`。

这是开发验收用的同步命令行程序，不是 Dashboard 实验提交、Run API 或 Optuna Trial。所有产品实验提交仍计划采用异步 Run ID；M3 后续 Worker 将复用适配器契约。此阶段不开放 HTTP 训练接口。

新增模块职责：

| 模块 | 职责 |
|---|---|
| `app/models/base.py` | 四种任务枚举、参数/资源规范、数据分区、评估结果、ModelAdapter 抽象接口及可选能力协议 |
| `app/models/registry.py` | 模型白名单、参数空间、默认值、任务/指标/资源校验、可信适配器工厂 |
| `app/models/sklearn_adapter.py` | 分类分层划分、预处理 Pipeline、两个 Estimator、训练、预测、评价及产物保存 |
| `app/models/pytorch_adapter.py` | PyTorchTrainingConfig 与不可实例化的抽象扩展位置；没有 torch 导入或训练代码 |
| `app/model_api.py` | 模型发现、详情、纯配置校验；无训练副作用 |
| `app/baseline.py` | 框架无关的离线基线协调、原始数据 SHA256 校验和结果落盘 |
| `app/telemetry.py` | 追加事件、进程 CPU/RSS 采样；后续可复用于 Worker/Trial/Tool Runtime |
| `scripts/03_baseline.py` | 命令行入口及可选 Iris 演示数据导入 |

API：`GET /api/models`、`GET /api/models?task_type=classification`、`GET /api/models/{model_name}`、`POST /api/models/{model_name}/validate`。校验请求示例：`{"task_type":"classification","parameters":{"C":1.0},"metrics":["accuracy","roc_auc"]}`。错误同时提供可读 `detail` 和结构化 `error.code/message/details`。未实现的任务列表为空，不伪造支持能力。

分类流程：先固定种子分层划分训练、验证、测试集，默认比例 60%/20%/20%；数值/类别类型仅从训练分区推断。数值中位数填补、逻辑回归标准化、类别众数填补及 OneHotEncoder 均在训练分区 fit。类别编码使用稀疏输出、`handle_unknown=ignore`、每列最多 64 个编码类别（包含可能的合并类别），控制展开规模。验证/测试集只做 transform 和评价，目标列不进入特征。

当前限制：分类目标不能缺失，需 2–50 个类别且每类至少 5 条样本；每个分区还必须覆盖全部类别。类别名保留为原始文本；目标任务须由用户明确指定，不按数值类型自动判定。验证/测试数值类型与训练推断不一致时明确报错。默认随机分层切分不适用于时间序列、分组实体或重复实体泄漏，正式数据需要对应切分策略。当前内存限制是输入表示估算，不能保证约束树模型的实际峰值内存；CPU 线程默认固定 1（包含 BLAS/OpenMP 限制），不运行 GPU。

指标：Accuracy、macro Precision/Recall/F1、ROC-AUC。二分类 ROC-AUC 正类为排序后的 `classes_[1]`，在结果说明中记录；多分类使用 OvR macro。评价分区缺少类别时 ROC-AUC 为 null 并返回原因，不伪装成 0。默认参数中的 LogisticRegression `max_iter=500` 是项目为降低不收敛风险设置的值，其他注册参数默认值见 API；不宣称与 sklearn 所有原厂默认参数完全相同。

产物保存在 `data/runs/baseline_<uuid>/`：

- `manifest.json`：数据 SHA256、目标列、模型参数、特征 schema、种子/资源配置、环境版本。
- `split_indices.json`：三个分区的原始数据行索引（去掉表头、忽略空物理行后从 0 开始）。
- `results.json`、`metrics.csv`：验证/测试指标、训练及评价耗时、混淆矩阵与采样到的进程峰值 RSS。
- `events.jsonl`：开始/结束、模型训练、分区评价及失败事件，附 UTC 时间、单调计时和本地执行 ID。
- `resources.jsonl`：约每 0.1 秒采样当前进程 CPU/RSS、主机 CPU/可用内存；短训练可能只有首尾样本，首个 CPU 样本为基准，采样峰值不是精确峰值，GPU 指标为 null。
- 每个模型独立目录：由 Adapter 决定格式的模型产物（当前 `model.joblib`）和带分区/行号的 `predictions.csv`。
- 失败时保留 `failure.json`、已有事件及产物；不自动删除。模型文件只保存可信后端创建的模型，禁止加载用户上传的 pickle/joblib。

同一基线中两个模型使用相同划分，验证集用于比较；此阶段没有搜索或根据测试集选择最优配置。未来 Optuna Objective 必须只使用开发数据/交叉验证，不将这里的测试指标接入调参。基线结果不是“优化后结果”，也不是完整 Agent 实验报告。

阶段验证：Iris 150 条、三分类、种子 2021，90/30/30 划分；两个模型测试 Accuracy 均为 0.90。合并测试 41 项通过，覆盖参数/任务/资源拒绝、数据划分互斥、训练分区预处理、未知类别、ROC-AUC 不可计算场景、模型重新加载、结果复现、数据完整性拒绝和故障产物保留。这些仅是开发验收，不替代正式性能或可靠性实验。

依赖新增 sklearn 1.9.1、NumPy 2.4.6、psutil 7.2.2（记录资源采样），以及 sklearn 必需的 SciPy/joblib 等传递依赖。无需 Pandas、PyTorch、Optuna 或 MCP 来运行当前分类基线。参考：[sklearn Pipeline 与数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html)、[Iris 数据说明](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)。

## M3 Optuna 与单 Worker 后台执行

M3 已接入 Optuna 5.1.0。开发命令行提交立即返回 Run ID；独立持久 Worker 取出排队 Run，使用 `Study.optimize(n_jobs=1)` 执行 Trial。不是 ask/tell + Trial 队列。应用状态在 `data/db/app.db`，Optuna Study/Trial 在独立的 `data/db/optuna.db`。本地互斥槽保证同一项目配置只有一个训练 Worker；多个 Run 顺序执行。

由于审批尚未实现，**只允许用户显式使用开发 CLI 提交**。没有 HTTP 创建、取消、恢复实验接口，Agent 也尚无这些写工具；当前 CLI 不代表已经实现审批。为满足 M3 异步执行，提前加入最小 Run 元数据/队列；完整计划版本与用户审批、权限及 Orchestrator 仍在 M5。

```powershell
cd E:\agent-hpo
# 此 ID 来自 M2 Iris 演示；若不存在，先运行 scripts/03_baseline.py --demo iris 获取新 ID。
.\.venv\Scripts\python.exe scripts\04_hpo.py submit --dataset-id dataset_ab847cf9ace74acd9315f9e67ade09c5 --trials 8 --metric f1_macro --idempotency-key my-first-hpo
# 将 RUN_ID 替换成提交返回的 run_...。
.\.venv\Scripts\python.exe scripts\04_hpo.py status RUN_ID
.\.venv\Scripts\python.exe scripts\04_hpo.py trials RUN_ID
.\.venv\Scripts\python.exe scripts\04_hpo.py results RUN_ID
.\.venv\Scripts\python.exe scripts\04_hpo.py list
```

可选参数：`--target`、`--models LogisticRegression RandomForestClassifier`、`--metric accuracy|precision_macro|recall_macro|f1_macro|roc_auc`、`--seed 2021`、`--timeout 600`、`--fixed-params-json`。固定参数的 JSON 结构与 M2 参数映射相同。模型/参数/指标/资源及目标列在提交时验证；冻结数据 SHA256、模型定义和执行配置。M3 仅支持分类和 maximize 指标，不宣称已支持其他任务。

默认 Trial 预算 10，当前限 1–200，且至少每个选定模型一次。每个模型的项目默认配置（叠加显式固定参数）先作为 baseline Trial 入队，**包含在总 Trial 预算内**。后续 Trial 选择模型及模型自己的参数空间，使用名字前缀避免不同模型参数冲突。TPE 启动阶段为 `min(5,max(模型数,总预算//3))`，小预算也能进入自适应搜索；此设置在 Study 和结果中记录。

每个 Trial 创建独立 Pipeline，只 fit 训练分区并评价验证分区。Objective 接口不接收测试分区。搜索结束并根据验证指标选定模型/参数后，分别重新训练该模型的默认配置与最优配置，才评价测试集。这额外的两次最终训练不计为搜索 Trial，事件中标记 `final_default/final_optimized`；不要把 Trial 数当作全部模型 fit 次数。优化后测试指标不保证提升。

失败的模型训练记为 Optuna FAIL，消耗一个 Trial 配额；不会以重试的名义增加预算。全部失败则 Run=FAILED，没有最优模型和分数。没有为非迭代的 sklearn fit 伪造中间指标或 pruning，当前使用 NopPruner。

```powershell
# 协作式取消：当前 Trial 仍会完成并记录，然后停止取下一个 Trial。
.\.venv\Scripts\python.exe scripts\04_hpo.py cancel RUN_ID
# 显式恢复 CANCELLED/FAILED，复用同一 Study 和剩余 Trial 配额。
.\.venv\Scripts\python.exe scripts\04_hpo.py resume RUN_ID
# Worker 管理：后台进程空闲时会保留，停止命令在当前 Run 完成后生效。
.\.venv\Scripts\python.exe scripts\04_hpo.py worker-status
.\.venv\Scripts\python.exe scripts\04_hpo.py worker-stop
.\.venv\Scripts\python.exe scripts\04_hpo.py worker-start
```

Run 状态：QUEUED → RUNNING → COMPLETED/FAILED；取消为 CANCELLING → CANCELLED。尚未运行的 QUEUED Run 可立即取消。重复取消不会修改已完成的结果。恢复不增加冻结预算、不重复 baseline，也不清除历史 Trial；原来的参数与指标保持不变。TPE sampler 重建后的后续采样序列不承诺与未中断运行完全一致。

`--timeout` 是每次 Study.optimize 的搜索软时限，不能中断已经进入的单次模型 fit，也不包括最终两次训练；显式 resume 后会获得新的搜索软时限。它不是硬 CPU/墙钟/计费限制。取消同样不能硬中断 fit，恢复也不会自动发生。Worker 意外退出可通过本机 PID+进程创建时间识别，Run 标为 FAILED；若遗留 Optuna RUNNING Trial，当前拒绝自动重放并返回 `TRIAL_RECOVERY_REQUIRED`，避免结果不明时重复执行。远程心跳、硬超时、重试策略与完整故障恢复尚未实现。

幂等键与 Run ID、Study Name、Trial Number 分开管理：同键同配置返回原 Run 及真实状态；同键不同配置返回 `IDEMPOTENCY_CONFLICT`。每个项目最多排队/执行 20 个 Run。M3 没有基于用户身份的所有权模型，是仅监听回环地址的本地单用户开发模式。

只读 HTTP：`GET /api/runs`、`GET /api/runs/{run_id}`、`GET /api/runs/{run_id}/trials`、`GET /api/runs/{run_id}/results`。结果未就绪返回 409；不存在的 Run 返回 404。查询进度直接读取应用元数据，不需要反复启动 LLM 或在 HTTP 中训练。Dashboard 的普通用户工作流和结果可视化尚未接入，本轮仅更新阶段说明。

新增模块：

| 模块 | 职责 |
|---|---|
| `app/runs.py` | 冻结计划、Run/Trial 元数据、幂等键、本地 Worker 单槽、队列及状态变更 |
| `app/hpo.py` | 参数采样、只接收训练/验证数据的 Objective、Study 恢复、Trial 配额、最终模型/测试评价 |
| `app/worker.py` | Windows 隐藏子进程启动、单槽注册、顺序取 Run 并调用 HPO 引擎 |
| `app/run_api.py` | Run、Trial、结果的只读查询 |
| `scripts/04_hpo.py` | 用户显式提交、查询、取消、恢复和 Worker 管理的开发命令 |
| `tests/test_04_hpo.py` | 真实 Optuna/SQLite 与真实子进程集成测试、可控模型训练故障注入 |

每个 `data/runs/run_<uuid>/` 保存 `plan.json`、`split_indices.json`、追加 `events.jsonl`、按执行尝试隔离的资源采样和模型目录。完成目录包含 `results.json`、`trials.csv`、default/optimized 两份模型；结果相对路径保存到应用数据库中。失败原因在 Run 的结构化 error 与事件日志中保留，可能存在未完成产物，不自动删除。子进程输出在 `logs/worker_launcher_*.log`。

Trial CSV/只读 API 提供状态、模型、参数、指标、Worker ID、开始/结束时间和耗时；采样记录训练 fit 与验证评价耗时，Optuna Trial 总耗时包含存储/调度开销。Optuna 的本机时间在导出时转为 UTC，事件用 UTC 和单调计时。资源采样仍有 M2 的频率/峰值限制，尚非正式分布式性能测量。

验收：全量 50 项测试通过，包括真实后台进程串行执行两个 Run、只读 HTTP 边界、默认 Trial 预算、测试集隔离、幂等冲突、训练中协作取消、取消后恢复及预算不超限、部分/全部 Trial 失败。Iris 双模型实测 8 个 Trial 成功完成；这只是功能验收，不能当作多 Worker 加速或可靠性实验结论。

参考：[Optuna Study.optimize](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html)、[RDBStorage](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html)。Optuna SQLite 当前仅由一个优化 Worker 写入；M7 多 Worker 必须迁移 PostgreSQL。

## 框架扩展契约

Registry 的模型规范包含 `model_name`、`framework`、`task_type`、`adapter`、`parameter_space`、`supported_metrics`、`execution_requirements`。框架适配器只由后端可信注册表选择，禁止接受用户或 LLM 提供的任意导入路径。

配置验证和参数空间由 Registry/ParameterSpec 统一管理；ModelAdapter 负责框架相关的数据准备、模型创建、训练、评估及保存。`predict`、`predict_proba`、`fit_predict`、`transform` 是分别声明的能力。监督学习上下文包含训练/验证数据与目标列；聚类上下文允许没有 y，且可没有独立验证/测试分区；聚类评估要处理单簇/全噪声；降维使用解释方差、重建误差或已批准的下游评估，包含维度预算。这些无监督流程还未实现。

Worker 只接收经验证的 Run 配置，解析 Registry，再调用适配器执行 Trial；Orchestrator 仅管理生命周期和全局预算，Optuna 负责采样和 Trial 状态。统一结果包含指标、产物相对路径和资源信息。预处理在训练分区上 fit，测试集不参与调参。

未来 PyTorch 配置分成模型参数、训练参数和执行要求：`device=cpu|cuda|auto`、`batch_size`、`epochs`、`learning_rate`、`optimizer`、`loss`、随机种子、CPU/GPU 数量及内存需求。每个 Optuna Trial 独立初始化模型/优化器/随机状态，使用独立输出目录；指标回调支持按 epoch 上报和 pruning，checkpoint 使用 Trial 范围内的路径并明确 best/last 策略。硬件不满足要求时在调度前拒绝，不默默占用 GPU。PyTorch 作为单独可选依赖，仅由该适配器延迟导入，不进入 sklearn 核心依赖。

## 路线与当前边界

实施顺序为 M0 初始化 → M1 数据集 → M2 Registry → M3 Optuna/Worker → M4 Qwen/ReAct/MCP → M5 审批/Orchestrator → M6 结果 → M7 PostgreSQL/多 Worker → M8 MatPool。

当前 M0、M1、M2 分类核心及 M3 单 Worker 后台 HPO 已实现。M2 的其他三类任务以及 M4–M8 尚未实现；没有 LLM 调用、MCP、审批或云资源操作。单 Worker 使用本地 SQLite Optuna Storage；多 Worker 才引入 PostgreSQL，并坚持 Worker Pull + Shared Storage。产品实验提交将绑定审批并异步返回 Run ID，当前仅有开发 CLI 提交。MatPool 写操作必须单独获得用户明确确认。

Qwen 通过阿里云商用 API Key 调用。后续性能实验可使用矩池云 CPU/GPU 资源。本地 sklearn 开发先用 CPU，资源租赁及收费前需展示具体配置和预计费用并取得明确确认；GPU 仅在任务确实需要时考虑。PyTorch、MatPool 写操作属于后续扩展，不为追求模型数量或界面效果延迟单 Worker 主流程、共享存储及可靠性实验。

版本核验来源：[FastAPI PyPI](https://pypi.org/project/fastapi/)、[MCP 官方 SDK](https://github.com/modelcontextprotocol/python-sdk)、[Optuna PyPI](https://pypi.org/project/optuna/)、[Scikit-learn 安装文档](https://scikit-learn.org/stable/install.html)。当前 sklearn 1.9.1、Optuna 5.1.0 已安装；后续候选 MCP 2.3.0 尚未安装，接入时重新验证兼容性。

## 课程报告预定证据

1. **工具调用与权限。** 用 Iris 的正常调用轨迹和调用序列图说明服务入口、MCP 工具发现、schema 校验、鉴权和审批，配上无效请求的处理结果。
2. **故障与重试。** 注入“Run 已写入、响应延迟到客户端超时”的故障。同键同方案重试应取回原 Run ID，数据库和 Worker 事件应能核对创建数量及执行轨迹。同键改方案应返回冲突。
3. **性能。** 对同一只读工具比较直接调用与 MCP 调用，统计延迟和成功率，单独记录 Qwen 推理耗时。多 Worker 阶段用固定 Trial 配置和中等规模的公开分类数据测吞吐与加速比，计划规模为 1/2/4/8，实际范围和报告结论由预实验及资源决定。

以上是计划验收项。Iris 单 Worker 是现有功能基线，正式性能和故障实验尚未运行。
