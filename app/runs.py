"""Local single-worker run metadata. No estimator or Optuna imports here.

Only the explicit development CLI can submit. HTTP exposes read-only state until
M5 adds approval/version checks. Historical runs, workers and trial records stay.
"""

import hashlib
import json
import re
from uuid import uuid4

import psutil
from pydantic import Field, StrictInt

from app.baseline import write_json
from app.datasets import DatasetStore
from app.models.base import ExecutionConfig, FrozenConfig, ModelError, TaskType
from app.models.registry import get_model_spec, validate_config
from app.telemetry import utc_now

RUN_ID = re.compile(r"run_[0-9a-f]{32}\Z")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}


class RunPlan(FrozenConfig):
    dataset_id: str
    dataset_sha256: str
    target_column: str
    models: tuple[str, ...]
    total_trials: StrictInt = Field(default=10, ge=1, le=200)
    metric: str = "accuracy"
    timeout_seconds: StrictInt = Field(default=600, ge=1, le=86400)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    fixed_parameters: dict[str, dict] = Field(default_factory=dict)
    model_specs: dict[str, dict]
    direction: str = "maximize"


def build_plan(settings, dataset_id, target_column=None, models=("LogisticRegression", "RandomForestClassifier"),
               total_trials=10, metric="accuracy", timeout_seconds=600, execution=None, fixed_parameters=None):
    metadata = DatasetStore(settings).detail(dataset_id)
    target = target_column if target_column is not None else metadata["target_column"]
    if target not in {column["name"] for column in metadata["columns"]}:
        raise ModelError("TARGET_REQUIRED", "请指定数据集中存在的分类目标列。")
    if not models or len(models) != len(set(models)) or total_trials < len(models):
        raise ModelError("INVALID_RUN_BUDGET", "模型不能重复，Trial 预算至少覆盖每个模型的一次基线。")
    fixed_parameters = fixed_parameters or {}
    if set(fixed_parameters) - set(models):
        raise ModelError("INVALID_MODEL_PARAMETERS", "固定参数包含未选择的模型。")
    execution = execution or ExecutionConfig()
    specs = {}
    for name in models:
        validate_config(name, TaskType.CLASSIFICATION, fixed_parameters.get(name), (metric,), execution)
        specs[name] = get_model_spec(name).model_dump(mode="json")
    if len({spec["framework"] for spec in specs.values()}) != 1:
        raise ModelError("MIXED_FRAMEWORK_RUN", "当前单次 Run 只支持同一框架。")
    return RunPlan(dataset_id=dataset_id, dataset_sha256=metadata["sha256"], target_column=target,
                   models=tuple(models), total_trials=total_trials, metric=metric, timeout_seconds=timeout_seconds,
                   execution=execution, fixed_parameters=fixed_parameters, model_specs=specs)


def process_alive(pid, created):
    try:
        process = psutil.Process(pid)
        return abs(process.create_time() - created) < 0.01 and process.is_running()
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # Do not steal a lease when liveness is uncertain.


class RunStore(DatasetStore):
    def initialize(self):
        super().initialize()
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS hpo_runs (
                  run_id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE,
                  plan_json TEXT NOT NULL, plan_hash TEXT NOT NULL, status TEXT NOT NULL,
                  created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                  worker_id TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
                  completed_trials INTEGER NOT NULL DEFAULT 0, failed_trials INTEGER NOT NULL DEFAULT 0,
                  pruned_trials INTEGER NOT NULL DEFAULT 0, best_score REAL, best_model TEXT,
                  results_path TEXT, error_json TEXT);
                CREATE TABLE IF NOT EXISTS hpo_trial_records (
                  run_id TEXT NOT NULL, number INTEGER NOT NULL, record_json TEXT NOT NULL,
                  PRIMARY KEY(run_id,number));
                CREATE TABLE IF NOT EXISTS local_workers (
                  worker_id TEXT PRIMARY KEY, pid INTEGER NOT NULL, process_created REAL NOT NULL,
                  state TEXT NOT NULL, started_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                  stop_requested INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS local_worker_slot (slot INTEGER PRIMARY KEY, worker_id TEXT);
                INSERT OR IGNORE INTO local_worker_slot(slot,worker_id) VALUES(1,NULL);
            """)

    def folder(self, run_id):
        if not RUN_ID.fullmatch(run_id):
            raise ModelError("RUN_NOT_FOUND", "Run 不存在。", status_code=404)
        return self.settings.data_dir / "runs" / run_id

    def submit(self, plan: RunPlan, idempotency_key=None):
        key = idempotency_key or uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", key):
            raise ModelError("INVALID_IDEMPOTENCY_KEY", "幂等键限 1–128 位字母、数字、点、横线或下划线。")
        payload = json.dumps(plan.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, allow_nan=False)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT run_id,plan_hash FROM hpo_runs WHERE idempotency_key=?", (key,)).fetchone()
            if existing:
                if existing["plan_hash"] != digest:
                    raise ModelError("IDEMPOTENCY_CONFLICT", "该幂等键已用于不同的实验配置。", status_code=409)
                run_id = existing["run_id"]
            else:
                queued = db.execute("SELECT count(*) FROM hpo_runs WHERE status IN ('QUEUED','RUNNING','CANCELLING')").fetchone()[0]
                if queued >= 20:
                    raise ModelError("LOCAL_QUEUE_FULL", "本地未完成 Run 已达到 20 个。", status_code=409)
                self.settings.check_disk()
                run_id = "run_" + uuid4().hex
                self.folder(run_id).mkdir(exist_ok=False)
                write_json(self.folder(run_id) / "plan.json", plan.model_dump(mode="json"))
                db.execute("""INSERT INTO hpo_runs(run_id,idempotency_key,plan_json,plan_hash,status,created_at)
                              VALUES(?,?,?,?,'QUEUED',?)""", (run_id, key, payload, digest, utc_now()))
        return self.get(run_id)

    def _lost_worker(self, db, worker):
        if worker and not process_alive(worker["pid"], worker["process_created"]):
            db.execute("UPDATE local_workers SET state='FAILED',updated_at=? WHERE worker_id=?", (utc_now(), worker["worker_id"]))
            error = json.dumps({"code": "WORKER_LOST", "message": "本地 Worker 已退出；当前版本不自动重放未确定结果的 Trial。"})
            db.execute("""UPDATE hpo_runs SET status='FAILED',finished_at=?,error_json=?
                          WHERE worker_id=? AND status IN ('RUNNING','CANCELLING')""", (utc_now(), error, worker["worker_id"]))
            db.execute("UPDATE local_worker_slot SET worker_id=NULL WHERE worker_id=?", (worker["worker_id"],))
            return True
        return False

    def get(self, run_id):
        self.folder(run_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM hpo_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise ModelError("RUN_NOT_FOUND", "Run 不存在。", status_code=404)
            if row["worker_id"] and row["status"] in {"RUNNING", "CANCELLING"}:
                worker = db.execute("SELECT * FROM local_workers WHERE worker_id=?", (row["worker_id"],)).fetchone()
                if self._lost_worker(db, worker):
                    row = db.execute("SELECT * FROM hpo_runs WHERE run_id=?", (run_id,)).fetchone()
        result = dict(row)
        result["plan"] = json.loads(result.pop("plan_json"))
        result["error"] = json.loads(result.pop("error_json") or "null")
        result["finished_trials"] = sum(result[key] for key in ("completed_trials", "failed_trials", "pruned_trials"))
        result["total_trials"] = result["plan"]["total_trials"]
        result["study_name"] = "study_" + run_id
        return result

    def list_runs(self):
        with self.connect() as db:
            ids = [row[0] for row in db.execute("SELECT run_id FROM hpo_runs ORDER BY created_at DESC LIMIT 100")]
        return [self.get(run_id) for run_id in ids]

    def active_worker(self):
        with self.connect() as db:
            worker = db.execute("SELECT w.* FROM local_worker_slot s JOIN local_workers w ON w.worker_id=s.worker_id WHERE s.slot=1").fetchone()
            if not worker or self._lost_worker(db, worker):
                return None
            return dict(worker)

    def register_worker(self):
        process = psutil.Process()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            worker = db.execute("SELECT w.* FROM local_worker_slot s JOIN local_workers w ON w.worker_id=s.worker_id WHERE s.slot=1").fetchone()
            if worker and not self._lost_worker(db, worker):
                return None
            worker_id = "worker_" + uuid4().hex
            db.execute("INSERT INTO local_workers VALUES(?,?,?,'IDLE',?,?,0)",
                       (worker_id, process.pid, process.create_time(), utc_now(), utc_now()))
            db.execute("UPDATE local_worker_slot SET worker_id=? WHERE slot=1", (worker_id,))
        return worker_id

    def release_worker(self, worker_id, state="STOPPED"):
        with self.connect() as db:
            db.execute("UPDATE local_workers SET state=?,updated_at=? WHERE worker_id=?", (state, utc_now(), worker_id))
            db.execute("UPDATE local_worker_slot SET worker_id=NULL WHERE worker_id=?", (worker_id,))

    def claim_next(self, worker_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            worker = db.execute("SELECT w.* FROM local_worker_slot s JOIN local_workers w ON w.worker_id=s.worker_id WHERE s.slot=1 AND w.worker_id=?", (worker_id,)).fetchone()
            if worker is None:
                return None, True
            if worker["stop_requested"]:
                db.execute("UPDATE local_workers SET state='STOPPED',updated_at=? WHERE worker_id=?", (utc_now(), worker_id))
                db.execute("UPDATE local_worker_slot SET worker_id=NULL WHERE worker_id=?", (worker_id,))
                return None, True
            row = db.execute("SELECT run_id FROM hpo_runs WHERE status='QUEUED' ORDER BY created_at LIMIT 1").fetchone()
            state = "BUSY" if row else "IDLE"
            db.execute("UPDATE local_workers SET state=?,updated_at=? WHERE worker_id=?", (state, utc_now(), worker_id))
            if row:
                db.execute("UPDATE hpo_runs SET status='RUNNING',worker_id=?,started_at=COALESCE(started_at,?) WHERE run_id=?", (worker_id, utc_now(), row[0]))
                return row[0], False
        return None, False

    def cancel(self, run_id):
        self.get(run_id)
        with self.connect() as db:
            db.execute("""UPDATE hpo_runs SET cancel_requested=1,
                       status=CASE WHEN status='QUEUED' THEN 'CANCELLED' ELSE 'CANCELLING' END,
                       finished_at=CASE WHEN status='QUEUED' THEN ? ELSE finished_at END
                       WHERE run_id=? AND status IN ('QUEUED','RUNNING','CANCELLING')""", (utc_now(), run_id))
        return self.get(run_id)

    def resume(self, run_id):
        run = self.get(run_id)
        if run["status"] not in {"CANCELLED", "FAILED"}:
            raise ModelError("RUN_NOT_RESUMABLE", "仅允许显式恢复 CANCELLED 或 FAILED Run。", status_code=409)
        with self.connect() as db:
            changed = db.execute("""UPDATE hpo_runs SET status='QUEUED',cancel_requested=0,finished_at=NULL,error_json=NULL
                                  WHERE run_id=? AND status IN ('CANCELLED','FAILED')""", (run_id,)).rowcount
            if not changed:
                raise ModelError("RUN_STATE_CHANGED", "Run 状态已变化，请重新查询。", status_code=409)
        return self.get(run_id)

    def stop_worker(self):
        with self.connect() as db:
            db.execute("UPDATE local_workers SET stop_requested=1 WHERE worker_id=(SELECT worker_id FROM local_worker_slot WHERE slot=1)")

    def enable_worker(self):
        # Serialize against claim_next's stop/release so new submissions cannot get stranded.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            worker = db.execute("SELECT w.* FROM local_worker_slot s JOIN local_workers w ON w.worker_id=s.worker_id WHERE s.slot=1").fetchone()
            if worker and not self._lost_worker(db, worker):
                db.execute("UPDATE local_workers SET stop_requested=0 WHERE worker_id=?", (worker["worker_id"],))
                return dict(worker)
        return None

    def trial_records(self, run_id):
        self.get(run_id)
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT record_json FROM hpo_trial_records WHERE run_id=? ORDER BY number", (run_id,))]

    def sync_trials(self, run_id, records):
        complete = [record for record in records if record["state"] == "COMPLETE"]
        best = max(complete, key=lambda record: record["value"]) if complete else None
        with self.connect() as db:
            db.executemany("""INSERT INTO hpo_trial_records VALUES(?,?,?) ON CONFLICT(run_id,number)
                              DO UPDATE SET record_json=excluded.record_json""",
                           [(run_id, record["number"], json.dumps(record, ensure_ascii=False, allow_nan=False)) for record in records])
            db.execute("""UPDATE hpo_runs SET completed_trials=?,failed_trials=?,pruned_trials=?,best_score=?,best_model=? WHERE run_id=?""",
                       (len(complete), sum(record["state"] == "FAIL" for record in records),
                        sum(record["state"] == "PRUNED" for record in records),
                        best["value"] if best else None, best["model_name"] if best else None, run_id))

    def finish(self, run_id, status, results_path=None, error=None):
        if status not in TERMINAL:
            raise ValueError("Expected terminal Run status")
        with self.connect() as db:
            # Cancel wins if requested during final fitting/evaluation.
            db.execute("""UPDATE hpo_runs SET status=CASE WHEN cancel_requested=1 AND ?='COMPLETED' THEN 'CANCELLED' ELSE ? END,
                          finished_at=?,results_path=?,error_json=? WHERE run_id=?""",
                       (status, status, utc_now(), results_path, json.dumps(error) if error else None, run_id))
