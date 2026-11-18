"""Optuna execution through ModelAdapter; never imports a concrete ML framework."""

import csv
import json
import math
import time
from dataclasses import asdict
from datetime import timezone
from importlib.metadata import version
from uuid import uuid4

import optuna
from optuna.trial import TrialState

from app.baseline import load_dataset, write_json
from app.models.base import ModelError
from app.models.registry import get_adapter, get_model_spec, validate_model_params
from app.runs import RunPlan, RunStore
from app.telemetry import EventLog, ResourceMonitor, utc_now


class TrialExecutionError(Exception):
    """A failed model evaluation consumes its Trial quota; it is not auto-retried."""


def storage_url(settings):
    return "sqlite:///" + (settings.data_dir / "db" / "optuna.db").as_posix()


def trial_record(trial):
    def utc(value):
        return value.astimezone(timezone.utc).isoformat() if value else None

    return {"number": trial.number, "state": trial.state.name, "value": trial.value,
            "model_name": trial.user_attrs.get("model_name", trial.params.get("model_name")),
            "parameters": trial.user_attrs.get("parameters", {}), "is_baseline": trial.user_attrs.get("is_baseline", False),
            "created_at": trial.user_attrs.get("created_at", utc(trial.datetime_start)),
            "started_at": utc(trial.datetime_start), "finished_at": utc(trial.datetime_complete),
            "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
            "worker_id": trial.user_attrs.get("worker_id"), "metrics": trial.user_attrs.get("validation_metrics", {}),
            "fit_seconds": trial.user_attrs.get("fit_seconds"), "evaluation_seconds": trial.user_attrs.get("evaluation_seconds"),
            "error": trial.user_attrs.get("error")}


def sample_parameters(trial, model_name, fixed):
    spec = get_model_spec(model_name)
    parameters = dict(fixed)
    for name, definition in spec.parameter_space.items():
        if name in fixed:
            continue
        key = model_name + "." + name
        if definition.kind == "float":
            value = trial.suggest_float(key, definition.low, definition.high, log=definition.log)
        elif definition.kind == "int":
            value = trial.suggest_int(key, int(definition.low), int(definition.high), log=definition.log)
        else:
            value = trial.suggest_categorical(key, definition.choices)
        parameters[name] = value
    return validate_model_params(model_name, parameters)


class Objective:
    """Receives train/validation only: no test partition can enter optimization."""

    def __init__(self, plan, adapter, schema, train, validation, worker_id, events, store, run_id):
        self.plan, self.adapter, self.schema = plan, adapter, schema
        self.train, self.validation = train, validation
        self.worker_id, self.events, self.store, self.run_id = worker_id, events, store, run_id

    def __call__(self, trial):
        if self.store.get(self.run_id)["cancel_requested"]:
            raise optuna.TrialPruned("Cancelled before fitting")
        name = trial.suggest_categorical("model_name", self.plan.models)
        trial.set_user_attr("model_name", name)
        trial.set_user_attr("worker_id", self.worker_id)
        if "created_at" not in trial.user_attrs:
            trial.set_user_attr("created_at", utc_now())
        self.events.emit("trial_started", {"trial_number": trial.number, "model_name": name})
        self.store.sync_trials(self.run_id, [trial_record(t) for t in trial.study.get_trials(deepcopy=False)])
        try:
            parameters = sample_parameters(trial, name, self.plan.fixed_parameters.get(name, {}))
            trial.set_user_attr("parameters", parameters)
            model = self.adapter.create_model(get_model_spec(name), parameters, self.schema, self.plan.execution)
            started = time.perf_counter()
            training = self.adapter.train(model, self.train, self.plan.execution,
                lambda event, details: self.events.emit(event, {"trial_number": trial.number, "model_name": name, **details}))
            trial.set_user_attr("fit_seconds", time.perf_counter() - started)
            trial.set_user_attr("training_warnings", training.get("warnings", []))
            evaluation_start = time.perf_counter()
            result = self.adapter.evaluate(model, self.validation, get_model_spec(name).supported_metrics)
            trial.set_user_attr("evaluation_seconds", time.perf_counter() - evaluation_start)
            trial.set_user_attr("validation_metrics", result.metrics)
            value = result.metrics[self.plan.metric]
            if value is None or not math.isfinite(value):
                raise ModelError("UNDEFINED_OBJECTIVE", "当前验证分区的目标指标不可计算。")
            return value
        except Exception as error:
            public = {"code": getattr(error, "code", type(error).__name__), "message": "模型训练或验证失败。"}
            trial.set_user_attr("error", public)
            self.events.emit("trial_failed", {"trial_number": trial.number, **public})
            raise TrialExecutionError(public["code"]) from error


def execute_run(settings, run_id, worker_id, after_trial=None):
    """Caller holds the exclusive local Worker slot. after_trial supports tests only."""
    store = RunStore(settings)
    run = store.get(run_id)
    if run["status"] != "RUNNING" or run["worker_id"] != worker_id:
        raise ModelError("RUN_NOT_CLAIMED", "Worker 尚未取得此 Run。", status_code=409)
    folder = store.folder(run_id)
    events = EventLog(folder / "events.jsonl", {"run_id": run_id, "worker_id": worker_id})
    events.emit("run_started")
    storage = None
    try:
        plan = RunPlan.model_validate(run["plan"])
        if json.loads((folder / "plan.json").read_text(encoding="utf-8")) != plan.model_dump(mode="json"):
            raise ModelError("PLAN_CHANGED", "计划文件与提交快照不一致。")
        for name in plan.models:
            if get_model_spec(name).model_dump(mode="json") != plan.model_specs[name]:
                raise ModelError("REGISTRY_CHANGED", "模型定义与提交时快照不一致，请新建 Run。")
        data, metadata = load_dataset(store, plan.dataset_id, plan.target_column)
        if metadata["sha256"] != plan.dataset_sha256:
            raise ModelError("DATASET_CHANGED", "数据与提交时快照不一致。")
        adapter = get_adapter(plan.model_specs[plan.models[0]]["framework"])
        prepared = adapter.prepare_data(data, plan.execution)
        if prepared.validation is None or prepared.test is None:
            raise ModelError("INVALID_SPLIT", "当前 HPO 需要训练、验证和测试分区。")
        splits = {key: list(getattr(prepared, key).row_indices) for key in ("train", "validation", "test")}
        split_path = folder / "split_indices.json"
        if split_path.exists():
            if json.loads(split_path.read_text()) != splits:
                raise ModelError("SPLIT_CHANGED", "恢复时的数据划分发生变化。")
        else:
            write_json(split_path, splits)
        events.emit("data_prepared", {key: len(value) for key, value in splits.items()})
        storage = optuna.storages.RDBStorage(storage_url(settings), engine_kwargs={"connect_args": {"timeout": 30}})
        startup_trials = min(5, max(len(plan.models), plan.total_trials // 3))
        study = optuna.create_study(study_name=run["study_name"], storage=storage, direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=plan.execution.random_seed, n_startup_trials=startup_trials),
            pruner=optuna.pruners.NopPruner(), load_if_exists=True)
        if study.user_attrs.get("plan_hash", run["plan_hash"]) != run["plan_hash"]:
            raise ModelError("STUDY_CONFLICT", "Study 绑定的计划与此 Run 不一致。")
        study.set_user_attr("plan_hash", run["plan_hash"])
        study.set_user_attr("run_id", run_id)
        study.set_user_attr("sampler", {"name": "TPESampler", "seed": plan.execution.random_seed, "n_startup_trials": startup_trials})
        trials = study.get_trials(deepcopy=False)
        if any(trial.state == TrialState.RUNNING for trial in trials):
            raise ModelError("TRIAL_RECOVERY_REQUIRED", "存在结果未确定的 RUNNING Trial，M3 不自动重放；保留现场供后续恢复。")
        if not trials:
            # Each selected model gets one project-default baseline within the same quota.
            for name in plan.models:
                defaults = validate_model_params(name, plan.fixed_parameters.get(name, {}))
                parameters = {"model_name": name, **{name + "." + key: value for key, value in defaults.items()
                               if key not in plan.fixed_parameters.get(name, {})}}
                study.enqueue_trial(parameters, user_attrs={"is_baseline": True, "model_name": name, "created_at": utc_now()})
        trials = study.get_trials(deepcopy=False)
        consumed = sum(trial.state != TrialState.WAITING for trial in trials)
        if len(trials) > plan.total_trials or consumed > plan.total_trials:
            raise ModelError("TRIAL_BUDGET_CONFLICT", "Study 已有 Trial 超过冻结预算。")
        remaining = plan.total_trials - consumed
        objective = Objective(plan, adapter, prepared.schema, prepared.train, prepared.validation,
                              worker_id, events, store, run_id)

        def callback(current_study, trial):
            records = [trial_record(item) for item in current_study.get_trials(deepcopy=False)]
            store.sync_trials(run_id, records)
            events.emit("trial_finished", {"trial_number": trial.number, "state": trial.state.name, "value": trial.value})
            if after_trial:
                after_trial(current_study, trial)
            if store.get(run_id)["cancel_requested"]:
                current_study.stop()

        attempt_id = "attempt_" + uuid4().hex
        with ResourceMonitor(folder / (attempt_id + "_resources.jsonl")) as monitor:
            if remaining and not store.get(run_id)["cancel_requested"]:
                study.optimize(objective, n_trials=remaining, n_jobs=1, timeout=plan.timeout_seconds,
                               catch=(TrialExecutionError,), callbacks=[callback])
            records = [trial_record(trial) for trial in study.get_trials(deepcopy=False)]
            store.sync_trials(run_id, records)
            if store.get(run_id)["cancel_requested"]:
                store.finish(run_id, "CANCELLED")
                events.emit("run_cancelled")
                return
            completed = [record for record in records if record["state"] == "COMPLETE"]
            if not completed:
                raise ModelError("NO_SUCCESSFUL_TRIAL", "没有成功 Trial，无法生成最优模型。")
            best = study.best_trial
            name = best.user_attrs["model_name"]
            final_dir = folder / attempt_id
            final_dir.mkdir(exist_ok=False)
            evaluations = {}
            for label, parameters in (("default", validate_model_params(name, plan.fixed_parameters.get(name, {}))),
                                      ("optimized", best.user_attrs["parameters"])):
                if store.get(run_id)["cancel_requested"]:
                    store.finish(run_id, "CANCELLED")
                    events.emit("run_cancelled")
                    return
                output = final_dir / label
                output.mkdir(exist_ok=False)
                model = adapter.create_model(get_model_spec(name), parameters, prepared.schema, plan.execution)
                adapter.train(model, prepared.train, plan.execution,
                    lambda event, details: events.emit(event, {"phase": "final_" + label, **details}))
                # Held-out test is touched only AFTER the search has selected the winner.
                evaluation = asdict(adapter.evaluate(model, prepared.test, get_model_spec(name).supported_metrics))
                evaluation.pop("predictions")
                artifact = adapter.save_model(model, output)
                evaluations[label] = {"parameters": parameters, "test": evaluation,
                                      "model_artifact": artifact.relative_to(folder).as_posix()}
            result = {"run_id": run_id, "study_name": study.study_name, "best_trial_number": best.number,
                      "best_model": name, "best_validation_score": best.value, "optimization_metric": plan.metric,
                      "total_trial_budget": plan.total_trials, "finished_trials": len(completed) + sum(record["state"] in {"FAIL", "PRUNED"} for record in records),
                      "completion_reason": "TRIAL_BUDGET_REACHED" if all(record["state"] not in {"WAITING", "RUNNING"} for record in records) and len(records) == plan.total_trials else "SEARCH_TIME_LIMIT",
                      "sampler": study.user_attrs["sampler"], "resources_path": monitor.path.relative_to(folder).as_posix(),
                      "evaluations": evaluations, "trials": records, "ended_at": utc_now(),
                      "versions": {package: version(package) for package in ("optuna", "scikit-learn", "numpy")}}
            write_json(final_dir / "results.json", result)
            with (final_dir / "trials.csv").open("x", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["trial_number", "state", "model_name", "value", "is_baseline", "started_at_utc", "finished_at_utc", "duration_seconds", "parameters_json"])
                for record in records:
                    writer.writerow([record["number"], record["state"], record["model_name"], record["value"], record["is_baseline"],
                                     record["started_at"], record["finished_at"], record["duration_seconds"], json.dumps(record["parameters"])])
        store.finish(run_id, "COMPLETED", (final_dir / "results.json").relative_to(folder).as_posix())
        events.emit("run_" + store.get(run_id)["status"].lower(),
                    {"best_trial": best.number, "sampled_peak_rss_bytes": monitor.peak_rss_bytes})
    except Exception as error:
        public = {"code": getattr(error, "code", type(error).__name__),
                  "message": str(error) if isinstance(error, ModelError) else "Run 执行失败，详见 Worker 日志。"}
        events.emit("run_failed", public)
        store.finish(run_id, "FAILED", error=public)
    finally:
        if storage is not None:
            storage.remove_session()
            storage.engine.dispose()
