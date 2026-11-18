"""Offline M2 acceptance runner, NOT an HTTP Run/Optuna/approval implementation.

This module depends on adapter contracts rather than a concrete ML framework.
M3's Worker can reuse those same contracts without importing estimator classes.
"""

import csv
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from uuid import uuid4

from app.config import Settings
from app.datasets import DatasetStore
from app.models.base import DatasetInput, ExecutionConfig, ModelError, TaskType
from app.models.registry import get_adapter, get_model_spec, validate_config
from app.telemetry import EventLog, ResourceMonitor, utc_now


def load_dataset(store: DatasetStore, dataset_id: str, target_column: str | None):
    metadata = store.detail(dataset_id)
    target = target_column if target_column is not None else metadata["target_column"]
    if target is None:
        raise ModelError("TARGET_REQUIRED", "请在 Dashboard 保存目标列，或通过命令指定 --target。")
    path = store.settings.datasets_dir / dataset_id / "original.csv"
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != metadata["sha256"]:
        raise ModelError("DATASET_INTEGRITY_ERROR", "原始 CSV 已变化，与登记的 SHA256 不一致。")
    with path.open(encoding=metadata["encoding"], newline="") as source:
        reader = csv.reader(source, strict=True)
        columns = tuple(next(reader))
        rows = tuple(tuple(row) for row in reader if row)
    return DatasetInput(columns, rows, target, TaskType.CLASSIFICATION), metadata


def write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, allow_nan=False, indent=2)


def run_baseline(settings: Settings, dataset_id: str, target_column: str | None = None,
                 model_names: tuple[str, ...] = ("LogisticRegression", "RandomForestClassifier"),
                 parameters: dict | None = None, execution: ExecutionConfig | None = None):
    settings.prepare()
    execution = execution or ExecutionConfig()
    parameters = parameters or {}
    if not model_names or len(model_names) != len(set(model_names)):
        raise ModelError("INVALID_MODEL_SELECTION", "请选择不重复的已注册模型。")
    if set(parameters) - set(model_names):
        raise ModelError("INVALID_MODEL_PARAMETERS", "参数映射包含本次未选择的模型。")
    # Reject all invalid model/task/parameter/resource configurations before training.
    specs = [get_model_spec(name) for name in model_names]
    configs = {spec.model_name: validate_config(spec.model_name, TaskType.CLASSIFICATION,
               parameters.get(spec.model_name), execution=execution) for spec in specs}
    if len({spec.framework for spec in specs}) != 1:
        raise ModelError("MIXED_FRAMEWORK_BASELINE", "当前离线基线一次只比较同一框架的模型。")
    adapter = get_adapter(specs[0].framework)
    store = DatasetStore(settings)
    store.initialize()
    baseline_id = "baseline_" + uuid4().hex
    folder = settings.data_dir / "runs" / baseline_id
    folder.mkdir(exist_ok=False)
    worker_id = f"local-cli-{os.getpid()}"
    events = EventLog(folder / "events.jsonl", {"baseline_id": baseline_id, "worker_id": worker_id})
    created_at, started = utc_now(), time.perf_counter()
    events.emit("baseline_created", {"dataset_id": dataset_id, "execution_kind": "offline_acceptance"})
    results = []
    try:
        with ResourceMonitor(folder / "resources.jsonl") as monitor:
            data, metadata = load_dataset(store, dataset_id, target_column)
            events.emit("dataset_loaded", {"rows": len(data.rows), "columns": len(data.columns)})
            prepared = adapter.prepare_data(data, execution)
            if prepared.validation is None or prepared.test is None:
                raise ModelError("INVALID_BASELINE_SPLIT", "分类基线需要训练、验证、测试三个分区。")
            splits = {name: list(getattr(prepared, name).row_indices) for name in ("train", "validation", "test")}
            write_json(folder / "split_indices.json", splits)
            events.emit("split_created", {name: len(indices) for name, indices in splits.items()})
            manifest = {
                "baseline_id": baseline_id, "created_at": created_at, "started_at": utc_now(),
                "worker_id": worker_id, "dataset_id": dataset_id, "dataset_sha256": metadata["sha256"],
                "target_column": data.target_column, "task_type": data.task_type,
                "feature_schema": asdict(prepared.schema), "execution": execution.model_dump(),
                "models": [spec.model_dump(mode="json") for spec in specs], "parameters": configs,
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                **{name: version(name) for name in ("scikit-learn", "numpy", "scipy", "joblib", "psutil")}},
                "evaluation_policy": "Fixed-parameter baselines; validation for comparison; held-out test is not used to fit or tune.",
                "resource_policy": "One CPU thread; input memory estimate is advisory, not an OS-enforced memory limit.",
                "warnings": (["数据有重复行；当前使用逐行分层划分，实体/时间/分组泄漏需另行设计切分。"] if metadata["duplicate_rows"] else []),
            }
            write_json(folder / "manifest.json", manifest)
            for spec in specs:
                name = spec.model_name
                model_dir = folder / name
                model_dir.mkdir(exist_ok=False)
                events.emit("model_started", {"model_name": name})
                model = adapter.create_model(spec, configs[name], prepared.schema, execution)
                fit_start = time.perf_counter()
                training = adapter.train(model, prepared.train, execution,
                    lambda event, details: events.emit(event, {"model_name": name, **details}))
                fit_seconds = time.perf_counter() - fit_start
                evaluations, evaluate_seconds = {}, {}
                with (model_dir / "predictions.csv").open("x", encoding="utf-8", newline="") as output:
                    writer = csv.writer(output)
                    writer.writerow(["partition", "row_index", "target", "prediction"])
                    for partition_name in ("validation", "test"):
                        partition = getattr(prepared, partition_name)
                        events.emit("evaluation_started", {"model_name": name, "partition": partition_name})
                        evaluation_start = time.perf_counter()
                        result = adapter.evaluate(model, partition, spec.supported_metrics)
                        evaluate_seconds[partition_name] = time.perf_counter() - evaluation_start
                        evaluations[partition_name] = asdict(result)
                        evaluations[partition_name].pop("predictions")
                        writer.writerows(zip([partition_name] * len(partition.row_indices), partition.row_indices,
                                             partition.targets, result.predictions))
                        events.emit("evaluation_completed", {"model_name": name, "partition": partition_name,
                                                              "duration_seconds": evaluate_seconds[partition_name]})
                artifact = adapter.save_model(model, model_dir)
                results.append({"model_name": name, "parameters": configs[name], "fit_seconds": fit_seconds,
                                "model_artifact": artifact.relative_to(folder).as_posix(),
                                "evaluation_seconds": evaluate_seconds, "training": training, **evaluations})
                events.emit("model_completed", {"model_name": name, "fit_seconds": fit_seconds})
        result = {"baseline_id": baseline_id, "status": "COMPLETED", "created_at": created_at, "ended_at": utc_now(),
                  "duration_seconds": time.perf_counter() - started, "sampled_peak_rss_bytes": monitor.peak_rss_bytes,
                  "results": results}
        write_json(folder / "results.json", result)
        with (folder / "metrics.csv").open("x", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["model_name", "partition", "metric", "value", "fit_seconds"])
            for model in results:
                for partition in ("validation", "test"):
                    for metric, value in model[partition]["metrics"].items():
                        writer.writerow([model["model_name"], partition, metric, value, model["fit_seconds"]])
        events.emit("baseline_completed", {"duration_seconds": result["duration_seconds"]})
        return {**result, "output_dir": str(folder)}
    except Exception as error:
        public = {"code": error.code, "message": str(error)} if isinstance(error, ModelError) else {"code": type(error).__name__, "message": "离线基线执行失败。"}
        write_json(folder / "failure.json", {"baseline_id": baseline_id, "status": "FAILED", "ended_at": utc_now(), "error": public})
        events.emit("baseline_failed", public)
        raise
