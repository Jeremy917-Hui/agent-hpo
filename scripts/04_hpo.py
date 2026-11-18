"""Explicit local development submission; no approval claims or cloud/API calls."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.datasets import DatasetError
from app.models.base import ExecutionConfig, ModelError
from app.runs import RunStore, build_plan
from app.worker import ensure_worker


def main():
    parser = argparse.ArgumentParser(description="M3 single-worker asynchronous HPO (development CLI)")
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--dataset-id", required=True)
    submit.add_argument("--target")
    submit.add_argument("--models", nargs="+", default=["LogisticRegression", "RandomForestClassifier"])
    submit.add_argument("--trials", type=int, default=10)
    submit.add_argument("--metric", default="accuracy")
    submit.add_argument("--seed", type=int, default=2021)
    submit.add_argument("--timeout", type=int, default=600)
    submit.add_argument("--idempotency-key")
    submit.add_argument("--fixed-params-json", default="{}")
    for action in ("status", "trials", "results", "cancel", "resume"):
        commands.add_parser(action).add_argument("run_id")
    for action in ("list", "worker-status", "worker-start", "worker-stop"):
        commands.add_parser(action)
    args = parser.parse_args()
    try:
        settings = Settings.load()
        settings.prepare()
        store = RunStore(settings)
        store.initialize()
        if args.command == "submit":
            fixed = json.loads(args.fixed_params_json)
            if not isinstance(fixed, dict) or any(not isinstance(value, dict) for value in fixed.values()):
                raise ValueError("--fixed-params-json must map model names to parameter objects")
            plan = build_plan(settings, args.dataset_id, args.target, tuple(args.models), args.trials,
                              args.metric, args.timeout, ExecutionConfig(random_seed=args.seed), fixed)
            run = store.submit(plan, args.idempotency_key)
            worker = ensure_worker(settings) if run["status"] == "QUEUED" else None
            result = {"run_id": run["run_id"], "status": run["status"], "idempotency_key": run["idempotency_key"],
                      "total_trials": run["total_trials"], "worker": worker}
        elif args.command == "resume":
            result = store.resume(args.run_id)
            result = {"run_id": args.run_id, "status": result["status"], "worker": ensure_worker(settings)}
        elif args.command == "cancel":
            result = store.cancel(args.run_id)
        elif args.command == "status":
            result = store.get(args.run_id)
            result.pop("plan")
        elif args.command == "trials":
            result = store.trial_records(args.run_id)
        elif args.command == "results":
            run = store.get(args.run_id)
            if run["status"] != "COMPLETED" or not run["results_path"]:
                raise ModelError("RESULTS_NOT_READY", "Run 尚未成功完成。", status_code=409)
            result = json.loads((store.folder(args.run_id) / run["results_path"]).read_text(encoding="utf-8"))
        elif args.command == "list":
            result = [{key: run[key] for key in ("run_id", "status", "finished_trials", "total_trials", "best_score")}
                      for run in store.list_runs()]
        elif args.command == "worker-start":
            result = ensure_worker(settings)
        elif args.command == "worker-stop":
            store.stop_worker()
            result = {"stop_requested": True, "message": "Worker 将在当前 Run 结束后停止；剩余排队任务保留。"}
        else:
            result = store.active_worker()
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (ModelError, DatasetError, ValueError, OSError) as error:
        print(json.dumps({"error": getattr(error, "code", "INVALID_INPUT"), "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
