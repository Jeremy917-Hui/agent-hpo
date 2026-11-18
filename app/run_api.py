"""Read-only Run/Trial APIs. Submission, cancel and resume stay in the dev CLI."""

import json

from fastapi import APIRouter, Request

from app.models.base import ModelError

router = APIRouter(prefix="/api/runs", tags=["runs (read-only in M3)"])


@router.get("")
def runs(request: Request):
    return {"runs": request.app.state.runs.list_runs(), "limit": 100}


@router.get("/{run_id}")
def run_status(run_id: str, request: Request):
    return request.app.state.runs.get(run_id)


@router.get("/{run_id}/trials")
def trials(run_id: str, request: Request):
    return {"run_id": run_id, "trials": request.app.state.runs.trial_records(run_id)}


@router.get("/{run_id}/results")
def results(run_id: str, request: Request):
    store = request.app.state.runs
    run = store.get(run_id)
    if run["status"] != "COMPLETED" or not run["results_path"]:
        raise ModelError("RESULTS_NOT_READY", "Run 尚未成功完成。", status_code=409)
    root = store.folder(run_id).resolve()
    path = (root / run["results_path"]).resolve()
    if not path.is_relative_to(root):
        raise ModelError("INVALID_ARTIFACT_PATH", "结果路径不合法。", status_code=409)
    return json.loads(path.read_text(encoding="utf-8"))
