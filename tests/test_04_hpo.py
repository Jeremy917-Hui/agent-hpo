import json
import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.hpo import execute_run
from app.main import create_app
from app.models.base import ModelError
from app.models.sklearn_adapter import ScikitLearnAdapter
from app.runs import RunStore, build_plan
from app.worker import ensure_worker
from test_00_foundation import isolated_settings
from test_03_baseline import ingest


def setup_run(trials=3, models=("LogisticRegression",)):
    settings = isolated_settings()
    dataset_id = ingest(settings)
    store = RunStore(settings)
    store.initialize()
    fixed = {name: {"n_estimators": 10} if name == "RandomForestClassifier" else {"max_iter": 100} for name in models}
    plan = build_plan(settings, dataset_id, models=models, total_trials=trials, fixed_parameters=fixed)
    return settings, store, plan


def claim(store, plan, key=None):
    run = store.submit(plan, key)
    worker = store.register_worker()
    assert worker is not None
    assert store.claim_next(worker) == (run["run_id"], False)
    return run["run_id"], worker


def test_persistent_real_optuna_budget_baselines_and_test_isolation(monkeypatch):
    settings, store, plan = setup_run(4, ("LogisticRegression", "RandomForestClassifier"))
    run_id, worker = claim(store, plan)
    original = ScikitLearnAdapter.evaluate
    calls = []

    def checked(self, model, data, metrics):
        splits = json.loads((store.folder(run_id) / "split_indices.json").read_text())
        complete = store.get(run_id)["finished_trials"]
        expected = "validation" if complete < plan.total_trials else "test"
        assert list(data.row_indices) == splits[expected]
        calls.append(expected)
        return original(self, model, data, metrics)

    monkeypatch.setattr(ScikitLearnAdapter, "evaluate", checked)
    try:
        execute_run(settings, run_id, worker)
        run = store.get(run_id)
        assert run["status"] == "COMPLETED", run["error"]
        records = store.trial_records(run_id)
        assert len(records) == run["finished_trials"] == 4
        assert sum(record["is_baseline"] for record in records) == 2
        assert {record["model_name"] for record in records[:2]} == set(plan.models)
        assert calls == ["validation"] * 4 + ["test"] * 2
        result = json.loads((store.folder(run_id) / run["results_path"]).read_text(encoding="utf-8"))
        assert result["evaluations"]["optimized"]["test"]["sample_count"] == 30
        assert result["completion_reason"] == "TRIAL_BUDGET_REACHED"
        for evaluation in result["evaluations"].values():
            assert (store.folder(run_id) / evaluation["model_artifact"]).exists()
        assert (settings.data_dir / "db" / "optuna.db").exists()
    finally:
        store.release_worker(worker)


def test_idempotency_returns_original_run_and_rejects_changed_plan():
    settings, store, plan = setup_run()
    first = store.submit(plan, "same-request")
    assert store.submit(plan, "same-request")["run_id"] == first["run_id"]
    assert len(store.list_runs()) == 1
    changed = plan.model_copy(update={"total_trials": 4})
    with pytest.raises(ModelError) as raised:
        store.submit(changed, "same-request")
    assert raised.value.code == "IDEMPOTENCY_CONFLICT"
    store.cancel(first["run_id"])
    assert store.submit(plan, "same-request")["status"] == "CANCELLED"


def test_cooperative_cancel_and_resume_preserve_history_and_total_budget():
    settings, store, plan = setup_run(5, ("LogisticRegression", "RandomForestClassifier"))
    run_id, worker = claim(store, plan)
    try:
        def stop_after_two(study, trial):
            if trial.number == 1:
                store.cancel(run_id)
        execute_run(settings, run_id, worker, after_trial=stop_after_two)
        assert store.get(run_id)["status"] == "CANCELLED"
        original = store.trial_records(run_id)
        assert len(original) == 2
        store.resume(run_id)
        assert store.claim_next(worker) == (run_id, False)
        execute_run(settings, run_id, worker)
        assert store.get(run_id)["status"] == "COMPLETED"
        resumed = store.trial_records(run_id)
        assert len(resumed) == store.get(run_id)["finished_trials"] == 5
        assert resumed[:2] == original
        assert [record["number"] for record in resumed] == list(range(5))
    finally:
        store.release_worker(worker)


def test_failed_trial_consumes_quota_and_search_continues(monkeypatch):
    settings, store, plan = setup_run(3)
    run_id, worker = claim(store, plan)
    original = ScikitLearnAdapter.train
    counter = 0
    def fail_once(self, *args):
        nonlocal counter
        counter += 1
        if counter == 1:
            raise ValueError("controlled first-Trial fit failure")
        return original(self, *args)
    monkeypatch.setattr(ScikitLearnAdapter, "train", fail_once)
    try:
        execute_run(settings, run_id, worker)
        run = store.get(run_id)
        assert run["status"] == "COMPLETED", run["error"]
        assert (run["completed_trials"], run["failed_trials"], run["finished_trials"]) == (2, 1, 3)
        assert store.trial_records(run_id)[0]["error"]["code"] == "ValueError"
    finally:
        store.release_worker(worker)


def test_all_failed_trials_fail_run_without_inventing_best_model(monkeypatch):
    settings, store, plan = setup_run(2)
    run_id, worker = claim(store, plan)
    def fail(*args):
        raise ValueError("controlled fit failure")
    monkeypatch.setattr(ScikitLearnAdapter, "train", fail)
    try:
        execute_run(settings, run_id, worker)
        run = store.get(run_id)
        assert run["status"] == "FAILED"
        assert run["failed_trials"] == 2 and run["best_score"] is None
        assert run["error"]["code"] == "NO_SUCCESSFUL_TRIAL"
        assert len(store.trial_records(run_id)) == 2
    finally:
        store.release_worker(worker)


def test_cancellation_during_fit_finishes_current_trial_without_touching_test(monkeypatch):
    settings, store, plan = setup_run(4)
    run_id, worker = claim(store, plan)
    original = ScikitLearnAdapter.train
    fits = []
    def request_cancel(self, *args):
        fits.append(1)
        assert store.cancel(run_id)["status"] == "CANCELLING"
        return original(self, *args)
    monkeypatch.setattr(ScikitLearnAdapter, "train", request_cancel)
    try:
        execute_run(settings, run_id, worker)
        assert fits == [1]
        assert store.get(run_id)["status"] == "CANCELLED"
        assert store.get(run_id)["completed_trials"] == 1
        assert store.trial_records(run_id)[0]["state"] == "COMPLETE"
        assert store.get(run_id)["results_path"] is None
    finally:
        store.release_worker(worker)


def test_single_worker_slot_and_queued_cancel():
    settings, store, plan = setup_run()
    run = store.submit(plan)
    store.cancel(run["run_id"])
    worker = store.register_worker()
    try:
        assert store.register_worker() is None
        assert store.claim_next(worker) == (None, False)
        assert store.get(run["run_id"])["status"] == "CANCELLED"
        assert not (settings.data_dir / "db" / "optuna.db").exists()
        store.stop_worker()
        assert store.claim_next(worker) == (None, True)
    finally:
        store.release_worker(worker)


def test_run_http_is_read_only_and_results_not_ready():
    settings, store, plan = setup_run()
    run = store.submit(plan)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/runs").json()["runs"][0]["run_id"] == run["run_id"]
        assert client.get("/api/runs/" + run["run_id"]).json()["status"] == "QUEUED"
        assert client.get("/api/runs/" + run["run_id"] + "/trials").json()["trials"] == []
        assert client.get("/api/runs/" + run["run_id"] + "/results").status_code == 409
        assert client.post("/api/runs", json={}).status_code == 405
        assert client.post("/api/runs/" + run["run_id"] + "/cancel", json={}).status_code == 404


def test_real_background_worker_serializes_two_runs_and_stops():
    settings, store, plan = setup_run(2)
    first, second = store.submit(plan), store.submit(plan)
    started = time.monotonic()
    launch = ensure_worker(settings)
    assert launch["started_new_process"]
    assert time.monotonic() - started < 5  # Must return before waiting for any training loop.
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            runs = [store.get(first["run_id"]), store.get(second["run_id"])]
            if all(run["status"] in {"COMPLETED", "FAILED"} for run in runs):
                break
            time.sleep(0.1)
        assert all(run["status"] == "COMPLETED" for run in runs), runs
        assert runs[0]["worker_id"] == runs[1]["worker_id"]
        assert datetime.fromisoformat(runs[1]["started_at"]) >= datetime.fromisoformat(runs[0]["finished_at"])
        assert not ensure_worker(settings)["started_new_process"]
    finally:
        store.stop_worker()
        deadline = time.monotonic() + 10
        while store.active_worker() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert store.active_worker() is None
