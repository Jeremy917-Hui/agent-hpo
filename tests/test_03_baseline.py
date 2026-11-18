import csv
import hashlib
import json
from pathlib import Path

import joblib
import pytest

from app.baseline import run_baseline
from app.datasets import DatasetStore
from app.models.base import ExecutionConfig, ModelError
from app.models.registry import get_adapter
from test_00_foundation import isolated_settings
from test_02_models import iris_input


def ingest(settings):
    settings.prepare()
    data = iris_input()
    staged = settings.tmp_dir / "iris.csv"
    with staged.open("x", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(data.columns)
        writer.writerows(data.rows)
    payload = staged.read_bytes()
    store = DatasetStore(settings)
    store.initialize()
    metadata = store.accept(staged, "iris.csv", len(payload), hashlib.sha256(payload).hexdigest())
    store.select_target(metadata["dataset_id"], "target")
    return metadata["dataset_id"]


def test_baseline_artifacts_reload_events_resources_and_reproducibility():
    settings = isolated_settings()
    dataset_id = ingest(settings)
    first = run_baseline(settings, dataset_id)
    second = run_baseline(settings, dataset_id)
    folder = Path(first["output_dir"])
    assert folder != Path(second["output_dir"]) and first["status"] == "COMPLETED"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_column"] == "target"
    assert manifest["execution"]["cpu_threads"] == 1
    assert "scikit-learn" in manifest["environment"]
    assert (folder / "split_indices.json").read_bytes() == (Path(second["output_dir"]) / "split_indices.json").read_bytes()
    prepared = get_adapter("sklearn").prepare_data(iris_input(), ExecutionConfig())
    for left, right in zip(first["results"], second["results"]):
        assert left["test"]["metrics"] == right["test"]["metrics"]
        model = joblib.load(folder / left["model_artifact"])  # Only a trusted artifact created by this test.
        assert len(model.predict(prepared.test.features)) == 30
        predictions = list(csv.DictReader((folder / left["model_name"] / "predictions.csv").open(encoding="utf-8")))
        assert len(predictions) == 60
        test_predictions = [row["prediction"] for row in predictions if row["partition"] == "test"]
        assert test_predictions == model.predict(prepared.test.features).tolist()
    assert len(list(csv.DictReader((folder / "metrics.csv").open(encoding="utf-8")))) == 20
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[0]["event"] == "baseline_created" and events[-1]["event"] == "baseline_completed"
    assert len([event for event in events if event["event"] == "fit_completed"]) == 2
    resources = [json.loads(line) for line in (folder / "resources.jsonl").read_text().splitlines()]
    assert len(resources) >= 2 and first["sampled_peak_rss_bytes"] > 0


def test_changed_dataset_is_rejected_and_failure_record_preserved():
    settings = isolated_settings()
    dataset_id = ingest(settings)
    original = settings.datasets_dir / dataset_id / "original.csv"
    with original.open("ab") as output:
        output.write(b"\n")
    with pytest.raises(ModelError) as raised:
        run_baseline(settings, dataset_id)
    assert raised.value.code == "DATASET_INTEGRITY_ERROR"
    folder = next((settings.data_dir / "runs").iterdir())
    assert json.loads((folder / "failure.json").read_text(encoding="utf-8"))["status"] == "FAILED"
    assert original.exists()
