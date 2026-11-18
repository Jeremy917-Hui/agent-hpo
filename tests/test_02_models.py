import inspect
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.datasets import load_iris

from app.main import create_app
from app.models.base import (DataPartition, DatasetInput, ExecutionConfig, ModelError,
                             PredictionCapability, ProbabilityCapability, TaskType)
from app.models.pytorch_adapter import PyTorchAdapter, PyTorchTrainingConfig
from app.models.registry import (create_estimator, get_adapter, get_model_spec, list_models,
                                 validate_config, validate_model_params)
from test_00_foundation import isolated_settings


def iris_input():
    iris = load_iris()
    rows = tuple(tuple([*(str(value) for value in row), str(iris.target_names[label])])
                 for row, label in zip(iris.data, iris.target))
    return DatasetInput((*iris.feature_names, "target"), rows, "target", TaskType.CLASSIFICATION)


def test_registry_contract_deep_copy_and_optional_pytorch():
    assert {spec.model_name for spec in list_models()} == {"LogisticRegression", "RandomForestClassifier"}
    assert list_models(TaskType.REGRESSION) == []
    spec = get_model_spec("LogisticRegression")
    spec.parameter_space.clear()
    assert get_model_spec("LogisticRegression").parameter_space["C"].log
    assert inspect.isabstract(PyTorchAdapter)
    assert PyTorchTrainingConfig().optimizer == "adam"
    assert "torch" not in sys.modules
    with pytest.raises(ModelError, match="PyTorch"):
        get_adapter("pytorch")
    with pytest.raises(ValidationError):
        PyTorchTrainingConfig(batch_size=0)


@pytest.mark.parametrize("name,parameters,code", [
    ("unknown", {}, "MODEL_NOT_FOUND"),
    ("LogisticRegression", {"C": 0}, "INVALID_PARAMETER"),
    ("LogisticRegression", {"C": float("nan")}, "INVALID_PARAMETER"),
    ("LogisticRegression", {"C": True}, "INVALID_PARAMETER"),
    ("LogisticRegression", {"solver": "arbitrary"}, "UNKNOWN_PARAMETER"),
    ("RandomForestClassifier", {"n_estimators": 1}, "INVALID_PARAMETER"),
    ("RandomForestClassifier", {"n_estimators": 10.5}, "INVALID_PARAMETER"),
    ("RandomForestClassifier", {"n_jobs": -1}, "UNKNOWN_PARAMETER"),
    ("RandomForestClassifier", {"max_depth": "unlimited"}, "INVALID_PARAMETER"),
])
def test_invalid_registry_parameters(name, parameters, code):
    with pytest.raises(ModelError) as raised:
        validate_model_params(name, parameters)
    assert raised.value.code == code


def test_task_metric_and_resource_validation():
    for options, code in [
        ({"task_type": TaskType.REGRESSION}, "TASK_MISMATCH"),
        ({"task_type": TaskType.CLASSIFICATION, "metrics": ("rmse",)}, "UNSUPPORTED_METRIC"),
        ({"task_type": TaskType.CLASSIFICATION, "execution": ExecutionConfig(device="cuda")}, "UNSUPPORTED_RESOURCE"),
        ({"task_type": TaskType.CLASSIFICATION, "execution": ExecutionConfig(cpu_threads=2)}, "UNSUPPORTED_RESOURCE"),
    ]:
        with pytest.raises(ModelError) as raised:
            validate_config("LogisticRegression", **options)
        assert raised.value.code == code


@pytest.mark.parametrize("name", ["LogisticRegression", "RandomForestClassifier"])
def test_real_multiclass_training_and_disjoint_deterministic_splits(name):
    adapter, execution = get_adapter("sklearn"), ExecutionConfig()
    assert isinstance(adapter, PredictionCapability) and isinstance(adapter, ProbabilityCapability)
    prepared = adapter.prepare_data(iris_input(), execution)
    partitions = [set(getattr(prepared, key).row_indices) for key in ("train", "validation", "test")]
    assert [len(part) for part in partitions] == [90, 30, 30]
    assert not (partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2])
    assert set.union(*partitions) == set(range(150))
    assert prepared.train.row_indices == adapter.prepare_data(iris_input(), execution).train.row_indices
    model = create_estimator(name, {}, prepared.schema, execution)
    events = []
    adapter.train(model, prepared.train, execution, lambda event, details: events.append(event))
    result = adapter.evaluate(model, prepared.test, get_model_spec(name).supported_metrics)
    assert events == ["fit_started", "fit_completed"]
    assert result.metrics["accuracy"] >= 0.8
    assert 0 <= result.metrics["roc_auc"] <= 1
    assert sum(map(sum, result.confusion_matrix)) == 30
    assert result.metrics["accuracy"] == np.mean(np.asarray(result.predictions) == prepared.test.targets)
    if name == "RandomForestClassifier":
        assert model.named_steps["estimator"].n_jobs == 1


def test_preprocessing_is_fit_on_train_only_and_unknown_categories_work():
    rows = tuple(("" if index % 7 == 0 else str(index), f"id_{index}", "yes" if index >= 60 else "no") for index in range(120))
    data = DatasetInput(("number", "category", "target"), rows, "target", TaskType.CLASSIFICATION)
    adapter, execution = get_adapter("sklearn"), ExecutionConfig()
    prepared = adapter.prepare_data(data, execution)
    model = create_estimator("LogisticRegression", {}, prepared.schema, execution)
    adapter.train(model, prepared.train, execution, lambda *_: None)
    preprocess = model.named_steps["preprocess"]
    imputer = preprocess.named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[0] == np.nanmedian(prepared.train.features[:, 0].astype(float))
    categories = set(preprocess.named_transformers_["categorical"].named_steps["encode"].categories_[0])
    assert categories == set(prepared.train.features[:, 1])
    assert categories.isdisjoint(set(prepared.test.features[:, 1]))
    assert len(adapter.predict(model, prepared.test.features)) == len(prepared.test.row_indices)
    assert prepared.schema.names == ("number", "category")
    one_class = DataPartition(prepared.test.features[:1], prepared.test.targets[:1], prepared.test.row_indices[:1])
    assert adapter.evaluate(model, one_class, ("roc_auc",)).metrics["roc_auc"] is None


@pytest.mark.parametrize("data,code", [
    (DatasetInput(("x", "y"), (("1", ""),) * 10, "y", TaskType.CLASSIFICATION), "MISSING_TARGET"),
    (DatasetInput(("x", "y"), (("1", "a"),) * 10, "y", TaskType.CLASSIFICATION), "INVALID_CLASS_DISTRIBUTION"),
    (DatasetInput(("x", "y"), (("1", "a"),) * 10, None, TaskType.CLASSIFICATION), "TARGET_REQUIRED"),
    (DatasetInput(("x", "y"), (("1", "a"),) * 10, "y", TaskType.REGRESSION), "TASK_NOT_IMPLEMENTED"),
])
def test_rejects_unusable_training_data(data, code):
    with pytest.raises(ModelError) as raised:
        get_adapter("sklearn").prepare_data(data, ExecutionConfig())
    assert raised.value.code == code


def test_registry_http_contract_and_no_training_endpoint():
    with TestClient(create_app(isolated_settings())) as client:
        assert len(client.get("/api/models").json()["models"]) == 2
        assert client.get("/api/models?task_type=regression").json()["models"] == []
        assert client.get("/api/models/unknown").json()["error"]["code"] == "MODEL_NOT_FOUND"
        valid = client.post("/api/models/LogisticRegression/validate", json={"task_type": "classification", "parameters": {"C": 2}})
        assert valid.status_code == 200 and valid.json()["parameters"]["C"] == 2
        invalid = client.post("/api/models/LogisticRegression/validate", json={"task_type": "regression"})
        assert invalid.status_code == 422 and invalid.json()["error"]["code"] == "TASK_MISMATCH"
        assert client.post("/api/models/LogisticRegression/train", json={}).status_code == 404
