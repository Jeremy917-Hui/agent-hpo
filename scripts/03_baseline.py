"""M2 offline acceptance CLI. For help: python scripts/03_baseline.py --help."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.baseline import run_baseline
from app.config import Settings
from app.datasets import DatasetError, DatasetStore
from app.models.base import ExecutionConfig, ModelError
from app.models.registry import list_models


def iris_demo(settings):
    # Bundled public dataset: no network download and no cloud/API credentials.
    from sklearn.datasets import load_iris
    iris = load_iris()
    staged = settings.tmp_dir / ("iris_" + uuid4().hex + ".csv")
    with staged.open("x", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([*iris.feature_names, "target"])
        writer.writerows([*features, iris.target_names[label]] for features, label in zip(iris.data, iris.target))
    store = DatasetStore(settings)
    store.initialize()
    content = staged.read_bytes()
    metadata = store.accept(staged, "iris_sklearn.csv", len(content), hashlib.sha256(content).hexdigest())
    store.select_target(metadata["dataset_id"], "target")
    return metadata["dataset_id"]


def main():
    parser = argparse.ArgumentParser(description="Offline classification baseline; not an asynchronous Run or HPO.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-id")
    source.add_argument("--demo", choices=["iris"])
    parser.add_argument("--target")
    parser.add_argument("--models", nargs="+", choices=[spec.model_name for spec in list_models()],
                        default=["LogisticRegression", "RandomForestClassifier"])
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--params-json", default="{}", help='JSON map keyed by model name; never executable code.')
    arguments = parser.parse_args()
    try:
        parameters = json.loads(arguments.params_json)
        if not isinstance(parameters, dict) or any(not isinstance(value, dict) for value in parameters.values()):
            raise ValueError("--params-json must map model names to parameter objects")
        settings = Settings.load()
        settings.prepare()
        dataset_id = iris_demo(settings) if arguments.demo else arguments.dataset_id
        result = run_baseline(settings, dataset_id, arguments.target, tuple(arguments.models), parameters,
                              ExecutionConfig(random_seed=arguments.seed))
        print(json.dumps({"baseline_id": result["baseline_id"], "dataset_id": dataset_id, "output_dir": result["output_dir"],
                          "results": [{"model": model["model_name"], "validation": model["validation"]["metrics"],
                                       "test": model["test"]["metrics"]} for model in result["results"]]}, ensure_ascii=False, indent=2))
        return 0
    except (ModelError, DatasetError, ValueError, OSError) as error:
        print(json.dumps({"error": getattr(error, "code", "INVALID_INPUT"), "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
