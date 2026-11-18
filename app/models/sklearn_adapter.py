"""sklearn-specific data splitting, leak-safe pipelines, training and metrics."""

import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from app.datasets import is_missing
from app.models.base import (
    DataPartition, DatasetInput, EvaluationResult, ExecutionConfig, FeatureSchema,
    ModelAdapter, ModelError, ModelSpec, PreparedData, TaskType,
)
from app.models.registry import validate_config


class ScikitLearnAdapter(ModelAdapter):
    framework = "sklearn"

    def prepare_data(self, data: DatasetInput, execution: ExecutionConfig) -> PreparedData:
        if data.task_type != TaskType.CLASSIFICATION:
            raise ModelError("TASK_NOT_IMPLEMENTED", "本阶段 sklearn 仅实现分类数据流程。")
        if data.target_column not in data.columns:
            raise ModelError("TARGET_REQUIRED", "分类任务必须指定存在的目标列。")
        if len(data.columns) < 2 or not data.rows:
            raise ModelError("INVALID_DATASET", "至少需要一个特征列及非空样本。")
        if any(len(row) != len(data.columns) for row in data.rows):
            raise ModelError("INVALID_DATASET", "样本列数与表头不一致。")
        estimate = len(data.rows) * len(data.columns) * 96
        available = min(execution.memory_limit_mb * 1024**2, psutil.virtual_memory().available // 2)
        if estimate > available:
            raise ModelError("MEMORY_ESTIMATE_EXCEEDED", "数据表示的预计内存超过预算。", {"estimated_bytes": estimate})
        target = data.columns.index(data.target_column)
        y = np.asarray([row[target] for row in data.rows], dtype=str)
        if any(is_missing(value) for value in y):
            raise ModelError("MISSING_TARGET", "目标列含缺失值，请先处理；系统不会静默删除样本。")
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) < 2 or len(labels) > 50 or min(counts) < 5:
            raise ModelError("INVALID_CLASS_DISTRIBUTION", "当前分层三划分要求 2–50 个类别，且每类至少 5 条样本。")
        indices = np.arange(len(y))
        try:
            development, test = train_test_split(indices, test_size=execution.test_fraction,
                                                 stratify=y, random_state=execution.random_seed)
            train, validation = train_test_split(
                development, test_size=execution.validation_fraction / (1 - execution.test_fraction),
                stratify=y[development], random_state=execution.random_seed,
            )
        except ValueError as error:
            raise ModelError("INVALID_SPLIT", "样本量与划分比例不足以覆盖所有类别。") from error
        if any(len(np.unique(y[part])) != len(labels) for part in (train, validation, test)):
            raise ModelError("INVALID_SPLIT", "每个分区必须包含所有类别，请增加样本或调整划分。")
        features = [index for index in range(len(data.columns)) if index != target]
        names = tuple(data.columns[index] for index in features)
        numeric, categorical = [], []
        # Infer column types exclusively from training rows, not the global profile.
        for output_index, source_index in enumerate(features):
            values = [data.rows[index][source_index] for index in train if not is_missing(data.rows[index][source_index])]
            try:
                is_numeric = all(math.isfinite(float(value)) for value in values)
            except ValueError:
                is_numeric = False
            (numeric if is_numeric else categorical).append(output_index)
        matrix = np.empty((len(y), len(features)), dtype=object)
        for row_index, row in enumerate(data.rows):
            for column, source_index in enumerate(features):
                value = row[source_index]
                if is_missing(value):
                    matrix[row_index, column] = np.nan
                elif column in numeric:
                    try:
                        number = float(value)
                        if not math.isfinite(number):
                            raise ValueError("nonfinite")
                        matrix[row_index, column] = number
                    except ValueError as error:
                        raise ModelError("FEATURE_SCHEMA_MISMATCH", "验证/测试数据与训练分区推断的数值类型不一致。", {"column": names[column]}) from error
                else:
                    matrix[row_index, column] = value

        def partition(part):
            return DataPartition(matrix[part], y[part], tuple(int(index) for index in part))

        schema = FeatureSchema(names, tuple(numeric), tuple(categorical))
        return PreparedData(data.task_type, schema, partition(train), partition(validation), partition(test))

    def create_model(self, spec: ModelSpec, parameters: dict, schema: FeatureSchema, execution: ExecutionConfig):
        if spec.framework != self.framework:
            raise ModelError("FRAMEWORK_MISMATCH", "模型规范与适配器不匹配。")
        params = validate_config(spec.model_name, spec.task_type, parameters, execution=execution)
        estimators = {
            "LogisticRegression": lambda: LogisticRegression(**params, solver="lbfgs", random_state=execution.random_seed),
            "RandomForestClassifier": lambda: RandomForestClassifier(**params, n_jobs=1, random_state=execution.random_seed),
        }
        if spec.model_name not in estimators:
            raise ModelError("MODEL_NOT_FOUND", "sklearn 适配器没有该模型定义。", status_code=404)
        transformers = []
        if schema.numeric_indices:
            steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
            if spec.model_name == "LogisticRegression":
                steps.append(("scale", StandardScaler()))
            transformers.append(("numeric", Pipeline(steps), list(schema.numeric_indices)))
        if schema.categorical_indices:
            transformers.append(("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True, max_categories=64)),
            ]), list(schema.categorical_indices)))
        if not transformers:
            raise ModelError("EMPTY_FEATURES", "没有可用特征。")
        preprocess = ColumnTransformer(transformers, sparse_threshold=1.0, n_jobs=1)
        return Pipeline([("preprocess", preprocess), ("estimator", estimators[spec.model_name]())])

    def train(self, model, data: DataPartition, execution: ExecutionConfig, report):
        if data.targets is None:
            raise ModelError("TARGET_REQUIRED", "分类训练需要目标值。")
        report("fit_started", {"rows": len(data.row_indices)})
        with threadpool_limits(limits=execution.cpu_threads), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(data.features, data.targets)
        messages = [{"category": item.category.__name__, "message": str(item.message)[:500]} for item in captured]
        report("fit_completed", {"warnings": messages})
        return {"warnings": messages}

    def predict(self, model, features):
        with threadpool_limits(limits=1):
            return model.predict(features)

    def predict_proba(self, model, features):
        with threadpool_limits(limits=1):
            return model.predict_proba(features)

    def evaluate(self, model, data: DataPartition, metrics: tuple[str, ...]) -> EvaluationResult:
        if data.targets is None:
            raise ModelError("TARGET_REQUIRED", "分类评价需要目标值。")
        allowed = {"accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"}
        if not metrics or set(metrics) - allowed:
            raise ModelError("UNSUPPORTED_METRIC", "未知分类评价指标。")
        predictions = self.predict(model, data.features)
        classes = model.classes_
        precision, recall, f1, _ = precision_recall_fscore_support(
            data.targets, predictions, labels=classes, average="macro", zero_division=0,
        )
        scores = {"accuracy": float(accuracy_score(data.targets, predictions)), "precision_macro": float(precision),
                  "recall_macro": float(recall), "f1_macro": float(f1)}
        notes = {}
        if "roc_auc" in metrics:
            probabilities = self.predict_proba(model, data.features)
            if set(data.targets) != set(classes):
                scores["roc_auc"] = None
                notes["roc_auc"] = "评价分区没有覆盖训练时的全部类别，ROC-AUC 不可计算。"
            elif len(classes) == 2:
                scores["roc_auc"] = float(roc_auc_score(data.targets == classes[1], probabilities[:, 1]))
                notes["roc_auc"] = f"二分类正类为 classes_[1]={classes[1]}。"
            else:
                scores["roc_auc"] = float(roc_auc_score(data.targets, probabilities, labels=classes, multi_class="ovr", average="macro"))
                notes["roc_auc"] = "多分类采用 one-vs-rest macro ROC-AUC。"
        return EvaluationResult(
            {metric: scores[metric] for metric in metrics}, notes, len(data.row_indices),
            predictions.tolist(), classes.tolist(), confusion_matrix(data.targets, predictions, labels=classes).tolist(),
        )

    def save_model(self, model, output_dir: Path) -> Path:
        # This project writes trusted artifacts; never load user-supplied pickle/joblib.
        output = output_dir / "model.joblib"
        with output.open("xb") as destination:
            joblib.dump(model, destination)
        return output
