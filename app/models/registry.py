"""Allowlist and parameter contracts. No estimator classes or user import paths."""

from copy import deepcopy

from app.models.base import ExecutionConfig, ModelError, ModelSpec, ParameterSpec, TaskType

METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc")
CAPABILITIES = ("train", "evaluate", "predict", "predict_proba", "save_model")

_SPECS = {
    "LogisticRegression": ModelSpec(
        model_name="LogisticRegression", framework="sklearn", task_type=TaskType.CLASSIFICATION,
        adapter="ScikitLearnAdapter", description="逻辑回归；数值标准化与稀疏类别编码。",
        parameter_space={
            "C": ParameterSpec(kind="float", low=1e-4, high=1000, log=True, default=1.0),
            "max_iter": ParameterSpec(kind="int", low=50, high=3000, default=500),
            "class_weight": ParameterSpec(kind="categorical", choices=(None, "balanced"), default=None),
        }, supported_metrics=METRICS, capabilities=CAPABILITIES,
    ),
    "RandomForestClassifier": ModelSpec(
        model_name="RandomForestClassifier", framework="sklearn", task_type=TaskType.CLASSIFICATION,
        adapter="ScikitLearnAdapter", description="随机森林；Estimator 内部固定 n_jobs=1。",
        parameter_space={
            "n_estimators": ParameterSpec(kind="int", low=10, high=500, default=100),
            "max_depth": ParameterSpec(kind="categorical", choices=(None, 3, 5, 10, 20, 30), default=None),
            "min_samples_split": ParameterSpec(kind="int", low=2, high=20, default=2),
            "min_samples_leaf": ParameterSpec(kind="int", low=1, high=20, default=1),
            "max_features": ParameterSpec(kind="categorical", choices=("sqrt", "log2", None), default="sqrt"),
            "class_weight": ParameterSpec(kind="categorical", choices=(None, "balanced"), default=None),
        }, supported_metrics=METRICS, capabilities=CAPABILITIES,
    ),
}


def list_models(task_type: TaskType | None = None) -> list[ModelSpec]:
    return [spec.model_copy(deep=True) for spec in _SPECS.values() if task_type is None or spec.task_type == task_type]


def get_model_spec(model_name: str) -> ModelSpec:
    if model_name not in _SPECS:
        raise ModelError("MODEL_NOT_FOUND", "该模型未注册。", {"model_name": model_name}, 404)
    return _SPECS[model_name].model_copy(deep=True)


def validate_model_params(model_name: str, parameters: dict | None = None) -> dict:
    spec = get_model_spec(model_name)
    parameters = {} if parameters is None else parameters
    unknown = set(parameters) - spec.parameter_space.keys()
    if unknown:
        raise ModelError("UNKNOWN_PARAMETER", "包含未注册的参数。", {"parameters": sorted(unknown)})
    return {name: definition.validate_value(name, parameters.get(name, deepcopy(definition.default)))
            for name, definition in spec.parameter_space.items()}


def validate_config(model_name: str, task_type: TaskType, parameters: dict | None = None,
                    metrics: tuple[str, ...] | None = None, execution: ExecutionConfig | None = None):
    spec = get_model_spec(model_name)
    if spec.task_type != task_type:
        raise ModelError("TASK_MISMATCH", "模型与任务类型不匹配。")
    resolved = validate_model_params(model_name, parameters)
    if metrics is not None and (not metrics or set(metrics) - set(spec.supported_metrics)):
        raise ModelError("UNSUPPORTED_METRIC", "评价指标不在模型支持范围内。")
    execution = execution or ExecutionConfig()
    requirements = spec.execution_requirements
    if execution.device not in requirements.supported_devices or execution.cpu_threads != requirements.cpu_threads:
        raise ModelError("UNSUPPORTED_RESOURCE", "当前 sklearn 适配器仅允许 CPU、每 Trial 一个计算线程。")
    if execution.memory_limit_mb < requirements.minimum_memory_mb:
        raise ModelError("INSUFFICIENT_MEMORY_BUDGET", "内存预算低于模型最低要求。")
    return resolved


def get_adapter(framework: str):
    # Only trusted, explicit factories. No dynamic imports derived from user input.
    if framework == "sklearn":
        from app.models.sklearn_adapter import ScikitLearnAdapter
        return ScikitLearnAdapter()
    if framework == "pytorch":
        raise ModelError("FRAMEWORK_NOT_IMPLEMENTED", "PyTorch 只预留扩展契约，当前没有训练实现。")
    raise ModelError("FRAMEWORK_NOT_SUPPORTED", "框架未注册。")


def create_estimator(model_name: str, parameters: dict, schema, execution: ExecutionConfig | None = None):
    spec = get_model_spec(model_name)
    execution = execution or ExecutionConfig()
    resolved = validate_config(model_name, spec.task_type, parameters, execution=execution)
    return get_adapter(spec.framework).create_model(spec, resolved, schema, execution)
