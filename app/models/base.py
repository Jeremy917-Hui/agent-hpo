"""Framework-neutral contracts; this module imports neither sklearn nor torch.

Core training/evaluation is shared. Prediction, clustering and transformation are
separate optional capabilities so an unsupervised adapter need not implement predict.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"


class ModelError(Exception):
    def __init__(self, code: str, message: str, details: dict | None = None, status_code=422):
        super().__init__(message)
        self.code, self.details, self.status_code = code, details or {}, status_code


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionConfig(FrozenConfig):
    device: str = Field(default="cpu", pattern=r"^(cpu|cuda|auto)$")
    cpu_threads: StrictInt = Field(default=1, ge=1, le=16)
    memory_limit_mb: StrictInt = Field(default=512, ge=64, le=4096)
    random_seed: StrictInt = Field(default=2021, ge=0, le=2**32 - 1)
    test_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    validation_fraction: float = Field(default=0.2, gt=0, lt=0.5)

    @model_validator(mode="after")
    def split_budget(self):
        if self.test_fraction + self.validation_fraction >= 0.8:
            raise ValueError("At least 20% of rows must remain for training")
        return self


class ExecutionRequirements(FrozenConfig):
    supported_devices: tuple[str, ...] = ("cpu",)
    cpu_threads: int = 1
    gpu_count: int = 0
    minimum_memory_mb: int = 64


class ParameterSpec(FrozenConfig):
    kind: str = Field(pattern=r"^(int|float|categorical)$")
    default: str | int | float | bool | None
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: tuple[str | int | float | bool | None, ...] = ()

    def validate_value(self, name: str, value):
        valid = False
        if self.kind == "categorical":
            valid = any(type(value) is type(choice) and value == choice for choice in self.choices)
        elif self.kind == "int":
            valid = type(value) is int and self.low <= value <= self.high
        elif self.kind == "float":
            valid = type(value) in (int, float) and self.low <= value <= self.high
        if not valid:
            raise ModelError("INVALID_PARAMETER", f"参数 {name} 的值不在注册表允许范围内。", {"parameter": name})
        return float(value) if self.kind == "float" else value


class ModelSpec(FrozenConfig):
    model_name: str
    framework: str
    task_type: TaskType
    adapter: str
    description: str
    parameter_space: dict[str, ParameterSpec]
    supported_metrics: tuple[str, ...]
    capabilities: tuple[str, ...]
    execution_requirements: ExecutionRequirements = Field(default_factory=ExecutionRequirements)


@dataclass(frozen=True)
class DatasetInput:
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    target_column: str | None
    task_type: TaskType


@dataclass(frozen=True)
class FeatureSchema:
    names: tuple[str, ...]
    numeric_indices: tuple[int, ...]
    categorical_indices: tuple[int, ...]


@dataclass(frozen=True)
class DataPartition:
    features: Any
    targets: Any | None  # None is valid for clustering and dimensionality reduction.
    row_indices: tuple[int, ...]


@dataclass(frozen=True)
class PreparedData:
    task_type: TaskType
    schema: FeatureSchema
    train: DataPartition
    validation: DataPartition | None = None
    test: DataPartition | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float | None]
    metric_notes: dict[str, str]
    sample_count: int
    predictions: list[Any]
    labels: list[Any] = field(default_factory=list)
    confusion_matrix: list[list[int]] = field(default_factory=list)


EventReporter = Callable[[str, dict], None]


class ModelAdapter(ABC):
    framework: str

    @abstractmethod
    def prepare_data(self, data: DatasetInput, execution: ExecutionConfig) -> PreparedData: ...

    @abstractmethod
    def create_model(self, spec: ModelSpec, parameters: dict, schema: FeatureSchema, execution: ExecutionConfig) -> Any: ...

    @abstractmethod
    def train(self, model: Any, data: DataPartition, execution: ExecutionConfig, report: EventReporter) -> dict: ...

    @abstractmethod
    def evaluate(self, model: Any, data: DataPartition, metrics: tuple[str, ...]) -> EvaluationResult: ...

    @abstractmethod
    def save_model(self, model: Any, output_dir: Path) -> Path: ...


@runtime_checkable
class PredictionCapability(Protocol):
    def predict(self, model: Any, features: Any) -> Any: ...


@runtime_checkable
class ProbabilityCapability(Protocol):
    def predict_proba(self, model: Any, features: Any) -> Any: ...


@runtime_checkable
class ClusteringCapability(Protocol):
    def fit_predict(self, model: Any, features: Any) -> Any: ...


@runtime_checkable
class TransformationCapability(Protocol):
    def transform(self, model: Any, features: Any) -> Any: ...
