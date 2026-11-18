"""Registry discovery and pure validation only; never trains in an HTTP handler."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from app.models.base import ExecutionConfig, TaskType
from app.models.registry import get_model_spec, list_models, validate_config

router = APIRouter(prefix="/api/models", tags=["models"])


class ModelValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_type: TaskType
    parameters: dict[str, Any] = Field(default_factory=dict)
    metrics: tuple[str, ...] | None = None
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


@router.get("")
def models(task_type: TaskType | None = None):
    return {"models": [spec.model_dump(mode="json") for spec in list_models(task_type)]}


@router.get("/{model_name}")
def model_spec(model_name: str):
    return get_model_spec(model_name).model_dump(mode="json")


@router.post("/{model_name}/validate")
def model_validation(model_name: str, request: ModelValidation):
    parameters = validate_config(model_name, request.task_type, request.parameters, request.metrics, request.execution)
    return {"valid": True, "model_name": model_name, "parameters": parameters, "execution": request.execution.model_dump()}
