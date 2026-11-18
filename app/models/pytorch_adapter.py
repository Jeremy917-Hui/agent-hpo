"""Reserved extension contract only. No torch imports, model registration or training.

A concrete implementation must implement ModelAdapter plus any optional model
capabilities. Training reports epoch metrics through EventReporter; each Trial
owns a newly initialized model/optimizer and an isolated checkpoint directory.
"""

from typing import Literal

from pydantic import Field, StrictInt

from app.models.base import FrozenConfig, ModelAdapter


class PyTorchTrainingConfig(FrozenConfig):
    batch_size: StrictInt = Field(default=32, ge=1, le=65536)
    epochs: StrictInt = Field(default=20, ge=1, le=10000)
    learning_rate: float = Field(default=1e-3, gt=0, le=1)
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    loss: str = Field(default="cross_entropy", pattern=r"^[a-z][a-z0-9_]{0,63}$")
    checkpoint_policy: Literal["best", "last", "best_and_last"] = "best"
    requested_gpus: StrictInt = Field(default=0, ge=0, le=8)


class PyTorchAdapter(ModelAdapter):
    """Abstract extension point, intentionally not instantiable in this release.

    Device/CPU/memory/seed come from ExecutionConfig. Future GPU scheduling must
    verify ModelSpec.execution_requirements before loading torch. This class is
    not a runnable adapter and is not registered by get_adapter().
    """

    framework = "pytorch"
