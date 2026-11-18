"""All writable paths and resource limits are resolved here."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

SOURCE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    log_dir: Path
    min_free_disk_gib: float = 2
    max_upload_mb: int = 20
    max_dataset_rows: int = 100_000
    max_dataset_columns: int = 256

    @classmethod
    def load(cls):
        values = {**dotenv_values(SOURCE_ROOT / ".env"), **os.environ}
        root = Path(values.get("PROJECT_ROOT") or SOURCE_ROOT).resolve()

        def path(key, default):
            value = Path(values.get(key) or default)
            return (value if value.is_absolute() else root / value).resolve()

        settings = cls(
            project_root=root,
            data_dir=path("DATA_DIR", "data"),
            log_dir=path("LOG_DIR", "logs"),
            min_free_disk_gib=float(values.get("MIN_FREE_DISK_GIB", 2)),
            max_upload_mb=int(values.get("MAX_UPLOAD_MB", 20)),
            max_dataset_rows=int(values.get("MAX_DATASET_ROWS", 100_000)),
            max_dataset_columns=int(values.get("MAX_DATASET_COLUMNS", 256)),
        )
        if settings.min_free_disk_gib < 0 or min(
            settings.max_upload_mb, settings.max_dataset_rows, settings.max_dataset_columns
        ) <= 0:
            raise ValueError("Resource limits must be positive; disk reserve may be zero")
        return settings

    @property
    def datasets_dir(self):
        return self.data_dir / "datasets"

    @property
    def database_path(self):
        return self.data_dir / "db" / "app.db"

    @property
    def tmp_dir(self):
        return self.project_root / "tmp"

    def prepare(self):
        for path in (
            self.project_root, self.datasets_dir, self.data_dir / "runs",
            self.database_path.parent, self.log_dir, self.tmp_dir,
            self.project_root / ".cache",
        ):
            path.mkdir(parents=True, exist_ok=True)
        # Keep dependency/runtime temporary files on the project drive.
        os.environ["TEMP"] = os.environ["TMP"] = str(self.tmp_dir)
        self.check_disk()

    def check_disk(self):
        free = shutil.disk_usage(self.project_root).free / 1024**3
        if free < self.min_free_disk_gib:
            raise OSError("Insufficient project disk space")
        return round(free, 2)
