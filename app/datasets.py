"""Bounded CSV inspection and persistence, reusable by HTTP and future MCP tools.

No training, code execution, or LLM access occurs here. Original bytes are preserved.
"""

import csv
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config import Settings

MISSING = {"", "na", "n/a", "nan", "null", "none"}
DATASET_ID = re.compile(r"dataset_[0-9a-f]{32}\Z")
csv.field_size_limit(1_048_576)


class DatasetError(Exception):
    def __init__(self, message, status_code=422):
        super().__init__(message)
        self.status_code = status_code


def validate_filename(filename: str):
    if (
        not filename or len(filename) > 180
        or any(character in filename for character in '/\\:')
        or any(ord(character) < 32 for character in filename)
        or not filename.lower().endswith(".csv")
    ):
        raise DatasetError("请选择合法的 CSV 文件名，不能包含路径或控制字符。", 400)


def is_missing(value):
    return value.strip().casefold() in MISSING


class ColumnAccumulator:
    """Welford numeric moments; bounded distinct-value tracking for text columns."""

    def __init__(self, name):
        self.name = name
        self.missing = 0
        self.present = 0
        self.numeric = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = None
        self.maximum = None
        self.values = set()
        self.distinct_truncated = False

    def add(self, value):
        if is_missing(value):
            self.missing += 1
            return
        self.present += 1
        # Hash tokens to avoid retaining large cell contents in memory.
        token = hashlib.sha256(value.encode("utf-8")).digest()
        if len(self.values) < 1024:
            self.values.add(token)
        elif token not in self.values:
            self.distinct_truncated = True
        try:
            number = float(value)
        except ValueError:
            return
        if not math.isfinite(number):
            return
        self.numeric += 1
        delta = number - self.mean
        self.mean += delta / self.numeric
        self.m2 += delta * (number - self.mean)
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)

    def summary(self, rows):
        numeric = self.present > 0 and self.numeric == self.present
        stats = None
        if numeric:
            variance = self.m2 / (self.numeric - 1) if self.numeric > 1 else 0

            def finite(value):
                return value if math.isfinite(value) else None

            stats = {
                "count": self.numeric, "mean": finite(self.mean),
                "std": finite(math.sqrt(max(0, variance))) if math.isfinite(variance) else None,
                "min": self.minimum, "max": self.maximum,
            }
        return {
            "name": self.name, "dtype": "numeric" if numeric else "text" if self.present else "empty",
            "missing_count": self.missing, "missing_ratio": self.missing / rows,
            "unique_count": len(self.values), "unique_count_is_lower_bound": self.distinct_truncated,
            "numeric_statistics": stats,
        }


def inspect_csv(path: Path, settings: Settings):
    """Scan full file, keeping only 20 preview rows, hashes and bounded counters."""
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open(encoding=encoding, newline="") as source:
                reader = csv.reader(source, strict=True)
                header = next(reader, None)
                if not header or not any(item.strip() for item in header):
                    raise DatasetError("CSV 为空或缺少表头。")
                if any(not item.strip() or "\x00" in item or len(item) > 256 for item in header):
                    raise DatasetError("列名不能为空、包含空字符或超过 256 个字符。")
                if len(set(header)) != len(header):
                    raise DatasetError("CSV 存在重复列名，请先明确区分各列。")
                if len(header) > settings.max_dataset_columns:
                    raise DatasetError(f"列数超过限制：{settings.max_dataset_columns}。", 413)
                columns = [ColumnAccumulator(name) for name in header]
                preview, seen = [], set()
                rows = duplicates = 0
                for row in reader:
                    if not row:  # Ignore blank physical lines, not rows of empty cells.
                        continue
                    if len(row) != len(header):
                        raise DatasetError(f"第 {reader.line_num} 行的字段数与表头不一致。")
                    if any("\x00" in value for value in row):
                        raise DatasetError("CSV 含有空字符，可能不是有效的文本文件。")
                    rows += 1
                    if rows > settings.max_dataset_rows:
                        raise DatasetError(f"行数超过限制：{settings.max_dataset_rows}。", 413)
                    digest = hashlib.sha256(json.dumps(row, ensure_ascii=False).encode("utf-8")).digest()
                    duplicates += digest in seen
                    seen.add(digest)
                    for column, value in zip(columns, row):
                        column.add(value)
                    if len(preview) < 20:
                        preview.append([None if is_missing(value) else value[:500] for value in row])
                if not rows:
                    raise DatasetError("CSV 只有表头，没有数据行。")
                summaries = [column.summary(rows) for column in columns]
                return {
                    "encoding": encoding, "row_count": rows, "column_count": len(header),
                    "columns": summaries, "duplicate_rows": duplicates,
                    "duplicate_definition": "all raw parsed cell values match (SHA256 comparison)",
                    "numeric_column_count": sum(column["dtype"] == "numeric" for column in summaries),
                    "categorical_column_count": sum(column["dtype"] == "text" for column in summaries),
                    "preview": preview, "preview_limit": 20, "preview_cell_limit": 500,
                    "missing_tokens": sorted(MISSING),
                    "task_hint": "尚未确定：监督任务需先选择目标列；聚类和降维可不选目标列。",
                }
        except UnicodeDecodeError:
            continue
        except csv.Error as error:
            raise DatasetError("CSV 格式错误或单个字段超过 1 MiB。") from error
    raise DatasetError("无法读取编码，请使用 UTF-8 或 GB18030 编码的 CSV。")


class DatasetStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.settings.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self):
        with self.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                target_column TEXT
            )""")

    def list(self):
        with self.connect() as connection:
            records = connection.execute(
                "SELECT * FROM datasets ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        return [dict(row) for row in records]

    def record(self, dataset_id):
        if not DATASET_ID.fullmatch(dataset_id):
            raise DatasetError("数据集不存在。", 404)
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if row is None:
            raise DatasetError("数据集不存在。", 404)
        return dict(row)

    def detail(self, dataset_id):
        record = self.record(dataset_id)
        try:
            metadata = json.loads((self.settings.datasets_dir / dataset_id / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise DatasetError("数据集元信息不可用，请检查本地存储。", 409) from error
        detail = {**metadata, **record}
        target = record["target_column"]
        if target is not None:
            column = next(item for item in detail["columns"] if item["name"] == target)
            target_summary = {"column": target, "missing_count": column["missing_count"], "numeric_statistics": column["numeric_statistics"]}
            if not column["unique_count_is_lower_bound"] and column["unique_count"] <= 50:
                counts = Counter()
                with (self.settings.datasets_dir / dataset_id / "original.csv").open(encoding=detail["encoding"], newline="") as source:
                    reader = csv.reader(source, strict=True)
                    index = next(reader).index(target)
                    for row in reader:
                        if row and not is_missing(row[index]):
                            counts[row[index]] += 1
                target_summary["value_counts"] = [{"value": value[:500], "count": count} for value, count in counts.most_common(20)]
                target_summary["value_counts_truncated"] = len(counts) > 20
            else:
                target_summary["value_counts"] = None
                target_summary["note"] = "目标列唯一值较多，省略类别分布；数值列可参考数值摘要。"
            detail["target_summary"] = target_summary
            detail["task_hint"] = "已选择目标列。分类或回归仍需根据业务含义确认；列类型和唯一值数量不能替代任务定义。"
        else:
            detail["target_summary"] = None
        return detail

    def accept(self, staged: Path, filename, size_bytes, sha256):
        self.settings.check_disk()
        summary = inspect_csv(staged, self.settings)
        dataset_id = "dataset_" + uuid4().hex
        folder = self.settings.datasets_dir / dataset_id
        # Verify both absolute paths before moving a generated staged upload.
        if not staged.resolve().is_relative_to(self.settings.tmp_dir.resolve()):
            raise DatasetError("非法暂存路径。", 400)
        if folder.resolve().parent != self.settings.datasets_dir.resolve():
            raise DatasetError("非法数据集路径。", 400)
        folder.mkdir(exist_ok=False)
        staged.rename(folder / "original.csv")  # New destination only; never overwrite.
        record = {
            "dataset_id": dataset_id, "original_filename": filename,
            "created_at": datetime.now(timezone.utc).isoformat(), "size_bytes": size_bytes,
            "row_count": summary["row_count"], "column_count": summary["column_count"],
            "sha256": sha256, "target_column": None,
        }
        # Immutable ingestion metadata. Mutable target selection lives in SQLite.
        with (folder / "metadata.json").open("x", encoding="utf-8") as output:
            json.dump({**record, **summary}, output, ensure_ascii=False, allow_nan=False, indent=2)
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO datasets VALUES (:dataset_id,:original_filename,:created_at,:size_bytes,:row_count,:column_count,:sha256,:target_column)", record,
            )
        return self.detail(dataset_id)

    def select_target(self, dataset_id, target_column):
        metadata = self.detail(dataset_id)
        if target_column is not None and target_column not in {column["name"] for column in metadata["columns"]}:
            raise DatasetError("目标列不在该数据集中。")
        with self.connect() as connection:
            connection.execute("UPDATE datasets SET target_column=? WHERE dataset_id=?", (target_column, dataset_id))
        return self.detail(dataset_id)
