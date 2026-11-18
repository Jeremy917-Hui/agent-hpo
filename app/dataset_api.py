"""Small HTTP transport layer for DatasetStore; uploads use raw CSV bodies."""

import hashlib
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from app.datasets import DatasetError, validate_filename

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class TargetSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_column: str | None = Field(max_length=256)


@router.get("")
def list_datasets(request: Request):
    return {"datasets": request.app.state.datasets.list(), "limit": 100}


@router.get("/{dataset_id}")
def dataset_detail(dataset_id: str, request: Request):
    return request.app.state.datasets.detail(dataset_id)


@router.patch("/{dataset_id}/target")
def select_target(dataset_id: str, selection: TargetSelection, request: Request):
    result = request.app.state.datasets.select_target(dataset_id, selection.target_column)
    request.app.state.logger.info("dataset_target_selected dataset_id=%s", dataset_id)
    return result


@router.post("", status_code=201, openapi_extra={"requestBody": {"required": True, "content": {"text/csv": {"schema": {"type": "string", "format": "binary"}}}}})
async def upload_dataset(request: Request, filename: str):
    """Send CSV bytes with ?filename=name.csv, Content-Type: text/csv (not multipart)."""
    validate_filename(filename)
    settings = request.app.state.settings
    store = request.app.state.datasets
    if request.headers.get("content-type", "").split(";")[0].strip() not in {
        "text/csv", "application/octet-stream", "application/vnd.ms-excel",
    }:
        raise DatasetError("上传请求必须携带 CSV 文件内容。", 415)
    limit = settings.max_upload_mb * 1024**2
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            length = int(content_length)
        except ValueError as error:
            raise DatasetError("Content-Length 格式错误。", 400) from error
        if length < 0:
            raise DatasetError("Content-Length 不能为负。", 400)
        if length > limit:
            raise DatasetError(f"文件超过 {settings.max_upload_mb} MiB 限制。", 413)
    settings.check_disk()
    staged = settings.tmp_dir / ("upload_" + uuid4().hex + ".part")
    size = 0
    digest = hashlib.sha256()
    try:
        with staged.open("xb") as output:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise DatasetError(f"文件超过 {settings.max_upload_mb} MiB 限制。", 413)
                digest.update(chunk)
                await run_in_threadpool(output.write, chunk)
        result = await run_in_threadpool(store.accept, staged, filename, size, digest.hexdigest())
    except Exception:
        # Retain staged bytes or an unindexed ingestion folder for manual recovery.
        request.app.state.logger.warning("dataset_upload_rejected staging_id=%s", staged.stem)
        raise
    request.app.state.logger.info("dataset_uploaded dataset_id=%s size_bytes=%s", result["dataset_id"], size)
    return result
