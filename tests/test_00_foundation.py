import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import SOURCE_ROOT, Settings
from app.main import create_app


def isolated_settings(**overrides):
    # Preserve test artifacts; do not use automatic temporary-directory cleanup.
    root = SOURCE_ROOT / "tmp" / "tests" / uuid4().hex
    return Settings(project_root=root, data_dir=root / "data", log_dir=root / "logs", **overrides)


def test_dashboard_health_and_project_directory_log():
    settings = isolated_settings()
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["status"] == "ok"
        page = client.get("/")
        assert page.status_code == 200
        assert "HPO Agent" in page.text
        assert client.get("/static/dashboard.js").status_code == 200
        assert client.get("/static/dashboard.css").status_code == 200
        assert client.get("/docs").status_code == 200
    records = [json.loads(line) for line in (settings.log_dir / "backend.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any("backend_started" in item["message"] for item in records)
    assert Path(settings.log_dir).resolve().is_relative_to(settings.project_root.resolve())


def test_startup_refuses_insufficient_disk():
    settings = isolated_settings(min_free_disk_gib=10**9)
    try:
        with TestClient(create_app(settings)):
            raise AssertionError("Startup should fail")
    except OSError as error:
        assert "disk" in str(error)


def test_rejects_unknown_host_and_cross_origin_write():
    with TestClient(create_app(isolated_settings())) as client:
        assert client.get("/health", headers={"host": "evil.example"}).status_code == 400
        assert client.post("/", headers={"origin": "https://evil.example"}).status_code == 403
