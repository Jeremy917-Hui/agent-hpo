import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from test_00_foundation import isolated_settings


CSV = "age,income,city,target\n20,100,A,0\n30,,B,1\n20,100,A,0\n,NA,A,1\n".encode()


def upload(client, content=CSV, filename="sample.csv", **kwargs):
    return client.post("/api/datasets", params={"filename": filename}, content=content,
                       headers={"content-type": "text/csv"}, **kwargs)


def test_upload_profile_target_and_persistence():
    settings = isolated_settings()
    with TestClient(create_app(settings)) as client:
        response = upload(client)
        assert response.status_code == 201
        data = response.json()
        dataset_id = data["dataset_id"]
        assert (data["row_count"], data["column_count"], data["duplicate_rows"]) == (4, 4, 1)
        assert data["columns"][0]["missing_count"] == 1
        assert data["columns"][1]["missing_ratio"] == 0.5
        assert data["columns"][1]["numeric_statistics"]["mean"] == 100
        assert data["preview"][1][1] is None
        assert data["sha256"] == hashlib.sha256(CSV).hexdigest()
        folder = settings.datasets_dir / dataset_id
        assert (folder / "original.csv").read_bytes() == CSV
        assert json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["row_count"] == 4
        assert client.get("/api/datasets").json()["datasets"][0]["dataset_id"] == dataset_id
        selection = client.patch(f"/api/datasets/{dataset_id}/target", json={"target_column": "target"})
        assert selection.status_code == 200
        assert selection.json()["target_column"] == "target"
        assert selection.json()["target_summary"]["value_counts"] == [{"value": "0", "count": 2}, {"value": "1", "count": 2}]
        assert client.patch(f"/api/datasets/{dataset_id}/target", json={"target_column": "missing"}).status_code == 422
        assert client.patch(f"/api/datasets/{dataset_id}/target", json={"target_column": "target", "unexpected": True}).status_code == 422
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get(f"/api/datasets/{dataset_id}").json()["target_column"] == "target"
        cleared = restarted.patch(f"/api/datasets/{dataset_id}/target", json={"target_column": None})
        assert cleared.json()["target_summary"] is None


@pytest.mark.parametrize("content,filename,status", [
    (b"", "empty.csv", 422),
    (b"a,b\n", "header.csv", 422),
    (b"a,a\n1,2", "duplicate.csv", 422),
    (b"a,\n1,2", "blank-header.csv", 422),
    (b"a,b\n1,2,3", "wide-row.csv", 422),
    (b'a,b\n"unterminated,2', "broken.csv", 422),
    (b"a,b\n1,\x00", "binary.csv", 422),
    (CSV, "sample.txt", 400),
    (CSV, "../escape.csv", 400),
    (CSV, "..\\escape.csv", 400),
    (CSV, "C:\\escape.csv", 400),
])
def test_rejects_invalid_uploads_without_registering(content, filename, status):
    settings = isolated_settings()
    with TestClient(create_app(settings)) as client:
        assert upload(client, content, filename).status_code == status
        assert client.get("/api/datasets").json()["datasets"] == []
        assert list(settings.datasets_dir.iterdir()) == []
        assert not (settings.project_root / "escape.csv").exists()


def test_size_limit_with_and_without_content_length():
    settings = isolated_settings(max_upload_mb=1)
    with TestClient(create_app(settings)) as client:
        payload = b"x\n" + b"1\n" * 600_000
        assert upload(client, payload).status_code == 413
        def chunks():
            yield payload[:700_000]
            yield payload[700_000:]
        assert upload(client, chunks()).status_code == 413
        assert client.get("/api/datasets").json()["datasets"] == []


def test_row_and_column_limits():
    with TestClient(create_app(isolated_settings(max_dataset_rows=2))) as client:
        assert upload(client).status_code == 413
    with TestClient(create_app(isolated_settings(max_dataset_columns=2))) as client:
        assert upload(client).status_code == 413


def test_gb18030_quoted_fields_and_html_treated_as_data():
    payload = '城市,备注,值\r\n北京,"逗号,换行\n仍为同一条记录",2\r\n上海,<script>alert(1)</script>,3\r\n'.encode("gb18030")
    with TestClient(create_app(isolated_settings())) as client:
        response = upload(client, payload, "中文.csv")
        assert response.status_code == 201
        data = response.json()
        assert data["encoding"] == "gb18030"
        assert data["row_count"] == 2
        assert data["preview"][1][1] == "<script>alert(1)</script>"
        assert "innerHTML" not in client.get("/static/dashboard.js").text


def test_unknown_id_content_type_and_origin():
    with TestClient(create_app(isolated_settings())) as client:
        assert client.get("/api/datasets/not-a-dataset").status_code == 404
        assert client.get("/api/datasets/dataset_" + "a" * 32).status_code == 404
        assert client.post("/api/datasets?filename=test.csv", json={"a": 1}).status_code == 415
        assert client.post("/api/datasets?filename=test.csv", content=CSV, headers={"content-type": "text/csv", "origin": "https://evil.example"}).status_code == 403


def test_bounded_preview_and_numeric_overflow_is_json_safe():
    content = b"number,text\n" + b"1," + b"x" * 600 + b"\n" + b"2,y\n" * 25
    with TestClient(create_app(isolated_settings())) as client:
        response = upload(client, content)
        assert response.status_code == 201
        data = response.json()
        assert len(data["preview"]) == 20
        assert len(data["preview"][0][1]) == 500
        assert upload(client, b"number\n1e308\n-1e308\n").status_code == 201
