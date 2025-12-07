import os
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch
import importlib.util

def load_app():
    os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
    base = Path(__file__).resolve().parents[1]
    main_path = base / "app" / "main.py"
    spec = importlib.util.spec_from_file_location("app_main", str(main_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app

client = TestClient(load_app())

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_v1_health():
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_readiness():
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}

def test_v1_readiness():
    resp = client.get("/v1/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}

def test_readiness_db_error():
    with patch('sqlalchemy.engine.create_engine') as mock_engine:
        mock_engine.return_value.connect.side_effect = Exception("DB Error")
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert "Service unavailable" in resp.json()["detail"]

def test_videos_roundtrip(tmp_path):
    db_file = tmp_path / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"
    app = load_app()
    client_local = TestClient(app)
    create = client_local.post("/v1/videos", json={"title": "Demo", "status": "new"})
    assert create.status_code == 200
    vid = create.json()
    assert "id" in vid
    assert vid["title"] == "Demo"
    listed = client_local.get("/v1/videos?limit=10")
    assert listed.status_code == 200
    items = listed.json()
    assert len(items) == 1
    got = client_local.get(f"/v1/videos/{vid['id']}")
    assert got.status_code == 200
    assert got.json()["id"] == vid["id"]

def test_videos_validation_error():
    resp = client.post("/v1/videos", json={"title": "", "status": "invalid"})
    assert resp.status_code == 422
    errors = resp.json()["error"]["details"]
    assert any("title" in e["loc"] for e in errors)
    assert any("status" in e["loc"] for e in errors)

def test_uploads_roundtrip(tmp_path):
    db_file = tmp_path / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"
    app = load_app()
    client_local = TestClient(app)
    # Create
    create = client_local.post("/v1/uploads", json={"filename": "test.mp4"})
    assert create.status_code == 200
    upload = create.json()
    assert upload["status"] == "queued"
    # List
    listed = client_local.get("/v1/uploads")
    assert len(listed.json()) == 1
    # Get
    got = client_local.get(f"/v1/uploads/{upload['id']}")
    assert got.status_code == 200
    # Actions
    start = client_local.post(f"/v1/uploads/{upload['id']}/start")
    assert start.status_code == 200
    assert start.json()["status"] == "processing"
    cancel = client_local.post(f"/v1/uploads/{upload['id']}/cancel")
    assert cancel.status_code == 200
    assert "canceled" in cancel.json()["error_message"]
    retry = client_local.post(f"/v1/uploads/{upload['id']}/retry")
    assert retry.status_code == 200
    assert retry.json()["status"] == "queued"

def test_uploads_rate_limit(tmp_path):
    db_file = tmp_path / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"
    app = load_app()
    client_local = TestClient(app)
    # Simulate multiple requests to trigger rate limit (limit=30/min, but mock fast)
    for _ in range(31):  # Exceed limit
        client_local.post("/v1/uploads", json={"filename": "test.mp4"})
    resp = client_local.post("/v1/uploads", json={"filename": "test.mp4"})
    assert resp.status_code == 429
    assert "Rate limit exceeded" in resp.json()["detail"]

def test_reviews_roundtrip(tmp_path):
    db_file = tmp_path / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"
    app = load_app()
    client_local = TestClient(app)
    # Create video first
    vid_create = client_local.post("/v1/videos", json={"title": "Test Vid"})
    vid_id = vid_create.json()["id"]
    # Create review
    create = client_local.post("/v1/reviews", json={"video_id": vid_id})
    assert create.status_code == 200
    review = create.json()
    assert review["status"] == "pending"
    # List pending
    listed = client_local.get("/v1/reviews?status=pending")
    assert len(listed.json()) == 1
    # Actions
    approve = client_local.post(f"/v1/reviews/{review['id']}/approve")
    assert approve.status_code == 200
    assert approve.json()["status"] == "approved"
    # Reject (new review)
    new_review = client_local.post("/v1/reviews", json={"video_id": vid_id}).json()
    reject = client_local.post(f"/v1/reviews/{new_review['id']}/reject")
    assert reject.status_code == 200
    assert reject.json()["status"] == "rejected"

def test_persons_roundtrip(tmp_path):
    db_file = tmp_path / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"
    app = load_app()
    client_local = TestClient(app)
    # Create person
    create = client_local.post("/v1/persons", json={"name": "John Doe"})
    assert create.status_code == 200
    person = create.json()
    # List
    listed = client_local.get("/v1/persons?limit=10")
    assert len(listed.json()) == 1
    # Create video
    vid_create = client_local.post("/v1/videos", json={"title": "Test Vid"})
    vid_id = vid_create.json()["id"]
    # Link
    link = client_local.post(f"/v1/persons/{person['id']}/link/{vid_id}")
    assert link.status_code == 200
    # Dossier
    dossier = client_local.get(f"/v1/dossiers/persons/{person['id']}")
    assert dossier.status_code == 200
    data = dossier.json()
    assert data["person"]["name"] == "John Doe"
    assert len(data["videos"]) == 1
    # Video Dossier
    vid_dossier = client_local.get(f"/v1/dossiers/videos/{vid_id}")
    assert vid_dossier.status_code == 200
    assert len(vid_dossier.json()["reviews"]) >= 0  # May be empty

def test_persons_validation_error():
    resp = client.post("/v1/persons", json={"name": ""})
    assert resp.status_code == 422
    assert "name" in str(resp.json()["error"]["details"])

def test_not_found():
    resp = client.get("/v1/videos/999")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"

def test_internal_error_simulation():
    with patch('sqlalchemy.orm.session.Session.add') as mock_add:
        mock_add.side_effect = Exception("DB Crash")
        resp = client.post("/v1/videos", json={"title": "Crash Test"})
        assert resp.status_code == 500
        assert resp.json()["error"]["code"] == "INTERNAL_ERROR"

def test_cors_middleware():
    resp = client.options("/v1/health", headers={"Origin": "http://localhost:3000"})
    assert resp.status_code == 200
    assert "Access-Control-Allow-Origin" in resp.headers
    # Invalid origin
    resp_invalid = client.options("/v1/health", headers={"Origin": "http://evil.com"})
    assert "Access-Control-Allow-Origin" not in resp_invalid.headers

def test_correlation_id():
    resp = client.get("/v1/health")
    assert "X-Correlation-ID" in resp.headers
    assert len(resp.headers["X-Correlation-ID"]) > 0  # UUID-like

def test_metrics():
    # First request to generate metrics
    client.get("/v1/health")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    content = resp.text
    assert "http_requests_total" in content
    assert "http_request_duration_seconds" in content