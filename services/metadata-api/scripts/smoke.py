import os
from pathlib import Path


def load_app(db_url: str):
    os.environ["DATABASE_URL"] = db_url
    base = Path(__file__).resolve().parents[1]
    main_path = base / "app" / "main.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("app_main", str(main_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


def run():
    db_file = Path(__file__).resolve().parent / "smoke.db"
    app = load_app(f"sqlite:///{db_file}")
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    c = client.post("/videos", json={"title": "Smoke", "status": "new"})
    assert c.status_code == 200
    vid = c.json()["id"]
    l = client.get("/videos")
    assert l.status_code == 200 and len(l.json()) >= 1
    g = client.get(f"/videos/{vid}")
    assert g.status_code == 200
    print("smoke: ok")


if __name__ == "__main__":
    run()
