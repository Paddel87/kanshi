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
    db_file = Path(__file__).resolve().parent / "seed.db"
    app = load_app(f"sqlite:///{db_file}")
    from fastapi.testclient import TestClient
    client = TestClient(app)
    items = [
        {"title": "Intro", "status": "new"},
        {"title": "Scene 1", "status": "processing"},
        {"title": "Outro", "status": "done"},
    ]
    for it in items:
        r = client.post("/videos", json=it)
        assert r.status_code == 200
    l = client.get("/videos")
    assert l.status_code == 200 and len(l.json()) >= 3
    print("seed: ok")


if __name__ == "__main__":
    run()
