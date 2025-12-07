import os
from pathlib import Path
from alembic import command
from alembic.config import Config


def test_alembic_upgrade_tmpdb(tmp_path):
    db_file = tmp_path / "migrations.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"
    base = Path(__file__).resolve().parents[1]
    cfg = Config(str(base / "alembic.ini"))
    cfg.set_main_option("script_location", str(base / "alembic"))
    command.upgrade(cfg, "head")
    assert db_file.exists()
