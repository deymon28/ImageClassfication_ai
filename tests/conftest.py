import pytest


@pytest.fixture(autouse=True)
def isolated_app_dirs(tmp_path, monkeypatch):
    """Tests never write caches or logs into the user's folders."""
    monkeypatch.setenv("FRAMELABEL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FRAMELABEL_CACHE_DIR", str(tmp_path / "cache"))
