from pathlib import Path

from pitsense import config


def test_checkout_data_does_not_depend_on_launch_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("PITSENSE_DATA", raising=False)
    expected = Path(config.__file__).resolve().parents[2] / "data"
    monkeypatch.chdir(tmp_path)
    assert config.data_dir() == expected
    monkeypatch.chdir(expected.parent.parent)
    assert config.data_dir() == expected


def test_data_override_is_resolved_from_launch_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PITSENSE_DATA", "custom-data")
    assert config.data_dir() == tmp_path / "custom-data"


def test_installed_package_uses_local_data(tmp_path, monkeypatch):
    monkeypatch.delenv("PITSENSE_DATA", raising=False)
    monkeypatch.setattr(config, "__file__", str(tmp_path / "site-packages" / "pitsense" / "config.py"))
    monkeypatch.chdir(tmp_path)
    assert config.data_dir() == tmp_path / "data"
