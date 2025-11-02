from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _restore_module(name: str, module) -> None:
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


def test_pydantic_fallback_basics() -> None:
    fallback = importlib.import_module("agi.src.pydantic._fallback")
    assert fallback.USING_PYDANTIC_FALLBACK is True

    class Item(fallback.BaseModel):
        value: int

    instance = Item(value=7)
    assert instance.model_dump()["value"] == 7


def test_pydantic_loader_prefers_real_package(tmp_path: Path) -> None:
    backup = {name: sys.modules.get(name) for name in ("pydantic", "agi.src.pydantic")}
    for name in list(backup):
        sys.modules.pop(name, None)

    fake_root = tmp_path / "realdeps"
    package_dir = fake_root / "pydantic"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "USING_PYDANTIC_FALLBACK = False\nFLAG = 'real'\nclass BaseModel:\n    def __init__(self, **kwargs):\n        self.payload = kwargs\n    def model_dump(self):\n        return dict(self.payload)\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(fake_root))

    try:
        module = importlib.import_module("pydantic")
        assert getattr(module, "FLAG", None) == "real"
        assert module.USING_PYDANTIC_FALLBACK is False
        instance = module.BaseModel(answer=42)
        assert instance.model_dump()["answer"] == 42
    finally:
        sys.path.remove(str(fake_root))
        for name in ("pydantic", "agi.src.pydantic"):
            sys.modules.pop(name, None)
        for name, module in backup.items():
            _restore_module(name, module)
        if backup["pydantic"] is None:
            importlib.import_module("pydantic")


def test_typer_fallback_basics() -> None:
    fallback = importlib.import_module("agi.src.typer._fallback")
    assert fallback.USING_TYPER_FALLBACK is True
    app = fallback.Typer()

    @app.command()
    def ping() -> None:  # pragma: no cover - executed via Typer
        pass

    commands = getattr(app, "registered_commands", None)  # type: ignore[attr-defined]
    if commands is None:
        commands = list(getattr(app, "_commands").keys())
    assert "ping" in list(commands)


def test_typer_loader_prefers_real_package(tmp_path: Path) -> None:
    backup = {
        name: sys.modules.get(name)
        for name in ("typer", "agi.src.typer", "typer.testing")
    }
    for name in list(backup):
        sys.modules.pop(name, None)

    fake_root = tmp_path / "realdeps"
    package_dir = fake_root / "typer"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "USING_TYPER_FALLBACK = False\nFLAG = 'real'\nclass Typer:\n    def __init__(self):\n        self.commands = {}\n    def command(self, name=None):\n        def decorator(fn):\n            self.commands[name or fn.__name__] = fn\n            return fn\n        return decorator\n    @property\n    def registered_commands(self):\n        return list(self.commands)\n",
        encoding="utf-8",
    )
    (package_dir / "testing.py").write_text(
        "class DummyRunner:\n    def __init__(self):\n        self.invocations = []\n\n__all__ = ['DummyRunner']\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(fake_root))

    try:
        module = importlib.import_module("typer")
        assert getattr(module, "FLAG", None) == "real"
        assert module.USING_TYPER_FALLBACK is False
        app = module.Typer()

        @app.command()
        def hello() -> None:  # pragma: no cover - executed via fake Typer
            pass

        assert "hello" in app.registered_commands
        testing = importlib.import_module("typer.testing")
        assert hasattr(testing, "DummyRunner")
    finally:
        sys.path.remove(str(fake_root))
        for name in ("typer", "agi.src.typer", "typer.testing"):
            sys.modules.pop(name, None)
        for name, module in backup.items():
            _restore_module(name, module)
        if backup["typer"] is None:
            importlib.import_module("typer")
