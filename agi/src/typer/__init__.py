from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType


def _load_real_module() -> ModuleType | None:
    module_name = __name__
    current_file = os.path.abspath(__file__)
    for entry in list(sys.path):
        if not entry:
            continue
        try:
            candidate_dir = os.path.join(entry, module_name.replace('.', os.sep))
            candidate_file = os.path.join(candidate_dir, '__init__.py')
        except TypeError:
            continue
        if not os.path.exists(candidate_file):
            continue
        try:
            if os.path.samefile(candidate_file, current_file):
                continue
        except OSError:
            continue
        spec = importlib.util.spec_from_file_location(
            module_name,
            candidate_file,
            submodule_search_locations=[candidate_dir],
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
    return None


_real_module = _load_real_module()
if _real_module is not None:
    globals().update(_real_module.__dict__)
    globals().setdefault('USING_TYPER_FALLBACK', False)
else:
    from ._fallback import *  # type: ignore[F401,F403]
