from __future__ import annotations

import importlib
import sys

_module = sys.modules.get('typer')
_use_fallback = bool(getattr(_module, 'USING_TYPER_FALLBACK', False))

if not _use_fallback:
    real = importlib.import_module('typer.testing')
    globals().update(real.__dict__)
    __all__ = getattr(real, '__all__', [name for name in real.__dict__ if not name.startswith('_')])
else:
    from ._testing_fallback import *  # type: ignore[F401,F403]
    __all__ = [name for name in globals().keys() if not name.startswith('_')]
