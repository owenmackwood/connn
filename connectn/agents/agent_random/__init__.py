import sys
import importlib

__dependent_modules__ = [f"{__name__}.{name}" for name in ["main",]]

for module in __dependent_modules__:
    if module in sys.modules:
        importlib.reload(sys.modules[module])

from .main import generate_move
