import pkgutil
import importlib

from .model_meta import get_model_meta
from .model_meta import ModelMeta

# Scan all packages
for _, module_name, is_pkg in pkgutil.walk_packages(__path__):
    if is_pkg:
        importlib.import_module(".{}".format(module_name), __package__)

__all__ = ["get_model_meta", "ModelMeta"]
