# -*- coding: utf-8 -*-
from collections import defaultdict


__all__ = ["ModelMeta", "get_model_meta", "model"]


class ModelMeta(object):
    def __init__(self, config_parser=None, data_loader=None, model_builder=None):
        self._config_parser = config_parser
        self._data_loader = data_loader
        self._model_builder = model_builder

    @property
    def arch_config_parser(self):
        return self._config_parser

    def set_arch_config_parser(self, parser):
        self._check(self._config_parser, "Config parser has been set")
        self._config_parser = parser

    @property
    def data_loader_builder(self):
        return self._data_loader

    def set_data_loader_builder(self, loader):
        self._check(self._data_loader, "Data loader builder has been set")
        self._data_loader = loader

    @property
    def model_builder(self):
        return self._model_builder

    def set_model_builder(self, model_builder):
        self._check(self._model_builder, "Model builder has been set")
        self._model_builder = model_builder

    def _check(self, value, message):
        if value is not None:
            raise ValueError(message)

    def __setitem__(self, k, v):
        self.k = v


class MetaType(object):
    ConfigParser = ModelMeta.set_arch_config_parser
    ModelBuilder = ModelMeta.set_model_builder


class _ModelMetaRegister(object):
    def __init__(self):
        self._register_map = defaultdict(ModelMeta)

    def get(self, name):
        return self._register_map.get(name)

    def __call__(self, name, setter):
        model_meta = self._register_map[name]

        def _executor(func):
            setter(model_meta, func)
            return func

        return _executor


model = _ModelMetaRegister()
get_model_meta = model.get
