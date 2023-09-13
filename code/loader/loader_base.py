# coding=utf-8
import os
import sys
import logging

import torch
# import common_io
import pandas as pd


logger = logging.getLogger(__name__)


class LoaderBase(torch.utils.data.IterableDataset):
    def __init__(self, table_name, slice_id, slice_count, columns, is_train):
        super(LoaderBase).__init__()
        self._table_name = table_name
        self._slice_id_init = slice_id
        self._slice_count = slice_count
        self._is_train = is_train
        self._columns = columns
        print("dataset init")
        self._line_count = self._get_dataset_info()

        if self._is_train:
            self._repeat = 999999999
        else:
            self._repeat = 0

    def parse_data(self, data):
        raise NotImplementedError

    def batchify(self):
        raise NotImplementedError

    def _get_dataset_info(self):
        _line_count = 0
        with open(self._table_name, "r") as reader:
            for line in reader:
                _line_count += 1
        print("table name: ", self._table_name)
        print("_line_count: ", _line_count)
        return _line_count

    def _reopen_reader(self, original_reader):
        if original_reader is not None:
            original_reader.close()
        reader = open(self._table_name, "r")
        print("_reopen_reader")
        self._get_dataset_info()
        return reader

    def __iter__(self):
        with open(self._table_name, "r") as reader:
            for data in reader:
                sample = self.parse_data(data)
                yield sample
                