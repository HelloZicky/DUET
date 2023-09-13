import numpy as np
import torch

from loader.loader_base import LoaderBase

from util import consts


class MetaSequenceDataLoader(LoaderBase):
    def __init__(self, table_name, slice_id, slice_count, is_train):
        super(MetaSequenceDataLoader, self).__init__(
            table_name=table_name,
            slice_id=slice_id,
            slice_count=slice_count,
            columns=[consts.FIELD_USER_ID, consts.FIELD_TARGET_ID, consts.FIELD_TRIGGER_SEQUENCE,
                     consts.FIELD_CLK_SEQUENCE, consts.FIELD_LABEL],
            is_train=is_train
        )

    def parse_data(self, data):
        data = data.strip("\n").split(";")
        return (
            int(data[0]),
            int(data[1]),
            np.fromstring(data[2], dtype=np.int32, sep=","),
            np.fromstring(data[3], dtype=np.int32, sep=","),
            float(data[4])
        )

    @staticmethod
    def batchify(data):
        return {
            consts.FIELD_USER_ID: torch.from_numpy(np.stack([item[0] for item in data], axis=0)),
            consts.FIELD_TARGET_ID: torch.from_numpy(np.stack([item[1] for item in data], axis=0)),
            consts.FIELD_TRIGGER_SEQUENCE: torch.from_numpy(np.stack([item[2] for item in data], axis=0)),
            consts.FIELD_CLK_SEQUENCE: torch.from_numpy(np.stack([item[3] for item in data], axis=0)),
            consts.FIELD_LABEL: torch.from_numpy(np.stack([item[4] for item in data], axis=0))
        }

