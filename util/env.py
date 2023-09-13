import logging
import os

import torch

from . import consts

logger = logging.getLogger(__name__)


def get_device():
    try:
        logger.info(torch.cuda.get_device_name(torch.cuda.current_device()))
        device = torch.device("cuda")
    except:
        device = torch.device("cpu")

    return device


def get_cluster_info():
    return int(os.environ[consts.ENVIRON_RANK]), int(os.environ[consts.ENVIRON_WORLD_SIZE])