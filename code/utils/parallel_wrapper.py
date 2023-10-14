import torch
from torch import nn
import torch.distributed as dist
from torch.multiprocessing import spawn

from logzero import logger
import os

logger.info('Distribution is multi-threaded but not multi-processor')
class DistributedDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)