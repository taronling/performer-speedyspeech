import torch
from torch import nn
import torch.distributed as dist
from torch.multiprocessing import spawn

from logzero import logger
import os

if torch.cuda.is_available(): 
    logger.info('Multi-processing successful')

    world_size = torch.cuda.device_count() 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='mpi', rank=0, world_size=world_size)

    class DistributedDataParallel(nn.parallel.DistributedDataParallel):
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
            
else:
    logger.info('Distribution is multi-threaded but not multi-processor')
    class DistributedDataParallel(nn.DataParallel):
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)