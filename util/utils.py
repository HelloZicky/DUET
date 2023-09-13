from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist
import logging
from io import BytesIO
logger = logging.getLogger(__name__)


def setup_seed(seed):
    import numpy as np
    import random
    from torch.backends import cudnn
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        print("count / total: {} / {}".format(self.count, self.total))
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
        print("count / total: {} / {}".format(self.count, self.total))

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        # for meter in self.meters.values():
        for name, meter in self.meters.items():
            print("Measure: {}".format(name))
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, total_iter, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(total_iter))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj, i
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (total_iter - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(log_msg.format(
                        i, total_iter, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logger.info(log_msg.format(
                        i, total_iter, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('{} Total time: {}'.format(header, total_time_str))


def ce_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def bce_accuracy(output, target, threshold=0.5):
    with torch.no_grad():
        output = torch.nn.Sigmoid()(output)
        print(output)
        zeros = torch.zeros_like(output)
        ones = torch.ones_like(output)
        output = torch.where(output >= threshold, ones, zeros)
        print(output)
        pred = output * target
        # batch_size = target.size(0)
        # num_classes = target.size(1)

        TP = pred.flatten().sum(dtype=torch.float32)
        TPFP = output.flatten().sum(dtype=torch.float32)
        TPFN = target.flatten().sum(dtype=torch.float32)

        return TP * (100.0 / TPFP), TP * (100.0 / TPFN)


def kldiv_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        TPFN = target.flatten().sum(dtype=torch.float32)

        res = []
        for k in topk:
            values, indices = output.topk(k, 1, True, True)
            pred = torch.zeros_like(output)
            ones = torch.ones_like(output)
            pred.scatter_(1, indices, ones)
            match_result = pred * target

            TP = match_result.flatten().sum(dtype=torch.float32)
            TPFP = pred.flatten().sum(dtype=torch.float32)
            res.append((TP, TPFP))

        return res, TPFN


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    def print_pass(*args, **kwargs):
        pass

    __builtin__.print = print
    if not is_master:
        logger.info = print_pass


def init_distributed_mode(args, gpu, ngpus_per_node, world_size, rank):
    assert (torch.cuda.is_available())
    torch.cuda.set_device(gpu)
    args.gpu = gpu
    args.dist_rank = rank * ngpus_per_node + gpu
    args.world_size = world_size
    args.rank = rank
    if world_size == 1:
        logger.info('Not using distributed mode')
        args.distributed = False
        return

    args.dist_backend = 'nccl'
    dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.dist_rank)
    args.distributed = True

