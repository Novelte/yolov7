# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
PyTorch utils
"""

import math
import os
import platform
import subprocess
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from torchinfo import summary
import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.general import LOGGER, check_version, colorstr, file_date, git_describe

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', message='User provided device_type of \'cuda\', but CUDA is not available. Disabling')


def smart_inference_mode(torch_1_9=check_version(torch.__version__, '1.9.0')):
    # Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator
    def decorate(fn):
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)

    return decorate


def smartCrossEntropyLoss(label_smoothing=0.0):
    # Returns nn.CrossEntropyLoss with label smoothing enabled for torch>=1.10.0
    if check_version(torch.__version__, '1.10.0'):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        LOGGER.warning(f'WARNING: label smoothing {label_smoothing} requires torch>=1.10.0')
    return nn.CrossEntropyLoss()


def smart_DDP(model):
    # Model DDP creation with checks
    assert not check_version(torch.__version__, '1.12.0', pinned=True), \
        'torch==1.12.0 torchvision==0.13.0 DDP training is not supported due to a known issue. ' \
        'Please upgrade or downgrade torch to use DDP. See https://github.com/ultralytics/yolov5/issues/8395'
    if check_version(torch.__version__, '1.11.0'):
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
    else:
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


def reshape_classifier_output(model, n=1000):
    # Update a TorchVision classification model to class count 'n' if required
    from models.common import Classify
    name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]  # last module
    if isinstance(m, Classify):  # YOLOv5 Classify() head
        if m.linear.out_features != n:
            m.linear = nn.Linear(m.linear.in_features, n)
    elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
        if m.out_features != n:
            setattr(model, name, nn.Linear(m.in_features, n))
    elif isinstance(m, nn.Sequential):
        types = [type(x) for x in m]
        if nn.Linear in types:
            i = types.index(nn.Linear)  # nn.Linear index
            if m[i].out_features != n:
                m[i] = nn.Linear(m[i].in_features, n)
        elif nn.Conv2d in types:
            i = types.index(nn.Conv2d)  # nn.Conv2d index
            if m[i].out_channels != n:
                m[i] = nn.Conv2d(m[i].in_channels, n, m[i].kernel_size, m[i].stride, bias=m[i].bias)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def device_count():
    # Returns number of CUDA devices available. Safe version of torch.cuda.device_count(). Supports Linux and Windows
    assert platform.system() in ('Linux', 'Windows'), 'device_count() only supported on Linux or Windows'
    try:
        cmd = 'nvidia-smi -L | wc -l' if platform.system() == 'Linux' else 'nvidia-smi -L | find /c /v ""'  # Windows
        return int(subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1])
    except Exception:
        return 0


# def select_device(device='', batch_size=0, newline=True):
#     # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
#     s = f'YOLOv5 🚀 {git_describe() or file_date()} Python-{platform.python_version()} torch-{torch.__version__} '
#     device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
#     cpu = device == 'cpu'
#     mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
#     if cpu or mps:
#         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
#     elif device:  # non-cpu device requested
#         os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
#         assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
#             f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

#     if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
#         devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
#         n = len(devices)  # device count
#         if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
#             assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
#         space = ' ' * (len(s) + 1)
#         for i, d in enumerate(devices):
#             p = torch.cuda.get_device_properties(i)
#             s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
#         arg = 'cuda:0'
#     elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
#         s += 'MPS\n'
#         arg = 'mps'
#     else:  # revert to CPU
#         s += 'CPU\n'
#         arg = 'cpu'

#     if not newline:
#         s = s.rstrip()
#     LOGGER.info(s)
#     return torch.device(arg)

def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOR 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    LOGGER.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(input, ops, n=10, device=None):
    """ YOLOv5 speed/memory/FLOPs profiler
    Usage:
        input = torch.randn(16, 3, 640, 640)
        m1 = lambda x: x * torch.sigmoid(x)
        m2 = nn.SiLU()
        profile(input, [m1, m2], n=100)  # profile over 100 iterations
    """
    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m  # device
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPs
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:  # no backward method
                        # print(e)  # for debug
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else 'list' for x in (x, y))  # shapes
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # parameters
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    LOGGER.info(f'Model pruned to {sparsity(model):.3g} global sparsity')


def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, imgsz=640):

    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32  # max stride
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
        fs = f', {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs'  # 640x640 GFLOPs
    except Exception:
        fs = ''

    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    LOGGER.info(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")
    summary(model, (1, int(model.yaml.get('ch', 3)),imgsz,imgsz))


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # Scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                g[1].append(v.im.implicit)
            else:
                for iv in v.im:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                g[1].append(v.ia.implicit)
            else:
                for iv in v.ia:
                    g[1].append(iv.implicit)

    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias")
    return optimizer

class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
        # is this method that is overwritten by the sub-class
        # This original goal of this method was for tensor sanity checks
        # If you're ok bypassing those sanity checks (eg. if you trust your inference
        # to provide the right dimensional inputs), then you can just use this method
        # for easy conversion from SyncBatchNorm
        # (unfortunately, SyncBatchNorm does not store the original class - if it did
        #  we could return the one that was originally created)
        return

def revert_sync_batchnorm(module):
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        new_cls = BatchNormXd
        module_output = BatchNormXd(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output

def smart_hub_load(repo='ultralytics/yolov5', model='yolov5s', **kwargs):
    # YOLOv5 torch.hub.load() wrapper with smart error/issue handling
    if check_version(torch.__version__, '1.9.1'):
        kwargs['skip_validation'] = True  # validation causes GitHub API rate limit errors
    if check_version(torch.__version__, '1.12.0'):
        kwargs['trust_repo'] = True  # argument required starting in torch 0.12
    try:
        return torch.hub.load(repo, model, **kwargs)
    except Exception:
        return torch.hub.load(repo, model, force_reload=True, **kwargs)


def smart_resume(ckpt, optimizer, ema=None, weights='yolov5s.pt', epochs=300, resume=True):
    # Resume training from a partially trained checkpoint
    best_fitness = 0.0
    start_epoch = ckpt['epoch'] + 1
    if ckpt['optimizer'] is not None:
        optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
        best_fitness = ckpt['best_fitness']
    if ema and ckpt.get('ema'):
        ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # EMA
        ema.updates = ckpt['updates']
    if resume:
        assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.\n' \
                                f"Start a new training without --resume, i.e. 'python train.py --weights {weights}'"
        LOGGER.info(f'Resuming training from {weights} from epoch {start_epoch} to {epochs} total epochs')
    if epochs < start_epoch:
        LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
        epochs += ckpt['epoch']  # finetune additional epochs
    return best_fitness, start_epoch, epochs


class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.')
        return stop


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

class NNDctDetect(nn.Module):

    def __init__(self, m, nl):
        super(NNDctDetect, self).__init__()
        self.m = m
        self.nl = nl
        from pytorch_nndct.nn import QuantStub, DeQuantStub
        dequant = []
        for i in range(self.nl):
            dequant.append(DeQuantStub())
        self.dequant = nn.ModuleList(dequant)
            
    
    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.dequant[i](self.m[i](x[i]))  # conv
        return x

class NNDctSegment(nn.Module):
    def __init__(self, m, nl, ia, proto):
        super(NNDctSegment, self).__init__()
        self.m = m
        self.nl = nl
        self.ia = deepcopy(ia)
        self.proto = deepcopy(proto)
        
        from pytorch_nndct.nn import DeQuantStub
        self.dequant = nn.ModuleList(DeQuantStub() for _ in range(nl))
        self.proto_dequant = DeQuantStub()

    def forward(self, x):
        p = self.proto_dequant(self.proto(x[0]))
        for i in range(self.nl):
            x[i] = self.dequant[i](self.m[i](self.ia[i](x[i])))  # conv
        return (p, x)

class NNDctModel(nn.Module):

    def __init__(self, model=None, device=None, img_size=(640,640), nndct_bitwidth=8, output_dir='nndct'): 
        super(NNDctModel, self).__init__()
        model = deepcopy(model)
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--quant_mode', default='calib', choices=['float', 'calib', 'test'], help='quant mode')
        parser.add_argument('--dump_model', action='store_true', help='dump model')
        opt, _ = parser.parse_known_args()
        self.quant_mode = opt.quant_mode
        self.dump_model = opt.dump_model
        if self.dump_model:
            if self.quant_mode != 'test':
                raise ValueError
        from models.yolo import Detect, IDetect, Segment, ISegment, IAuxDetect, IKeypoint, IBin
        from pytorch_nndct.apis import torch_quantizer
        
        print(" Convert model to Traced-model... ") 
        self.stride = model.stride
        self.names = model.names
        self.model = model

        # self.model = revert_sync_batchnorm(self.model)
        # fuse multi time will cause invalid param value
        # with torch.no_grad():
        #     self.model = model.fuse() # make sure the model is fused
        from pytorch_nndct.nn import QuantStub, DeQuantStub
        from pytorch_nndct import QatProcessor
        quant = QuantStub()
        setattr(self.model, 'quant', quant)
        with torch.no_grad():
            for k, v in self.model.named_parameters():
                v.requires_grad = True  # train all layers
                if 'implicit' in k:
                    print('freezing %s' % k)
                    v.requires_grad = False
        self.model.to('cpu')
        self.model.eval()

        self.detect_layer = self.model.model[-1]
        if type(self.detect_layer) in (Detect, IDetect):
            modules = list(self.model.model)
            m_ = NNDctDetect(self.detect_layer.m, self.detect_layer.nl)
            m_.type = 'NNDctDetect'
            m_.i = modules[-1].i
            modules[-1].i += 1
            m_.f = modules[-1].f
            m_.np = sum([x.numel() for x in m_.parameters()])  # number params
            modules.insert(-1, m_)
            self.detect_layer.m = nn.ModuleList(nn.Identity() for _ in self.detect_layer.m)
            modules[-1].f = -1 # from previous
            modules[-1].np = sum([x.numel() for x in modules[-1].parameters()])
            self.model.model = nn.Sequential(*modules)

        elif type(self.detect_layer) in (Segment, ISegment):
            modules = list(self.model.model)

            m_ = NNDctSegment(self.detect_layer.m, self.detect_layer.nl, 
                              self.detect_layer.ia, self.detect_layer.proto)
            m_.type = 'NNDctSegment'
            m_.i = modules[-1].i
            modules[-1].i += 1
            m_.f = modules[-1].f
            m_.np = sum([x.numel() for x in m_.parameters()])  # number params
            modules.insert(-1, m_)

            # Make m Identity
            self.detect_layer.m = nn.ModuleList(nn.Identity() for _ in self.detect_layer.m)
            self.detect_layer.ia = nn.ModuleList(nn.Identity() for _ in self.detect_layer.ia)
            self.detect_layer.im = nn.ModuleList(nn.Identity() for _ in self.detect_layer.im)
            self.detect_layer.proto = nn.Identity()
            setattr(self.detect_layer, 'dequant', True)
            setattr(modules[-1], 'dequant', True)
            modules[-1].f = -1 # from previous
            modules[-1].np = sum([x.numel() for x in modules[-1].parameters()])
            self.model.model = nn.Sequential(*modules)

        elif isinstance(self.detect_layer, (IAuxDetect, IKeypoint, IBin)):
            raise NotImplementedError

        self.model.traced = True
        self.output_dir = output_dir
        
        rand_example = torch.rand(1, 3, img_size, img_size)
        # Dry run
        self.model(rand_example)
        print("done dry run model")

        print(f"\n\n\n Modify Model:")
        self.model.info(verbose=False)

        if self.quant_mode == 'float':
            # traced_script_module = torch.jit.trace(self.model, rand_example, strict=False)
            #traced_script_module = torch.jit.script(self.model)
            # traced_script_module.save("traced_model.pt")
            # print(" traced_script_module saved! ")
            self.quantizer = None
            self.model = model
        else:
            quantizer = torch_quantizer(quant_mode=self.quant_mode,
                                        bitwidth=nndct_bitwidth,
                                        module=model,
                                        input_args=rand_example,
                                        output_dir=output_dir)
            quant_model = quantizer.quant_model
            self.quantizer = quantizer
            self.model = quant_model
        self.model.to(device)
        self.detect_layer.to(device)
        print(" model is traced! \n") 

    def forward(self, x, augment=False, profile=False):
        out = self.model(x)
        out = list(out)
        out = self.detect_layer(out)
        return out

    def export(self):
        if self.quant_mode == 'calib':
            self.quantizer.export_quant_config()
        elif self.quant_mode == 'test':
            self.quantizer.export_onnx_model(output_dir=self.output_dir, verbose=False, dynamic_batch=True, opset_version=12)
            self.quantizer.export_torch_script(output_dir=self.output_dir, verbose=False)
            self.quantizer.export_xmodel(output_dir=self.output_dir, deploy_check=True, dynamic_batch=True)

def get_qat_model(model, device=None, img_size=640, nndct_bitwidth=8, output_dir='nndct'):
    from models.yolo import Detect, IDetect, Segment, ISegment, IAuxDetect, IKeypoint, IBin
    from pytorch_nndct.nn import QuantStub, DeQuantStub
    from pytorch_nndct import QatProcessor
    model = deepcopy(model)
    quant = QuantStub()
    model.train()
    setattr(model, 'quant', quant)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if 'implicit' in k:
            print('freezing %s' % k)
            v.requires_grad = False

    detect_layer = model.model[-1]
    if type(self.detect_layer) in (Detect, IDetect):
        modules = list(model.model)
        m_ = NNDctDetect(detect_layer.m, detect_layer.nl)
        m_.type = 'NNDctDetect'
        m_.i = modules[-1].i
        modules[-1].i += 1
        m_.f = modules[-1].f
        m_.np = sum([x.numel() for x in m_.parameters()])  # number params
        modules.insert(-1, m_)
        detect_layer.m = nn.ModuleList(nn.Identity() for _ in detect_layer.m)
        modules[-1].f = -1 # from previous
        modules[-1].np = sum([x.numel() for x in modules[-1].parameters()])
        model.model = nn.Sequential(*modules)
    elif type(detect_layer) in (Segment, ISegment):
        modules = list(model.model)
        m_dequant = nn.ModuleList(DeQuantStub() for x in range(modules[-1].nl))
        setattr(modules[-1], 'dequant', m_dequant)
        setattr(modules[-1].proto.cv1, 'dequant', DeQuantStub())
        setattr(modules[-1].proto.cv2, 'dequant', DeQuantStub())
        setattr(modules[-1].proto.cv3, 'dequant', DeQuantStub())
    elif isinstance(detect_layer, (IAuxDetect, IKeypoint, IBin)):
        raise NotImplementedError
    model.traced = True   
    model.to(device)
    # Image sizes
    rand_example = torch.rand(1, 3, img_size, img_size).to(next(model.parameters()).device)
    # Dry run
    model(rand_example)
    qat_processor = QatProcessor(model, (rand_example,), bitwidth=nndct_bitwidth, mix_bit=False)
    qat_model = qat_processor.trainable_model()
    qat_model.stride = model.stride
    qat_model.names = model.names
    qat_model.origin_forward = qat_model.forward
    def forward(instance, x):
        x = instance.origin_forward(x)
        x = detect_layer(x)
        return x
    from types import MethodType
    qat_model.new_forward = MethodType(forward, qat_model)
    qat_model.forward = qat_model.new_forward
    return qat_model, qat_processor
