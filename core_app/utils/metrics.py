"""
Metric utilities for training and evaluation.
"""
import torch
import time
from typing import Dict, Any


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class MetricLogger:
    """Logs metrics during training."""
    
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                f"{name}: {meter.avg:.4f}"
            )
        return self.delimiter.join(loss_str)
    
    def avg_dict(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}


class SmoothedValue:
    """Track a series of values and provide access to smoothed values."""
    
    def __init__(self, window_size=20):
        self.deque = []
        self.count = 0
        self.sum = 0.0
        self.window_size = window_size
    
    def update(self, value):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.popleft()
        self.count += 1
        self.sum += value
    
    @property
    def median(self):
        import numpy as np
        return np.median(self.deque)
    
    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0
    
    @property
    def smoothed(self):
        if len(self.deque) == 0:
            return 0
        return sum(self.deque) / len(self.deque)


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
