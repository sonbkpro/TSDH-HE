from __future__ import annotations
from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer=None, scheduler=None, step=0, config=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {'model': model.state_dict(), 'step': step, 'config': config}
    if optimizer is not None: payload['optimizer'] = optimizer.state_dict()
    if scheduler is not None: payload['scheduler'] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model'], strict=True)
    if optimizer is not None and 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt
