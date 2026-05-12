#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config
from src.engine.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/train_default.yaml')
    ap.add_argument('--resume', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    Trainer(cfg, resume=args.resume).train()

if __name__ == '__main__':
    main()
