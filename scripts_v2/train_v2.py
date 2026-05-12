#!/usr/bin/env python
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_v2.models.v2_model import V2MultiHypothesisHomographyNet
from src_v2.engine.trainer_v2 import V2Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/train_v2.yaml')
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = V2MultiHypothesisHomographyNet(
        feature_channels=int(cfg['model'].get('feature_channels', 1)),
        num_hypotheses=int(cfg['model'].get('num_hypotheses', 4)),
    )
    trainer = V2Trainer(model, cfg)
    trainer.train()


if __name__ == '__main__':
    main()
