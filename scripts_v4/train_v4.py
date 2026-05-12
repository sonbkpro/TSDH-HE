from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config
from src_v4.engine.trainer_v4 import TrainerV4


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/train_v4.yaml')
    p.add_argument('--resume', default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    TrainerV4(cfg, resume=args.resume).train()


if __name__ == '__main__':
    main()
