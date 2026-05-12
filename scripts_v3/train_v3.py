from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config
from src_v3.engine.trainer_v3 import TrainerV3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_v3.yaml')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    TrainerV3(cfg, resume=args.resume).train()


if __name__ == '__main__':
    main()
