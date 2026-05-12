#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_v2.models.v2_model import V2MultiHypothesisHomographyNet


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, H, W = 1, 32, 48
    model = V2MultiHypothesisHomographyNet(num_hypotheses=4).to(device)
    model.eval()
    ia = torch.rand(B, 1, H, W, device=device)
    ib = torch.rand(B, 1, H, W, device=device)
    with torch.no_grad():
        out = model.forward_pair(ia, ib)
    assert out['Hs'].shape == (B, 4, 3, 3)
    assert out['offsets'].shape == (B, 4, 8)
    assert out['assignments'].shape == (B, 4, H, W)
    assert torch.isfinite(out['Hs']).all()
    print({
        'ok': True,
        'Hs_shape': tuple(out['Hs'].shape),
        'offsets_shape': tuple(out['offsets'].shape),
        'assignments_shape': tuple(out['assignments'].shape),
    })


if __name__ == '__main__':
    main()
