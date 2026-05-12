from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.video_pair_dataset import VideoFramePairDataset, LabeledPointPairsDataset
from src.utils.checkpoint import save_checkpoint
from src_v2.data.video_triplet_dataset import VideoFrameTripletDataset
from src_v2.engine.evaluator_v2 import evaluate_labeled_points_v2
from src_v2.losses.multi_homography_loss import V2MultiHomographyLoss, V2TemporalCycleLoss


def _sample_to_device(batch: dict, prefix: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch[f'{prefix}_org_images'].to(device, non_blocking=True).float(),
        batch[f'{prefix}_input_tensors'].to(device, non_blocking=True).float(),
        batch[f'{prefix}_h4p'].to(device, non_blocking=True).float(),
        batch[f'{prefix}_patch_indices'].to(device, non_blocking=True).float(),
    )


class V2Trainer:
    def __init__(self, model, cfg: dict):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.out_dir = Path(cfg['train']['out_dir'])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg.get('amp', False)) and self.device.type == 'cuda')
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=float(cfg['train']['lr']),
            betas=(float(cfg['train'].get('beta1', 0.9)), float(cfg['train'].get('beta2', 0.999))),
            eps=float(cfg['train'].get('eps', 1e-8)),
            weight_decay=float(cfg['train'].get('weight_decay', 1e-4)),
            amsgrad=bool(cfg['train'].get('amsgrad', True)),
        )
        self.sched = torch.optim.lr_scheduler.StepLR(
            self.optim,
            step_size=int(cfg['train'].get('lr_decay_every', 12000)),
            gamma=float(cfg['train'].get('lr_decay_gamma', 0.8)),
        )
        lcfg = cfg.get('loss', {})
        self.criterion = V2MultiHomographyLoss(
            margin=float(lcfg.get('triplet_margin', 1.0)),
            lambda_diversity=float(lcfg.get('diversity', 0.02)),
            lambda_entropy=float(lcfg.get('mask_entropy', 0.01)),
            lambda_balance=float(lcfg.get('mask_balance', 0.001)),
            diversity_scale=float(lcfg.get('diversity_scale', 16.0)),
        )
        self.cycle_loss = V2TemporalCycleLoss(weight=float(lcfg.get('temporal_cycle', 0.1)))

    def _make_dataset(self):
        dcfg = self.cfg['data']
        if bool(self.cfg['train'].get('use_temporal_cycle', False)):
            return VideoFrameTripletDataset(
                dcfg['train_video_dir'],
                dcfg['crop_h'], dcfg['crop_w'],
                dcfg.get('temporal_gap_min', 1), dcfg.get('temporal_gap_max', 3),
                dcfg.get('pairs_per_epoch', 12000),
                None if not dcfg.get('deterministic_sampling', False) else self.cfg.get('seed', 42),
                img_h=dcfg.get('img_h', 360), img_w=dcfg.get('img_w', 640), rho=dcfg.get('rho', 16),
            )
        return VideoFramePairDataset(
            dcfg['train_video_dir'], dcfg['crop_h'], dcfg['crop_w'],
            dcfg.get('temporal_gap_min', 1), dcfg.get('temporal_gap_max', 5),
            dcfg.get('pairs_per_epoch', 12000),
            None if not dcfg.get('deterministic_sampling', False) else self.cfg.get('seed', 42),
            img_h=dcfg.get('img_h', 360), img_w=dcfg.get('img_w', 640), rho=dcfg.get('rho', 16),
            official_oneline=True,
        )

    def train(self):
        ds = self._make_dataset()
        dcfg = self.cfg['data']
        loader = DataLoader(
            ds,
            batch_size=int(dcfg['batch_size']),
            shuffle=True,
            num_workers=int(dcfg.get('num_workers', 4)),
            pin_memory=self.device.type == 'cuda',
            drop_last=True,
        )
        total = int(self.cfg['train']['total_iters'])
        pbar = tqdm(total=total, initial=self.step, dynamic_ncols=True)
        self.model.train()
        temporal = bool(self.cfg['train'].get('use_temporal_cycle', False))

        while self.step < total:
            for batch in loader:
                if self.step >= total:
                    break
                self.optim.zero_grad(set_to_none=True)
                with torch.amp.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                    if not temporal:
                        org_images = batch['org_images'].to(self.device, non_blocking=True).float()
                        input_tensors = batch['input_tensors'].to(self.device, non_blocking=True).float()
                        h4p = batch['h4p'].to(self.device, non_blocking=True).float()
                        patch_indices = batch['patch_indices'].to(self.device, non_blocking=True).float()
                        out = self.model.forward_oneline(org_images, input_tensors, h4p, patch_indices)
                        losses = self.criterion(out)
                        loss = losses['loss']
                    else:
                        s01 = _sample_to_device(batch, 'p01', self.device)
                        s12 = _sample_to_device(batch, 'p12', self.device)
                        s02 = _sample_to_device(batch, 'p02', self.device)
                        out01 = self.model.forward_oneline(*s01)
                        out12 = self.model.forward_oneline(*s12)
                        out02 = self.model.forward_oneline(*s02)
                        losses01 = self.criterion(out01)
                        losses12 = self.criterion(out12)
                        losses02 = self.criterion(out02)
                        cyc = self.cycle_loss(out01, out12, out02)
                        loss = (losses01['loss'] + losses12['loss'] + losses02['loss']) / 3.0 + cyc
                        losses = {'loss': loss, 'loss_align': losses01['loss_align'], 'loss_cycle': cyc.detach()}

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg['train'].get('grad_clip', 10.0)))
                before = self.scaler.get_scale()
                self.scaler.step(self.optim)
                self.scaler.update()
                if not self.scaler.is_enabled() or self.scaler.get_scale() >= before:
                    self.sched.step()

                self.step += 1
                pbar.update(1)

                if self.step % int(self.cfg['train'].get('log_every', 50)) == 0:
                    pbar.set_postfix(
                        loss=f'{loss.item():.4f}',
                        align=f"{float(losses.get('loss_align', loss)):.4f}",
                        lr=f'{self.sched.get_last_lr()[0]:.2e}',
                        mode='temporal' if temporal else 'pair',
                    )
                if self.step % int(self.cfg['train'].get('ckpt_every', 5000)) == 0:
                    save_checkpoint(self.out_dir / f'ckpt_{self.step:07d}.pt', self.model, self.optim, self.sched, self.step, self.cfg)
                if self.step % int(self.cfg['train'].get('val_every', 2000)) == 0:
                    self._maybe_validate()

        save_checkpoint(self.out_dir / 'last.pt', self.model, self.optim, self.sched, self.step, self.cfg)
        pbar.close()

    def _maybe_validate(self):
        dcfg = self.cfg['data']
        try:
            ds = LabeledPointPairsDataset(
                dcfg['val_npy_dir'], dcfg['val_image_root'], dcfg['crop_h'], dcfg['crop_w'],
                dcfg.get('img_h', 360), dcfg.get('img_w', 640), dcfg.get('eval_crop_x', 40),
                dcfg.get('eval_crop_y', 23),
            )
            metrics = evaluate_labeled_points_v2(self.model, ds, self.device, max_points=dcfg.get('eval_max_points', 6))
            print(f"\nvalidation_v2: {metrics}")
        except Exception as e:
            print(f"\nvalidation_v2 skipped: {e}")
