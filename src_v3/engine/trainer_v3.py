from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.video_pair_dataset import VideoFramePairDataset, LabeledPointPairsDataset
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src_v3.data.video_triplet_dataset import VideoFrameTripletDataset
from src_v3.models.dominant_homography_v3 import DominantHomographyV3Net
from src_v3.losses.v3_losses import V3DominantHomographyLoss
from .evaluator_v3 import evaluate_labeled_points_v3


class TrainerV3:
    """Accuracy-first V3 trainer.

    Important correction compared with the first V3:
    - Do not train temporal cycle, consensus, mask weighting and uncertainty from step 0.
    - Stage 1 matches the stable V1 protocol as closely as possible.
    - Extra V3 components are introduced only after the single-H baseline is stable.
    """

    def __init__(self, cfg, resume=None):
        self.cfg = cfg
        requested = cfg.get('device', 'cuda')
        self.device = torch.device(requested if requested == 'cpu' or torch.cuda.is_available() else 'cpu')
        self.out_dir = Path(cfg['train']['out_dir']); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model = DominantHomographyV3Net(
            feature_channels=cfg['model'].get('feature_channels', 1),
            pretrained_backbone=bool(cfg['model'].get('pretrained_backbone', False)),
            consensus_radius=float(cfg['model'].get('consensus_radius', 1.0)),
            consensus_temperature=float(cfg['model'].get('consensus_temperature', 0.25)),
            consensus_max_candidates=int(cfg['model'].get('consensus_max_candidates', 5)),
        ).to(self.device)
        if torch.cuda.device_count() > 1 and bool(cfg['train'].get('data_parallel', True)) and self.device.type == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
        self.criterion = V3DominantHomographyLoss(
            margin=cfg['loss'].get('triplet_margin', 1.0),
            weights=cfg['loss'].get('weights', {}),
            min_support=cfg['loss'].get('min_support', 0.15),
        )
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=cfg['train']['lr'],
            betas=(cfg['train'].get('beta1', 0.9), cfg['train'].get('beta2', 0.999)),
            eps=cfg['train'].get('eps', 1e-8), amsgrad=bool(cfg['train'].get('amsgrad', True)),
            weight_decay=float(cfg['train'].get('weight_decay', 1e-4)),
        )
        self.sched = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=cfg['train'].get('lr_decay_every', 12000), gamma=cfg['train'].get('lr_decay_gamma', 0.8)
        )
        self.scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg.get('amp', False) and self.device.type == 'cuda'))
        self.step = 0
        self._stage2_lr_applied = False
        if resume:
            ckpt = load_checkpoint(resume, self._model_for_ckpt(), self.optim, self.sched, self.device)
            self.step = int(ckpt.get('step', 0))
            self._stage2_lr_applied = self.step >= int(cfg['train'].get('stage1_iters', 60000))

    def _model_for_ckpt(self):
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

    def _make_loader(self):
        d = self.cfg['data']
        # Keep triplet dataset available, but the stage schedule can use only p01 during warm-up.
        if bool(self.cfg['train'].get('use_temporal_triplets', True)):
            ds = VideoFrameTripletDataset(
                d['train_video_dir'], d['crop_h'], d['crop_w'], d.get('temporal_gap_min', 1), d.get('temporal_gap_max', 3),
                d['pairs_per_epoch'], seed=self.cfg.get('seed') if d.get('deterministic_sampling', False) else None,
                img_h=d.get('img_h', 360), img_w=d.get('img_w', 640), rho=d.get('rho', 16),
            )
        else:
            ds = VideoFramePairDataset(
                d['train_video_dir'], d['crop_h'], d['crop_w'], d.get('temporal_gap_min', 1), d.get('temporal_gap_max', 3),
                d['pairs_per_epoch'], seed=self.cfg.get('seed') if d.get('deterministic_sampling', False) else None,
                img_h=d.get('img_h', 360), img_w=d.get('img_w', 640), rho=d.get('rho', 16), official_oneline=True,
            )
        return DataLoader(ds, batch_size=d['batch_size'], shuffle=True, num_workers=d['num_workers'],
                          pin_memory=self.device.type == 'cuda', drop_last=True)

    def _pair_from_prefix(self, batch, p: str):
        return {
            'org_images': batch[f'{p}_org_images'].to(self.device, non_blocking=True).float(),
            'input_tensors': batch[f'{p}_input_tensors'].to(self.device, non_blocking=True).float(),
            'h4p': batch[f'{p}_h4p'].to(self.device, non_blocking=True).float(),
            'patch_indices': batch[f'{p}_patch_indices'].to(self.device, non_blocking=True).float(),
        }

    def _pair_plain(self, batch):
        return {
            'org_images': batch['org_images'].to(self.device, non_blocking=True).float(),
            'input_tensors': batch['input_tensors'].to(self.device, non_blocking=True).float(),
            'h4p': batch['h4p'].to(self.device, non_blocking=True).float(),
            'patch_indices': batch['patch_indices'].to(self.device, non_blocking=True).float(),
        }

    def _stage(self) -> str:
        s1 = int(self.cfg['train'].get('stage1_iters', 60000))
        s2 = int(self.cfg['train'].get('stage2_iters', 90000))
        s3 = int(self.cfg['train'].get('stage3_iters', 110000))
        if self.step < s1:
            return 's1_v1_warmup'
        if self.step < s2:
            return 's2_mask_finetune'
        if self.step < s3:
            return 's3_consensus'
        return 's4_temporal'

    def _stage_flags(self, stage: str) -> dict:
        if stage == 's1_v1_warmup':
            return dict(use_attention=True, use_mask_weighting=False, use_consensus=False, use_temporal=False)
        if stage == 's2_mask_finetune':
            return dict(use_attention=True, use_mask_weighting=True, use_consensus=False, use_temporal=False)
        if stage == 's3_consensus':
            return dict(use_attention=True, use_mask_weighting=True, use_consensus=True, use_temporal=False)
        return dict(use_attention=True, use_mask_weighting=True, use_consensus=True, use_temporal=True)

    def _apply_stage_weights(self, stage: str):
        # Avoid corrupting the main homography objective early. V1-like triplet must dominate.
        if stage == 's1_v1_warmup':
            w = dict(triplet=1.0, init_triplet=0.0, support=0.0, uncertainty=0.0, consensus=0.0, temporal_cycle=0.0, point=0.0)
        elif stage == 's2_mask_finetune':
            w = dict(triplet=1.0, init_triplet=0.0, support=0.005, uncertainty=0.0, consensus=0.0, temporal_cycle=0.0, point=0.0)
        elif stage == 's3_consensus':
            w = dict(triplet=1.0, init_triplet=0.05, support=0.005, uncertainty=0.0, consensus=0.005, temporal_cycle=0.0, point=0.0)
        else:
            w = dict(triplet=1.0, init_triplet=0.05, support=0.005, uncertainty=0.001, consensus=0.005, temporal_cycle=0.005, point=0.0)
        self.criterion.weights.update(w)

    def _maybe_apply_stage2_lr(self, stage: str):
        stage2_lr = self.cfg['train'].get('stage2_lr')
        if stage != 's2_mask_finetune' or stage2_lr is None or self._stage2_lr_applied:
            return
        mode = str(self.cfg['train'].get('stage2_lr_mode', 'min')).lower()
        for group in self.optim.param_groups:
            group['lr'] = min(float(group['lr']), float(stage2_lr)) if mode == 'min' else float(stage2_lr)
        self._stage2_lr_applied = True

    def train(self):
        loader = self._make_loader()
        pbar = tqdm(total=self.cfg['train']['total_iters'], initial=self.step, dynamic_ncols=True)
        model_core = self._model_for_ckpt()
        self.model.train()
        use_triplet_dataset = bool(self.cfg['train'].get('use_temporal_triplets', True))
        while self.step < self.cfg['train']['total_iters']:
            for batch in loader:
                if self.step >= self.cfg['train']['total_iters']:
                    break
                stage = self._stage()
                flags = self._stage_flags(stage)
                self._apply_stage_weights(stage)
                self._maybe_apply_stage2_lr(stage)
                self.optim.zero_grad(set_to_none=True)
                with torch.amp.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                    if use_triplet_dataset:
                        p01 = self._pair_from_prefix(batch, 'p01')
                        out01 = self.model(**p01, use_attention=flags['use_attention'],
                                           use_mask_weighting=flags['use_mask_weighting'],
                                           use_consensus=flags['use_consensus'])
                        if flags['use_temporal']:
                            p12, p02 = self._pair_from_prefix(batch, 'p12'), self._pair_from_prefix(batch, 'p02')
                            out12 = self.model(**p12, use_attention=flags['use_attention'],
                                               use_mask_weighting=flags['use_mask_weighting'],
                                               use_consensus=flags['use_consensus'])
                            out02 = self.model(**p02, use_attention=flags['use_attention'],
                                               use_mask_weighting=flags['use_mask_weighting'],
                                               use_consensus=flags['use_consensus'])
                            losses = self.criterion(out01, temporal={'out01': out01, 'out12': out12, 'out02': out02})
                        else:
                            losses = self.criterion(out01)
                    else:
                        p = self._pair_plain(batch)
                        out01 = self.model(**p, use_attention=flags['use_attention'],
                                           use_mask_weighting=flags['use_mask_weighting'],
                                           use_consensus=flags['use_consensus'])
                        losses = self.criterion(out01)
                    loss = losses['loss']
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['train'].get('clip_grad_norm', 10.0))
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optim)
                self.scaler.update()
                if not self.scaler.is_enabled() or self.scaler.get_scale() >= scale_before:
                    self.sched.step()
                self.step += 1; pbar.update(1)
                if self.step % self.cfg['train']['log_every'] == 0:
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}", trip=f"{losses['triplet'].item():.4f}",
                        stage=stage, lr=f"{self.sched.get_last_lr()[0]:.2e}",
                        cyc=f"{losses.get('temporal_cycle', torch.tensor(0.)).item():.2e}",
                        sup=f"{out01['support_ap'].detach().float().mean().item():.3f}",
                        conf=f"{out01['confidence'].detach().float().mean().item():.3f}",
                    )
                if self.step % self.cfg['train']['save_every'] == 0:
                    save_checkpoint(self.out_dir / f'ckpt_{self.step:07d}.pt', model_core, self.optim, self.sched, self.step, self.cfg)
                    save_checkpoint(self.out_dir / 'last.pt', model_core, self.optim, self.sched, self.step, self.cfg)
                if self.step % self.cfg['train']['val_every'] == 0:
                    self._validate()
                    self.model.train()
        save_checkpoint(self.out_dir / 'last.pt', model_core, self.optim, self.sched, self.step, self.cfg)
        pbar.close()

    def _validate(self):
        d = self.cfg['data']
        try:
            ds = LabeledPointPairsDataset(
                d['val_npy_dir'], d['val_image_root'], d['crop_h'], d['crop_w'],
                d.get('img_h', 360), d.get('img_w', 640), d.get('eval_crop_x', 40), d.get('eval_crop_y', 23),
            )
            metrics = evaluate_labeled_points_v3(self._model_for_ckpt(), ds, self.device, max_points=d.get('eval_max_points', 6))
            print(f"\nvalidation_v3: {metrics}")
        except Exception as e:
            print(f"\nvalidation_v3 skipped: {e}")
