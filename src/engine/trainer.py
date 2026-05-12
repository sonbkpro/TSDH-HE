from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.video_pair_dataset import ImagePairListDataset, VideoFramePairDataset, LabeledPointPairsDataset
from src.models.content_aware_homography import ContentAwareHomographyNet
from src.losses.triplet_homography_loss import ContentAwareTripletLoss
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from .evaluator import evaluate_labeled_points


class Trainer:
    def __init__(self, cfg, resume=None):
        self.cfg = cfg
        requested = cfg.get('device', 'cuda')
        self.device = torch.device(requested if requested == 'cpu' or torch.cuda.is_available() else 'cpu')
        self.out_dir = Path(cfg['train']['out_dir']); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model = ContentAwareHomographyNet(
            cfg['model'].get('feature_channels', 1),
            pretrained_backbone=bool(cfg['model'].get('pretrained_backbone', False)),
        ).to(self.device)
        self.criterion = ContentAwareTripletLoss(margin=cfg['loss'].get('triplet_margin', 1.0))
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg['train']['lr'], betas=(cfg['train']['beta1'], cfg['train']['beta2']), eps=cfg['train']['eps'], amsgrad=bool(cfg['train'].get('amsgrad', False)), weight_decay=float(cfg['train'].get('weight_decay', 0.0)))
        self.sched = torch.optim.lr_scheduler.StepLR(self.optim, step_size=cfg['train']['lr_decay_every'], gamma=cfg['train']['lr_decay_gamma'])
        self.scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg.get('amp', True) and self.device.type == 'cuda'))
        self.step = 0
        self._stage2_lr_applied = False
        if resume:
            ckpt = load_checkpoint(resume, self.model, self.optim, self.sched, self.device)
            self.step = int(ckpt.get('step', 0))
            self._stage2_lr_applied = self.step > int(cfg['train'].get('stage1_iters', 0))

    def train(self):
        dcfg = self.cfg['data']
        if dcfg.get('train_list') and dcfg.get('train_image_root'):
            dataset = ImagePairListDataset(
                dcfg['train_list'], dcfg['train_image_root'], dcfg['crop_h'], dcfg['crop_w'],
                dcfg.get('img_h', 360), dcfg.get('img_w', 640), dcfg.get('rho', 16),
                seed=self.cfg.get('seed') if dcfg.get('deterministic_sampling', False) else None,
            )
        else:
            dataset = VideoFramePairDataset(
                dcfg['train_video_dir'], dcfg['crop_h'], dcfg['crop_w'], dcfg['temporal_gap_min'], dcfg['temporal_gap_max'],
                dcfg['pairs_per_epoch'], seed=self.cfg.get('seed') if dcfg.get('deterministic_sampling', False) else None, img_h=dcfg.get('img_h', 360),
                img_w=dcfg.get('img_w', 640), rho=dcfg.get('rho', 16), official_oneline=True)
        loader = DataLoader(dataset, batch_size=dcfg['batch_size'], shuffle=True, num_workers=dcfg['num_workers'], pin_memory=self.device.type == 'cuda', drop_last=True)
        pbar = tqdm(total=self.cfg['train']['total_iters'], initial=self.step, dynamic_ncols=True)
        self.model.train()
        while self.step < self.cfg['train']['total_iters']:
            for batch in loader:
                if self.step >= self.cfg['train']['total_iters']: break
                org_images = batch['org_images'].to(self.device, non_blocking=True).float()
                input_tensors = batch['input_tensors'].to(self.device, non_blocking=True).float()
                h4p = batch['h4p'].to(self.device, non_blocking=True).float()
                patch_indices = batch['patch_indices'].to(self.device, non_blocking=True).float()
                stage1 = self.step < int(self.cfg['train']['stage1_iters'])
                if stage1:
                    use_attention = bool(self.cfg['train'].get('stage1_use_attention', True))
                    use_mask_weighting = bool(self.cfg['train'].get('stage1_use_mask_weighting', False))
                else:
                    self._maybe_apply_stage2_lr()
                    use_attention = bool(self.cfg['train'].get('stage2_use_attention', True))
                    use_mask_weighting = bool(self.cfg['train'].get('stage2_use_mask_weighting', True))
                self.optim.zero_grad(set_to_none=True)
                with torch.amp.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                    out = self.model.forward_oneline(
                        org_images, input_tensors, h4p, patch_indices,
                        use_attention=use_attention,
                        use_mask_weighting=use_mask_weighting,
                    )
                    losses = self.criterion(out)
                    loss = losses['loss']
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optim)
                self.scaler.update()
                if not self.scaler.is_enabled() or self.scaler.get_scale() >= scale_before:
                    self.sched.step()
                self.step += 1; pbar.update(1)
                if self.step % self.cfg['train']['log_every'] == 0:
                    stage_name = 'finetune-mask' if not stage1 else 'warmup-mask-ap-ones'
                    mask_mean = out['mask_ap'].detach().float().mean().item()
                    raw_mask_mean = 0.5 * (
                        out['ma_full'].detach().float().mean().item()
                        + out['mb_full'].detach().float().mean().item()
                    )
                    pbar.set_postfix(
                        loss=f'{loss.item():.4f}',
                        stage=stage_name,
                        lr=self.sched.get_last_lr()[0],
                        mask=f'{mask_mean:.3g}',
                        rawM=f'{raw_mask_mean:.3g}',
                    )
                if self.step % self.cfg['train']['ckpt_every'] == 0:
                    save_checkpoint(self.out_dir / f'ckpt_{self.step:07d}.pt', self.model, self.optim, self.sched, self.step, self.cfg)
                if self.step % self.cfg['train']['val_every'] == 0:
                    self._maybe_validate()
        save_checkpoint(self.out_dir / 'last.pt', self.model, self.optim, self.sched, self.step, self.cfg)
        pbar.close()

    def _maybe_apply_stage2_lr(self):
        stage2_lr = self.cfg['train'].get('stage2_lr')
        if stage2_lr is None or self._stage2_lr_applied:
            return
        mode = str(self.cfg['train'].get('stage2_lr_mode', 'set')).lower()
        for group in self.optim.param_groups:
            if mode == 'min':
                group['lr'] = min(float(group['lr']), float(stage2_lr))
            else:
                group['lr'] = float(stage2_lr)
        self._stage2_lr_applied = True

    def _maybe_validate(self):
        dcfg = self.cfg['data']
        try:
            ds = LabeledPointPairsDataset(
                dcfg['val_npy_dir'], dcfg['val_image_root'], dcfg['crop_h'], dcfg['crop_w'],
                dcfg.get('img_h', 360), dcfg.get('img_w', 640), dcfg.get('eval_crop_x', 40),
                dcfg.get('eval_crop_y', 23),
            )
            metrics = evaluate_labeled_points(self.model, ds, self.device, max_points=dcfg.get('eval_max_points', 6))
            print(f"\nvalidation: {metrics}")
        except Exception as e:
            print(f"\nvalidation skipped: {e}")
