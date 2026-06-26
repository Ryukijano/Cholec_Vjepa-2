"""
MOTTrainer — training loop for the Surgical MOT system.

Supports four stages mapped to config key ``meta.stage``:

  * ``stage1_supervised`` — joint supervised training on CholecTrack20
    (``L_det + λ_track · L_track + λ_reid · L_reid``).
  * ``stage2_jepa``       — GOT-JEPA teacher-student predictor
    pretraining (invariance + covariance losses only; teacher frozen).
  * ``stage3_joint``      — load JEPA-pretrained student and fine-tune
    jointly with detection + track + reid losses.
  * ``stage4_full``       — enable VGGT / OccuSolver branches; train
    jointly (detection + track + reid + occusolver + consistency).

Each stage shares the same outer skeleton — only the per-step forward
call and loss composition change. The design follows the existing
``core_app.trainers.world_model_trainer.WorldModelTrainer`` so that
configs and entry points stay familiar.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from contextlib import nullcontext
from tqdm import tqdm

from ..utils.checkpoint import load_checkpoint, save_checkpoint


def resolve_amp_dtype(config: Dict[str, Any]) -> Optional[torch.dtype]:
    """Map ``meta.dtype`` to a torch AMP dtype (bf16 preferred on GB10/Blackwell)."""
    raw = str(config.get('meta', {}).get('dtype', 'bfloat16')).lower()
    if raw in ('none', 'null', 'fp32', 'float32', '32'):
        return None
    if raw in ('bf16', 'bfloat16', 'bfloat'):
        return torch.bfloat16
    if raw in ('fp16', 'float16', '16', 'half'):
        return torch.float16
    _LOG.warning("Unknown meta.dtype=%r — defaulting to bfloat16", raw)
    return torch.bfloat16
from ..utils.metrics import MetricLogger
from ..utils.wandb_logger import WandbLogger
from .system import SurgicalMOTSystem

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Builder helpers                                                         #
# ---------------------------------------------------------------------- #


def build_model_from_config(config: Dict[str, Any]) -> SurgicalMOTSystem:
    """Instantiate ``SurgicalMOTSystem`` from a YAML config dict."""
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    detr_cfg = config.get('detr', {})
    reid_cfg = config.get('reid', {})
    pred_cfg = config.get('predictor', {})
    track_cfg = config.get('tracker', {})
    stage_cfg = config.get('stage_flags', {})
    occusolver_cfg = dict(config.get('occusolver', {}))
    occusolver_cfg.setdefault('cotracker_version', 3)
    occusolver_cfg.setdefault('num_refine_steps', 0)
    loss_cfg = config.get('losses', {})

    # Backward-compatible support for older configs placing occu hooks under
    # stage_flags (instead of the dedicated occusolver section).
    for key in (
        'cotracker_version',
        'cotracker_mode',
        'cotracker_model_variant',
        'cotracker_num_query_points',
        'num_refine_steps',
        'stub',
        'hub_name',
    ):
        if key in stage_cfg and key not in occusolver_cfg:
            occusolver_cfg[key] = stage_cfg[key]

    return SurgicalMOTSystem(
        # Encoder / neck
        encoder_type=model_cfg.get('encoder_type', 'dinov2'),
        encoder_checkpoint=model_cfg.get('encoder_checkpoint'),
        model_name=model_cfg.get('model_name', 'dinov2_vitb14'),
        encoder_dim=model_cfg.get('encoder_dim', 768),
        neck_dim=model_cfg.get('neck_dim', 256),
        pred_dim=pred_cfg.get('dim', 256),
        img_size=data_cfg.get('img_size', 392),
        num_frames=data_cfg.get('clip_length', 3),
        layer_indices=model_cfg.get('layer_indices', [-1]),
        use_torch_hub=model_cfg.get('use_torch_hub', True),
        encoder_lora=model_cfg.get('encoder_lora', None),
        # DETR
        num_tools=detr_cfg.get('num_tools', 7),
        num_queries=detr_cfg.get('num_queries', 16),
        num_decoder_layers=detr_cfg.get('num_decoder_layers', 5),
        detr_nheads=detr_cfg.get('nheads', 8),
        detr_dropout=detr_cfg.get('dropout', 0.15),
        detr_class_weight=detr_cfg.get('class_weight', 1.0),
        detr_bbox_weight=detr_cfg.get('bbox_weight', 5.0),
        detr_giou_weight=detr_cfg.get('giou_weight', 2.0),
        detr_focal_alpha=detr_cfg.get('focal_alpha', 0.25),
        detr_focal_gamma=detr_cfg.get('focal_gamma', 2.0),
        detr_use_denoising=detr_cfg.get('use_denoising', False),
        detr_num_denoising_groups=detr_cfg.get('num_denoising_groups', 5),
        detr_num_noise_per_group=detr_cfg.get('num_noise_per_group', 4),
        detr_label_noise_prob=detr_cfg.get('label_noise_prob', 0.2),
        detr_box_noise_scale=detr_cfg.get('box_noise_scale', 0.4),
        detr_denoising_weight=detr_cfg.get('denoising_weight', 1.0),
        # Per-track predictor
        pred_num_heads=pred_cfg.get('num_heads', 8),
        pred_num_encoder_layers=pred_cfg.get('num_encoder_layers', 4),
        pred_num_decoder_layers=pred_cfg.get('num_decoder_layers', 2),
        pred_dim_feedforward=pred_cfg.get('dim_feedforward', 1024),
        pred_dropout=pred_cfg.get('dropout', 0.1),
        # ReID
        reid_embedding_dim=reid_cfg.get('embedding_dim', 256),
        reid_supcon_weight=reid_cfg.get('supcon_weight', 1.0),
        reid_supcon_temperature=reid_cfg.get('supcon_temperature', 0.07),
        reid_cross_consistency_weight=reid_cfg.get('cross_consistency_weight', 0.0),
        reid_dropout=reid_cfg.get('dropout', 0.15),
        # Loss weights (detector_only: Stage-1 teacher for pseudo-labels — DETR only)
        track_loss_weight=(
            0.0 if loss_cfg.get('detector_only', False) else loss_cfg.get('track_weight', 1.0)
        ),
        track_cls_weight=loss_cfg.get('track_cls_weight', 1.0),
        track_giou_weight=loss_cfg.get('track_giou_weight', 2.0),
        reid_loss_weight=(
            0.0 if loss_cfg.get('detector_only', False) else loss_cfg.get('reid_weight', 0.5)
        ),
        occu_loss_weight=loss_cfg.get('occu_weight', 0.0),
        consist_loss_weight=loss_cfg.get('consist_weight', 0.0),
        occu_gate_features=loss_cfg.get('occu_gate_features', True),
        occu_max_tracks_per_batch=loss_cfg.get('occu_max_tracks_per_batch'),
        # Track manager
        tracker_birth_score=track_cfg.get('birth_score', 0.6),
        tracker_min_hits=track_cfg.get('min_hits', 3),
        tracker_max_age=track_cfg.get('max_age', 30),
        tracker_reentry_ttl=track_cfg.get('reentry_ttl', 300),
        tracker_cost_threshold=track_cfg.get('cost_threshold', 0.7),
        tracker_infer_score_threshold=track_cfg.get('infer_score_threshold', 0.3),
        # Stage 4 flags
        use_geometry=stage_cfg.get('use_geometry', False),
        use_occusolver=stage_cfg.get('use_occusolver', False),
        occusolver_kwargs=occusolver_cfg,
        use_depth=stage_cfg.get('use_depth', False),
        geometry_kwargs=dict(config.get('geometry', {})),
        # Architecture toggle
        use_deformable_detr=detr_cfg.get('use_deformable_detr', False),
    )


# ---------------------------------------------------------------------- #
# MOTTrainer                                                              #
# ---------------------------------------------------------------------- #


class MOTTrainer:
    """Staged trainer for the Surgical MOT system."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[SurgicalMOTSystem] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda',
        rank: int = 0,
        world_size: int = 1,
        is_main: bool = True,
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.rank = rank
        self.world_size = world_size
        self.is_main = is_main
        self.is_ddp = world_size > 1

        self.stage = config.get('meta', {}).get('stage', 'stage1_supervised')
        self.logger = _LOG
        self.amp_dtype = resolve_amp_dtype(config)
        self.grad_scaler = (
            torch.amp.GradScaler('cuda', enabled=(self.amp_dtype == torch.float16))
            if self.amp_dtype is not None
            else None
        )
        if self.is_main:
            amp_name = {torch.bfloat16: 'bfloat16', torch.float16: 'float16'}.get(
                self.amp_dtype, 'float32'
            )
            self.logger.info(
                "AMP enabled: %s%s",
                amp_name,
                " (GradScaler on)" if self.grad_scaler and self.grad_scaler.is_enabled() else "",
            )

        if model is None:
            model = build_model_from_config(config)
        self.model = model.to(self.device)

        # DDP wrapping (Stage 1/3/4 only; Stage 2 uses manual grad sync).
        # find_unused_parameters=True because SurgicalMOTSystem has conditional
        # branches (per-track predictor only fires when ≥1 track exists,
        # geometry/occusolver are stage-gated) which leave some params without
        # grad on certain batches.
        self._ddp_model: Optional[nn.Module] = None
        if self.is_ddp and self.stage != 'stage2_jepa':
            self._ddp_model = DDP(
                self.model,
                device_ids=[self.device],
                find_unused_parameters=True,
                static_graph=False,
            )

        self.train_loader = train_loader
        self.val_loader = val_loader

        # JEPA wrapper is set up lazily in Stage 2.
        self.jepa_wrapper = None
        if self.stage == 'stage2_jepa':
            self._setup_jepa_wrapper()
            init_ckpt = config.get('meta', {}).get('load_checkpoint')
            if init_ckpt:
                self._load_stage1_weights_for_jepa(init_ckpt)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        self.output_dir = Path(config.get('meta', {}).get('folder', 'outputs/mot'))
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # W&B logger (enabled via config or env var) — only on main rank.
        wb_cfg = config.get('wandb', {})
        self.wb = WandbLogger(
            config=config,
            enabled=wb_cfg.get('enabled', True) and self.is_main,
            project=wb_cfg.get('project', 'surgical-mot'),
            entity=wb_cfg.get('entity'),
            name=wb_cfg.get('name'),
            group=wb_cfg.get('group'),
            job_type=wb_cfg.get('job_type', 'train'),
            tags=wb_cfg.get('tags'),
        )
        if self.is_main and self.stage in (
            'stage1_supervised',
            'stage3_joint',
            'stage4_full',
            'stage4_got_edit',
        ):
            # wandb.watch(log='gradients') crashes in DDP with
            # find_unused_parameters=True because some params have grad=None.
            # Safer to only log parameters (not gradients) under DDP.
            watch_log = wb_cfg.get('watch', 'gradients')
            if self.is_ddp and watch_log in ('gradients', 'all'):
                self.logger.info(
                    "Skipping wandb.watch(log='gradients') under DDP "
                    "(would crash on unused parameters)."
                )
            else:
                self.wb.watch_model(
                    self.model,
                    log=watch_log,
                    log_freq=wb_cfg.get('watch_freq', 100),
                )

        if config.get('losses', {}).get('detector_only', False) and self.is_main:
            self.logger.info(
                "detector_only=True — training DETR head only (track/reid losses disabled) "
                "for pseudo-label generation on unlabelled video."
            )

        if self.is_main:
            self.logger.info(
                f"MOTTrainer ready | stage={self.stage} | device={self.device} | "
                f"ddp={self.is_ddp} | ws={self.world_size} | out={self.output_dir}"
            )

    # --- Stage-specific setup ---------------------------------------- #

    def _load_stage1_weights_for_jepa(self, path: str) -> None:
        """Load frozen backbone + teacher predictor from Stage 1 (meta.load_checkpoint)."""
        ckpt = load_checkpoint(path, map_location=str(self.device))
        missing, unexpected = self.model.load_state_dict(ckpt['model'], strict=False)
        if self.is_main:
            if missing:
                self.logger.info("Stage-1 init: %d missing keys (expected for partial load)", len(missing))
            if unexpected:
                self.logger.warning("Stage-1 init: %d unexpected keys in checkpoint", len(unexpected))
            self.logger.info("Loaded Stage-1 weights for JEPA from %s (epoch %s)", path, ckpt.get('epoch'))
        # Refresh teacher copy after Stage-1 weights are in the student predictor.
        if self.jepa_wrapper is not None:
            self.jepa_wrapper.teacher.load_state_dict(self.model.per_track_predictor.state_dict())
            self.jepa_wrapper._freeze_teacher()

    def _setup_jepa_wrapper(self) -> None:
        from .jepa import GOTJEPAWrapper
        self.jepa_wrapper = GOTJEPAWrapper(
            student_predictor=self.model.per_track_predictor,
            inv_weight=self.config.get('losses', {}).get('jepa_inv_weight', 1.0),
            cov_weight=self.config.get('losses', {}).get('jepa_cov_weight', 0.5),
        ).to(self.device)

        # Corruption bank for the student branch.
        from .augment import SurgicalCorruption
        aug_cfg = self.config.get('augmentation', {})
        self.corruption = SurgicalCorruption(
            smoke_p=aug_cfg.get('smoke_p', 0.4),
            blood_p=aug_cfg.get('blood_p', 0.2),
            spec_p=aug_cfg.get('spec_p', 0.3),
            blur_p=aug_cfg.get('blur_p', 0.3),
            jitter_p=aug_cfg.get('jitter_p', 0.5),
            cutout_p=aug_cfg.get('cutout_p', 0.2),
        ).to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_cfg = self.config.get('optimization', {})
        lr = opt_cfg.get('lr', 1e-4)
        lora_lr = opt_cfg.get('lora_lr', lr)
        wd = opt_cfg.get('weight_decay', 1e-2)

        if self.stage == 'stage2_jepa':
            # Only student + ProjNet + Expander are trainable.
            assert self.jepa_wrapper is not None
            all_params = [p for p in self.jepa_wrapper.parameters() if p.requires_grad]
            self.logger.info(f"Trainable params: {sum(p.numel() for p in all_params):,}")
            return AdamW(all_params, lr=lr, weight_decay=wd)

        # Separate LoRA parameters so they can use a higher learning rate.
        from ..models.lora import get_lora_params
        lora_params = set(get_lora_params(self.model))
        lora_params_list = [p for p in lora_params if p.requires_grad]
        non_lora_params = [
            p for p in self.model.parameters()
            if p.requires_grad and p not in lora_params
        ]

        param_groups = [
            {'params': non_lora_params, 'lr': lr, 'weight_decay': wd},
        ]
        if lora_params_list:
            param_groups.append({
                'params': lora_params_list,
                'lr': lora_lr,
                'weight_decay': wd,
            })
            total = sum(p.numel() for p in non_lora_params) + sum(p.numel() for p in lora_params_list)
            self.logger.info(
                f"Trainable params: {total:,} (non-LoRA {len(non_lora_params)} groups, "
                f"LoRA {len(lora_params_list)} params, lora_lr={lora_lr:.2e})"
            )
        else:
            total = sum(p.numel() for p in non_lora_params)
            self.logger.info(f"Trainable params: {total:,} (no LoRA active)")

        return AdamW(param_groups, lr=lr, weight_decay=wd)

    def _build_scheduler(self):
        opt_cfg = self.config.get('optimization', {})
        epochs = opt_cfg.get('epochs', 15)
        min_lr = opt_cfg.get('min_lr', 1e-6)
        warmup_epochs = opt_cfg.get('warmup_epochs', 0)

        base = CosineAnnealingLR(self.optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=min_lr)
        if warmup_epochs > 0:
            warmup = LinearLR(
                self.optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            sched = SequentialLR(
                self.optimizer,
                schedulers=[warmup, base],
                milestones=[warmup_epochs],
            )
            return sched
        return base

    def _align_scheduler_to_epoch(self, epoch: int) -> None:
        """Advance LR schedule so epoch ``epoch`` uses the correct learning rate."""
        if epoch <= 0:
            return
        for _ in range(epoch):
            self.scheduler.step()

    def _log_optimizer_lr(self) -> None:
        lrs = {pg['lr'] for pg in self.optimizer.param_groups}
        lr_str = ', '.join(f'{lr:.2e}' for lr in sorted(lrs))
        self.logger.info(f"Optimizer LR: {lr_str}")

    def _autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=self.amp_dtype)

    def _ddp_safe_zero_backward(self) -> None:
        """Finite backward that satisfies DDP all-reduce without touching NaN loss."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            self.optimizer.zero_grad()
            return
        zero = sum(p.reshape(-1)[0] * 0.0 for p in params)
        if self.grad_scaler is not None and self.grad_scaler.is_enabled():
            self.grad_scaler.scale(zero).backward()
            self.grad_scaler.update()
        else:
            zero.backward()
        self.optimizer.zero_grad()

    # --- Training step dispatch -------------------------------------- #

    def _step_stage1_or_3_or_4(self, batch: Dict[str, Any]) -> Dict[str, float]:
        current_video = batch['current_video'].to(self.device, non_blocking=True)
        detr_targets = batch['detr_targets']
        per_track_targets = batch['per_track_targets']
        reid_labels = batch['reid_labels']

        # Use DDP wrapper if active (Stages 1/3/4); Stage 2 skips DDP and uses manual grad sync.
        forward_model = self._ddp_model if self._ddp_model is not None else self.model
        with self._autocast():
            out = forward_model(
                current_video=current_video,
                per_track_targets=per_track_targets,
                detr_targets=detr_targets,
                reid_labels=reid_labels,
                mode='train',
            )

        loss = out['total_loss']
        if loss is None or loss.requires_grad is False:
            self.logger.warning("Total loss is None or has no grad — skipping step.")
            return {}

        # DDP-safe NaN/Inf handling: all ranks must agree on skip-vs-backward,
        # otherwise ranks with valid loss call backward() and hang on all-reduce
        # while the skipping rank never contributes gradients.
        is_bad_local = torch.tensor(
            [1.0 if (torch.isnan(loss) or torch.isinf(loss)) else 0.0],
            device=self.device,
        )
        if self.is_ddp:
            dist.all_reduce(is_bad_local, op=dist.ReduceOp.SUM)
        any_bad = bool(is_bad_local.item() > 0)

        if any_bad:
            if self.is_main:
                self.logger.warning(
                    f"NaN/Inf loss on ≥1 rank at epoch {self.current_epoch}, batch skipped. "
                    f"loss_dict={out.get('loss_dict', {})}"
                )
            # DDP still needs a finite backward; never use loss * 0.0 (NaN * 0 => NaN).
            self._ddp_safe_zero_backward()
            return {}

        if self.grad_scaler is not None and self.grad_scaler.is_enabled():
            self.grad_scaler.scale(loss).backward()
            clip_norm = self.config.get('optimization', {}).get('gradient_clip_norm', 1.0)
            if clip_norm > 0:
                self.grad_scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            clip_norm = self.config.get('optimization', {}).get('gradient_clip_norm', 1.0)
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.global_step += 1

        loss_dict = out.get('loss_dict', {})
        loss_dict['total'] = float(loss.item())
        return loss_dict

    def _step_stage2_jepa(self, batch: Dict[str, Any], backward: bool = True) -> Dict[str, float]:
        assert self.jepa_wrapper is not None

        current_video = batch['current_video'].to(self.device, non_blocking=True)
        per_track_targets = batch['per_track_targets']

        # Corrupt the current frame — last temporal slice — only for the student.
        clean_video = current_video
        dirty_video = current_video.clone()
        last = dirty_video[:, :, -1]                               # (B, C, H, W)
        dirty_video[:, :, -1] = self.corruption(last)

        # Encode both branches to get token representations.
        with self._autocast():
            reality_clean, _ = self.model.encode_frames(clean_video)
            reality_dirty, _ = self.model.encode_frames(dirty_video)
            clean_tokens = self.model.pred_token_proj(reality_clean)   # (B, T, N, C_pred)
            dirty_tokens = self.model.pred_token_proj(reality_dirty)   # (B, T, N, C_pred)

        N_tokens = clean_tokens.size(2)
        hw = int(N_tokens ** 0.5)
        H_pred = W_pred = hw

        from .predictor import gaussian_label_encoding

        # Gather all tracks across the batch into one pseudo-batch for efficient
        # training of the predictor.
        ref_feats_list = []
        ref_labels_list = []
        clean_cur_list = []
        dirty_cur_list = []
        for b, tracks in enumerate(per_track_targets):
            if not tracks:
                continue
            ref0 = clean_tokens[b, 0]
            ref1 = clean_tokens[b, 1]
            for sample in tracks:
                heat0 = gaussian_label_encoding(
                    sample.ref_bbox_0.to(self.device).view(1, 4), H_pred, W_pred
                ).view(1, -1, 1)
                heat1 = gaussian_label_encoding(
                    sample.ref_bbox_1.to(self.device).view(1, 4), H_pred, W_pred
                ).view(1, -1, 1)
                ref_feats_list.append(torch.cat([ref0.unsqueeze(0), ref1.unsqueeze(0)], dim=1))
                ref_labels_list.append(torch.cat([heat0, heat1], dim=1))
                clean_cur_list.append(clean_tokens[b, -1].unsqueeze(0))
                dirty_cur_list.append(dirty_tokens[b, -1].unsqueeze(0))

        if not ref_feats_list:
            self._jepa_skips_epoch = getattr(self, '_jepa_skips_epoch', 0) + 1
            if backward and self.is_ddp:
                # Must participate in all_reduce even with no tracks.
                # Create a dummy zero-loss backward to ensure every param has .grad,
                # then zero them so the all_reduce is a no-op.
                dummy = torch.zeros(1, device=self.device, requires_grad=True)
                dummy.backward()
                for p in self.jepa_wrapper.parameters():
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    else:
                        p.grad.zero_()
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data /= self.world_size
            return {}

        ref_feats = torch.cat(ref_feats_list, dim=0)               # (N_tracks, 2N, C_pred)
        ref_labels = torch.cat(ref_labels_list, dim=0)             # (N_tracks, 2N, 1)
        clean_cur = torch.cat(clean_cur_list, dim=0)               # (N_tracks, N, C_pred)
        dirty_cur = torch.cat(dirty_cur_list, dim=0)               # (N_tracks, N, C_pred)

        with self._autocast():
            result = self.jepa_wrapper(
                reference_features=ref_feats,
                label_encoding=ref_labels,
                current_features_clean=clean_cur,
                current_features_dirty=dirty_cur,
            )

        if backward:
            loss = result['loss']
            if self.grad_scaler is not None and self.grad_scaler.is_enabled():
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            # Manual gradient sync for Stage 2 (variable track counts per rank).
            if self.is_ddp:
                for p in self.jepa_wrapper.parameters():
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data /= self.world_size

            clip_norm = self.config.get('optimization', {}).get('gradient_clip_norm', 1.0)
            params = [p for p in self.jepa_wrapper.parameters() if p.requires_grad]
            if self.grad_scaler is not None and self.grad_scaler.is_enabled():
                if clip_norm > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(params, max_norm=clip_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                if clip_norm > 0:
                    nn.utils.clip_grad_norm_(params, max_norm=clip_norm)
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.global_step += 1

        return result['loss_dict']

    # --- Training loop ----------------------------------------------- #

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        if self.jepa_wrapper is not None:
            self.jepa_wrapper.train()
        self._jepa_skips_epoch = 0

        meter = MetricLogger()
        assert self.train_loader is not None

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch} train",
            disable=not self.is_main,
            unit="batch",
        )

        for batch_idx, batch in pbar:
            if self.stage == 'stage2_jepa':
                stats = self._step_stage2_jepa(batch)
            else:
                stats = self._step_stage1_or_3_or_4(batch)

            if stats:
                meter.update(**stats)
                if self.is_main:
                    # Log per-batch scalars to W&B
                    prefix = f"train/{self.stage}"
                    self.wb.log_scalars(
                        {f"{prefix}/{k}": v for k, v in stats.items()},
                        step=self.global_step,
                    )
                    # LR & grad norm
                    lr = self.optimizer.param_groups[0]['lr']
                    self.wb.log_lr(lr, step=self.global_step)
                    if self.stage == 'stage2_jepa':
                        self.wb.log_grad_norm(self.jepa_wrapper, step=self.global_step)
                    else:
                        self.wb.log_grad_norm(self.model, step=self.global_step)

                    # Stage-2: log clean vs corrupted every N steps
                    if self.stage == 'stage2_jepa' and batch_idx % 100 == 0 and 'current_video' in batch:
                        clean = batch['current_video'].to(self.device)
                        dirty = clean.clone()
                        dirty[:, :, -1] = self.corruption(dirty[:, :, -1])
                        self.wb.log_clean_vs_corrupted(clean, dirty, step=self.global_step, max_images=2)

                    # Stage-2: collapse monitor
                    if self.stage == 'stage2_jepa':
                        self.wb.log_jepa_collapse_monitor(
                            stats.get('jepa_inv', stats.get('inv', 0.0)),
                            stats.get('jepa_cov', stats.get('cov', 0.0)),
                            step=self.global_step,
                        )

                    # Live tqdm postfix
                    avg = meter.avg_dict()
                    if self.stage == 'stage2_jepa':
                        disp_loss = avg.get('jepa_total', float('nan'))
                    else:
                        disp_loss = avg.get('total', avg.get('jepa_total', 0.0))
                    pbar.set_postfix(loss=f"{disp_loss:.4f}" if disp_loss == disp_loss else "skip")

            log_freq = self.config.get('optimization', {}).get('log_freq', 10)
            if batch_idx % log_freq == 0 and self.is_main:
                if stats:
                    loss_val = stats.get('jepa_total', stats.get('total', 0.0))
                else:
                    loss_val = float('nan')
                skip_note = " (skipped — no tracks)" if not stats else ""
                loss_str = f"{loss_val:.4f}" if loss_val == loss_val else "nan"
                self.logger.info(
                    f"[{self.stage}] epoch {epoch} | batch {batch_idx}/{len(self.train_loader)} | "
                    f"loss={loss_str}{skip_note}"
                )

        if self.is_main:
            pbar.close()
            if self.stage == 'stage2_jepa' and self._jepa_skips_epoch > 0:
                n_batches = len(self.train_loader)
                self.logger.warning(
                    "Stage-2 epoch %d: skipped %d/%d batches (no per-track targets). "
                    "Rebuild ssl_corpus or set per_track_min_visible_frames: 1.",
                    epoch,
                    self._jepa_skips_epoch,
                    n_batches,
                )
        self.scheduler.step()
        return meter.avg_dict()

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        if self.val_loader is None:
            return {}

        meter = MetricLogger()
        vis_logged = False
        # Accumulate predictions for end-of-epoch mAP
        all_pred_logits: List[torch.Tensor] = []
        all_pred_boxes: List[torch.Tensor] = []
        all_targets: List[Dict] = []
        pbar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc=f"Epoch {epoch} val",
            disable=not self.is_main,
            unit="batch",
        )
        for batch_idx, batch in pbar:
            if self.stage == 'stage2_jepa':
                # Re-use training step — no gradient, but computes losses.
                stats = self._step_stage2_jepa(batch, backward=False)
                if stats:
                    meter.update(**stats)
                    if self.is_main:
                        prefix = f"val/{self.stage}"
                        self.wb.log_scalars(
                            {f"{prefix}/{k}": v for k, v in stats.items()},
                            step=self.global_step,
                        )
                        self.wb.log_jepa_collapse_monitor(
                            stats.get('inv', 0.0),
                            stats.get('cov', 0.0),
                            step=self.global_step,
                        )
                        if batch_idx == 0 and 'current_video' in batch:
                            clean = batch['current_video'].to(self.device)
                            dirty = clean.clone()
                            dirty[:, :, -1] = self.corruption(dirty[:, :, -1])
                            self.wb.log_clean_vs_corrupted(clean, dirty, step=self.global_step, max_images=2)
            else:
                current_video = batch['current_video'].to(self.device)
                forward_model = self._ddp_model if self._ddp_model is not None else self.model
                with self._autocast():
                    out = forward_model(
                        current_video=current_video,
                        per_track_targets=batch['per_track_targets'],
                        detr_targets=batch['detr_targets'],
                        reid_labels=batch['reid_labels'],
                        mode='train',   # re-use train path for loss computation
                    )
                if out.get('total_loss') is not None:
                    loss_dict = out.get('loss_dict', {})
                    meter.update(**loss_dict)
                    # Log per-batch val scalars
                    if self.is_main:
                        prefix = f"val/{self.stage}"
                        self.wb.log_scalars(
                            {f"{prefix}/{k}": v for k, v in loss_dict.items()},
                            step=self.global_step,
                        )

                    # Accumulate for end-of-epoch mAP
                    if 'detr' in out:
                        detr = out['detr']
                        pred = detr.get('pred', detr)
                        if 'class_logits' in pred and 'pred_boxes' in pred:
                            all_pred_logits.append(pred['class_logits'].cpu())
                            all_pred_boxes.append(pred['pred_boxes'].cpu())
                            all_targets.extend(batch['detr_targets'])

                    # Visualise one batch per epoch: DETR predictions vs GT
                    if not vis_logged and 'detr' in out and self.is_main:
                        try:
                            detr = out['detr']
                            pred_logits = detr.get('pred_logits', detr.get('class_embed'))
                            if pred_logits is not None:
                                pred_scores, pred_labels = pred_logits.softmax(-1).max(-1)
                                pred_boxes = detr.get('pred_boxes')
                                # Filter by confidence
                                keep = pred_scores[0] > 0.3
                                if keep.sum() > 0 and pred_boxes is not None:
                                    pboxes = pred_boxes[0][keep]
                                    plabels = pred_labels[0][keep]
                                    pscores = pred_scores[0][keep]
                                    # GT from detr_targets
                                    gt = batch['detr_targets'][0] if batch['detr_targets'] else None
                                    gt_boxes = gt['boxes'] if gt and 'boxes' in gt else None
                                    gt_labels = gt['labels'] if gt and 'labels' in gt else None
                                    self.wb.log_images_with_boxes(
                                        images=current_video[:, :, -1],  # last frame (B,C,H,W)
                                        pred_boxes=pboxes,
                                        pred_labels=plabels,
                                        pred_scores=pscores,
                                        gt_boxes=gt_boxes,
                                        gt_labels=gt_labels,
                                        img_size=self.config.get('data', {}).get('img_size', 392),
                                        step=self.global_step,
                                        max_images=2,
                                        key="val/predictions",
                                    )
                                    # Bbox distribution histogram
                                    if gt_boxes is not None:
                                        self.wb.log_bbox_distribution(gt_boxes, step=self.global_step)
                                    vis_logged = True
                        except Exception as e:
                            self.logger.warning(f"W&B val visualisation failed: {e}")

            if self.is_main:
                avg = meter.avg_dict()
                pbar.set_postfix(loss=f"{avg.get('total', avg.get('jepa_total', 0.0)):.4f}")

        if self.is_main:
            pbar.close()

        # --- End-of-epoch mAP computation ---
        if all_pred_logits and self.stage != 'stage2_jepa':
            try:
                from .det_metrics import compute_map_from_detr_outputs
                pred_logits_cat = torch.cat(all_pred_logits)
                pred_boxes_cat = torch.cat(all_pred_boxes)
                num_classes = self.config.get('detr', {}).get('num_tools', 7)
                map_metrics = compute_map_from_detr_outputs(
                    pred_logits=pred_logits_cat,
                    pred_boxes=pred_boxes_cat,
                    targets=all_targets,
                    num_classes=num_classes,
                )
                if self.is_main:
                    prefix = f"val/{self.stage}"
                    self.wb.log_scalars(
                        {f"{prefix}/{k}": v for k, v in map_metrics.items()},
                        step=self.global_step,
                    )
                # Merge into meter for return value
                for k, v in map_metrics.items():
                    meter.update(**{k: v})
            except Exception as e:
                self.logger.warning(f"mAP computation failed: {e}")

        return meter.avg_dict()

    def _epoch_end(self, epoch: int, train_stats: Dict[str, float], val_stats: Dict[str, float]) -> None:
        """Logging, checkpointing, and W&B summary — call only on rank 0."""
        if not self.is_main:
            return

        self.logger.info(
            f"Epoch {epoch} | train: {train_stats} | val: {val_stats}"
        )

        # Epoch-level W&B summary (use global_step to stay monotonic with per-batch logs)
        self.wb.log_epoch_summary(train_stats, val_stats, epoch, step=self.global_step)
        is_best = False
        if val_stats:
            val_loss = val_stats.get('total', float('inf'))
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.wb.log_scalars({"epoch/best_val_loss": self.best_val_loss}, step=self.global_step)

        # Save latest.
        save_checkpoint(
            {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'jepa': self.jepa_wrapper.state_dict() if self.jepa_wrapper is not None else None,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'config': self.config,
                'stage': self.stage,
            },
            path=str(self.output_dir / 'latest.pth.tar'),
        )

        # Checkpoint best-val.
        if is_best:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'jepa': self.jepa_wrapper.state_dict() if self.jepa_wrapper is not None else None,
                    'config': self.config,
                    'stage': self.stage,
                },
                path=str(self.output_dir / 'best.pth.tar'),
                is_best=True,
            )

    def train(self, num_epochs: int) -> None:
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                train_stats = self.train_epoch(epoch)
                val_stats = self.validate(epoch) if self.val_loader is not None else {}
                self._epoch_end(epoch, train_stats, val_stats)
        finally:
            if self.is_main:
                self.wb.finish()

    def load_checkpoint(
        self,
        path: str,
        *,
        reset_optimizer: bool = False,
        reset_scheduler: bool = False,
        start_epoch: Optional[int] = None,
    ) -> None:
        ckpt = load_checkpoint(path, map_location=str(self.device))
        self.model.load_state_dict(ckpt['model'], strict=False)
        if ckpt.get('jepa') is not None and self.jepa_wrapper is not None:
            self.jepa_wrapper.load_state_dict(ckpt['jepa'], strict=False)

        ckpt_epoch = int(ckpt.get('epoch', -1))
        if start_epoch is not None:
            self.current_epoch = start_epoch
        else:
            self.current_epoch = ckpt_epoch + 1

        if reset_optimizer:
            self.optimizer = self._build_optimizer()
            reset_scheduler = True
            self.logger.info(
                "Optimizer reset from config (fresh AdamW; LR not taken from checkpoint)."
            )
        elif 'optimizer' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as e:  # pragma: no cover
                self.logger.warning(f"Optimizer state not restored: {e}")

        if reset_scheduler:
            self.scheduler = self._build_scheduler()
            self._align_scheduler_to_epoch(self.current_epoch)
            self.logger.info(
                f"LR scheduler rebuilt and aligned to epoch {self.current_epoch}."
            )
        elif 'scheduler' in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            except Exception as e:  # pragma: no cover
                self.logger.warning(f"Scheduler state not restored: {e}")

        if self.is_main:
            self._log_optimizer_lr()
        self.logger.info(
            f"Resumed weights from {path} (checkpoint epoch {ckpt_epoch}) "
            f"→ training from epoch {self.current_epoch}"
        )
