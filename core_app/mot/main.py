"""
Entry point for training the Surgical MOT system.

Single-GPU usage:

    python -m core_app.mot.main \
        --fname configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml \
        --devices cuda:0 \
        --debugmode False

Multi-GPU DDP usage (3 L40S):

    torchrun --standalone --nproc_per_node=3 \
        -m core_app.mot.main \
        --fname configs/... \
        --devices cuda

The config's ``meta.stage`` key chooses the stage (``stage1_supervised``,
``stage2_jepa``, ``stage3_joint``, or ``stage4_full``).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# DINOv2 torch.hub uses xFormers MemEffAttention when import succeeds, but current
# xformers builds lack GB10 (sm_120) kernels and reject float32. Force PyTorch SDPA.
os.environ.setdefault("XFORMERS_DISABLED", "1")

import yaml
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core_app.mot.data import MOTCholecDataset, mot_collate_fn
from core_app.mot.trainer import MOTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Surgical MOT system (GOT-stack)')
    parser.add_argument('--fname', type=str, required=True, help='Path to config YAML.')
    parser.add_argument('--devices', nargs='+', default=['cuda:0'], help='GPU devices (first used).')
    parser.add_argument('--debugmode', type=lambda s: s.lower() == 'true', default=False,
                        help='Reduce dataset size for quick smoke tests.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from.')
    parser.add_argument(
        '--reset-optimizer',
        action='store_true',
        help='On resume: discard Adam state and use optimization.lr from config (recommended after instability).',
    )
    parser.add_argument(
        '--reset-scheduler',
        action='store_true',
        help='On resume: rebuild cosine schedule (auto-enabled with --reset-optimizer).',
    )
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=None,
        metavar='N',
        help='On resume: begin training at epoch N (e.g. 62 to re-run an epoch). Default: checkpoint_epoch + 1.',
    )
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for DDP (set automatically by torchrun).')
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _is_valid_cholectrack_root(path: Path) -> bool:
    from core_app.data.paths import is_mot_dataset_root
    return is_mot_dataset_root(path)


def apply_dataset_root_overrides(config: dict) -> None:
    data_cfg = config.setdefault('data', {})
    datasets = data_cfg.get('datasets')
    if not isinstance(datasets, list) or not datasets:
        return

    def _parse_root_list(raw_env: str) -> list[str]:
        return [
            Path(p).expanduser().resolve().as_posix()
            for p in raw_env.split(',')
            if p.strip()
        ]

    # Optional explicit multi-root override for training-only data augmentation:
    #   MOT_TRAIN_DATA_ROOTS=/path/ct20,/path/cholec80_train
    # The first entry should remain CT20 (for validation/eval continuity).
    explicit_train_roots = os.getenv('MOT_TRAIN_DATA_ROOTS')
    if explicit_train_roots:
        train_roots = _parse_root_list(explicit_train_roots)
        if not train_roots:
            raise FileNotFoundError('[CONFIG] MOT_TRAIN_DATA_ROOTS is empty.')
        for root in train_roots:
            if not Path(root).exists():
                raise FileNotFoundError(f"[CONFIG] Requested training root does not exist: {root}")
        data_cfg['train_datasets'] = train_roots
        data_cfg['datasets'][0] = train_roots[0]
        return

    # Fallback that supports overriding all configured dataset roots for both train/val
    # via a comma-separated env var.
    explicit_root = os.getenv('MOT_DATA_ROOT')
    if explicit_root:
        explicit_root = Path(explicit_root).expanduser().resolve()
        if explicit_root.exists():
            datasets[0] = str(explicit_root)
            return
        raise FileNotFoundError(
            f"[CONFIG] Requested MOT_DATA_ROOT does not exist: {explicit_root}"
        )

    explicit_roots = os.getenv('MOT_DATA_ROOTS')
    if explicit_roots:
        explicit_list = _parse_root_list(explicit_roots)
        if not explicit_list:
            raise FileNotFoundError('[CONFIG] MOT_DATA_ROOTS is empty.')
        for root in explicit_list:
            if not Path(root).exists():
                raise FileNotFoundError(f"[CONFIG] Requested data root does not exist: {root}")
        data_cfg['datasets'] = explicit_list
        return

    from core_app.data.paths import resolve_mot_dataset_root

    resolved = []
    for root in datasets:
        p = resolve_mot_dataset_root(root)
        if p is not None:
            resolved.append(str(p))
        else:
            resolved.append(str(Path(root).expanduser()))
    data_cfg['datasets'] = resolved


def build_dataloaders(config: dict, debug: bool = False, ddp: bool = False):
    data_cfg = config.get('data', {})
    clip_length = data_cfg.get('clip_length', 3)
    img_size = data_cfg.get('img_size', 392)
    roots = data_cfg.get('datasets', [])
    train_roots = data_cfg.get('train_datasets', None)
    if isinstance(train_roots, str):
        train_roots = [train_roots]
    if not train_roots:
        train_roots = roots[:1]
    assert roots, "config.data.datasets must contain at least one root path."

    stage = config.get('meta', {}).get('stage', 'stage1_supervised')
    per_track_min = int(data_cfg.get('per_track_min_visible_frames', 3 if stage != 'stage2_jepa' else 1))

    train_datasets = [
        MOTCholecDataset(
            data_root=root,
            split='train',
            clip_length=clip_length,
            img_size=img_size,
            training=True,
            per_track_min_visible_frames=per_track_min,
        )
        for root in train_roots
    ]
    train_ds = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)

    val_root = roots[0]
    val_ds = MOTCholecDataset(
        data_root=val_root,
        split='val',
        clip_length=clip_length,
        img_size=img_size,
        training=False,
        per_track_min_visible_frames=per_track_min,
    )

    if debug:
        train_ds.clips = train_ds.clips[:32]
        val_ds.clips = val_ds.clips[:16]
        if not ddp or dist.get_rank() == 0:
            print(f"[debug] {len(train_ds.clips)} train / {len(val_ds.clips)} val clips")

    batch_size = data_cfg.get('batch_size', 2)
    num_workers = data_cfg.get('num_workers', 4)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=data_cfg.get('pin_memory', True),
        collate_fn=mot_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=data_cfg.get('pin_memory', True),
        collate_fn=mot_collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler, val_sampler


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    )

    args = parse_args()
    config = load_config(args.fname)
    apply_dataset_root_overrides(config)

    # Detect DDP via torchrun env vars (RANK, LOCAL_RANK, WORLD_SIZE).
    ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if ddp:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        # Prefer NCCL for GPU; fall back to gloo if PyTorch lacks NCCL build.
        if dist.is_nccl_available() and torch.cuda.is_available():
            backend = 'nccl'
        else:
            backend = 'gloo'
            if torch.cuda.is_available() and not dist.is_nccl_available():
                print(
                    "[WARN] PyTorch was built without NCCL. Using 'gloo' backend instead. "
                    "GPU communication will be slower. To fix, reinstall PyTorch with CUDA support:\n"
                    "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
                )
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
        is_main = rank == 0
    else:
        device = args.devices[0]
        rank = 0
        world_size = 1
        is_main = True

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if is_main:
        print(f"Loading config: {args.fname}")
        print(f"Stage: {config.get('meta', {}).get('stage', 'stage1_supervised')}")
        print(f"Device: {device} | DDP: {ddp} | World size: {world_size}")

    train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(
        config, debug=args.debugmode, ddp=ddp
    )
    if is_main:
        print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Pre-download DINOv2 on rank 0 to avoid torch.hub cache race across DDP ranks.
    model_cfg = config.get('model', {})
    if is_main and model_cfg.get('encoder_type') == 'dinov2' and model_cfg.get('use_torch_hub', True):
        print("Pre-downloading DINOv2 on rank 0...")
        _model_name = model_cfg.get('model_name', 'dinov2_vits14')
        torch.hub.load('facebookresearch/dinov2', _model_name)
        print("DINOv2 download complete.")
    if ddp:
        dist.barrier()

    trainer = MOTTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        rank=rank,
        world_size=world_size,
        is_main=is_main,
    )

    if args.resume:
        if is_main:
            print(f"Resuming from {args.resume}")
            if args.reset_optimizer:
                print("  → reset optimizer + scheduler from config")
            if args.start_epoch is not None:
                print(f"  → start at epoch {args.start_epoch}")
        trainer.load_checkpoint(
            args.resume,
            reset_optimizer=args.reset_optimizer,
            reset_scheduler=args.reset_scheduler or args.reset_optimizer,
            start_epoch=args.start_epoch,
        )

    num_epochs = config.get('optimization', {}).get('epochs', 15)
    start_epoch = trainer.current_epoch if args.resume else 0
    if is_main and args.resume:
        print(f"Training epochs {start_epoch}..{num_epochs - 1} (resume)")
    import traceback
    exit_reason = "normal completion"
    try:
        for epoch in range(start_epoch, num_epochs):
            if ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_stats = trainer.train_epoch(epoch)
            val_stats = trainer.validate(epoch) if val_loader is not None else {}
            if is_main:
                trainer._epoch_end(epoch, train_stats, val_stats)
    except KeyboardInterrupt:
        exit_reason = "KeyboardInterrupt (Ctrl+C)"
        if is_main:
            print(f"[rank {rank}] Training interrupted by user (KeyboardInterrupt)")
    except Exception as e:
        exit_reason = f"exception: {type(e).__name__}: {e}"
        print(f"[rank {rank}] Training crashed: {exit_reason}", flush=True)
        traceback.print_exc()
        raise
    finally:
        if is_main:
            print(f"Training ended. Reason: {exit_reason}")
        if ddp:
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
