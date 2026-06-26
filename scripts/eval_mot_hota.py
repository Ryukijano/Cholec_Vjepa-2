#!/usr/bin/env python3
"""
HOTA / MOTA evaluation on CholecTrack20 with smoke-stratified breakdown.

Usage:
    python scripts/eval_mot_hota.py \
        --config configs/train_mot/dinov2/cholec20-mot-stage3-joint-finetune-vits.yaml \
        --checkpoint outputs/mot/cholec20-stage3-joint-finetune-vits/best.pth.tar \
        --split test \
        --max-frames 120
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from core_app.data.splits import CT20_TEST
from core_app.mot.eval import (
    _ct20_json_path,
    build_ct20_gt_mot_rows,
    build_frame_challenges,
    ct20_split_video_dirs,
    evaluate_stratified_mot,
    load_ct20_video_tensor,
    mot_rollout,
    resolve_ct20_frame_source,
)
from core_app.mot.trainer import build_model_from_config
from core_app.utils.checkpoint import load_checkpoint

_LOG = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate MOT checkpoint with smoke-stratified HOTA')
    p.add_argument('--config', required=True, help='YAML config path')
    p.add_argument('--checkpoint', required=True, help='Checkpoint .pth.tar')
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--data-root', default=None, help='Override CT20 root')
    p.add_argument('--output-dir', default='outputs/mot/eval_hota', help='Eval artefact root')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-frames', type=int, default=0, help='Limit frames per video (0=all)')
    p.add_argument('--max-videos', type=int, default=0, help='Limit number of videos (0=all)')
    p.add_argument(
        '--strata',
        nargs='+',
        default=['overall', 'smoke', 'no_smoke'],
        help='Challenge strata from evaluate_stratified_mot',
    )
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config.get('data', {})
    img_size = int(data_cfg.get('img_size', 392))
    clip_length = int(data_cfg.get('clip_length', 3))
    data_root = Path(args.data_root or data_cfg['datasets'][0])

    model = build_model_from_config(config)
    model.eval()
    model.to(args.device)
    ckpt = load_checkpoint(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt['model'], strict=False)
    _LOG.info('Loaded checkpoint epoch %s from %s', ckpt.get('epoch', '?'), args.checkpoint)

    model.configure_tracker_for_eval()

    video_dirs = ct20_split_video_dirs(data_root, args.split)
    if args.split == 'test':
        video_dirs = [d for d in video_dirs if d.name in CT20_TEST]
    if args.max_videos > 0:
        video_dirs = video_dirs[: args.max_videos]

    per_video: Dict[str, Dict[str, Dict[str, float]]] = {}
    aggregate: Dict[str, List[Dict[str, float]]] = {s: [] for s in args.strata}

    for video_dir in video_dirs:
        json_path = _ct20_json_path(video_dir)
        frame_numbers, frames_dir, mp4_path = resolve_ct20_frame_source(video_dir)
        if args.max_frames > 0:
            frame_numbers = frame_numbers[: args.max_frames]
        if not frame_numbers:
            _LOG.warning('Skipping %s — no frames', video_dir.name)
            continue

        with open(json_path) as f:
            annotations = json.load(f).get('annotations', {})
        frame_challenges = build_frame_challenges(annotations, frame_numbers)
        gt_rows = build_ct20_gt_mot_rows(json_path, frame_numbers, img_size, img_size)

        frames = load_ct20_video_tensor(
            frames_dir or (video_dir / 'Frames'),
            frame_numbers,
            img_size=img_size,
            mp4_path=mp4_path,
        )
        pred_rows = mot_rollout(
            model,
            frames,
            clip_len=clip_length,
            img_size=img_size,
            device=torch.device(args.device),
        )

        strata_metrics = evaluate_stratified_mot(
            gt_rows,
            pred_rows,
            frame_challenges,
            seq_length=len(frame_numbers),
            width=img_size,
            height=img_size,
            strata=tuple(args.strata),
        )
        per_video[video_dir.name] = strata_metrics
        for stratum, vals in strata_metrics.items():
            aggregate.setdefault(stratum, []).append(vals)

        smoky = sum(1 for flags in frame_challenges.values() if flags.get('smoke'))
        _LOG.info(
            '%s: %d frames (%d smoky), GT=%d pred=%d',
            video_dir.name,
            len(frame_numbers),
            smoky,
            len(gt_rows),
            len(pred_rows),
        )

    if not per_video:
        raise RuntimeError('No sequences evaluated — check data paths and split.')

    headline: Dict[str, float] = {}
    for stratum, vals_list in aggregate.items():
        if not vals_list:
            continue
        for key in ('HOTA', 'MOTA', 'IDF1'):
            nums = [v[key] for v in vals_list if key in v]
            if nums:
                headline[f'{stratum}_{key}'] = sum(nums) / len(nums)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        'checkpoint': str(args.checkpoint),
        'split': args.split,
        'max_frames': args.max_frames,
        'per_video': per_video,
        'headline': headline,
    }
    summary_path = out_root / 'smoke_stratified_metrics.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    _LOG.info('Headline metrics: %s', headline)
    _LOG.info('Wrote %s', summary_path)


if __name__ == '__main__':
    main()
