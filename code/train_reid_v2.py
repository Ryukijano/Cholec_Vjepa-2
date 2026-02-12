#!/usr/bin/env python3
"""
Train direction-aware Re-ID head with:
  - Frozen RF-DETR detector/query features
  - LoRA-tuned V-JEPA2 encoder
  - Triplet + contrastive + direction losses
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
vjepa2_path = project_root / "vjepa2"
src_path = project_root / "vjepa2" / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(vjepa2_path))
sys.path.insert(0, str(current_dir))

from vjepa2.app.vjepa.utils import init_video_model
from vjepa2.app.vjepa_cholec80.lora import apply_lora_to_encoder, load_checkpoint_into_lora_model

from dataset import CholecDetectDataset, collate_fn
from reid_head import EmbeddingMemoryBank, TrackContrastiveLoss, TripletLoss
from reid_head_v2 import ReIDHeadV2, direction_proxy_targets
from rfdetr_wrapper import RFDETRFeatureWrapper


logging.basicConfig(level=logging.INFO, format="[%(levelname)-8s][%(name)-20s] %(message)s")
logger = logging.getLogger("train_reid_v2")


def _build_rfdetr_model(model_size: str):
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase

    mapping = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
        "base": RFDETRBase,
    }
    return mapping[model_size]()


def _extract_last_frames_as_pil(clips: torch.Tensor) -> List:
    """
    clips: [B, C, T, H, W] float in [0,1]
    """
    from PIL import Image
    import numpy as np

    frames = []
    b = clips.shape[0]
    for i in range(b):
        arr = clips[i, :, -1].detach().cpu().permute(1, 2, 0).numpy()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(arr))
    return frames


def _gather_valid_reid_samples(embs, tids, dir_logits=None, dir_targets=None):
    emb_rows = []
    tid_rows = []
    dir_logit_rows = [] if dir_logits is not None else None
    dir_tgt_rows = [] if dir_targets is not None else None

    for i in range(len(embs)):
        if embs[i].numel() == 0:
            continue
        valid = tids[i] >= 0
        if valid.sum() == 0:
            continue
        emb_rows.append(embs[i][valid])
        tid_rows.append(tids[i][valid])
        if dir_logits is not None:
            dir_logit_rows.append(dir_logits[i][valid])
        if dir_targets is not None:
            dir_tgt_rows.append(dir_targets[i][valid])

    if not emb_rows:
        return None
    emb_cat = torch.cat(emb_rows, dim=0)
    tid_cat = torch.cat(tid_rows, dim=0)
    dir_logit_cat = torch.cat(dir_logit_rows, dim=0) if dir_logit_rows is not None and dir_logit_rows else None
    dir_tgt_cat = torch.cat(dir_tgt_rows, dim=0) if dir_tgt_rows is not None and dir_tgt_rows else None
    return emb_cat, tid_cat, dir_logit_cat, dir_tgt_cat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="../cholec_dataset/Training")
    parser.add_argument("--val_dir", type=str, default="../cholec_dataset/Validation")
    parser.add_argument("--checkpoint", type=str, default="", help="V-JEPA2 SSL/detection/joint checkpoint")
    parser.add_argument("--out_dir", type=str, default="../outputs/reid_v2")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--lr_lora", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.7)
    parser.add_argument("--memory_bank_size", type=int, default=8192)
    parser.add_argument("--rfdetr_model_size", choices=["nano", "small", "medium", "large", "base"], default="medium")
    parser.add_argument("--rfdetr_score_thresh", type=float, default=0.3)
    parser.add_argument("--num_direction_bins", type=int, default=3)
    parser.add_argument("--reid_dim", type=int, default=128)
    parser.add_argument("--phase_a_epochs", type=int, default=10,
                        help="Epochs with App+Motion+Dir in fusion, Box+Class disabled.")
    parser.add_argument("--phase_b_epochs", type=int, default=25,
                        help="After this, enable Box+Class in fusion as well.")
    parser.add_argument("--gate_entropy_lambda", type=float, default=0.0,
                        help="Optional entropy regularization on fusion gates.")
    parser.add_argument("--box_noise_std", type=float, default=0.0,
                        help="Std of Gaussian noise added to (cx,cy) when training.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    train_ds = CholecDetectDataset(args.train_dir, load_track_ids=True)
    val_ds = CholecDetectDataset(args.val_dir, load_track_ids=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # V-JEPA2 encoder + LoRA
    encoder, _ = init_video_model(
        device=device,
        model_name="vit_large",
        patch_size=16,
        max_num_frames=16,
        tubelet_size=2,
        crop_size=224,
        pred_depth=12,
        pred_embed_dim=384,
        use_mask_tokens=True,
        use_sdpa=True,
    )
    apply_lora_to_encoder(encoder, rank=16, alpha=16.0, start_layer=12)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if "encoder_lora" in ckpt:
            load_checkpoint_into_lora_model(encoder, ckpt["encoder_lora"])
        elif "encoder" in ckpt:
            load_checkpoint_into_lora_model(encoder, ckpt["encoder"])
    encoder.to(device).train()

    # RF-DETR is only used during inference (run_tracking.py), not training
    # During training we use GT boxes + V-JEPA2 features only
    # rf_model = _build_rfdetr_model(args.rfdetr_model_size)
    # rf_wrap = RFDETRFeatureWrapper(rf_model)

    # ReID head v2
    reid_head = ReIDHeadV2(
        embed_dim=1024,
        reid_dim=args.reid_dim,
        num_direction_bins=args.num_direction_bins,
    ).to(device)

    # Losses
    triplet = TripletLoss(margin=args.margin)
    contrastive = TrackContrastiveLoss(temp_start=0.15, temp_final=0.07, warmup_epochs=5)
    direction_ce = torch.nn.CrossEntropyLoss()
    mem_bank = EmbeddingMemoryBank(capacity=args.memory_bank_size, embed_dim=args.reid_dim, device=device)

    params = []
    lora_params = [p for n, p in encoder.named_parameters() if p.requires_grad]
    params.append({"params": lora_params, "lr": args.lr_lora, "weight_decay": args.weight_decay})
    params.append({"params": reid_head.parameters(), "lr": args.lr_head, "weight_decay": args.weight_decay})
    optimizer = optim.AdamW(params)

    best_val = -1.0
    for epoch in range(args.epochs):
        encoder.train()
        reid_head.train()
        contrastive.set_epoch(epoch)
        running = {"trip": 0.0, "cont": 0.0, "dir": 0.0, "n": 0}
        gate_sums = torch.zeros(4)
        gate_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            clips = batch[0].to(device, non_blocking=True)
            boxes = [b.to(device) for b in batch[1]]
            tids = [t.to(device) for t in batch[3]]
            classes = [c.to(device) for c in batch[2]]

            # Curriculum on branches
            # Note: Appearance branch is always disabled during training (RF-DETR features not used)
            reid_head.enable_app_in_fusion = False
            
            if epoch < args.phase_a_epochs:
                # Phase A: Dir + Motion only (App disabled, Box+Class off)
                reid_head.enable_dir_in_fusion = True
                reid_head.enable_box_in_fusion = False
            elif epoch < args.phase_b_epochs:
                # Phase B: still keep Box+Class off for safety
                reid_head.enable_dir_in_fusion = True
                reid_head.enable_box_in_fusion = False
            else:
                # Phase C: Dir + Motion + Box+Class (App still disabled)
                reid_head.enable_dir_in_fusion = True
                reid_head.enable_box_in_fusion = True

            # Optional coordinate noise for Box+Class robustness
            if args.box_noise_std > 0 and reid_head.enable_box_in_fusion:
                noisy_boxes = []
                for b in boxes:
                    if b.numel() == 0:
                        noisy_boxes.append(b)
                        continue
                    nb = b.clone()
                    noise = torch.randn_like(nb[:, 0:2]) * args.box_noise_std
                    nb[:, 0:2] = (nb[:, 0:2] + noise).clamp(0.0, 1.0)
                    noisy_boxes.append(nb)
                boxes_used = noisy_boxes
            else:
                boxes_used = boxes

            dir_tgts = direction_proxy_targets(boxes_used, bins=args.num_direction_bins)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                # V-JEPA2 full clip tokens
                full_tokens = encoder([clips])[0]

                # During training, we use GT boxes and V-JEPA2 features only.
                # RF-DETR query features are only used during inference in run_tracking.py
                # (predictions don't align with GT boxes, causing shape mismatch)
                embs, dir_logits = reid_head(
                    full_tokens=full_tokens,
                    boxes=boxes_used,
                    query_features=None,  # Skip RF-DETR features during training
                    classes=classes,
                )

                packed = _gather_valid_reid_samples(embs, tids, dir_logits, dir_tgts)
                if packed is None:
                    continue
                emb_all, tid_all, dir_logit_all, dir_tgt_all = packed

                bank_embs, bank_tids = mem_bank.get()
                loss_trip = triplet(emb_all, tid_all, bank_embs=bank_embs, bank_tids=bank_tids)
                loss_cont = contrastive(emb_all, tid_all, bank_embs=bank_embs, bank_tids=bank_tids)
                loss_dir = direction_ce(dir_logit_all, dir_tgt_all)

                # Optional gate entropy regularization
                loss_gate = 0.0
                if args.gate_entropy_lambda > 0.0 and reid_head._last_gates is not None:
                    g = reid_head._last_gates  # [N_det, 4]
                    # clamp for numerical stability
                    g_safe = g.clamp(min=1e-6)
                    entropy = -(g_safe * g_safe.log()).sum(dim=-1)  # per-sample H
                    loss_gate = -args.gate_entropy_lambda * entropy.mean()

                loss = 0.5 * loss_trip + 0.5 * loss_cont + 0.2 * loss_dir + loss_gate

            loss.backward()
            torch.nn.utils.clip_grad_norm_(reid_head.parameters(), 5.0)
            optimizer.step()

            with torch.no_grad():
                mem_bank.enqueue(emb_all, tid_all)

            running["trip"] += float(loss_trip.item())
            running["cont"] += float(loss_cont.item())
            running["dir"] += float(loss_dir.item())
            running["n"] += 1
            # Accumulate gate statistics
            if reid_head._last_gates is not None and reid_head._last_gates.numel() > 0:
                gate_sums += reid_head._last_gates.mean(dim=0).detach().cpu()
                gate_count += 1
            pbar.set_postfix(
                trip=f"{running['trip']/max(running['n'],1):.3f}",
                cont=f"{running['cont']/max(running['n'],1):.3f}",
                dir=f"{running['dir']/max(running['n'],1):.3f}",
            )

        # quick validation proxy: positive-pair cosine separation
        encoder.eval()
        reid_head.eval()
        pos_sim = []
        neg_sim = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                clips = batch[0].to(device, non_blocking=True)
                boxes = [b.to(device) for b in batch[1]]
                tids = [t.to(device) for t in batch[3]]
                classes = [c.to(device) for c in batch[2]]
                qf = [None] * len(boxes)

                full_tokens = encoder([clips])[0]
                embs, _ = reid_head(full_tokens, boxes, query_features=qf, classes=classes)
                packed = _gather_valid_reid_samples(embs, tids)
                if packed is None:
                    continue
                emb_all, tid_all, _, _ = packed
                if emb_all.shape[0] < 2:
                    continue
                sim = emb_all @ emb_all.t()
                same = tid_all.unsqueeze(0) == tid_all.unsqueeze(1)
                eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
                pos = sim[same & ~eye]
                neg = sim[~same]
                if pos.numel() > 0:
                    pos_sim.append(pos.mean().item())
                if neg.numel() > 0:
                    neg_sim.append(neg.mean().item())

        pos_m = sum(pos_sim) / max(len(pos_sim), 1)
        neg_m = sum(neg_sim) / max(len(neg_sim), 1)
        score = pos_m - neg_m

        if gate_count > 0:
            gate_mean = gate_sums / gate_count
            logger.info(
                "Epoch %d gate means [dir, motion, app, box]=[%.3f, %.3f, %.3f, %.3f]",
                epoch + 1,
                gate_mean[0].item(),
                gate_mean[1].item(),
                gate_mean[2].item(),
                gate_mean[3].item(),
            )
        else:
            logger.info("Epoch %d gate means: (no detections this epoch)", epoch + 1)

        logger.info(
            "Epoch %d | trip=%.4f cont=%.4f dir=%.4f | val pos=%.4f neg=%.4f sep=%.4f",
            epoch + 1,
            running["trip"] / max(running["n"], 1),
            running["cont"] / max(running["n"], 1),
            running["dir"] / max(running["n"], 1),
            pos_m,
            neg_m,
            score,
        )

        ckpt = {
            "epoch": epoch + 1,
            "encoder_lora": {k: v.cpu() for k, v in encoder.state_dict().items() if "lora" in k.lower()},
            "reid_head_v2": reid_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_separation": score,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if score > best_val:
            best_val = score
            torch.save(ckpt, out_dir / "best.pt")

    rf_wrap.close()
    logger.info("Training finished. Best separation: %.4f", best_val)


if __name__ == "__main__":
    main()
