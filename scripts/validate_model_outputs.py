#!/usr/bin/env python3
"""Validate MOT model with visual diagnostics: overlays, activation maps, t-SNE, depth."""
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
import numpy as np
from PIL import Image, ImageDraw

from core_app.mot.trainer import build_model_from_config

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from sklearn.manifold import TSNE
    HAS_SKL = True
except Exception:
    HAS_SKL = False

CONFIG_PATH = REPO_ROOT / "configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml"
CKPT_PATH = REPO_ROOT / "outputs/mot/cholec20-stage1-supervised/best.pth.tar"
VIDEO_DIR = Path("/scratch/kcwp264/data/surgi_world_track/cholectrack20/Training/VID01/Frames")


def load_frames(d, n=12):
    f = sorted(d.glob("*.png")) or sorted(d.glob("*.jpg"))
    print(f"Found {len(f)} frames, loading {min(len(f), n)}...")
    return [Image.open(p).convert("RGB") for p in f[:n]]


def preprocess(frames, sz=392):
    tr = T.Compose([T.Resize((sz, sz)), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return torch.stack([tr(f) for f in frames], dim=1).unsqueeze(0)


def _cxcywh_to_xyxy(box, w, h):
    cx, cy, bw, bh = box
    return (max(0, (cx - bw / 2) * w), max(0, (cy - bh / 2) * h),
            min(w - 1, (cx + bw / 2) * w), min(h - 1, (cy + bh / 2) * h))


def draw_overlay(im, tracks, thresh):
    out = im.convert("RGB").copy()
    dr = ImageDraw.Draw(out)
    pal = [(255, 90, 90), (90, 140, 255), (90, 220, 130), (230, 170, 60), (170, 90, 220), (60, 200, 230), (220, 60, 170)]
    W, H = out.size
    for i, t in enumerate(tracks):
        s = float(t.get("score", 0.0))
        if s < thresh:
            continue
        x1, y1, x2, y2 = _cxcywh_to_xyxy(t.get("bbox", []), W, H)
        c = pal[i % len(pal)]
        dr.rectangle([x1, y1, x2, y2], outline=c, width=2)
        dr.text((x1 + 2, max(0, y1 - 12)), f"id={t.get('id', '?')} cls={t.get('class', 'n/a')} s={s:.2f}", fill=c)
    return out


def save_act(tokens, img_sz, path):
    if not HAS_MPL:
        return
    N, C = tokens.shape
    g = tokens[1:] if N > 500 else tokens
    hw = int(g.size(0) ** 0.5)
    if hw * hw != g.size(0):
        return
    a = g.mean(-1).reshape(hw, hw).cpu().numpy()
    a = F.interpolate(torch.from_numpy(a)[None, None], (img_sz, img_sz), mode="bilinear", align_corners=False)[0, 0].numpy()
    plt.imsave(path, a, cmap="viridis")
    print(f"  Activation map: {path}")


def save_detr_hm(logits, boxes, img_sz, path, thresh=0.1):
    if not HAS_MPL:
        return
    s, cl = logits.sigmoid().max(-1)
    s_np, b_np = s.cpu().numpy(), boxes.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.ones((img_sz, img_sz, 3)) * 0.2)
    cm = plt.cm.plasma
    for sc, b in zip(s_np, b_np):
        if sc < thresh:
            continue
        cx, cy, w, h = b
        x1 = (cx - w / 2) * img_sz
        y1 = (cy - h / 2) * img_sz
        rect = plt.Rectangle((x1, y1), w * img_sz, h * img_sz, linewidth=2, edgecolor=cm(sc), facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1, f"{sc:.2f}", color="white", fontsize=7)
    ax.set_xlim(0, img_sz)
    ax.set_ylim(img_sz, 0)
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  DETR heatmap: {path}")


def save_tsne(embs, cls, path):
    if not HAS_SKL or not HAS_MPL or embs.size(0) < 3:
        return
    e, c = embs.cpu().numpy(), cls.cpu().numpy()
    p = min(30, e.shape[0] - 1)
    pts = TSNE(n_components=2, perplexity=p, random_state=42, init="pca").fit_transform(e)
    fig, ax = plt.subplots(figsize=(6, 6))
    pal = plt.cm.tab10(np.linspace(0, 1, 10))
    for u in np.unique(c):
        m = c == u
        ax.scatter(pts[m, 0], pts[m, 1], c=[pal[u % 10]], label=f"cls {u}", s=60, alpha=0.8, edgecolors="k", linewidths=0.5)
    ax.legend(title="Class", fontsize=8)
    ax.set_title("ReID t-SNE")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ReID t-SNE: {path}")


def save_depth(dm, path):
    if not HAS_MPL:
        return
    d = dm.squeeze().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(d, cmap="magma")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Depth map: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--save-dir", type=Path, default=REPO_ROOT / "outputs/validation")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 70 + "\nModel Validation — Visual Inference Diagnostics\n" + "=" * 70)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading model...")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = build_model_from_config(cfg)
    if args.ckpt.exists():
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd.get("model", sd.get("model_state_dict", sd)), strict=False)
        print(f"  CKPT: {args.ckpt}")
    else:
        print("  WARNING: no checkpoint found — random init!")
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    img_sz = cfg["data"].get("img_size", 392)
    clip_len = cfg["data"].get("clip_length", 3)
    print(f"  Device: {dev}, img_size={img_sz}, clip_len={clip_len}")

    print(f"\n[2/4] Loading frames from {args.video_dir}...")
    frames = load_frames(args.video_dir, args.max_frames)
    if len(frames) < clip_len:
        print(f"Need >= {clip_len} frames. Exiting.")
        return

    print(f"\n[3/4] Running inference...")
    overlays, embs, clss, scrs, acts, dms, detr_logits, detr_boxes = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for i in range(len(frames) - clip_len + 1):
            clip = preprocess(frames[i:i + clip_len], img_sz).to(dev)
            reality_seq, neck_out = model.encode_frames(clip)
            acts.append(reality_seq[0, -1].cpu())
            out = model(current_video=clip, mode="infer")
            active = out.get("active_tracks", [])
            tracks = []
            for t in active:
                tracks.append(t if isinstance(t, dict) else {
                    "id": getattr(t, "id", None), "bbox": getattr(t, "bbox", []),
                    "score": float(getattr(t, "score", 0.0)), "class": getattr(t, "cls", None),
                })
            overlays.append((i, tracks))
            print(f"  Window [{i:02d}:{i + clip_len:02d}] tracks={len(active)}")

            detr = out.get("detr", {})
            if "pred" in detr:
                logits = detr["pred"]["class_logits"][0]
                boxes = detr["pred"]["pred_boxes"][0]
                s, c = logits.sigmoid().max(-1)
                k = s > args.score_threshold
                if k.any():
                    re = out.get("reid", {}).get("embeddings")
                    if re is not None:
                        embs.append(re[0][k].cpu())
                        clss.append(c[k].cpu())
                        scrs.append(s[k].cpu())
                detr_logits.append(logits.cpu())
                detr_boxes.append(boxes.cpu())

            if model.occusolver is not None and hasattr(model.occusolver, "depth_estimator"):
                try:
                    dm = model.occusolver.depth_estimator(clip[:, -1])
                    dms.append(dm.cpu())
                except Exception:
                    pass

    print(f"\n[4/4] Saving visualizations to {args.save_dir}...")
    for i, tracks in overlays:
        draw_overlay(frames[i + clip_len - 1], tracks, args.score_threshold).save(args.save_dir / f"overlay_{i:02d}.png")
    print(f"  Overlays: {len(overlays)} frames")

    if acts:
        save_act(acts[-1], img_sz, args.save_dir / "encoder_activation.png")
    if detr_logits and detr_boxes:
        save_detr_hm(detr_logits[-1], detr_boxes[-1], img_sz, args.save_dir / "detr_heatmap.png")
    if embs:
        all_e = torch.cat(embs, dim=0)
        all_c = torch.cat(clss, dim=0)
        save_tsne(all_e, all_c, args.save_dir / "reid_tsne.png")
    if dms:
        save_depth(dms[-1][0] if dms[-1].dim() == 4 else dms[-1], args.save_dir / "depth_map.png")

    print("\n" + "=" * 70 + "\nValidation complete!\n" + "=" * 70)


if __name__ == "__main__":
    main()
