# DINOv2 pretrained weights (Git LFS)

| File | Backbone |
|------|----------|
| `dinov2_vits14_pretrain.pth` | ViT-S/14 (384-dim) — Stage 1–4 ViT-S runs |
| `dinov2_vitb14_pretrain.pth` | ViT-B/14 (768-dim) — LoRA detection experiment |

Loaded via `torch.hub.load('facebookresearch/dinov2', model_name)` or local path when offline.

```bash
git lfs pull --include="weights/dinov2/*"
```
