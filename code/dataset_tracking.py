"""
Track-aware dataset for Re-ID training on CholecTrack20.

Loads consecutive frame pairs with track ID annotations for
contrastive Re-ID learning. Each sample contains two frames
from the same video with shared track IDs for positive pairs.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CholecTrackPairDataset(Dataset):
    """Dataset that yields frame pairs with track ID annotations.
    
    For Re-ID training, we need pairs of frames where the same tool
    instance appears in both frames (positive pair) and different
    instances appear (negative pair).
    
    Each sample returns:
        clip_a, clip_b: [C, T, H, W] video clips for two frames
        boxes_a, boxes_b: [N, 4] cxcywh boxes
        labels_a, labels_b: [N] class labels
        track_ids_a, track_ids_b: [N] intraoperative track IDs
    """
    
    def __init__(self, split_dir: str, clip_len: int = 16, frame_size: int = 224,
                 max_frame_gap: int = 5, track_perspective: str = 'intraoperative'):
        self.clip_len = clip_len
        self.frame_size = frame_size
        self.max_frame_gap = max_frame_gap
        self.track_key = f'{track_perspective}_track'
        
        split_dir = Path(split_dir)
        self.pairs = []
        
        video_dirs = sorted([p for p in split_dir.iterdir()
                             if p.is_dir() and p.name.startswith('VID')])
        
        for vdir in video_dirs:
            json_path = next(vdir.glob('*.json'), None)
            if not json_path:
                continue
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            anns = data.get('annotations', {})
            fids = sorted([int(k) for k in anns.keys()])
            
            # Build frame index
            frame_data = {}
            for fid in fids:
                frame_anns = anns[str(fid)]
                boxes, labels, track_ids = [], [], []
                for ann in frame_anns:
                    inst = ann.get('instrument', -1)
                    bbox = ann.get('tool_bbox')
                    tid = ann.get(self.track_key, -1)
                    if inst is None or inst < 0 or bbox is None or tid < 0:
                        continue
                    if bbox[2] <= 0 or bbox[3] <= 0 or bbox[0] < -0.5:
                        continue
                    boxes.append(bbox)
                    labels.append(int(inst))
                    track_ids.append(int(tid))
                
                if len(labels) > 0:
                    frame_data[fid] = {
                        'boxes': boxes, 'labels': labels,
                        'track_ids': track_ids, 'vdir': vdir
                    }
            
            # Create pairs: consecutive frames with shared track IDs
            valid_fids = sorted(frame_data.keys())
            for i in range(len(valid_fids)):
                fid_a = valid_fids[i]
                tids_a = set(frame_data[fid_a]['track_ids'])
                
                # Find nearby frames with shared tracks
                for j in range(i + 1, min(i + max_frame_gap + 1, len(valid_fids))):
                    fid_b = valid_fids[j]
                    tids_b = set(frame_data[fid_b]['track_ids'])
                    
                    shared = tids_a & tids_b
                    if len(shared) > 0:
                        self.pairs.append({
                            'vdir': vdir,
                            'fid_a': fid_a,
                            'fid_b': fid_b,
                            'data_a': frame_data[fid_a],
                            'data_b': frame_data[fid_b],
                            'all_fids': fids,
                        })
        
        print(f"Loaded {len(self.pairs)} frame pairs from {len(video_dirs)} videos")
    
    def __len__(self):
        return len(self.pairs)
    
    def _load_clip(self, vdir: Path, target_fid: int, all_fids: List[int]) -> torch.Tensor:
        """Load a clip of clip_len frames ending at target_fid."""
        # Find index of target_fid
        idx = all_fids.index(target_fid) if target_fid in all_fids else 0
        
        clip_fids = []
        for j in range(idx - self.clip_len + 1, idx + 1):
            k = max(0, j)
            clip_fids.append(all_fids[k])
        
        imgs = []
        for fid in clip_fids:
            path = vdir / 'Frames' / f"{fid:06d}.png"
            if path.exists():
                img = Image.open(path).convert('RGB')
                img = img.resize((self.frame_size, self.frame_size), Image.BILINEAR)
                imgs.append(np.array(img, dtype=np.float32) / 255.0)
            else:
                imgs.append(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.float32))
        
        clip = torch.from_numpy(np.stack(imgs, axis=0))  # [T, H, W, C]
        clip = clip.permute(3, 0, 1, 2)  # [C, T, H, W]
        return clip
    
    def _process_boxes(self, boxes_tlwh: List, labels: List, track_ids: List):
        """Convert tlwh boxes to cxcywh tensors."""
        boxes = torch.tensor(boxes_tlwh, dtype=torch.float32)
        # tlwh → cxcywh
        boxes_cx = torch.zeros_like(boxes)
        boxes_cx[:, 0] = boxes[:, 0] + 0.5 * boxes[:, 2]
        boxes_cx[:, 1] = boxes[:, 1] + 0.5 * boxes[:, 3]
        boxes_cx[:, 2] = boxes[:, 2]
        boxes_cx[:, 3] = boxes[:, 3]
        boxes_cx = boxes_cx.clamp(0, 1)
        
        labels_t = torch.tensor(labels, dtype=torch.long)
        tids_t = torch.tensor(track_ids, dtype=torch.long)
        return boxes_cx, labels_t, tids_t
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        vdir = pair['vdir']
        all_fids = pair['all_fids']
        
        clip_a = self._load_clip(vdir, pair['fid_a'], all_fids)
        clip_b = self._load_clip(vdir, pair['fid_b'], all_fids)
        
        boxes_a, labels_a, tids_a = self._process_boxes(
            pair['data_a']['boxes'], pair['data_a']['labels'], pair['data_a']['track_ids'])
        boxes_b, labels_b, tids_b = self._process_boxes(
            pair['data_b']['boxes'], pair['data_b']['labels'], pair['data_b']['track_ids'])
        
        return (clip_a, boxes_a, labels_a, tids_a,
                clip_b, boxes_b, labels_b, tids_b)


def collate_fn(batch):
    """Custom collate for track pair dataset."""
    clips_a, boxes_a, labels_a, tids_a = [], [], [], []
    clips_b, boxes_b, labels_b, tids_b = [], [], [], []
    
    for ca, ba, la, ta, cb, bb, lb, tb in batch:
        clips_a.append(ca)
        boxes_a.append(ba)
        labels_a.append(la)
        tids_a.append(ta)
        clips_b.append(cb)
        boxes_b.append(bb)
        labels_b.append(lb)
        tids_b.append(tb)
    
    # clips_a and clips_b remain as lists of tensors
    
    return (clips_a, boxes_a, labels_a, tids_a,
            clips_b, boxes_b, labels_b, tids_b)
