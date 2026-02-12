"""
Dataset loader for CholecTrack20 surgical tool detection.
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image


class IdentitySampler(Sampler):
    """
    Randomly sample N videos, then for each video,
    randomly sample K frames. This ensures that each batch
    contains multiple frames from the same video, providing
    positive pairs for Triplet Loss.
    """
    def __init__(self, data_source, batch_size=8, num_instances=4):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        
        # Build index dict: {video_name: [frame_idx1, frame_idx2, ...]}
        self.index_dic = {}
        for i in range(len(data_source)):
            vid = data_source.items[i]['video'] 
            if vid not in self.index_dic:
                self.index_dic[vid] = []
            self.index_dic[vid].append(i)
            
        self.pids = list(self.index_dic.keys())

    def __iter__(self):
        # Produce one full epoch: num_batches = len(dataset) // batch_size
        # Each batch has num_pids_per_batch videos with num_instances frames each (positive pairs for Re-ID)
        num_batches = len(self.data_source) // self.batch_size
        n_pids = self.num_pids_per_batch  # e.g. 2
        k = self.num_instances  # e.g. 4

        for _ in range(num_batches):
            batch = []
            # Sample n_pids videos (with replacement so we can have many batches)
            pid_indices = np.random.randint(0, len(self.pids), size=n_pids)
            for pid_idx in pid_indices:
                pid = self.pids[pid_idx]
                t_idxs = self.index_dic[pid]
                if len(t_idxs) >= k:
                    sampled = np.random.choice(t_idxs, size=k, replace=False)
                else:
                    sampled = np.random.choice(t_idxs, size=k, replace=True)
                batch.extend(sampled.tolist())
            yield batch[: self.batch_size]

    def __len__(self):
        # Full epoch: as many batches as standard sampling
        return len(self.data_source) // self.batch_size


class CholecDetectDataset(Dataset):
    """Dataset for surgical tool detection with frame-level annotations.

    When ``load_track_ids=True`` the dataset also returns per-detection
    intraoperative track IDs (needed for joint detection + Re-ID training).
    Track IDs are read from the ``intraoperative_track`` field in the
    CholecTrack20 annotation JSON.  Frames without *any* valid track ID
    are still included (track_ids will be all -1) so that detection training
    is not affected.
    """
    
    def __init__(self, split_dir: str, clip_len: int = 16, frame_size: int = 224,
                 load_track_ids: bool = False,
                 track_perspective: str = 'intraoperative'):
        self.items = []
        self.clip_len = clip_len
        self.frame_size = frame_size
        self.load_track_ids = load_track_ids
        self.track_key = f'{track_perspective}_track'
        split_dir = Path(split_dir)
        
        # Find all video directories (VID01, VID02, etc.)
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
            
            for i in range(len(fids)):
                target_fid = fids[i]
                # Gather clip_len frames ending at target_fid
                clip_fids = []
                for j in range(i - clip_len + 1, i + 1):
                    idx = max(0, j)  # Pad with first frame at beginning
                    clip_fids.append(fids[idx])
                
                # Load frame paths
                img_paths = [vdir / 'Frames' / f"{fid:06d}.png" for fid in clip_fids]
                if not all(p.exists() for p in img_paths):
                    continue
                
                # Get annotations for target frame
                frame_anns = anns[str(target_fid)]
                boxes, labels, track_ids = [], [], []
                for ann in frame_anns:
                    inst = ann.get('instrument', -1)
                    bbox = ann.get('tool_bbox')  # [x, y, w, h] normalized tlwh
                    if inst is None or inst < 0 or bbox is None:
                        continue
                    # Skip sentinel/invalid boxes (e.g. [-1,-1,-1,-1] for off-screen)
                    if bbox[2] <= 0 or bbox[3] <= 0 or bbox[0] < -0.5:
                        continue
                    boxes.append(bbox)
                    labels.append(int(inst))
                    if load_track_ids:
                        tid = ann.get(self.track_key, -1)
                        track_ids.append(int(tid) if tid is not None else -1)
                
                if len(labels) == 0:
                    continue  # Skip frames with no tools
                
                item = {
                    'img_paths': img_paths,
                    'boxes': boxes,
                    'labels': labels,
                    'video': vdir.name,
                }
                if load_track_ids:
                    item['track_ids'] = track_ids
                self.items.append(item)
        
        print(f"Loaded {len(self.items)} annotated frames from {len(video_dirs)} videos"
              f"{' (with track IDs)' if load_track_ids else ''}")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        imgs = []
        for p in item['img_paths']:
            img = Image.open(p).convert('RGB')
            img = img.resize((self.frame_size, self.frame_size), Image.BILINEAR)
            imgs.append(np.array(img, dtype=np.float32) / 255.0)
        
        # Stack to [T, H, W, C] then permute to [C, T, H, W] for V-JEPA
        clip = torch.from_numpy(np.stack(imgs, axis=0))  # [T, H, W, C]
        clip = clip.permute(3, 0, 1, 2)  # [C, T, H, W]
        
        # GT boxes are normalized tlwh (top-left x, y, width, height) in [0,1]
        boxes_tlwh = torch.tensor(item['boxes'], dtype=torch.float32)  # [N, 4]
        
        # Check if boxes are normalized (allow small margin for partially off-screen tools)
        if boxes_tlwh.numel() > 0:
            if boxes_tlwh.max() > 1.1 or boxes_tlwh.min() < -0.1:
                # Potential pixel coordinates? Log a warning (only once per video to avoid spam)
                if not hasattr(self, '_warned_videos'):
                    self._warned_videos = set()
                if item['video'] not in self._warned_videos:
                    print(f"WARNING: Boxes for video {item['video']} appear to be in pixel coordinates (max={boxes_tlwh.max().item():.2f}, min={boxes_tlwh.min().item():.2f}). Expected normalized [0, 1].")
                    self._warned_videos.add(item['video'])
        
        # Convert to normalized cxcywh for compatibility with sigmoid decoder output
        boxes_cxcywh = torch.zeros_like(boxes_tlwh)
        boxes_cxcywh[:, 0] = boxes_tlwh[:, 0] + 0.5 * boxes_tlwh[:, 2]  # cx
        boxes_cxcywh[:, 1] = boxes_tlwh[:, 1] + 0.5 * boxes_tlwh[:, 3]  # cy
        boxes_cxcywh[:, 2] = boxes_tlwh[:, 2]  # w
        boxes_cxcywh[:, 3] = boxes_tlwh[:, 3]  # h
        
        # Clamp to [0, 1] — handles tools partially off-screen
        boxes_cxcywh = boxes_cxcywh.clamp(0, 1)
        
        labels = torch.tensor(item['labels'], dtype=torch.long)
        
        if self.load_track_ids:
            track_ids = torch.tensor(item['track_ids'], dtype=torch.long)
            return clip, boxes_cxcywh, labels, track_ids
        
        return clip, boxes_cxcywh, labels


def collate_fn(batch):
    """Custom collate for variable-length annotations.

    Handles both 3-tuple (clip, boxes, labels) when ``load_track_ids=False``
    and 4-tuple (clip, boxes, labels, track_ids) when ``load_track_ids=True``.
    """
    if len(batch[0]) == 4:
        clips, boxes, labels, track_ids = zip(*batch)
        clips = torch.stack(clips, dim=0)  # [B, C, T, H, W]
        return clips, list(boxes), list(labels), list(track_ids)
    else:
        clips, boxes, labels = zip(*batch)
        clips = torch.stack(clips, dim=0)  # [B, C, T, H, W]
        return clips, list(boxes), list(labels)
