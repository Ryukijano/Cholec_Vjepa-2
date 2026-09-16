"""
Multi-Object Tracking package for surgical tools.

Adapted from the GOT stack (GOT-Edit ICLR'26, GOT-JEPA TCSVT'26) into a
multi-object tracker for CholecTrack20. Reuses encoder / neck / DETR /
ReID primitives from ``core_app.models`` and layers per-track
hypernetwork filters, VGGT geometry with null-space editing (Stage 4),
and OccuSolver visibility gating (Stage 4) on top.

Stage roadmap:
  - Stage 0: scaffold + smoke tests
  - Stage 1: per-track filter + track manager + Hungarian association
  - Stage 2: GOT-JEPA teacher-student predictor pretraining
  - Stage 3: joint MOT fine-tune with JEPA-pretrained student
  - Stage 4: VGGT geometry + null-space editor + OccuSolver

See:
  - ``.windsurf/plans/got-mot-stack-plan-d8f505.md``
  - ``docs/multi_object_tracking_research.md``
"""

from .track import Track, LongTermMemoryBank
from .assoc import hungarian_match, compute_cost_matrix, box_iou, box_giou
from .manager import TrackManager
from .predictor import PerTrackModelPredictor, gaussian_label_encoding
from .localizer import ClsDec, RegDec, TrackLocalizationLoss
from .system import SurgicalMOTSystem

__all__ = [
    'Track',
    'LongTermMemoryBank',
    'hungarian_match',
    'compute_cost_matrix',
    'box_iou',
    'box_giou',
    'TrackManager',
    'PerTrackModelPredictor',
    'gaussian_label_encoding',
    'ClsDec',
    'RegDec',
    'TrackLocalizationLoss',
    'SurgicalMOTSystem',
]
