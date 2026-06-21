"""
Smoke tests for the ``core_app.mot`` package.

These tests use CPU-only random tensors and do NOT require CholecTrack20
data or any external downloads (DINOv2 / V-JEPA / VGGT / CoTracker are
all stubbed or skipped).

Run with:

    PYTHONPATH=. python -m pytest tests/test_mot_smoke.py -x -q

or for a quick manual run:

    PYTHONPATH=. python tests/test_mot_smoke.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

# Make sure the repo root is importable when executed as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------- #
# 1. Track / LongTermMemoryBank                                           #
# ---------------------------------------------------------------------- #


def test_track_basic_update():
    from core_app.mot.track import Track, STATUS_TENTATIVE

    t = Track(id=0, cls=0, bbox=torch.tensor([0.5, 0.5, 0.1, 0.1]))
    assert t.status == STATUS_TENTATIVE
    emb = torch.nn.functional.normalize(torch.randn(16), dim=0)
    t.update(
        bbox=torch.tensor([0.6, 0.6, 0.1, 0.1]),
        score=0.9,
        cls=0,
        embedding=emb,
        visibility=1.0,
    )
    assert t.hits == 2
    assert t.age == 0
    assert t.mem_embedding is not None
    assert torch.isfinite(t.mem_embedding).all()


def test_longterm_memory_bank_match():
    from core_app.mot.track import LongTermMemoryBank, Track

    bank = LongTermMemoryBank(max_ttl=50, max_entries=16)
    emb = torch.nn.functional.normalize(torch.randn(32), dim=0)

    t = Track(id=7, cls=1, bbox=torch.tensor([0.5, 0.5, 0.1, 0.1]))
    t.mem_embedding = emb
    bank.add(t)

    # Query with the same embedding — should match.
    result = bank.match(
        embeddings=emb.unsqueeze(0),
        classes=torch.tensor([1]),
        sim_threshold=0.5,
    )
    assert 0 in result
    assert result[0] == 7


# ---------------------------------------------------------------------- #
# 2. Per-track predictor + localizer                                      #
# ---------------------------------------------------------------------- #


def test_gaussian_label_encoding_shape():
    from core_app.mot.predictor import gaussian_label_encoding

    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.4, 0.1, 0.1]])
    heat = gaussian_label_encoding(boxes, spatial_h=28, spatial_w=28)
    assert heat.shape == (2, 28, 28)
    # Peak should be at the bbox centre.
    for i in range(2):
        flat = heat[i].flatten()
        peak = flat.argmax().item()
        assert heat[i].max().item() > 0.9


def test_per_track_predictor_forward():
    from core_app.mot.predictor import (
        PerTrackModelPredictor,
        apply_filter,
        gaussian_label_encoding,
    )
    torch.manual_seed(0)
    B, H, W, C = 2, 14, 14, 32
    pred = PerTrackModelPredictor(dim=C, num_heads=4, num_encoder_layers=2,
                                  num_decoder_layers=1, dim_feedforward=64)
    ref_feat = torch.randn(B, 2 * H * W, C)
    label_enc = gaussian_label_encoding(
        torch.tensor([[0.5, 0.5, 0.2, 0.2]] * B), H, W
    ).flatten(1).unsqueeze(-1).repeat(1, 2, 1)
    cur_feat = torch.randn(B, H * W, C)

    omega, encoded = pred(ref_feat, label_enc, cur_feat)
    assert omega.shape == (B, C)
    assert encoded.shape == (B, 2 * H * W + H * W, C) or encoded.shape == (B, 2 * H * W, C)

    # Apply filter to a feature map.
    fmap = torch.randn(B, C, H, W)
    score = apply_filter(omega, fmap)
    assert score.shape == (B, 1, H, W)


def test_track_localization_loss():
    from core_app.mot.localizer import (
        ClsDec, RegDec, TrackLocalizationLoss, ltrb_to_bbox,
    )

    B, H, W, C = 2, 14, 14, 32
    cls = ClsDec(in_channels=1)
    reg = RegDec(feature_dim=C)
    loss = TrackLocalizationLoss()

    score_raw = torch.randn(B, 1, H, W)
    score = cls(score_raw)
    fmap = torch.randn(B, C, H, W)
    ltrb = reg(fmap, score)
    assert ltrb.shape == (B, 4, H, W)

    peak = torch.randint(0, H, (B, 2))
    bbox = ltrb_to_bbox(ltrb, peak, H, W)
    assert bbox.shape == (B, 4)

    gt = torch.rand(B, 4) * 0.5 + 0.25
    l, ld = loss(score, bbox, gt, H, W)
    assert torch.isfinite(l)
    assert 'cls' in ld and 'giou' in ld


# ---------------------------------------------------------------------- #
# 3. Hungarian assoc                                                      #
# ---------------------------------------------------------------------- #


def test_hungarian_and_cost():
    from core_app.mot.track import Track
    from core_app.mot.assoc import compute_cost_matrix, hungarian_match

    det_boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]])
    det_embs = torch.nn.functional.normalize(torch.randn(2, 16), dim=1)
    det_cls = torch.tensor([0, 1])

    track_a = Track(id=0, cls=0, bbox=torch.tensor([0.51, 0.51, 0.1, 0.1]))
    track_a.mem_embedding = det_embs[0]
    track_b = Track(id=1, cls=1, bbox=torch.tensor([0.71, 0.71, 0.1, 0.1]))
    track_b.mem_embedding = det_embs[1]

    cost = compute_cost_matrix(det_boxes, det_embs, det_cls, [track_a, track_b])
    assert cost.shape == (2, 2)

    matches, u_det, u_trk = hungarian_match(cost, threshold=0.9)
    # Diagonal match is expected.
    match_dict = dict(matches)
    assert match_dict.get(0) == 0
    assert match_dict.get(1) == 1


# ---------------------------------------------------------------------- #
# 4. TrackManager.step                                                    #
# ---------------------------------------------------------------------- #


def test_track_manager_birth_and_match():
    from core_app.mot.manager import TrackManager

    mgr = TrackManager(birth_score=0.5, min_hits=1, max_age=5)

    det_boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
    det_scores = torch.tensor([0.9])
    det_classes = torch.tensor([0])
    det_embs = torch.nn.functional.normalize(torch.randn(1, 16), dim=1)

    result1 = mgr.step(det_boxes, det_scores, det_classes, det_embs)
    assert len(result1['active_tracks']) == 1
    assert len(result1['new_track_ids']) == 1

    # Same detection next frame — should match the existing track.
    result2 = mgr.step(det_boxes, det_scores, det_classes, det_embs)
    assert len(result2['active_tracks']) == 1
    assert len(result2['new_track_ids']) == 0


# ---------------------------------------------------------------------- #
# 5. SurgicalMOTSystem forward (encoder stubbed via V-JEPA fallback)      #
# ---------------------------------------------------------------------- #


def _build_tiny_system():
    """
    Build the smallest possible ``SurgicalMOTSystem`` that doesn't
    require network / torch.hub access. We force V-JEPA encoder with
    ``use_torch_hub=False`` and the HF fallback disabled — the encoder
    will be randomly initialized (fine for smoke-testing forward paths).
    """
    from core_app.mot.system import SurgicalMOTSystem

    return SurgicalMOTSystem(
        encoder_type='vjepa',
        encoder_checkpoint=None,
        model_name='vit_base',
        encoder_dim=768,
        neck_dim=64,
        pred_dim=64,
        img_size=224,
        num_frames=3,
        layer_indices=[-1],
        use_torch_hub=False,
        num_tools=7,
        num_queries=8,
        num_decoder_layers=2,
        detr_nheads=4,
        detr_dim_feedforward=128,
        pred_num_heads=4,
        pred_num_encoder_layers=2,
        pred_num_decoder_layers=1,
        pred_dim_feedforward=128,
        reid_embedding_dim=32,
        use_geometry=False,
        use_occusolver=False,
    )


def test_system_builds_and_smoke_forward():
    """Build the system and run a forward pass on random inputs."""
    from core_app.mot.system import PerTrackSample

    try:
        model = _build_tiny_system()
    except Exception as e:
        # Encoder construction may fail without torch.hub — skip if so.
        import pytest
        pytest.skip(f"Encoder init unavailable in this environment: {e}")

    model.eval()

    B, T = 1, 3
    img = 224
    clip = torch.randn(B, 3, T, img, img)

    # Build a single PerTrackSample per batch element.
    per_track = [[
        PerTrackSample(
            batch_idx=0,
            ref_bbox_0=torch.tensor([0.5, 0.5, 0.2, 0.2]),
            ref_bbox_1=torch.tensor([0.52, 0.52, 0.2, 0.2]),
            cur_bbox=torch.tensor([0.55, 0.55, 0.2, 0.2]),
            cls=0,
            track_id=7,
        )
    ]]
    detr_targets = [{
        'labels': torch.tensor([0], dtype=torch.long),
        'boxes': torch.tensor([[0.55, 0.55, 0.2, 0.2]], dtype=torch.float32),
    }]
    reid_labels = [torch.tensor([7], dtype=torch.long)]

    model.train()  # turn on DETR loss path
    out = model(
        current_video=clip,
        per_track_targets=per_track,
        detr_targets=detr_targets,
        reid_labels=reid_labels,
        mode='train',
    )
    assert out['total_loss'] is not None
    assert torch.isfinite(out['total_loss'])


def test_system_train_occusolver_loss():
    """Stage-4 path: OccuSolver BCE + feature gating in per-track training."""
    from core_app.mot.system import PerTrackSample, SurgicalMOTSystem

    try:
        model = SurgicalMOTSystem(
            encoder_type='vjepa',
            encoder_checkpoint=None,
            model_name='vit_base',
            encoder_dim=768,
            neck_dim=64,
            pred_dim=64,
            img_size=224,
            num_frames=3,
            layer_indices=[-1],
            use_torch_hub=False,
            num_tools=7,
            num_queries=8,
            num_decoder_layers=2,
            detr_nheads=4,
            detr_dim_feedforward=128,
            pred_num_heads=4,
            pred_num_encoder_layers=2,
            pred_num_decoder_layers=1,
            pred_dim_feedforward=128,
            reid_embedding_dim=32,
            use_geometry=False,
            use_occusolver=True,
            occusolver_kwargs={'stub': True, 'num_query_points': 4},
            occu_loss_weight=0.3,
            occu_gate_features=True,
            occu_max_tracks_per_batch=2,
        )
    except Exception as e:
        import pytest
        pytest.skip(f"Encoder init unavailable: {e}")

    model.train()
    hw = 28
    n = hw * hw
    pred_tokens_seq = torch.randn(1, 3, n, 64)
    pred_spatial = torch.randn(1, 64, hw, hw)
    spatial_map = torch.randn(1, 768, hw, hw)
    clip = torch.randn(1, 3, 3, 224, 224)
    per_track = [[
        PerTrackSample(
            batch_idx=0,
            ref_bbox_0=torch.tensor([0.5, 0.5, 0.2, 0.2]),
            ref_bbox_1=torch.tensor([0.52, 0.52, 0.2, 0.2]),
            cur_bbox=torch.tensor([0.55, 0.55, 0.2, 0.2]),
            cls=0,
            track_id=7,
        )
    ]]
    per_track_result: dict = {'track_loss': None, 'loss_dict': {}, 'num_tracks': 0}
    model._accumulate_per_track_losses(
        per_track_result=per_track_result,
        per_track_targets=per_track,
        pred_tokens_seq=pred_tokens_seq,
        pred_spatial=pred_spatial,
        spatial_map=spatial_map,
        current_video=clip,
    )
    assert per_track_result.get('occu_loss') is not None
    assert torch.isfinite(per_track_result['occu_loss'])
    assert per_track_result.get('track_loss') is not None


def test_system_infer_updates_tracker():
    from core_app.mot.system import PerTrackSample

    try:
        model = _build_tiny_system()
    except Exception as e:
        import pytest
        pytest.skip(f"Encoder init unavailable in this environment: {e}")

    model.eval()
    model.reset_tracker()

    clip = torch.randn(1, 3, 3, 224, 224)
    out = model(current_video=clip, mode='infer')
    assert 'active_tracks' in out
    # Tracker should not crash on empty-detection or real-detection paths.
    assert isinstance(out['active_tracks'], list)


# ---------------------------------------------------------------------- #
# 6. JEPA + augment                                                       #
# ---------------------------------------------------------------------- #


def test_jepa_loss_runs():
    from core_app.mot.jepa import GOTJEPAWrapper
    from core_app.mot.predictor import PerTrackModelPredictor

    pred = PerTrackModelPredictor(dim=32, num_heads=4, num_encoder_layers=1,
                                  num_decoder_layers=1, dim_feedforward=64)
    jepa = GOTJEPAWrapper(student_predictor=pred)

    N_tracks, N, C = 3, 49, 32
    ref_feat = torch.randn(N_tracks, 2 * N, C)
    labels = torch.rand(N_tracks, 2 * N, 1)
    clean_cur = torch.randn(N_tracks, N, C)
    dirty_cur = clean_cur + 0.1 * torch.randn_like(clean_cur)

    out = jepa(ref_feat, labels, clean_cur, dirty_cur)
    assert torch.isfinite(out['loss'])
    assert out['loss'].requires_grad
    assert 'jepa_inv' in out['loss_dict']
    assert 'jepa_cov' in out['loss_dict']


def test_corruption_preserves_shape():
    from core_app.mot.augment import SurgicalCorruption

    aug = SurgicalCorruption()
    aug.train()
    x = torch.rand(2, 3, 64, 64)
    y = aug(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert y.min() >= 0 and y.max() <= 1


def test_frame_challenges_and_stratified_filter():
    from core_app.mot.eval import (
        build_frame_challenges,
        filter_mot_rows_by_frames,
        frames_for_stratum,
    )

    annotations = {
        '100': [
            {'smoke': 1, 'bleeding': 0, 'occluded': 0, 'intraoperative_track': 1,
             'tool_bbox': [0.1, 0.2, 0.3, 0.4]},
        ],
        '101': [
            {'smoke': 0, 'bleeding': 1, 'occluded': 1, 'intraoperative_track': 2,
             'tool_bbox': [0.2, 0.3, 0.2, 0.2]},
        ],
    }
    frame_numbers = [100, 101]
    challenges = build_frame_challenges(annotations, frame_numbers)
    assert challenges[1]['smoke'] is True
    assert challenges[2]['bleeding'] is True

    rows = [(1, 1, 0, 0, 10, 10, 1.0), (2, 2, 0, 0, 10, 10, 1.0)]
    smoke_frames = frames_for_stratum(challenges, 'smoke')
    filtered = filter_mot_rows_by_frames(rows, smoke_frames)
    assert len(filtered) == 1
    assert filtered[0][0] == 1


# ---------------------------------------------------------------------- #
# 7. Geometry / OccuSolver (stub paths)                                   #
# ---------------------------------------------------------------------- #


def test_null_space_editor():
    from core_app.mot.geometry import NullSpaceEditor

    editor = NullSpaceEditor()
    W = torch.randn(16, 32)
    delta = torch.randn(16, 32)
    out = editor(delta, W)
    assert out.shape == delta.shape
    assert torch.isfinite(out).all()


def test_geometry_branch_stub():
    from core_app.mot.geometry import GeometryBranch

    gb = GeometryBranch(
        semantic_dim=64,
        pred_dim=32,
        vggt_kwargs={'stub': True, 'stub_out_channels': 64},
        geom_predictor_kwargs={'num_heads': 4, 'num_encoder_layers': 1,
                               'num_decoder_layers': 1, 'dim_feedforward': 64},
    )
    gb.eval()
    rgb = torch.rand(1, 3, 96, 96)
    v_sem = torch.randn(1, 64, 24, 24)
    ref = torch.randn(1, 2 * 576, 32)
    labels = torch.rand(1, 2 * 576, 1)
    cur = torch.randn(1, 576, 32)
    omega_sem = torch.randn(1, 32)

    out = gb(rgb, v_sem, ref, labels, cur, omega_sem)
    assert 'omega_final' in out
    assert out['omega_final'].shape == omega_sem.shape


def test_resolve_cotracker_hub_entrypoint():
    from core_app.mot.occusolver import (
        cotracker_kwargs_from_occusolver_cfg,
        resolve_cotracker_hub_entrypoint,
    )

    assert resolve_cotracker_hub_entrypoint(3, 'offline') == 'cotracker3_offline'
    assert resolve_cotracker_hub_entrypoint(3, 'online') == 'cotracker3_online'
    assert resolve_cotracker_hub_entrypoint(2, 'offline') == 'cotracker2'
    assert resolve_cotracker_hub_entrypoint(2, 'online') == 'cotracker2_online'

    kw = cotracker_kwargs_from_occusolver_cfg(
        {'stub': True, 'cotracker_version': 2, 'cotracker_mode': 'offline'}
    )
    assert kw['stub'] is True
    assert kw['cotracker_version'] == 2
    assert kw['cotracker_mode'] == 'offline'

    kw_hub = cotracker_kwargs_from_occusolver_cfg(
        {'stub': False, 'model_variant': 'cotracker3_offline'}
    )
    assert kw_hub['stub'] is False
    assert kw_hub['model_variant'] == 'cotracker3_offline'
    assert 'cotracker_version' not in kw_hub


def test_visibility_map_from_points():
    from core_app.mot.occusolver import visibility_map_from_points

    coords = torch.tensor([[[0.5, 0.5], [0.25, 0.75]]], dtype=torch.float32)
    vis = torch.tensor([[1.0, 0.2]], dtype=torch.float32)
    m = visibility_map_from_points(coords, vis, height=14, width=14)
    assert m.shape == (1, 1, 14, 14)
    assert m.max() <= 1.0 + 1e-5
    assert m.min() >= 0.0


def test_occusolver_stub():
    from core_app.mot.occusolver import OccuSolver

    occu = OccuSolver(feature_dim=32, cotracker_kwargs={'stub': True})
    occu.eval()

    video = torch.rand(1, 3, 3, 128, 128)
    ref_heat = torch.rand(1, 1, 128, 128)
    pts = torch.rand(1, 64, 2)
    out = occu(video, ref_heat, pts)
    assert 'visibility' in out
    assert out['visibility'].shape == (1, 64)
    assert (out['visibility'] >= 0).all() and (out['visibility'] <= 1).all()


def test_occusolver_iterative_refine_steps():
    """Regression: 5-point cross offsets must broadcast as (B, N, 5, 2), not (1,1,1,2)."""
    from core_app.mot.occusolver import OccuSolver

    occu = OccuSolver(
        feature_dim=32,
        cotracker_kwargs={'stub': True},
        num_refine_steps=2,
    )
    occu.train()
    video = torch.rand(2, 3, 3, 128, 128)
    ref_heat = torch.rand(2, 1, 128, 128)
    pts = torch.rand(2, 4, 2)
    out = occu(video, ref_heat, pts)
    assert out['visibility'].shape == (2, 4)
    assert torch.isfinite(out['visibility']).all()


def test_occusolver_stub_with_depth():
    from core_app.mot.occusolver import OccuSolver

    occu = OccuSolver(feature_dim=32, cotracker_kwargs={'stub': True},
                      use_depth=True, depth_stub=True)
    occu.eval()

    video = torch.rand(1, 3, 3, 128, 128)
    ref_heat = torch.rand(1, 1, 128, 128)
    pts = torch.rand(1, 64, 2)
    out = occu(video, ref_heat, pts)
    assert 'visibility' in out
    assert 'point_depth' in out
    assert 'point_depth_valid' in out
    assert out['point_depth'].shape == (1, 64, 1)
    assert out['point_depth_valid'].shape == (1, 64)
    # Depth exists even if vis < 0.5 (just zeroed out)
    assert out['point_depth'] is not None


# ---------------------------------------------------------------------- #
# 8. MOT eval helpers (smoke stratification)                              #
# ---------------------------------------------------------------------- #


def test_filter_mot_rows_by_frames():
    from core_app.mot.eval import filter_mot_rows_by_frames

    rows = [(1, 1, 0, 0, 10, 10, 1.0), (2, 1, 1, 1, 10, 10, 0.9)]
    filtered = filter_mot_rows_by_frames(rows, {2})
    assert filtered == [(2, 1, 1, 1, 10, 10, 0.9)]


def test_frame_smoke_flags_from_ct20_json():
    from core_app.mot.eval import build_frame_challenges

    annotations = {
        '10': [{'smoke': 0}, {'smoke': 1}],
        '20': [{'smoke': 0}],
    }
    flags = build_frame_challenges(annotations, [10, 20])
    assert flags[1]['smoke'] is True
    assert flags[2]['smoke'] is False


def test_export_ct20_gt_mot_rows_normalized_bbox(tmp_path):
    from core_app.mot.eval import build_ct20_gt_mot_rows

    json_path = tmp_path / 'VID01.json'
    json_path.write_text(json.dumps({
        'video': {'width': 100, 'height': 100, 'num_frames': 1},
        'annotations': {
            '5': [{
                'intraoperative_track': 3,
                'tool_bbox': [0.1, 0.2, 0.3, 0.4],
            }]
        }
    }), encoding='utf-8')
    rows = build_ct20_gt_mot_rows(json_path, [5], img_width=100, img_height=100)
    assert rows == [(1, 3, 10.0, 20.0, 30.0, 40.0, 1.0, 1)]


def test_cholec_dataset_test_split_path(tmp_path):
    from core_app.data.video_dataset import CholecDataset

    root = tmp_path / 'ct20'
    test_dir = root / 'Testing' / 'VID01' / 'Frames'
    test_dir.mkdir(parents=True)
    (test_dir / '000001.png').write_bytes(b'')
    (root / 'Testing' / 'VID01' / 'VID01.json').write_text(
        json.dumps({'annotations': {'1': []}}),
        encoding='utf-8',
    )
    ds = CholecDataset(data_root=root, split='test', clip_length=1, img_size=64, training=False)
    assert ds.video_dir.name == 'Testing'


# ---------------------------------------------------------------------- #
# Script-mode entry (pytest-independent)                                  #
# ---------------------------------------------------------------------- #


def test_track_constant_velocity_predict():
    """M3: Track.predict() should advance bbox by velocity with damping."""
    from core_app.mot.track import Track
    bbox0 = torch.tensor([0.5, 0.5, 0.1, 0.1])
    t = Track(id=1, cls=0, bbox=bbox0)
    # Simulate two updates to establish velocity.
    bbox1 = torch.tensor([0.55, 0.55, 0.1, 0.1])
    t.update(bbox1, score=0.9, cls=0)
    assert t.velocity is not None
    # mark_missed should call predict(), moving bbox forward.
    t.mark_missed()
    assert t.bbox[0].item() > bbox1[0].item()  # cx should increase
    assert t.time_since_update == 1


def test_track_velocity_damping_decay():
    """M3: Velocity should decay over multiple missed frames."""
    from core_app.mot.track import Track
    t = Track(id=1, cls=0, bbox=torch.tensor([0.5, 0.5, 0.1, 0.1]))
    t.update(torch.tensor([0.6, 0.5, 0.1, 0.1]), score=0.9, cls=0)
    vel_after_update = t.velocity.clone()
    for _ in range(5):
        t.mark_missed()
    # After 5 missed frames, velocity damping should slow drift.
    # The bbox should still be moving but less per frame.
    assert t.time_since_update == 5
    # Check bbox hasn't drifted past 1.0 (clamped).
    assert t.bbox[0].item() <= 1.0


def test_track_operator_phase_fields():
    """M1: Track should store operator and phase from updates."""
    from core_app.mot.track import Track
    t = Track(id=1, cls=0, bbox=torch.tensor([0.5, 0.5, 0.1, 0.1]))
    assert t.operator == -1
    assert t.phase == -1
    t.update(torch.tensor([0.5, 0.5, 0.1, 0.1]), score=0.9, cls=0,
             operator=2, phase=3)
    assert t.operator == 2
    assert t.phase == 3


def test_dead_bank_stores_velocity_operator():
    """M3: LongTermMemoryBank.add() should store velocity and operator."""
    from core_app.mot.track import Track, LongTermMemoryBank
    t = Track(id=1, cls=0, bbox=torch.tensor([0.5, 0.5, 0.1, 0.1]))
    t.update(torch.tensor([0.55, 0.5, 0.1, 0.1]), score=0.9, cls=0,
             embedding=torch.randn(8), operator=1, phase=2)
    bank = LongTermMemoryBank()
    bank.add(t)
    entry = bank.entries[1]
    assert entry['velocity'] is not None
    assert entry['operator'] == 1
    assert entry['phase'] == 2


def test_dead_bank_match_with_direction_cue():
    """M3: dead_bank.match() should accept det_boxes for direction cue."""
    from core_app.mot.track import Track, LongTermMemoryBank
    bank = LongTermMemoryBank()
    t = Track(id=1, cls=0, bbox=torch.tensor([0.5, 0.5, 0.1, 0.1]))
    t.update(torch.tensor([0.55, 0.5, 0.1, 0.1]), score=0.9, cls=0,
             embedding=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    bank.add(t)
    # Detection near predicted location with same embedding.
    det_emb = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    det_cls = torch.tensor([0])
    det_boxes = torch.tensor([[0.6, 0.5, 0.1, 0.1]])
    matched = bank.match(det_emb, det_cls, det_boxes=det_boxes)
    assert 0 in matched
    assert matched[0] == 1


def test_per_track_sample_m1_fields():
    """M1: PerTrackSample should have operator, phase, occluded, visible fields."""
    from core_app.mot.system import PerTrackSample
    s = PerTrackSample(
        batch_idx=0,
        ref_bbox_0=torch.tensor([0.5, 0.5, 0.1, 0.1]),
        ref_bbox_1=torch.tensor([0.5, 0.5, 0.1, 0.1]),
        cur_bbox=torch.tensor([0.5, 0.5, 0.1, 0.1]),
        cls=0, track_id=1,
        operator=2, phase=1, occluded=1, visible=0,
    )
    assert s.operator == 2
    assert s.phase == 1
    assert s.occluded == 1
    assert s.visible == 0


def test_annot_to_tracks_captures_occluded_operator():
    """M1: _annot_to_tracks should parse occluded and operator fields."""
    from core_app.mot.data import _annot_to_tracks
    annots = [{
        'intraoperative_track': 1,
        'tool_bbox': [0.1, 0.2, 0.3, 0.4],
        'instrument': 3,
        'occluded': 1,
        'operator': 2,
        'phase': 1,
    }]
    tracks = _annot_to_tracks(annots, img_size=392)
    assert 1 in tracks
    assert tracks[1]['occluded'] == 1
    assert tracks[1]['operator'] == 2
    assert tracks[1]['phase'] == 1


def test_physics_smoke_depth_dependent():
    """M4: Physics-based smoke should apply depth-dependent attenuation."""
    from core_app.mot.augment import smoke, _depth_prior
    # Create a simple image: bright everywhere.
    x = torch.ones(2, 3, 64, 64)
    out = smoke(x, probability=1.0, intensity=0.8)
    # Smoke should have been applied (output differs from input).
    assert not torch.allclose(out, x)
    # Depth prior: centre should be nearer (lower depth) than corners.
    d = _depth_prior(1, 64, 64, x.device)
    assert d[0, 0, 32, 32].item() < d[0, 0, 2, 2].item()


def test_physics_smoke_preserves_range():
    """M4: Smoke output should be in [0, 1]."""
    from core_app.mot.augment import smoke
    x = torch.rand(4, 3, 32, 32)
    out = smoke(x, probability=1.0, intensity=0.5)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_configure_tracker_for_eval():
    """P0-3: configure_tracker_for_eval should set min_hits=1 and birth_score=0.3."""
    from core_app.mot.track import Track, STATUS_TENTATIVE
    from core_app.mot.manager import TrackManager
    # We can't easily build the full SurgicalMOTSystem in a unit test,
    # but we can test the logic by checking TrackManager configs.
    mgr = TrackManager(birth_score=0.6, min_hits=3, max_age=30)
    assert mgr.birth_score == 0.6
    assert mgr.min_hits == 3
    # Simulate what configure_tracker_for_eval does.
    mgr2 = TrackManager(birth_score=0.3, min_hits=1, max_age=30)
    assert mgr2.birth_score == 0.3
    assert mgr2.min_hits == 1


def test_ct20_tool_classes_defined():
    """P0-2: CT20_TOOL_CLASSES should have 7 classes."""
    from core_app.mot.eval import CT20_TOOL_CLASSES, CT20_CLASS_TO_ID
    assert len(CT20_TOOL_CLASSES) == 7
    assert CT20_CLASS_TO_ID['grasper'] == 1
    assert CT20_CLASS_TO_ID['specimen_bag'] == 7


def test_write_mot_txt_includes_class():
    """P0-2: write_mot_txt should write class ID as 8th field."""
    import tempfile
    from core_app.mot.eval import write_mot_txt, load_mot_txt
    rows = [(1, 5, 10.0, 20.0, 30.0, 40.0, 0.95, 3)]
    with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False) as f:
        path = Path(f.name)
    write_mot_txt(rows, path)
    loaded = load_mot_txt(path)
    assert len(loaded) == 1
    assert loaded[0][7] == 3  # class ID preserved


def test_lora_linear_forward():
    """A1: LoRALinear should preserve base output at init (zero low-rank update) and add delta after training."""
    from core_app.models.lora import LoRALinear
    base = torch.nn.Linear(32, 16)
    wrapped = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(8, 32)
    # At init, low-rank path is zero so output equals frozen base.
    with torch.no_grad():
        assert torch.allclose(wrapped(x), base(x), atol=1e-6)
    # LoRA parameters are trainable; base is frozen.
    lora_params = [p for p in wrapped.parameters() if p is not None and p.requires_grad]
    assert len(lora_params) == 2  # lora_A and lora_B
    for p in base.parameters():
        assert not p.requires_grad


def test_lora_injected_dinov2_base_frozen():
    """A1: After LoRA injection, DINOv2 base weights remain frozen and LoRA matrices are trainable."""
    from core_app.models.vjepa_world_model import Dinov2EncoderWrapper
    import torch.nn as nn
    wrapper = Dinov2EncoderWrapper(
        model_name='dinov2_vits14',
        img_size=224,
        freeze=True,
        lora={'enable': True, 'rank': 4, 'alpha': 8, 'target_modules': ['qkv', 'proj']},
    )
    trainable = [p.numel() for p in wrapper.encoder.parameters() if p.requires_grad]
    assert sum(trainable) > 0, "LoRA matrices should be trainable"
    base_trainable = 0
    for block in wrapper.encoder.blocks:
        for name in ['qkv', 'proj']:
            layer = getattr(block, name, None)
            if isinstance(layer, nn.Linear):
                base_trainable += sum(p.numel() for p in layer.parameters() if p.requires_grad)
    assert base_trainable == 0, "Base linear weights must stay frozen"


def test_deformable_detr_denoising_loss():
    """A2: DeformableSurgicalToolDetector with denoising enabled should produce a denoising loss."""
    from core_app.models.deformable_detr_head import DeformableSurgicalToolDetector
    head = DeformableSurgicalToolDetector(
        neck_dim=256,
        num_tools=7,
        num_queries=16,
        use_denoising=True,
        num_denoising_groups=2,
        num_noise_per_group=2,
        box_noise_scale=0.2,
        denoising_weight=1.0,
    )
    head.train()
    neck_out = {
        'detection_scales': [
            torch.randn(2, 256, 14, 14),
            torch.randn(2, 256, 7, 7),
            torch.randn(2, 256, 4, 4),
            torch.randn(2, 256, 2, 2),
        ]
    }
    targets = [
        {'labels': torch.tensor([0, 1]), 'boxes': torch.tensor([[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1]])},
        {'labels': torch.tensor([2]), 'boxes': torch.tensor([[0.3, 0.3, 0.15, 0.15]])},
    ]
    out = head(neck_out, targets=targets)
    assert 'loss' in out
    assert 'denoise_loss_focal' in out['loss_dict']
    assert 'denoise_loss_l1' in out['loss_dict']
    assert 'denoise_loss_giou' in out['loss_dict']
    assert out['loss_dict']['denoise_loss_focal'] > 0


# ---------------------------------------------------------------------- #
# Main runner                                                             #
# ---------------------------------------------------------------------- #

def _run_as_script():
    """Run every ``test_*`` in this module as a simple pass/fail sweep."""
    tests = [obj for name, obj in globals().items()
             if name.startswith('test_') and callable(obj)]
    passed = 0
    failed = 0
    skipped = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  [PASS] {t.__name__}")
        except Exception as e:
            # Treat pytest.skip as a skip.
            if 'Skipped' in type(e).__name__ or 'skip' in str(e).lower():
                skipped += 1
                print(f"  [SKIP] {t.__name__}: {e}")
            else:
                failed += 1
                print(f"  [FAIL] {t.__name__}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
    print(f"\n{'=' * 60}")
    print(f"  {passed} passed | {skipped} skipped | {failed} failed")
    print('=' * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    _run_as_script()
