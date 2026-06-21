"""
Checkpoint utilities for saving and loading models.
"""
import torch
from pathlib import Path
from typing import Dict, Any, Optional


def save_checkpoint(
    state: Dict[str, Any],
    path: str,
    is_best: bool = False
):
    """Save checkpoint to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(state, path)
    
    if is_best:
        best_path = path.parent / 'best.pth.tar'
        torch.save(state, best_path)


def load_checkpoint(
    path: str,
    map_location: str = 'cpu'
) -> Dict[str, Any]:
    """Load checkpoint from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    return torch.load(path, map_location=map_location)


def load_pretrained_encoder(
    model,
    checkpoint_path: str,
    strict: bool = False
):
    """
    Load pretrained encoder weights.
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to checkpoint
        strict: Whether to strictly enforce matching keys
    """
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Extract encoder state dict
    if 'encoder' in checkpoint:
        state_dict = checkpoint['encoder']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Filter for encoder keys only
        state_dict = {k.replace('encoder.', ''): v 
                     for k, v in state_dict.items() 
                     if k.startswith('encoder.')}
    else:
        state_dict = checkpoint
    
    # Load into encoder
    model.encoder.load_state_dict(state_dict, strict=strict)
    print(f"Loaded pretrained encoder from {checkpoint_path}")


def freeze_encoder(model):
    """Freeze encoder parameters."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()
    print("Encoder frozen")
