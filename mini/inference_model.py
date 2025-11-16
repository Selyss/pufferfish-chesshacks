# inference_model.py
"""
Loads the tiny model and provides inference functionality.
"""
import torch
from model import TinyModel


def load_model(path: str = "tiny_model.pt") -> TinyModel:
    """Load the saved model from disk."""
    model = TinyModel()
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_move_features(model: TinyModel, features: torch.Tensor) -> float:
    """
    Evaluate move features using the model.
    
    Args:
        model: Loaded TinyModel instance
        features: Tensor of shape (12,) with move features
    
    Returns:
        Float score for the move
    """
    with torch.no_grad():
        # Reshape to batch of 1
        x = features.unsqueeze(0) if features.dim() == 1 else features
        score = model(x)
        return float(score.item())
