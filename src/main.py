from .utils import chess_manager, GameContext
from chess import Move
import sys
import os

# Add mini folder to path
mini_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mini')
sys.path.insert(0, mini_path)

from inference_model import load_model
from search import find_best_move_tactical

# Load model once at startup
MODEL_PATH = os.path.join(mini_path, 'tiny_model.pt')
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Use the tiny neural network to find the best move.
    """
    try:
        best_move = find_best_move_tactical(ctx.board, model)
        if best_move is None:
            # Game is over, no legal moves available
            raise ValueError("No legal moves available (game is over)")
        return best_move
    except Exception as e:
        print(f"Error finding move: {e}")
        # Fallback to random legal move
        import random
        legal_moves = list(ctx.board.legal_moves)
        if legal_moves:
            return random.choice(legal_moves)
        raise ValueError("No legal moves available")


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called when a new game begins.
    No state to reset for this stateless model.
    """
    pass

