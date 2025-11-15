from .utils import chess_manager, GameContext
from chess import Move
import random
import time
import os
import subprocess
import shutil

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # Try to call the native pufferfish engine with current FEN
    def find_engine() -> str | None:
        here = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        candidates = [
            os.path.join(here, 'pufferfish', 'build', 'Release', 'pufferfish.exe'),
            os.path.join(here, 'pufferfish', 'build', 'Debug', 'pufferfish.exe'),
            os.path.join(here, 'pufferfish', 'pufferfish.exe'),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return shutil.which('pufferfish')

    def call_engine(fen: str, movetime_ms: int = 200, timeout_s: float | None = None) -> str | None:
        exe = find_engine()
        if not exe:
            return None
        cmd = [exe, '--fen', fen, '--movetime', str(movetime_ms)]
        try:
            out = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s or max(2.0, movetime_ms / 1000.0 + 1.0),
            )
        except Exception:
            return None
        if not out.stdout:
            return None
        for line in out.stdout.splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[0].lower() == 'bestmove':
                # Engine now returns SAN only after the keyword
                return parts[1]
        return None

    fen = ctx.board.fen()
    best_san = call_engine(fen, movetime_ms=200)
    if best_san:
        try:
            mv = ctx.board.parse_san(best_san)
            if mv in ctx.board.legal_moves:
                ctx.logProbabilities({mv: 1.0})
                return mv
        except Exception:
            pass

    # Fallback: random move
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities( {})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    move_probs = {move: w / total_weight for move, w in zip(legal_moves, move_weights)}
    ctx.logProbabilities(move_probs)
    return random.choices(legal_moves, weights=move_weights, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    # TODO: clear TT, free everything
    pass
