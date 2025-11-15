from .utils import chess_manager, GameContext
from chess import Move
import time
import os
import subprocess
import shutil
from huggingface_hub import hf_hub_download

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

# Download NNUE binary from Hugging Face
NNUE_PATH = hf_hub_download(
    repo_id="LegendaryAKx3/nnue-bin",
    filename="nnue_residual.bin",  # Adjust filename if different in your repo
    repo_type="model"
)


def find_engine() -> str | None:
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    candidates = [
        os.path.join(here, 'pufferfish', 'pufferfish'),
        os.path.join(here, 'pufferfish', 'build', 'Release', 'pufferfish.exe'),
        os.path.join(here, 'pufferfish', 'build', 'Debug', 'pufferfish.exe'),
        os.path.join(here, 'pufferfish', 'pufferfish.exe'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return shutil.which('pufferfish')


def call_engine(fen: str, movetime_ms: int = 1000, timeout_s: float | None = None) -> str | None:
    exe = find_engine()
    if not exe:
        return None
    # Pass FEN as six separate CLI tokens expected by the C++ program
    fen_tokens = fen.split()
    cmd = [exe, '--fen', *fen_tokens, '--movetime', str(movetime_ms)]
    # Optional tuning via environment variables
    tt_mb = os.getenv('PUFFERFISH_TT_MB')
    if tt_mb and tt_mb.isdigit():
        cmd += ['--tt', tt_mb]
    if os.getenv('PUFFERFISH_NO_QSEARCH', '').lower() in ('1', 'true', 'yes'):
        cmd += ['--no-qsearch']
    if os.getenv('PUFFERFISH_NO_TT', '').lower() in ('1', 'true', 'yes'):
        cmd += ['--no-tt']
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
            # Engine returns UCI after the keyword
            return parts[1]
    return None


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # Try to call the native pufferfish engine with current FEN
    fen = ctx.board.fen()
    best_uci = call_engine(fen, movetime_ms=1000)
    if best_uci:
        try:
            mv = Move.from_uci(best_uci)
            if mv in ctx.board.legal_moves:
                return mv
        except Exception:
            pass

    # No fallback: fail fast if engine didn't return a valid move
    # TODO: just give a valid move instead of erroring out, later
    raise ValueError("Engine did not return a valid move")


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    # TODO: clear TT, free everything
    pass
