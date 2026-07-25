"""NNUEState incremental-update correctness.

The search relies on push/pop restoring the feature vector exactly. If it drifts,
evaluations silently corrupt partway through a search and nothing visibly fails,
so these check it directly.

Run:  python -m pytest tests/test_nnue_state.py -q
      python tests/test_nnue_state.py          (no pytest needed)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import chess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.bot.dataset import FenFeatureEncoder  # noqa: E402
from src.bot.nnue import NNUEState  # noqa: E402

ENCODER = FenFeatureEncoder()

POSITIONS = [
    ("startpos", chess.STARTING_FEN),
    ("ep available", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"),
    ("castling both", "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1"),
    ("promotion ready", "8/PPP4k/8/8/8/8/4Kppp/8 w - - 0 1"),
    ("middlegame", "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"),
]


def _fresh(fen: str) -> NNUEState:
    return NNUEState(chess.Board(fen), ENCODER)


def test_push_pop_restores_features_exactly():
    """Every legal move, pushed then popped, must leave features bit-identical."""
    for name, fen in POSITIONS:
        state = _fresh(fen)
        before = state.features.copy()
        for move in list(state.board.legal_moves):
            state.push(move)
            state.pop()
            assert np.array_equal(state.features, before), \
                f"{name}: features drifted after push/pop of {move.uci()}"
            assert state.board.fen() == fen, f"{name}: board drifted after {move.uci()}"


def test_incremental_matches_full_recompute():
    """After a move, incremental features must equal encoding the resulting board."""
    for name, fen in POSITIONS:
        state = _fresh(fen)
        for move in list(state.board.legal_moves):
            state.push(move)
            incremental = state.features.copy()
            expected = ENCODER.encode_board(state.board)
            state.pop()
            assert np.allclose(incremental, expected, atol=1e-6), (
                f"{name}: incremental features disagree with full encode "
                f"after {move.uci()}; max diff "
                f"{np.abs(incremental - expected).max():.6g}"
            )


def test_push_null_round_trips():
    """push_null/pop must restore features exactly, like a real move."""
    for name, fen in POSITIONS:
        state = _fresh(fen)
        before = state.features.copy()
        state.push_null()
        state.pop()
        assert np.array_equal(state.features, before), f"{name}: null push/pop drifted"
        assert state.board.fen() == fen, f"{name}: board drifted after null"


def test_push_null_flips_side_to_move_and_clears_ep():
    """A null move must hand the turn over and wipe any en-passant feature."""
    stm_idx = ENCODER.side_to_move_index
    for name, fen in POSITIONS:
        state = _fresh(fen)
        turn_before = state.board.turn
        stm_before = float(state.features[stm_idx])
        state.push_null()
        assert state.board.turn != turn_before, f"{name}: null did not flip turn"
        assert float(state.features[stm_idx]) != stm_before, \
            f"{name}: side-to-move feature unchanged after null"
        ep_slice = state.features[
            ENCODER.ep_index_start:ENCODER.ep_index_start + 8]
        assert not ep_slice.any(), f"{name}: ep features set after null move"
        # and it must agree with a full re-encode of the null-moved board
        expected = ENCODER.encode_board(state.board)
        assert np.allclose(state.features, expected, atol=1e-6), \
            f"{name}: null-moved features disagree with full encode"
        state.pop()


def test_deep_sequence_round_trips():
    """Nested pushes unwind cleanly, which is what the search actually does."""
    state = _fresh("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9")
    snapshots = [state.features.copy()]
    moves = []
    for _ in range(6):
        legal = list(state.board.legal_moves)
        if not legal:
            break
        mv = legal[len(legal) // 2]
        moves.append(mv)
        state.push(mv)
        snapshots.append(state.features.copy())
    for i in range(len(moves) - 1, -1, -1):
        state.pop()
        assert np.array_equal(state.features, snapshots[i]), \
            f"drift unwinding ply {i} ({moves[i].uci()})"


if __name__ == "__main__":
    failures = 0
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}\n      {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
