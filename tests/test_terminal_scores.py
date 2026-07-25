"""Terminal positions must be scored exactly, not handed to the evaluator.

The search routed every `board.is_game_over()` position into quiescence, which
returns the raw NNUE evaluation. That produced two wrong answers:

  checkmate  scored ~-13.7 instead of a mate score. Mates were still usually found,
             but only because the network happens to dislike mated positions. Mate
             distance was never encoded, so the engine had no reason to prefer a
             faster mate, and the TT's mate-score adjustment was dead code.

  stalemate  scored ~-20.0 instead of 0.0, and the sign makes this actively
             dangerous. Scores are from the side to move's perspective, so a
             stalemated opponent scoring -20 is read by the parent node as +20 --
             a larger gain than any real advantage the evaluator can express. A
             winning engine would therefore steer into stalemate and throw the game.

Run:  python tests/test_terminal_scores.py
"""
from __future__ import annotations

import logging
import os
import sys

import chess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DRAW_TOL = 1e-9


def _searcher(ev):
    from src.bot.search import AlphaBetaSearch
    s = AlphaBetaSearch(evaluator=ev, max_depth=3, quiescence_depth=4)
    s.first_move_completed = False   # normally set inside search()
    return s


def _score(ev, fen: str, ply: int = 1) -> float:
    s = _searcher(ev)
    state = ev.initial_state(chess.Board(fen))
    return s._negamax(state, 2, -1e9, 1e9, ply, None, None)


CHECKMATES = [
    ("black mated, back rank", "R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1"),
    ("black mated, smothered", "6rk/5Npp/8/8/8/8/8/6K1 b - - 1 1"),
]

STALEMATES = [
    ("classic queen stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
    ("corner stalemate",        "k7/8/1Q6/8/8/8/8/7K b - - 0 1"),
]

INSUFFICIENT = [
    ("K vs K",   "7k/8/6K1/8/8/8/8/8 w - - 0 1"),
    ("K vs K+N", "7k/8/6K1/8/8/8/8/6N1 b - - 0 1"),
]


def main() -> int:
    logging.disable(logging.INFO)
    from src.main import engine
    engine.ensure_ready()
    ev = engine._evaluator

    failures = []

    for name, fen in CHECKMATES:
        b = chess.Board(fen)
        assert b.is_checkmate(), f"{name}: position is not actually checkmate"
        got = _score(ev, fen, ply=1)
        # side to move is mated: large negative, magnitude near mate_score
        ok = got < -9000
        print(f"{'PASS' if ok else 'FAIL'}  checkmate    {name:26s} score={got:+10.3f}")
        if not ok:
            failures.append(f"{name}: checkmate scored {got:+.3f}, expected < -9000")

    for name, fen in STALEMATES:
        b = chess.Board(fen)
        assert b.is_stalemate(), f"{name}: position is not actually stalemate"
        got = _score(ev, fen)
        ok = abs(got) < DRAW_TOL
        print(f"{'PASS' if ok else 'FAIL'}  stalemate    {name:26s} score={got:+10.3f}")
        if not ok:
            failures.append(f"{name}: stalemate scored {got:+.3f}, expected 0.0")

    for name, fen in INSUFFICIENT:
        b = chess.Board(fen)
        assert b.is_insufficient_material(), f"{name}: material is not insufficient"
        got = _score(ev, fen)
        ok = abs(got) < DRAW_TOL
        print(f"{'PASS' if ok else 'FAIL'}  dead draw    {name:26s} score={got:+10.3f}")
        if not ok:
            failures.append(f"{name}: dead draw scored {got:+.3f}, expected 0.0")

    # Mate distance. _terminal_score is from the perspective of the side that is
    # MATED, so being mated sooner must score *lower* (worse) for them. The parent
    # node negates it, which is what makes the mating side prefer the faster mate.
    s = _searcher(ev)
    near = s._terminal_score(chess.Board(CHECKMATES[0][1]), ply=2)
    far = s._terminal_score(chess.Board(CHECKMATES[0][1]), ply=8)
    ok = near < far and near < -9000 and far < -9000
    print(f"{'PASS' if ok else 'FAIL'}  mate distance  mated sooner scores worse "
          f"(ply2={near:+.1f} < ply8={far:+.1f});  "
          f"mating side prefers ply2 ({-near:+.1f} > {-far:+.1f})")
    if not ok:
        failures.append(
            f"mate distance not encoded: expected ply2 ({near:+.1f}) < ply8 ({far:+.1f})")

    print()
    if failures:
        print(f"{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all terminal scores correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
