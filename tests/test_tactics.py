"""Tactical regression suite.

Pruning heuristics (null-move, LMR) buy speed by declining to look at some moves, and
the risk is that they stop seeing forced tactics. A game-based Elo measurement shows
that only as diffuse noise, so this tests it directly.

Every position here is a forced mate in 2. Two reasons for that choice:

  - Ground truth is mechanical. The expected moves are not typed in by hand (an
    earlier version of this file had three wrong positions, one of which was Fool's
    Mate with White already checkmated). They are computed at runtime by an
    exhaustive solver, so a bad position fails loudly instead of silently scoring.

  - Mate in *one* would prove nothing: search() short-circuits those before the
    search runs, so they never exercise the pruning at all. Mate in 2 needs real
    depth, and these are dense middlegame positions full of quiet moves for LMR
    to reduce.

Run:  python tests/test_tactics.py
      python tests/test_tactics.py --depth 4
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import chess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Dense positions with a forced mate in 2 and a near-unique solution, found by
# scanning random games with the solver below (seed 20260724).
SUITE = [
    ("31 pieces, Nf3+",  "rnbq1bnr/pp2p2p/B2p4/2pk1PpP/5PP1/1P5N/P1PP4/RNBQK1R1 w Q - 1 11"),
    ("31 pieces, Bf7+",  "rn1qkbnr/p1p1pppp/b7/1pPB4/8/1Q4P1/PP1PPP1P/RNB1K1NR w KQkq - 0 7"),
    ("30 pieces, c6",    "1rb2bk1/ppqpp2p/n5r1/2p2p2/1PP2Pn1/N3QKPB/P2PP2P/5RNR b - - 9 15"),
    ("27 pieces, Qf2+",  "r1b1kbQ1/pp1pppB1/n1p5/8/P2qP1n1/5PPr/2PPN3/RN2KB2 b Qq - 0 13"),
    ("30 pieces, Nc7+",  "3rkbnr/3n1ppp/1Q1p4/pppNpq2/P1N4P/8/1P1PPPP1/1RB1KB1R w K - 0 15"),
    ("30 pieces, Nc2+",  "r1b1k1nr/1ppp4/p2b2pp/P7/Rn1Pp1P1/2Pq4/1P2PPBP/1NB1K1NR b kq - 2 14"),
    ("32 pieces, Qe4+",  "r1b1q1nr/p1ppbppp/1p2k3/8/NnPPpBPP/5N2/PPQ1PP2/R2K1B1R w - - 4 13"),
    ("28 pieces, Qf6+",  "rn1qkbn1/p1pp3r/b6p/1p2K3/P1P1P3/8/1P1P1PPP/RNB2BNR b q - 0 11"),
]

MATE_IN = 2


# ----------------------------------------------------------------------
# Exhaustive mate solver -- the suite's ground truth
# ----------------------------------------------------------------------

def forced_mate_moves(board: chess.Board, n: int) -> list[str]:
    """Every move for the side to move that forces mate in at most n moves."""
    out = []
    for move in board.legal_moves:
        board.push(move)
        forced = _is_mated_within(board, n - 1, attacker_to_move=False)
        board.pop()
        if forced:
            out.append(move.uci())
    return out


def _is_mated_within(board: chess.Board, n: int, attacker_to_move: bool) -> bool:
    if board.is_checkmate():
        return not attacker_to_move
    if n <= 0 or board.is_game_over():
        return False
    if attacker_to_move:
        for move in board.legal_moves:  # attacker needs one move that works
            board.push(move)
            ok = _is_mated_within(board, n - 1, False)
            board.pop()
            if ok:
                return True
        return False
    moves = list(board.legal_moves)      # defender: every reply must still lose
    if not moves:
        return False
    for move in moves:
        board.push(move)
        ok = _is_mated_within(board, n, True)
        board.pop()
        if not ok:
            return False
    return True


def build_ground_truth() -> list[tuple[str, str, list[str]]]:
    """Solve each position, and refuse to run if any of them is unsound."""
    resolved = []
    for name, fen in SUITE:
        board = chess.Board(fen)
        assert board.is_valid(), f"{name}: invalid position"
        assert not board.is_game_over(), f"{name}: position is already over"
        solutions = forced_mate_moves(chess.Board(fen), MATE_IN)
        assert solutions, f"{name}: no forced mate in {MATE_IN} -- position is bad"
        shallower = forced_mate_moves(chess.Board(fen), MATE_IN - 1)
        assert not shallower, (
            f"{name}: mate in {MATE_IN-1} exists ({shallower}), so the engine's "
            "mate fast-path would answer it without searching"
        )
        resolved.append((name, fen, solutions))
    return resolved


# ----------------------------------------------------------------------

def run_suite(engine_cls, evaluator, positions, depth: int, label: str):
    passed, misses, nodes, elapsed = 0, [], 0, 0.0
    for name, fen, solutions in positions:
        search = engine_cls(evaluator=evaluator, max_depth=depth, quiescence_depth=4)
        state = evaluator.initial_state(chess.Board(fen))
        t0 = time.perf_counter()
        res = search.search(state, 0)  # no deadline -> deterministic
        elapsed += time.perf_counter() - t0
        nodes += res.nodes
        if res.move.uci() in solutions:
            passed += 1
        else:
            misses.append((name, res.move.uci(), solutions))
    print(f"{label}: {passed}/{len(positions)} solved   "
          f"nodes={nodes:,}  time={elapsed:.1f}s")
    for name, got, want in misses:
        print(f"    MISS  {name:22s} played {got:6s} want {'/'.join(want)}")
    return passed, nodes, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    print(f"verifying {len(SUITE)} positions are sound mate-in-{MATE_IN}s...")
    positions = build_ground_truth()
    print("all positions verified\n")

    logging.disable(logging.INFO)
    from src.main import engine
    engine.ensure_ready()
    ev = engine._evaluator

    from src.bot.search_baseline import AlphaBetaSearch as Baseline
    from src.bot.search import AlphaBetaSearch as Current

    print(f"fixed depth {args.depth}, mate-in-{MATE_IN} suite")
    b_pass, b_nodes, b_time = run_suite(Baseline, ev, positions, args.depth, "baseline")
    c_pass, c_nodes, c_time = run_suite(Current, ev, positions, args.depth, "current ")

    n = len(positions)
    print(f"\n{'-'*62}")
    print(f"solved  baseline {b_pass}/{n}   current {c_pass}/{n}")
    print(f"nodes   {b_nodes:,} -> {c_nodes:,}   ({b_nodes/max(c_nodes,1):.2f}x fewer)")
    print(f"time    {b_time:.1f}s -> {c_time:.1f}s   ({b_time/max(c_time,1e-9):.2f}x faster)")
    if c_pass < b_pass:
        print("\nREGRESSION: pruning lost tactics the baseline found.")
        return 1
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
