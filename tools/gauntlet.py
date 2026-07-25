"""Self-play gauntlet: play two search implementations against each other and report Elo.

Every optimization gets measured here instead of guessed at. Game variety comes from the
opening book: each opening is played twice with colors swapped, which cancels out
first-move advantage.

TWO MEASUREMENT MODES -- pick the one that matches what you changed:

  Fixed depth (--movetime 0 --max-depth N)
      Deterministic: no deadline, so the search always reaches exactly depth N and
      picks the same move every run. Use this to measure whether a change makes the
      search *smarter* per node (better ordering, pruning, eval). Safe to parallelize.

  Fixed time (--movetime MS)
      Nondeterministic: wall-clock jitter changes the depth reached, which changes
      moves. Use this to measure whether a change makes the search *faster* -- it is
      the only mode that credits speed gains. Noisy, so it needs many more games, and
      you should run --workers 1, since CPU contention between parallel workers
      distorts the very timing you are trying to measure.

Passing the same engine twice is the harness self-test. In fixed-depth mode it must
report exactly 50.0% / 0.0 Elo (verified). In fixed-time mode it will NOT -- that is
the timing noise, not a bug.

Usage:
    python tools/gauntlet.py --games 200 --movetime 0 --max-depth 4   # search quality
    python tools/gauntlet.py --games 200 --movetime 200 --workers 1   # real strength
    python tools/gauntlet.py --games 6 --movetime 0 --max-depth 2 \
        --engine-a baseline --engine-b baseline                       # self-test
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import chess  # noqa: E402

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

MAX_PLIES = 300          # games longer than this are adjudicated as draws
RESIGN_MARGIN = 15       # material edge (pawns) that ends a hopeless game early
RESIGN_PLIES = 6         # ...sustained for this many plies


# --------------------------------------------------------------------------
# Engine registry
# --------------------------------------------------------------------------

def _build_baseline(evaluator, max_depth: int, quiescence_depth: int):
    from src.bot.search_baseline import AlphaBetaSearch
    return AlphaBetaSearch(evaluator=evaluator, max_depth=max_depth,
                           quiescence_depth=quiescence_depth)


def _build_current(evaluator, max_depth: int, quiescence_depth: int):
    from src.bot.search import AlphaBetaSearch
    return AlphaBetaSearch(evaluator=evaluator, max_depth=max_depth,
                           quiescence_depth=quiescence_depth)


ENGINES: dict[str, Callable] = {
    "baseline": _build_baseline,
    "current": _build_current,
}


# --------------------------------------------------------------------------
# Per-worker lazy state (the NNUE model is loaded once per process)
# --------------------------------------------------------------------------

_EVALUATOR = None


def _get_evaluator():
    global _EVALUATOR
    if _EVALUATOR is None:
        logging.disable(logging.WARNING)
        from src.main import engine as _engine
        _engine.ensure_ready()
        _EVALUATOR = _engine._evaluator
    return _EVALUATOR


# --------------------------------------------------------------------------
# Opening book
# --------------------------------------------------------------------------

def _material_balance(board: chess.Board) -> int:
    total = 0
    for sq, piece in board.piece_map().items():
        v = PIECE_VALUES[piece.piece_type]
        total += v if piece.color == chess.WHITE else -v
    return total


def build_openings(count: int, plies: int = 8, seed: int = 0xC0FFEE) -> list[str]:
    """Random-walk opening book. Reproducible, and filtered so neither side starts won."""
    rng = random.Random(seed)
    out: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < count and attempts < count * 200:
        attempts += 1
        board = chess.Board()
        ok = True
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                ok = False
                break
            board.push(rng.choice(moves))
        if not ok or board.is_game_over() or board.is_check():
            continue
        if abs(_material_balance(board)) > 1:
            continue
        fen = board.fen()
        if fen in seen:
            continue
        seen.add(fen)
        out.append(fen)
    if len(out) < count:
        raise RuntimeError(f"only generated {len(out)}/{count} openings")
    return out


# --------------------------------------------------------------------------
# Playing a game
# --------------------------------------------------------------------------

@dataclass
class GameResult:
    opening_idx: int
    a_is_white: bool
    score_a: float          # 1.0 win / 0.5 draw / 0.0 loss, from A's perspective
    plies: int
    reason: str
    a_depth_sum: int = 0
    a_moves: int = 0
    b_depth_sum: int = 0
    b_moves: int = 0
    error: str | None = None


def _play_game(job) -> GameResult:
    (opening_idx, fen, a_is_white, engine_a, engine_b,
     movetime, max_depth, qdepth) = job

    evaluator = _get_evaluator()
    search_a = ENGINES[engine_a](evaluator, max_depth, qdepth)
    search_b = ENGINES[engine_b](evaluator, max_depth, qdepth)

    board = chess.Board(fen)
    depth_sum = {True: 0, False: 0}
    move_count = {True: 0, False: 0}
    streak = 0
    reason = "unterminated"

    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        a_to_move = (board.turn == chess.WHITE) == a_is_white
        search = search_a if a_to_move else search_b
        try:
            state = evaluator.initial_state(board)
            res = search.search(state, movetime)
            move = res.move
            depth_sum[a_to_move] += res.depth
            move_count[a_to_move] += 1
        except Exception as exc:  # a crash is a loss for whoever crashed
            return GameResult(opening_idx, a_is_white,
                              0.0 if a_to_move else 1.0, board.ply(),
                              "crash", depth_sum[True], move_count[True],
                              depth_sum[False], move_count[False],
                              error=f"{'A' if a_to_move else 'B'}: {type(exc).__name__}: {exc}")
        if move not in board.legal_moves:
            return GameResult(opening_idx, a_is_white,
                              0.0 if a_to_move else 1.0, board.ply(),
                              "illegal_move", depth_sum[True], move_count[True],
                              depth_sum[False], move_count[False],
                              error=f"{'A' if a_to_move else 'B'} played {move.uci()} in {board.fen()}")
        board.push(move)

        bal = _material_balance(board)
        edge_white = bal >= RESIGN_MARGIN
        edge_black = bal <= -RESIGN_MARGIN
        if edge_white or edge_black:
            streak += 1
            if streak >= RESIGN_PLIES:
                white_score = 1.0 if edge_white else 0.0
                reason = "adjudicated_material"
                score_a = white_score if a_is_white else 1.0 - white_score
                return GameResult(opening_idx, a_is_white, score_a, board.ply(), reason,
                                  depth_sum[True], move_count[True],
                                  depth_sum[False], move_count[False])
        else:
            streak = 0

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        white_score, reason = 0.5, "ply_cap"
    elif outcome.winner is None:
        white_score, reason = 0.5, (outcome.termination.name.lower())
    else:
        white_score = 1.0 if outcome.winner == chess.WHITE else 0.0
        reason = outcome.termination.name.lower()

    score_a = white_score if a_is_white else 1.0 - white_score
    return GameResult(opening_idx, a_is_white, score_a, board.ply(), reason,
                      depth_sum[True], move_count[True],
                      depth_sum[False], move_count[False])


# --------------------------------------------------------------------------
# Elo math
# --------------------------------------------------------------------------

def elo_diff(score: float) -> float:
    if score <= 0.0:
        return float("-inf")
    if score >= 1.0:
        return float("inf")
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_report(results: list[GameResult]) -> str:
    n = len(results)
    if n == 0:
        return "no games played"
    wins = sum(1 for r in results if r.score_a == 1.0)
    draws = sum(1 for r in results if r.score_a == 0.5)
    losses = n - wins - draws
    score = (wins + 0.5 * draws) / n

    # standard error on the score rate, propagated to Elo
    var = sum((r.score_a - score) ** 2 for r in results) / max(n - 1, 1)
    se = math.sqrt(var / n)
    lo, hi = max(1e-9, score - 1.96 * se), min(1 - 1e-9, score + 1.96 * se)

    a_depth = sum(r.a_depth_sum for r in results) / max(sum(r.a_moves for r in results), 1)
    b_depth = sum(r.b_depth_sum for r in results) / max(sum(r.b_moves for r in results), 1)

    lines = [
        "",
        "=" * 62,
        f"games {n}   W {wins}  L {losses}  D {draws}   score {score*100:.1f}%",
        f"Elo(A-B)  {elo_diff(score):+.1f}   95% CI [{elo_diff(lo):+.1f}, {elo_diff(hi):+.1f}]",
        f"mean depth   A {a_depth:.2f}   B {b_depth:.2f}",
    ]
    reasons: dict[str, int] = {}
    for r in results:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
    lines.append("terminations  " + "  ".join(f"{k}={v}" for k, v in sorted(reasons.items())))

    errs = [r for r in results if r.error]
    if errs:
        lines.append(f"\n!! {len(errs)} game(s) ended in crash/illegal move:")
        for r in errs[:5]:
            lines.append(f"   [{r.reason}] {r.error}")
    lines.append("=" * 62)
    return "\n".join(lines)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--movetime", type=int, default=200, help="ms per move")
    ap.add_argument("--engine-a", default="current", choices=sorted(ENGINES))
    ap.add_argument("--engine-b", default="baseline", choices=sorted(ENGINES))
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--quiescence-depth", type=int, default=4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    ap.add_argument("--opening-plies", type=int, default=8)
    args = ap.parse_args()

    n_openings = (args.games + 1) // 2
    openings = build_openings(n_openings, args.opening_plies, args.seed)

    jobs = []
    for i, fen in enumerate(openings):
        for a_is_white in (True, False):
            if len(jobs) >= args.games:
                break
            jobs.append((i, fen, a_is_white, args.engine_a, args.engine_b,
                         args.movetime, args.max_depth,
                         args.quiescence_depth))

    print(f"A={args.engine_a}  vs  B={args.engine_b}")
    print(f"{len(jobs)} games from {len(openings)} openings, {args.movetime}ms/move, "
          f"{args.workers} worker(s)")
    if args.movetime == 0:
        print(f"mode: FIXED DEPTH {args.max_depth} (deterministic - measures search quality)")
    else:
        print("mode: FIXED TIME (nondeterministic - measures speed+quality)")
        if args.workers > 1:
            print(f"  WARNING: --workers {args.workers} with a time control lets CPU "
                  "contention\n           distort the timing being measured. "
                  "Prefer --workers 1.")
    if args.engine_a == args.engine_b:
        if args.movetime == 0:
            print("(self-test: expect exactly 50.0% / 0.0 Elo)")
        else:
            print("(self-test under a time control: expect ~50% with noise, not exactly 50%)")
    print()

    results: list[GameResult] = []
    t0 = time.perf_counter()
    if args.workers == 1:
        for j in jobs:
            results.append(_play_game(j))
            _progress(len(results), len(jobs), results, t0)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_play_game, j) for j in jobs]
            for f in as_completed(futs):
                results.append(f.result())
                _progress(len(results), len(jobs), results, t0)

    print(elo_report(results))
    return 0


def _progress(done: int, total: int, results: list[GameResult], t0: float) -> None:
    score = sum(r.score_a for r in results) / max(len(results), 1)
    rate = (time.perf_counter() - t0) / max(done, 1)
    eta = rate * (total - done)
    print(f"\r  {done}/{total} games   score {score*100:5.1f}%   "
          f"eta {eta/60:4.1f}m", end="", flush=True)
    if done == total:
        print()


if __name__ == "__main__":
    raise SystemExit(main())
