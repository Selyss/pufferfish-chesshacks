"""Play C++ engine revisions against each other and report the Elo difference.

Builds each revision from git into its own worktree, then plays a match. Openings
come from a seeded random-walk book, each played twice with colours swapped so
first-move advantage cancels.

Two engine protocols are supported, because UCI was only added recently and older
revisions predate it:

  uci       one process per game, transposition table stays warm across moves
  oneshot   one process per move via --fen/--movetime, ~30ms of startup each

The protocol is detected per binary, so you can bench against anything that still
compiles. Revisions before fe0d32d do not build on macOS at all (they call the MSVC
intrinsic __popcnt64), and revisions before f257f1c evaluate every position as 0.

Usage:
    python tools/selfplay.py --a HEAD --b f257f1c --games 40 --movetime 200
    python tools/selfplay.py --a HEAD --b HEAD --games 10 --movetime 100   # self-test
    python tools/selfplay.py --list                                        # candidates

The self-test (same revision twice) is the harness check. Engines are not
deterministic under a time control, so expect roughly 50%, not exactly 50%.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

import chess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_ROOT = os.path.join(REPO, ".selfplay")
MAX_PLIES = 300
RESIGN_MARGIN = 15
RESIGN_PLIES = 6
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------

def _run(cmd, cwd=None, check=True, timeout=600):
    return subprocess.run(cmd, cwd=cwd, check=check, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def resolve(rev: str) -> str:
    return _run(["git", "rev-parse", "--short", rev], cwd=REPO).stdout.strip()


def build_revision(rev: str, verbose: bool = True) -> str:
    """Check out `rev` into a worktree and build it. Returns the binary path."""
    sha = resolve(rev)
    work = os.path.join(BUILD_ROOT, sha)
    binary = os.path.join(work, "build", "pufferfish")
    if os.path.exists(binary):
        return binary

    os.makedirs(BUILD_ROOT, exist_ok=True)
    src = os.path.join(work, "src")
    if not os.path.exists(src):
        if verbose:
            print(f"  [{sha}] creating worktree")
        _run(["git", "worktree", "add", "--detach", src, sha], cwd=REPO)

    if verbose:
        print(f"  [{sha}] building")
    try:
        _run(["cmake", "-S", os.path.join(src, "pufferfish"),
              "-B", os.path.join(work, "build"),
              "-DCMAKE_BUILD_TYPE=Release"], cwd=REPO)
        _run(["cmake", "--build", os.path.join(work, "build"), "-j8"], cwd=REPO)
    except subprocess.CalledProcessError as exc:
        tail = "\n".join((exc.stdout or "").strip().splitlines()[-6:])
        raise SystemExit(f"revision {sha} does not build:\n{tail}")
    if not os.path.exists(binary):
        raise SystemExit(f"revision {sha} built but produced no binary")
    return binary


# --------------------------------------------------------------------------
# engine adapters
# --------------------------------------------------------------------------

class Engine:
    """Drives one engine binary, over UCI when available, else the one-shot CLI."""

    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name
        self.proc: subprocess.Popen | None = None
        self.protocol = "oneshot"
        self._try_uci()

    def _try_uci(self) -> None:
        try:
            p = subprocess.Popen([self.path, "--uci"], cwd=REPO, text=True,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, bufsize=1)
            p.stdin.write("uci\n")
            p.stdin.flush()
            deadline = time.time() + 5
            while time.time() < deadline:
                line = p.stdout.readline()
                if not line:
                    break
                if line.strip() == "uciok":
                    self.proc = p
                    self.protocol = "uci"
                    return
            p.kill()
        except Exception:
            pass

    def new_game(self) -> None:
        if self.protocol == "uci" and self.proc:
            self.proc.stdin.write("ucinewgame\nisready\n")
            self.proc.stdin.flush()
            deadline = time.time() + 5
            while time.time() < deadline:
                line = self.proc.stdout.readline()
                if not line or line.strip() == "readyok":
                    break

    def bestmove(self, fen: str, movetime: int) -> str | None:
        if self.protocol == "uci":
            return self._bestmove_uci(fen, movetime)
        return self._bestmove_oneshot(fen, movetime)

    def _bestmove_uci(self, fen: str, movetime: int) -> str | None:
        assert self.proc
        try:
            self.proc.stdin.write(f"position fen {fen}\ngo movetime {movetime}\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            return None
        deadline = time.time() + movetime / 1000.0 + 15
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                return None
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else None
        return None

    def _bestmove_oneshot(self, fen: str, movetime: int) -> str | None:
        try:
            r = subprocess.run([self.path, "--fen", fen, "--movetime", str(movetime)],
                               cwd=REPO, capture_output=True, text=True,
                               timeout=movetime / 1000.0 + 20)
        except subprocess.TimeoutExpired:
            return None
        for line in r.stdout.splitlines():
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else None
        return None

    def close(self) -> None:
        if self.proc:
            try:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()


# --------------------------------------------------------------------------
# match play
# --------------------------------------------------------------------------

def material(board: chess.Board) -> int:
    t = 0
    for _, p in board.piece_map().items():
        v = PIECE_VALUES[p.piece_type]
        t += v if p.color == chess.WHITE else -v
    return t


def build_openings(count: int, plies: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out, seen, tries = [], set(), 0
    while len(out) < count and tries < count * 400:
        tries += 1
        b = chess.Board()
        for _ in range(plies):
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rng.choice(ms))
        if b.is_game_over() or b.is_check() or abs(material(b)) > 1:
            continue
        f = b.fen()
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


@dataclass
class Result:
    score_a: float
    plies: int
    reason: str
    error: str | None = None
    depths: dict = field(default_factory=dict)


def play_game(ea: Engine, eb: Engine, fen: str, a_white: bool,
              movetime: int) -> Result:
    board = chess.Board(fen)
    ea.new_game()
    eb.new_game()
    streak = 0

    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        a_turn = (board.turn == chess.WHITE) == a_white
        eng = ea if a_turn else eb
        uci = eng.bestmove(board.fen(), movetime)
        if uci is None:
            return Result(0.0 if a_turn else 1.0, board.ply(), "no_move",
                          f"{eng.name} returned nothing")
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            return Result(0.0 if a_turn else 1.0, board.ply(), "bad_uci",
                          f"{eng.name} sent {uci!r}")
        if mv not in board.legal_moves:
            return Result(0.0 if a_turn else 1.0, board.ply(), "illegal_move",
                          f"{eng.name} played {uci} in {board.fen()}")
        board.push(mv)

        bal = material(board)
        if abs(bal) >= RESIGN_MARGIN:
            streak += 1
            if streak >= RESIGN_PLIES:
                white_score = 1.0 if bal > 0 else 0.0
                return Result(white_score if a_white else 1.0 - white_score,
                              board.ply(), "adjudicated")
        else:
            streak = 0

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        white_score, reason = 0.5, "ply_cap"
    elif outcome.winner is None:
        white_score, reason = 0.5, outcome.termination.name.lower()
    else:
        white_score = 1.0 if outcome.winner == chess.WHITE else 0.0
        reason = outcome.termination.name.lower()
    return Result(white_score if a_white else 1.0 - white_score, board.ply(), reason)


def elo(score: float) -> float:
    if score <= 0:
        return float("-inf")
    if score >= 1:
        return float("inf")
    return -400.0 * math.log10(1.0 / score - 1.0)


def report(results: list[Result], a: str, b: str) -> str:
    n = len(results)
    if not n:
        return "no games"
    w = sum(1 for r in results if r.score_a == 1.0)
    d = sum(1 for r in results if r.score_a == 0.5)
    l = n - w - d
    s = (w + 0.5 * d) / n
    var = sum((r.score_a - s) ** 2 for r in results) / max(n - 1, 1)
    se = math.sqrt(var / n)
    lo, hi = max(1e-9, s - 1.96 * se), min(1 - 1e-9, s + 1.96 * se)
    lines = ["", "=" * 64,
             f"A = {a}   B = {b}",
             f"games {n}   W {w}  L {l}  D {d}   score {s*100:.1f}%",
             f"Elo(A-B)  {elo(s):+.1f}   95% CI [{elo(lo):+.1f}, {elo(hi):+.1f}]"]
    if lo <= 0.5 <= hi:
        lines.append("  -> interval spans zero: this run cannot distinguish them")
    reasons: dict[str, int] = {}
    for r in results:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
    lines.append("terminations  " + "  ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    errs = [r for r in results if r.error]
    if errs:
        lines.append(f"\n!! {len(errs)} game(s) ended badly:")
        for r in errs[:5]:
            lines.append(f"   [{r.reason}] {r.error}")
    lines.append("=" * 64)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="HEAD", help="revision under test")
    ap.add_argument("--b", default="f257f1c", help="baseline revision")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--movetime", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--list", action="store_true", help="show buildable revisions")
    ap.add_argument("--clean", action="store_true", help="remove build worktrees")
    args = ap.parse_args()

    if args.clean:
        for d in os.listdir(BUILD_ROOT) if os.path.isdir(BUILD_ROOT) else []:
            src = os.path.join(BUILD_ROOT, d, "src")
            if os.path.exists(src):
                subprocess.run(["git", "worktree", "remove", "--force", src], cwd=REPO)
        shutil.rmtree(BUILD_ROOT, ignore_errors=True)
        print("cleaned")
        return 0

    if args.list:
        out = _run(["git", "log", "--oneline", "-15"], cwd=REPO).stdout
        print("recent revisions (only fe0d32d onward compiles on macOS;\n"
              "only f257f1c onward has a working evaluation):\n")
        print(out)
        return 0

    sha_a, sha_b = resolve(args.a), resolve(args.b)
    print(f"building A={args.a} ({sha_a})")
    bin_a = build_revision(args.a)
    print(f"building B={args.b} ({sha_b})")
    bin_b = build_revision(args.b)

    ea = Engine(bin_a, f"A/{sha_a}")
    eb = Engine(bin_b, f"B/{sha_b}")
    print(f"protocols: A={ea.protocol}  B={eb.protocol}")
    if sha_a == sha_b:
        print("(self-test: same revision both sides; expect ~50%, not exactly 50%,\n"
              " because the engines are time-limited and therefore nondeterministic)")

    openings = build_openings((args.games + 1) // 2, args.opening_plies, args.seed)
    jobs = []
    for i, f in enumerate(openings):
        for a_white in (True, False):
            if len(jobs) >= args.games:
                break
            jobs.append((f, a_white))

    results: list[Result] = []
    t0 = time.perf_counter()
    try:
        for i, (fen, a_white) in enumerate(jobs, 1):
            results.append(play_game(ea, eb, fen, a_white, args.movetime))
            sc = sum(r.score_a for r in results) / len(results)
            rate = (time.perf_counter() - t0) / len(results)
            print(f"\r  {i}/{len(jobs)}  score {sc*100:5.1f}%  "
                  f"eta {rate*(len(jobs)-i)/60:4.1f}m", end="", flush=True)
        print()
    finally:
        ea.close()
        eb.close()

    print(report(results, f"{args.a} ({sha_a})", f"{args.b} ({sha_b})"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
