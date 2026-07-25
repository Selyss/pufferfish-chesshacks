"""Behavioural tests for the C++ engine, driven over UCI.

These exist because every bug that mattered in this engine was a correctness
bug, and each one was found by checking a specific claim rather than by reading
code. They are the regression net for those bugs:

  illegal moves        castling through an attacked square was generated and
                       played; the whole move generator is cross-checked against
                       python-chess
  null moves           running out of time during depth 1 returned "bestmove
                       0000", an instant forfeit
  stalemate scoring    a node with no legal move kept its -INF seed, so
                       stalemating scored better than mating and won endgames
                       were deliberately drawn
  transposition table  cutoffs at the root returned a score with no principal
                       variation, so with a warm table every shallow iteration
                       was discarded and the engine answered a mate in one with
                       a mate in five
  hangs                "go infinite" never returned because the search owned the
                       only thread and never read "stop"

Run:  python tests/test_engine.py
      python tests/test_engine.py --engine /path/to/pufferfish

IMPORTANT for anyone adding tests here: keep stdin open until "bestmove"
arrives. Closing it (or sending "quit") aborts the search, because that is what
those commands mean. Writing a fixed command string and reading the output
afterwards silently truncates the search and makes correct code look broken --
it produced three false alarms in one afternoon. Engine.go() below does it
properly; use it.
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time

import chess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = [
    os.path.join(REPO, "pufferfish", "build", "pufferfish"),
    os.path.join(REPO, "build", "pufferfish"),
]


def find_engine(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    raise SystemExit(
        "engine binary not found; build it with\n"
        "  cmake -S pufferfish -B pufferfish/build -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build pufferfish/build -j8")


class Engine:
    """A UCI client that waits for replies instead of closing the pipe."""

    def __init__(self, path: str, cwd: str | None = None):
        self.proc = subprocess.Popen(
            [path, "--uci"], cwd=cwd or REPO, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.send("uci")
        self.read_until("uciok")

    def send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, token: str, timeout: float = 60.0) -> list[str]:
        end = time.time() + timeout
        seen: list[str] = []
        while time.time() < end:
            line = self.proc.stdout.readline()
            if not line:
                return seen
            seen.append(line.strip())
            if line.startswith(token):
                return seen
        raise TimeoutError(f"engine never sent {token!r} (hung?)")

    def go(self, fen: str | None = None, moves: str = "", limit: str = "depth 8"):
        """Search and return (bestmove, info lines). Holds stdin open throughout."""
        if fen:
            self.send(f"position fen {fen}" + (f" moves {moves}" if moves else ""))
        else:
            self.send("position startpos" + (f" moves {moves}" if moves else ""))
        self.send(f"go {limit}")
        lines = self.read_until("bestmove")
        best = None
        for ln in lines:
            if ln.startswith("bestmove"):
                parts = ln.split()
                best = parts[1] if len(parts) > 1 else None
        return best, [l for l in lines if l.startswith("info")]

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# ----------------------------------------------------------------------

def test_movegen_matches_python_chess(path, report):
    """No illegal move generated, no legal move missed. Catches the castling bug."""
    dumper = os.path.join(os.path.dirname(path), "movedump")
    if not os.path.exists(dumper):
        report("movegen cross-check", None, "skipped (movedump helper not built)")
        return
    rng = random.Random(99)
    fens = []
    for _ in range(1500):
        b = chess.Board()
        for _ in range(rng.randint(0, 40)):
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rng.choice(ms))
        if not b.is_game_over():
            fens.append(b.fen())
    fens += [
        "r3k2r/pp2pp2/q1p4b/8/P2PN1Q1/6P1/1PP4P/R3K2R w KQkq - 0 16",  # castle through check
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/5q2/8/R3K2R w KQkq - 0 1",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # en passant
        "8/PPP4k/8/8/8/8/4Kppp/8 w - - 0 1",  # promotions
    ]
    out = subprocess.run([dumper], input="\n".join(fens) + "\n",
                         capture_output=True, text=True, cwd=REPO).stdout.strip().split("\n")
    bad = 0
    for fen, line in zip(fens, out):
        ours = set(line.split())
        theirs = {m.uci() for m in chess.Board(fen).legal_moves}
        if ours != theirs:
            bad += 1
    report("movegen matches python-chess", bad == 0,
           f"{len(fens)} positions, {bad} with a discrepancy")


def test_never_returns_null_move(path, report):
    """Even at absurdly short budgets a legal move must come back."""
    eng = Engine(path)
    rng = random.Random(3)
    fens = []
    for _ in range(25):
        b = chess.Board()
        for _ in range(rng.randint(0, 50)):
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rng.choice(ms))
        if not b.is_game_over():
            fens.append(b.fen())
    bad = 0
    for f in fens:
        for lim in ("movetime 1", "movetime 5", "movetime 20"):
            mv, _ = eng.go(fen=f, limit=lim)
            if not mv or mv == "0000":
                bad += 1
    eng.close()
    report("never returns a null move", bad == 0,
           f"{len(fens)*3} searches at 1/5/20ms, {bad} null")


def test_finds_forced_mates(path, report):
    eng = Engine(path)
    cases = [
        "8/k1K5/8/8/4Q3/8/8/8 w - - 0 1",
        "5K1k/8/1Q6/8/8/8/8/8 w - - 0 1",
        "6rk/ppp2p1p/2b1p3/4bp2/7q/2P2P2/PP2B2P/R2Q3K b - - 0 22",
        "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    ]
    bad = []
    for fen in cases:
        mv, _ = eng.go(fen=fen, limit="depth 8")
        b = chess.Board(fen)
        b.push(chess.Move.from_uci(mv))
        if not b.is_checkmate():
            bad.append(f"{fen} -> {mv}")
    eng.close()
    report("finds forced mates", not bad, f"{len(cases)} positions" +
           ("" if not bad else f"; missed {bad}"))


def test_does_not_stalemate_won_endgames(path, report):
    """K+Q vs K was stalemated 6 times out of 6 before the terminal-score fix."""
    eng = Engine(path)
    cases = [
        "8/2Q5/8/8/8/8/3K4/5k2 w - - 0 1",
        "7K/8/8/8/8/8/2Q5/k7 w - - 0 1",
        "k7/8/8/1Q6/3K4/8/8/8 w - - 0 1",
        "7k/8/4Q3/8/8/1K6/8/8 w - - 0 1",
        "Q7/8/8/8/8/8/2K5/4k3 w - - 0 1",
        "7k/8/8/8/7K/8/5Q2/8 w - - 0 1",
    ]
    bad = []
    for fen in cases:
        mv, _ = eng.go(fen=fen, limit="depth 8")
        b = chess.Board(fen)
        b.push(chess.Move.from_uci(mv))
        if b.is_stalemate():
            bad.append(f"{fen} -> {mv}")
    eng.close()
    report("does not stalemate won endgames", not bad,
           f"{len(cases)} K+Q/K+R wins" + ("" if not bad else f"; threw away {len(bad)}"))


def test_warm_table_still_finds_mate_in_one(path, report):
    """The transposition bug: correct from a fresh process, wrong from a warm one."""
    import chess.pgn, io
    pgn = ("1. e4 c6 2. d4 d5 3. exd5 cxd5 4. Nf3 Bf5 5. Nc3 a6 6. Bf4 e6 7. Be2 Ne7 "
           "8. O-O Ng6 9. Bg3 Nc6 10. h3 Be7 11. Na4 O-O 12. c3 e5 13. Re1 e4 14. Nd2 Bh4 "
           "15. Bg4 Bxg4 16. Qxg4 f5 17. Qe2 Bxg3 18. fxg3 b6 19. c4 Nxd4 20. Qd1 dxc4 "
           "21. Nxc4 b5 22. Nc5 bxc4 23. Rc1 Qd6 24. Rxc4 Nf3+ 25. Kh1 Qe5 26. Nd7 Qxg3 "
           "27. Qd5+ Kh8 28. gxf3 Qxe1+ 29. Kg2 Nf4+ 30. Kh2 Qf2+ 31. Kh1")
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = chess.Board()
    seen = []
    for m in game.mainline_moves():
        board.push(m)
        if board.turn == chess.BLACK:
            seen.append(board.fen())
    mate_fen = seen[-1]

    eng = Engine(path)
    for f in seen[:-1]:
        eng.go(fen=f, limit="movetime 60")      # fill the table
    mv, _ = eng.go(fen=mate_fen, limit="wtime 30000 btime 30000")
    eng.close()
    report("mate in one found with a warm table", mv == "f2g2",
           f"warmed with {len(seen)-1} searches, played {mv} (want f2g2 = Qg2#)")


def test_go_infinite_then_stop(path, report):
    """go infinite must run until stop, and stop must be answered promptly."""
    eng = Engine(path)
    eng.send("position startpos moves e2e4 c7c6")
    eng.send("go infinite")
    time.sleep(1.5)
    t0 = time.perf_counter()
    eng.send("stop")
    try:
        lines = eng.read_until("bestmove", timeout=15)
        dt = (time.perf_counter() - t0) * 1000
        ok = any(l.startswith("bestmove") for l in lines)
    except TimeoutError:
        ok, dt = False, -1
    eng.close()
    report("go infinite then stop", ok,
           f"bestmove {dt:.0f}ms after stop" if ok else "never returned (hung)")


def test_isready_during_search(path, report):
    eng = Engine(path)
    eng.send("position startpos")
    eng.send("go infinite")
    time.sleep(0.8)
    eng.send("isready")
    try:
        eng.read_until("readyok", timeout=10)
        ok = True
    except TimeoutError:
        ok = False
    eng.send("stop")
    try:
        eng.read_until("bestmove", timeout=15)
    except TimeoutError:
        pass
    eng.close()
    report("isready answered during search", ok, "UCI requires this mid-search")


def test_weights_load_from_any_directory(path, report):
    """GUIs launch engines from arbitrary directories and cannot set one."""
    bad = []
    for cwd in ("/", os.path.expanduser("~"), "/var/tmp"):
        try:
            eng = Engine(path, cwd=cwd)
            mv, _ = eng.go(limit="depth 4")
            eng.close()
            if not mv or mv == "0000":
                bad.append(cwd)
        except Exception as exc:
            bad.append(f"{cwd} ({type(exc).__name__})")
    report("weights load from any working directory", not bad,
           "tried /, $HOME, /var/tmp" + ("" if not bad else f"; failed in {bad}"))


def test_plays_a_legal_game_against_itself(path, report):
    """End to end: a whole game with every move validated."""
    eng = Engine(path)
    board = chess.Board()
    bad = None
    for _ in range(120):
        if board.is_game_over(claim_draw=True):
            break
        mv, _ = eng.go(fen=board.fen(), limit="movetime 40")
        if not mv or mv == "0000":
            bad = f"null move at {board.fen()}"
            break
        m = chess.Move.from_uci(mv)
        if m not in board.legal_moves:
            bad = f"illegal {mv} at {board.fen()}"
            break
        board.push(m)
    eng.close()
    report("plays a legal game end to end", bad is None,
           f"{board.ply()} plies" + ("" if bad is None else f"; {bad}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None)
    args = ap.parse_args()
    path = find_engine(args.engine)
    print(f"engine: {path}\n")

    results = []

    def report(name, ok, detail=""):
        results.append((name, ok))
        mark = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  {mark:4s} {name:44s} {detail}")

    for fn in (test_movegen_matches_python_chess,
               test_never_returns_null_move,
               test_finds_forced_mates,
               test_does_not_stalemate_won_endgames,
               test_warm_table_still_finds_mate_in_one,
               test_go_infinite_then_stop,
               test_isready_during_search,
               test_weights_load_from_any_directory,
               test_plays_a_legal_game_against_itself):
        try:
            fn(path, report)
        except Exception as exc:
            report(fn.__name__, False, f"{type(exc).__name__}: {exc}")

    failed = [n for n, ok in results if ok is False]
    print()
    if failed:
        print(f"{len(failed)} failed: {', '.join(failed)}")
        return 1
    print(f"all {len([r for r in results if r[1] is not None])} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
