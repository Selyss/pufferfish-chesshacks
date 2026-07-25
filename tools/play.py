"""Play a game against the C++ engine in the terminal.

    python tools/play.py                     # you are White, 1s per engine move
    python tools/play.py --black             # engine moves first
    python tools/play.py --movetime 3000     # give it more thinking time
    python tools/play.py --depth 12          # fixed depth instead of time
    python tools/play.py --fen "<FEN>"       # start from a position

Enter moves as SAN (Nf3, exd5, O-O) or UCI (g1f3, e4d5, e1g1). Commands:

    moves       list every legal move
    board       redraw
    eval        what the engine thinks of the current position
    undo        take back your move and its reply
    fen         print the current FEN
    quit        resign and exit

The engine binary is found automatically, or pass --engine /path/to/pufferfish.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import chess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CANDIDATES = [
    os.path.join(REPO, "pufferfish", "build", "pufferfish"),
    os.path.join(REPO, "build", "pufferfish"),
    "/private/tmp/claude-501/-Users-selyss-code-pufferfish-chesshacks/"
    "ec73096d-3f4d-41d2-be57-594e6cd56095/scratchpad/build/pufferfish",
]

PIECES = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}


def find_engine(explicit: str | None) -> str:
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f"no engine at {explicit}")
        return explicit
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    raise SystemExit(
        "Could not find the engine binary. Build it first:\n"
        "    cmake -S pufferfish -B pufferfish/build\n"
        "    cmake --build pufferfish/build -j8\n"
        "or pass --engine /path/to/pufferfish")


class Engine:
    def __init__(self, path: str):
        self.proc = subprocess.Popen(
            [path, "--uci"], cwd=REPO, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._send("uci")
        self._wait_for("uciok")
        self._send("ucinewgame")

    def _send(self, s: str) -> None:
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str, timeout: float = 10.0):
        end = time.time() + timeout
        last = []
        while time.time() < end:
            line = self.proc.stdout.readline()
            if not line:
                break
            last.append(line.strip())
            if line.startswith(token):
                return last
        return last

    def analyse(self, board: chess.Board, movetime: int, depth: int | None):
        self._send(f"position fen {board.fen()}")
        self._send(f"go depth {depth}" if depth else f"go movetime {movetime}")
        lines = self._wait_for("bestmove", timeout=(depth and 120) or movetime / 1000 + 30)
        best, score, dep, nodes = None, None, None, None
        for ln in lines:
            if ln.startswith("info"):
                parts = ln.split()
                if "score" in parts:
                    try:
                        score = int(parts[parts.index("score") + 2])
                    except (ValueError, IndexError):
                        pass
                if "depth" in parts:
                    try:
                        dep = int(parts[parts.index("depth") + 1])
                    except (ValueError, IndexError):
                        pass
                if "nodes" in parts:
                    try:
                        nodes = int(parts[parts.index("nodes") + 1])
                    except (ValueError, IndexError):
                        pass
            if ln.startswith("bestmove"):
                p = ln.split()
                best = p[1] if len(p) > 1 else None
        return best, score, dep, nodes

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


def show(board: chess.Board, flip: bool) -> None:
    print()
    ranks = range(8) if flip else range(7, -1, -1)
    for r in ranks:
        files = range(7, -1, -1) if flip else range(8)
        row = [f"{r+1} "]
        for f in files:
            p = board.piece_at(chess.square(f, r))
            row.append((PIECES[p.symbol()] if p else "·") + " ")
        print("".join(row))
    files = "hgfedcba" if flip else "abcdefgh"
    print("  " + " ".join(files))
    if board.move_stack:
        print(f"\n  last: {board.peek().uci()}")
    print()


def score_text(score: int | None, pov_white: bool) -> str:
    if score is None:
        return "?"
    s = score if pov_white else -score
    if abs(s) > 30000:
        return "mate"
    if abs(s) > 29000:
        n = (31000 - abs(s) + 1) // 2
        return f"mate in {n}" + ("" if s > 0 else " (against it)")
    return f"{s/100:+.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--black", action="store_true", help="you play Black")
    ap.add_argument("--movetime", type=int, default=1000, help="ms per engine move")
    ap.add_argument("--depth", type=int, default=None, help="fixed depth instead of time")
    ap.add_argument("--fen", default=None)
    args = ap.parse_args()

    path = find_engine(args.engine)
    board = chess.Board(args.fen) if args.fen else chess.Board()
    human = chess.BLACK if args.black else chess.WHITE

    eng = Engine(path)
    print(f"engine: {path}")
    print(f"you are {'Black' if args.black else 'White'}; "
          f"engine thinks for {args.depth and str(args.depth)+' plies' or str(args.movetime)+'ms'}")
    print("type 'moves' for legal moves, 'undo' to take back, 'quit' to resign")
    show(board, flip=args.black)

    try:
        while not board.is_game_over(claim_draw=True):
            if board.turn == human:
                try:
                    raw = input("your move> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nbye")
                    return 0
                if not raw:
                    continue
                low = raw.lower()
                if low in ("quit", "exit", "resign"):
                    print("you resign. good game.")
                    return 0
                if low == "board":
                    show(board, args.black); continue
                if low == "fen":
                    print(board.fen()); continue
                if low == "moves":
                    print("  " + " ".join(sorted(board.san(m) for m in board.legal_moves)))
                    continue
                if low == "eval":
                    _, sc, dep, _ = eng.analyse(board, args.movetime, args.depth)
                    print(f"  engine: {score_text(sc, board.turn == chess.WHITE)} "
                          f"at depth {dep} (positive = good for the side to move)")
                    continue
                if low == "undo":
                    if len(board.move_stack) >= 2:
                        board.pop(); board.pop(); show(board, args.black)
                    else:
                        print("  nothing to undo")
                    continue
                mv = None
                for parse in (board.parse_san, chess.Move.from_uci):
                    try:
                        cand = parse(raw)
                        if cand in board.legal_moves:
                            mv = cand
                            break
                    except Exception:
                        pass
                if mv is None:
                    print(f"  '{raw}' is not a legal move here. try 'moves'.")
                    continue
                board.push(mv)
                show(board, args.black)
            else:
                t0 = time.perf_counter()
                best, sc, dep, nodes = eng.analyse(board, args.movetime, args.depth)
                dt = (time.perf_counter() - t0) * 1000
                if not best or best == "0000":
                    print("engine returned no move; treating as resignation.")
                    return 0
                mv = chess.Move.from_uci(best)
                if mv not in board.legal_moves:
                    print(f"!! engine tried an ILLEGAL move {best} in {board.fen()}")
                    return 1
                san = board.san(mv)
                board.push(mv)
                print(f"engine plays {san}   "
                      f"[{score_text(sc, board.turn != chess.WHITE)} depth {dep} "
                      f"{nodes or 0} nodes {dt:.0f}ms]")
                show(board, args.black)

        outcome = board.outcome(claim_draw=True)
        print(f"game over: {outcome.termination.name.lower()}", end="")
        if outcome.winner is None:
            print(" - draw")
        else:
            print(f" - {'White' if outcome.winner else 'Black'} wins")
    finally:
        eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
