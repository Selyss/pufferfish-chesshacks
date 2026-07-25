"""Drive the C++ engine over UCI, so the devtools board can play against it.

The devtools UI talks to serve.py, which calls the entrypoint in src/main.py.
That entrypoint has always run the Python search. This adapter lets it hand the
position to the C++ engine instead, which is several plies stronger, without
changing anything on the web side.

Select it with CHESSBOT_ENGINE=cpp. The binary is found automatically; override
with CHESSBOT_CPP_PATH.

One process is kept alive across moves so the transposition table stays warm,
and it is restarted transparently if it ever dies.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import chess

REPO = Path(__file__).resolve().parents[2]

CANDIDATES = [
    REPO / "pufferfish" / "build" / "pufferfish",
    REPO / "build" / "pufferfish",
]


def find_binary() -> Path:
    override = os.getenv("CHESSBOT_CPP_PATH")
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"CHESSBOT_CPP_PATH points at {p}, which does not exist")
        return p
    for c in CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError(
        "C++ engine binary not found. Build it with:\n"
        "  cmake -S pufferfish -B pufferfish/build -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build pufferfish/build -j8\n"
        "or set CHESSBOT_CPP_PATH.")


class CppEngine:
    """Minimal UCI client for the C++ engine."""

    def __init__(self, movetime_ms: int | None = None, depth: int | None = None):
        self.path = find_binary()
        self.movetime_ms = movetime_ms
        self.depth = depth
        self.proc: subprocess.Popen | None = None
        self._start()

    # -- process handling -------------------------------------------------

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            [str(self.path), "--uci"], cwd=str(REPO), text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._send("uci")
        if not self._await("uciok", 10.0):
            raise RuntimeError("engine did not answer uci")
        self._send("ucinewgame")

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _ensure(self) -> None:
        if not self._alive():
            self._start()

    def _send(self, line: str) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _await(self, token: str, timeout: float) -> list[str]:
        assert self.proc and self.proc.stdout
        end = time.time() + timeout
        seen: list[str] = []
        while time.time() < end:
            line = self.proc.stdout.readline()
            if not line:
                return []
            seen.append(line.strip())
            if line.startswith(token):
                return seen
        return []

    # -- public ------------------------------------------------------------

    def reset(self) -> None:
        self._ensure()
        self._send("ucinewgame")

    def select_move(self, board: chess.Board, time_left_ms: int | None) -> chess.Move:
        self._ensure()

        if self.depth:
            go = f"go depth {self.depth}"
            budget = 120.0
        elif self.movetime_ms:
            go = f"go movetime {self.movetime_ms}"
            budget = self.movetime_ms / 1000 + 20
        else:
            # Hand the clock over and let the engine's own time manager decide.
            left = int(time_left_ms) if time_left_ms and time_left_ms > 0 else 60000
            side = "wtime" if board.turn == chess.WHITE else "btime"
            other = "btime" if board.turn == chess.WHITE else "wtime"
            go = f"go {side} {left} {other} {left}"
            budget = left / 1000 + 20

        self._send(f"position fen {board.fen()}")
        self._send(go)
        lines = self._await("bestmove", budget)
        if not lines:
            raise RuntimeError("engine produced no bestmove (it may have crashed)")

        uci = None
        for ln in lines:
            if ln.startswith("bestmove"):
                parts = ln.split()
                uci = parts[1] if len(parts) > 1 else None
        if not uci or uci == "0000":
            raise RuntimeError(f"engine returned no move ({uci!r}) for {board.fen()}")

        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"engine returned illegal move {uci} in {board.fen()}")
        return move

    def close(self) -> None:
        if self._alive():
            try:
                self._send("quit")
                assert self.proc
                self.proc.wait(timeout=3)
            except Exception:
                if self.proc:
                    self.proc.kill()
