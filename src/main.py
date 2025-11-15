from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import chess
from .bot.nnue import NNUEEvaluator
from .bot.search import AlphaBetaSearch
from .utils import GameContext, chess_manager


@dataclass
class EngineConfig:
    checkpoint: str
    device: str = "cpu"
    max_depth: int = 8
    quiescence_depth: int = 4
    temperature: float = 0.6
    time_fraction: float = 0.02
    min_time_ms: int = 100
    max_time_ms: int = 4000
    default_time_ms: int = 1000


class SearchEngine:
    def __init__(self) -> None:
        self._config = self._load_config()
        self._evaluator: NNUEEvaluator | None = None
        self._search: AlphaBetaSearch | None = None

    def _load_config(self) -> EngineConfig:
        local_ckpt = Path(__file__).resolve().parent / "bot" / "checkpoints" / "nnue_epoch004.pt"
        hf_default = "hf://ebatu/ChessHacks-models/nnue_epoch004.pt"
        env_checkpoint = os.getenv("CHESSBOT_CHECKPOINT")
        if env_checkpoint:
            checkpoint = env_checkpoint
        elif local_ckpt.exists():
            checkpoint = str(local_ckpt)
        else:
            checkpoint = hf_default
        checkpoint = self._resolve_checkpoint_source(checkpoint)
        device = os.getenv("CHESSBOT_DEVICE", "cpu")
        max_depth = int(os.getenv("CHESSBOT_MAX_DEPTH", "8"))
        quiescence_depth = int(os.getenv("CHESSBOT_QUIESCENCE_DEPTH", "8"))
        temperature = float(os.getenv("CHESSBOT_TEMPERATURE", "0.6"))
        time_fraction = float(os.getenv("CHESSBOT_TIME_FRACTION", "0.02"))
        min_time_ms = int(os.getenv("CHESSBOT_MIN_TIME_MS", "100"))
        max_time_ms = int(os.getenv("CHESSBOT_MAX_TIME_MS", "4000"))
        default_time_ms = int(os.getenv("CHESSBOT_DEFAULT_TIME_MS", "1000"))
        if not checkpoint:
            raise ValueError("CHESSBOT_CHECKPOINT must point to a trained NNUE checkpoint file.")
        return EngineConfig(
            checkpoint=checkpoint,
            device=device,
            max_depth=max_depth,
            quiescence_depth=quiescence_depth,
            temperature=temperature,
            time_fraction=time_fraction,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            default_time_ms=default_time_ms,
        )

    def _resolve_checkpoint_source(self, checkpoint: str) -> str:
        checkpoint = checkpoint.strip()
        if checkpoint.startswith("hf://"):
            return self._download_checkpoint_from_hf(checkpoint)
        return checkpoint

    def _download_checkpoint_from_hf(self, uri: str) -> str:
        spec = uri.removeprefix("hf://").strip("/")
        parts = spec.split("/")
        if len(parts) < 3:
            raise ValueError(
                "hf:// URIs must follow hf://<user>/<repo>/<path/to/file>. "
                f"Received: {uri}"
            )
        owner = parts[0]
        repo_segment = parts[1]
        revision = None
        if "@" in repo_segment:
            repo_name, revision = repo_segment.split("@", 1)
        else:
            repo_name = repo_segment
        repo_id = f"{owner}/{repo_name}"
        filename = "/".join(parts[2:])
        if not filename:
            raise ValueError(f"No file specified in hf:// URI: {uri}")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface-hub is required for hf:// checkpoints. "
                "Install it with `pip install huggingface-hub`."
            ) from exc
        return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)

    def ensure_ready(self) -> None:
        if self._search is not None:
            return
        try:
            self._evaluator = NNUEEvaluator.from_checkpoint(
                self._config.checkpoint,
                device=self._config.device,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"NNUE checkpoint not found at {self._config.checkpoint}. "
                "Set CHESSBOT_CHECKPOINT to a valid path."
            ) from exc
        self._search = AlphaBetaSearch(
            evaluator=self._evaluator,
            max_depth=self._config.max_depth,
            quiescence_depth=self._config.quiescence_depth,
            temperature=self._config.temperature,
        )

    def select_move(self, ctx: GameContext) -> chess.Move:
        self.ensure_ready()
        assert self._evaluator is not None and self._search is not None
        state = self._evaluator.initial_state(ctx.board)
        budget = self._allocate_time(ctx.timeLeft)
        result = self._search.search(state, budget)
        ctx.logProbabilities(result.probabilities)
        return result.move

    def reset(self) -> None:
        if self._search is not None:
            self._search.nodes = 0

    def _allocate_time(self, remaining_ms: int) -> int:
        if remaining_ms is None or remaining_ms <= 0:
            return self._config.default_time_ms
        allocated = int(remaining_ms * self._config.time_fraction)
        return max(self._config.min_time_ms, min(self._config.max_time_ms, allocated))


engine = SearchEngine()


@chess_manager.entrypoint
def choose_move(ctx: GameContext) -> chess.Move:
    return engine.select_move(ctx)


@chess_manager.reset
def reset_engine(ctx: GameContext) -> None:
    engine.reset()
