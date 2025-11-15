"""Feature extraction utilities for the NNUE trainer."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import chess
import numpy as np

from config import COLORS, FEATURE_DIM, PIECE_TYPES, SQUARES


_PIECE_VECTOR_SIZE = SQUARES * PIECE_TYPES * COLORS
_AUX_FEATURES = FEATURE_DIM - _PIECE_VECTOR_SIZE
if _AUX_FEATURES < 5:
    raise ValueError(
        "FEATURE_DIM must be at least piece planes (12*64) + 5 aux bits"
    )

_PIECE_INDEX = {
    (color, piece_type): color * PIECE_TYPES + (piece_type - 1)
    for color in (chess.WHITE, chess.BLACK)
    for piece_type in range(1, PIECE_TYPES + 1)
}


def normalize_fen(fen: str) -> str:
    """Pad half-move/full-move counters so python-chess can parse the FEN."""

    fen = fen.strip()
    parts = fen.split()
    if len(parts) == 6:
        return fen
    if len(parts) == 4:
        return f"{fen} 0 1"
    if len(parts) == 5:
        return f"{fen} 1"
    raise ValueError(f"Unexpected FEN format: {fen}")


@lru_cache(maxsize=10000)
def _board_from_fen(fen: str) -> chess.Board:
    """Cache boards since repeated positions are common in self-play data."""

    normalized = normalize_fen(fen)
    try:
        return chess.Board(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid FEN '{fen}': {exc}") from exc


def fen_to_feature_vector(fen: str) -> np.ndarray:
    """Convert a FEN string into a float32 feature vector of length FEATURE_DIM."""

    board = _board_from_fen(fen)
    vector = np.zeros(FEATURE_DIM, dtype=np.float32)

    for square, piece in board.piece_map().items():
        plane = _PIECE_INDEX[(piece.color, piece.piece_type)]
        idx = plane * SQUARES + square
        vector[idx] = 1.0

    offset = _PIECE_VECTOR_SIZE
    vector[offset] = 1.0 if board.turn == chess.WHITE else 0.0
    offset += 1

    castling_flags: Iterable[bool] = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    for flag in castling_flags:
        vector[offset] = 1.0 if flag else 0.0
        offset += 1

    return vector


def fen_to_numpy(fen: str) -> np.ndarray:
    """Alias for clarity when numpy is explicitly required."""

    return fen_to_feature_vector(fen)
