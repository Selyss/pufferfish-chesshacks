from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import chess
import numpy as np
import torch

from .dataset import FenFeatureEncoder, _PIECE_TYPES
from .model import SimpleNNUE

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


@dataclass
class StateDelta:
    material_before: float
    ep_file_before: Optional[int]
    feature_changes: List[tuple[int, float]] = field(default_factory=list)
    touched: set[int] = field(default_factory=set)
    piece_counts_before: Dict[tuple[chess.Color, chess.PieceType], int] | None = None
    total_pieces_before: int | None = None


class NNUEState:
    """Tracks a board and feature vector that can be updated incrementally."""

    def __init__(self, board: chess.Board, encoder: FenFeatureEncoder) -> None:
        self.encoder = encoder
        self.board = board.copy(stack=False)
        self.features = encoder.encode_board(self.board)
        self._material_balance = self._compute_material(self.board)
        self._current_ep_file = self._ep_file_from_board(self.board)
        self._piece_counts = {
            (color, piece_type): len(self.board.pieces(piece_type, color))
            for color in (chess.WHITE, chess.BLACK)
            for piece_type in _PIECE_TYPES
        }
        self._total_pieces = sum(self._piece_counts.values())
        self._stack: List[StateDelta] = []

    @classmethod
    def from_board(cls, board: chess.Board, encoder: FenFeatureEncoder) -> "NNUEState":
        return cls(board, encoder)

    def push(self, move: chess.Move) -> None:
        delta = StateDelta(
            material_before=self._material_balance,
            ep_file_before=self._current_ep_file,
            piece_counts_before=self._piece_counts.copy(),
            total_pieces_before=self._total_pieces,
        )
        board = self.board
        moving_piece = board.piece_at(move.from_square)
        if moving_piece is None:
            raise ValueError(f"No piece on {move.from_square} to move.")

        capture_square = move.to_square
        captured_piece: Optional[chess.Piece] = None
        if board.is_en_passant(move):
            # Captured pawn sits behind the destination square.
            shift = -8 if board.turn == chess.WHITE else 8
            capture_square = move.to_square + shift
            captured_piece = chess.Piece(chess.PAWN, not board.turn)
        else:
            captured_piece = board.piece_at(move.to_square)

        self._remove_piece(moving_piece.color, moving_piece.piece_type, move.from_square, delta)
        if captured_piece is not None:
            self._remove_piece(captured_piece.color, captured_piece.piece_type, capture_square, delta)
            self._adjust_material(-self._signed_value(captured_piece), delta)
            self._adjust_piece_count(captured_piece.color, captured_piece.piece_type, -1, delta)

        castling_rook: Optional[tuple[int, int]] = None
        if board.is_castling(move):
            castling_rook = self._rook_squares_for_castle(move, board.turn)
            if castling_rook:
                rook_from, _ = castling_rook
                rook_piece = board.piece_at(rook_from)
                if rook_piece is not None:
                    self._remove_piece(rook_piece.color, rook_piece.piece_type, rook_from, delta)

        board.push(move)

        result_piece = board.piece_at(move.to_square)
        if result_piece is None:
            raise ValueError("Expected a piece after pushing move.")
        self._add_piece(result_piece.color, result_piece.piece_type, move.to_square, delta)

        if castling_rook:
            _, rook_to = castling_rook
            rook_piece = board.piece_at(rook_to)
            if rook_piece is not None:
                self._add_piece(rook_piece.color, rook_piece.piece_type, rook_to, delta)

        if result_piece.piece_type != moving_piece.piece_type:
            self._adjust_piece_count(moving_piece.color, moving_piece.piece_type, -1, delta)
            self._adjust_piece_count(result_piece.color, result_piece.piece_type, +1, delta)
            promotion_delta = self._signed_value(result_piece) - self._signed_value(moving_piece)
            self._adjust_material(promotion_delta, delta)

        self._update_side_to_move(delta)
        self._update_castling_rights(delta)
        self._update_en_passant(delta)
        self._set_material_feature(delta)

        self._stack.append(delta)

    def pop(self) -> None:
        if not self._stack:
            raise IndexError("Cannot pop from an empty NNUE state stack.")
        delta = self._stack.pop()
        self.board.pop()
        for index, prev_value in reversed(delta.feature_changes):
            self.features[index] = prev_value
        self._material_balance = delta.material_before
        self._current_ep_file = delta.ep_file_before
        if delta.piece_counts_before is not None:
            self._piece_counts = delta.piece_counts_before
        if delta.total_pieces_before is not None:
            self._total_pieces = delta.total_pieces_before

    def _remove_piece(self, color: chess.Color, piece_type: chess.PieceType, square: chess.Square, delta: StateDelta) -> None:
        index = self.encoder.piece_index(color, piece_type, square)
        self._set_feature(index, 0.0, delta)

    def _add_piece(self, color: chess.Color, piece_type: chess.PieceType, square: chess.Square, delta: StateDelta) -> None:
        index = self.encoder.piece_index(color, piece_type, square)
        self._set_feature(index, 1.0, delta)

    def _adjust_piece_count(
        self,
        color: chess.Color,
        piece_type: chess.PieceType,
        delta_count: int,
        delta: StateDelta,
    ) -> None:
        if delta_count == 0:
            return
        key = (color, piece_type)
        self._piece_counts[key] = self._piece_counts.get(key, 0) + delta_count
        idx = self.encoder.piece_count_indices[key]
        value = self._piece_counts[key] / self.encoder.piece_count_scale
        self._set_feature(idx, value, delta)
        self._total_pieces += delta_count
        phase_value = self._total_pieces / self.encoder.phase_scale
        self._set_feature(self.encoder.phase_index, phase_value, delta)

    def _update_side_to_move(self, delta: StateDelta) -> None:
        value = 1.0 if self.board.turn == chess.WHITE else 0.0
        self._set_feature(self.encoder.side_to_move_index, value, delta)

    def _update_castling_rights(self, delta: StateDelta) -> None:
        board = self.board
        self._set_feature(
            self.encoder.castling_indices["white_kingside"],
            1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
            delta,
        )
        self._set_feature(
            self.encoder.castling_indices["white_queenside"],
            1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
            delta,
        )
        self._set_feature(
            self.encoder.castling_indices["black_kingside"],
            1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
            delta,
        )
        self._set_feature(
            self.encoder.castling_indices["black_queenside"],
            1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
            delta,
        )

    def _update_en_passant(self, delta: StateDelta) -> None:
        new_file = self._ep_file_from_board(self.board)
        if self._current_ep_file is not None:
            idx = self.encoder.ep_index_start + self._current_ep_file
            self._set_feature(idx, 0.0, delta)
        if new_file is not None:
            idx = self.encoder.ep_index_start + new_file
            self._set_feature(idx, 1.0, delta)
        self._current_ep_file = new_file

    def _set_material_feature(self, delta: StateDelta) -> None:
        scaled = self._material_balance / self.encoder.material_scale
        self._set_feature(self.encoder.material_index, scaled, delta)

    def _set_feature(self, index: int, value: float, delta: StateDelta) -> None:
        current = float(self.features[index])
        if math.isclose(current, value, abs_tol=1e-6):
            return
        if index not in delta.touched:
            delta.feature_changes.append((index, current))
            delta.touched.add(index)
        self.features[index] = value

    def _adjust_material(self, delta_cp: float, delta: StateDelta) -> None:
        if delta_cp == 0:
            return
        self._material_balance += delta_cp
        self._set_material_feature(delta)

    @staticmethod
    def _ep_file_from_board(board: chess.Board) -> Optional[int]:
        if board.ep_square is None:
            return None
        return chess.square_file(board.ep_square)

    @staticmethod
    def _rook_squares_for_castle(move: chess.Move, color: chess.Color) -> Optional[tuple[int, int]]:
        file_to = chess.square_file(move.to_square)
        rank = chess.square_rank(move.from_square)
        if file_to == 6:  # g-file
            rook_from = chess.square(7, rank)  # h-file
            rook_to = chess.square(5, rank)  # f-file
            return rook_from, rook_to
        if file_to == 2:  # c-file
            rook_from = chess.square(0, rank)  # a-file
            rook_to = chess.square(3, rank)  # d-file
            return rook_from, rook_to
        return None

    @staticmethod
    def _signed_value(piece: chess.Piece) -> float:
        base = PIECE_VALUES[piece.piece_type]
        return float(base if piece.color == chess.WHITE else -base)

    @staticmethod
    def _compute_material(board: chess.Board) -> float:
        total = 0.0
        for square, piece in board.piece_map().items():
            total += NNUEState._signed_value(piece)
        return total


class NNUEEvaluator:
    """Wraps the trained NNUE for fast inference."""

    def __init__(
        self,
        model: SimpleNNUE,
        input_dim: int,
        encoder: FenFeatureEncoder | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.encoder = encoder or FenFeatureEncoder()
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        base_dim = self.encoder.feature_dim
        if input_dim < base_dim:
            raise ValueError(
                f"Model expects {input_dim} features but encoder provides {base_dim}. "
                "Retrain with matching encoder configuration."
            )
        self.extra_feature_dim = input_dim - base_dim
        self._warned_extra = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "NNUEEvaluator":
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = checkpoint.get("model_config")
        if not model_config:
            raise ValueError(
                "Checkpoint missing model_config metadata. "
                "Please retrain using the updated training script."
            )
        encoder = FenFeatureEncoder()
        input_dim = model_config.get("input_dim", encoder.feature_dim)
        hidden_dims = model_config.get("hidden_dims")
        dropout = model_config.get("dropout", 0.0)
        residual_repeats = model_config.get("residual_repeats")
        model = SimpleNNUE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            residual_repeats=residual_repeats,
        )
        model.load_state_dict(checkpoint["model_state"])
        evaluator = cls(model=model, input_dim=input_dim, encoder=encoder, device=device)
        return evaluator

    def initial_state(self, board: chess.Board) -> NNUEState:
        return NNUEState.from_board(board, self.encoder)

    def evaluate(self, state: NNUEState) -> float:
        features = torch.as_tensor(state.features, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.extra_feature_dim > 0:
            if not self._warned_extra:
                print(
                    f"[NNUE] Model expects {self.extra_feature_dim} additional feature(s); "
                    "padding zeros at inference. Consider retraining without depth/knodes metadata."
                )
                self._warned_extra = True
            padding = torch.zeros((1, self.extra_feature_dim), dtype=features.dtype, device=self.device)
            features = torch.cat([features, padding], dim=1)
        with torch.no_grad():
            value = self.model(features).item()
        # Convert to the perspective of the player to move
        return value if state.board.turn == chess.WHITE else -value
