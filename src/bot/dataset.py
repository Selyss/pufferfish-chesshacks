from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import gzip

import chess
import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import IterableDataset, load_dataset
from torch.utils.data import Dataset

# Mapping based on classical NNUE piece-square encodings.
_PIECE_TYPES: Tuple[chess.PieceType, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
_PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def normalize_fen(fen: str) -> str:
    """Ensure the provided FEN string has all 6 fields."""
    if fen.count(" ") >= 5:
        return fen
    return f"{fen.strip()} 0 1"


class FenFeatureEncoder:
    """Encodes FEN strings into NNUE-style feature vectors."""

    def __init__(self) -> None:
        self._piece_offsets = {}
        offset = 0
        for color in (chess.WHITE, chess.BLACK):
            for piece_type in _PIECE_TYPES:
                self._piece_offsets[(color, piece_type)] = offset
                offset += 64
        self.piece_feature_dim = offset

        self.side_to_move_index = offset
        offset += 1

        self.castling_indices = {
            "white_kingside": offset,
            "white_queenside": offset + 1,
            "black_kingside": offset + 2,
            "black_queenside": offset + 3,
        }
        offset += 4

        self.ep_index_start = offset
        offset += 8

        self.material_scale = 2000.0
        self.material_index = offset
        offset += 1

        self.piece_count_indices = {}
        self.piece_count_scale = 8.0
        for color in (chess.WHITE, chess.BLACK):
            for piece_type in _PIECE_TYPES:
                self.piece_count_indices[(color, piece_type)] = offset
                offset += 1

        self.phase_index = offset
        self.phase_scale = 32.0
        offset += 1

        self.extra_feature_dim = offset - self.piece_feature_dim
        self.feature_dim = offset
        self._board = chess.Board()

    def piece_index(self, color: chess.Color, piece_type: chess.PieceType, square: chess.Square) -> int:
        return self._piece_offsets[(color, piece_type)] + square

    def encode(self, fen: str) -> np.ndarray:
        board = self._board
        board.set_fen(normalize_fen(fen))
        return self.encode_board(board)

    def encode_board(self, board: chess.Board) -> np.ndarray:
        features = np.zeros(self.feature_dim, dtype=np.float32)
        # Piece-square table
        for square, piece in board.piece_map().items():
            features[self.piece_index(piece.color, piece.piece_type, square)] = 1.0

        # Side to move
        features[self.side_to_move_index] = 1.0 if board.turn == chess.WHITE else 0.0

        # Castling rights
        features[self.castling_indices["white_kingside"]] = (
            1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        )
        features[self.castling_indices["white_queenside"]] = (
            1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        )
        features[self.castling_indices["black_kingside"]] = (
            1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        )
        features[self.castling_indices["black_queenside"]] = (
            1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        )

        # En passant file
        if board.ep_square is not None:
            file_idx = chess.square_file(board.ep_square)
            features[self.ep_index_start + file_idx] = 1.0

        # Material balance (white - black scaled)
        material = 0.0
        for piece_type, value in _PIECE_VALUES.items():
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            material += value * (white_count - black_count)
            features[self.piece_count_indices[(chess.WHITE, piece_type)]] = (
                white_count / self.piece_count_scale
            )
            features[self.piece_count_indices[(chess.BLACK, piece_type)]] = (
                black_count / self.piece_count_scale
            )
        total_pieces = sum(
            len(board.pieces(piece_type, color))
            for color in (chess.WHITE, chess.BLACK)
            for piece_type in _PIECE_TYPES
        )
        features[self.material_index] = material / self.material_scale  # keep the value within [-5, 5]
        features[self.phase_index] = total_pieces / self.phase_scale

        return features


def load_light_preprocessed_dataset(
    split: str = "train",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> IterableDataset | HFDataset:
    """Load the LegendaryAKx3/rebalanced-preprocessed dataset."""
    return load_dataset(
        path="LegendaryAKx3/heavy-preprocessed",
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )


@dataclass
class DatasetConfig:
    label_key: str = "cp_label"
    target_scale: float = 0.01
    include_depth_feature: bool = True
    include_knodes_feature: bool = True
    weight_by_depth: bool = False
    weight_by_knodes: bool = False
    depth_norm: float = 64.0
    knodes_log_norm: float = math.log(1e8)


class LightPreprocessedDataset(Dataset):
    """PyTorch dataset wrapper with NNUE feature encoding."""

    def __init__(
        self,
        dataset: HFDataset,
        encoder: Optional[FenFeatureEncoder] = None,
        config: Optional[DatasetConfig] = None,
        file_path: Optional[str] = None
    ) -> None:        
        self.dataset = dataset
        self.encoder = encoder or FenFeatureEncoder()
        self.config = config or DatasetConfig()

        assert self.config.label_key == "cp_label", "Only cp_label is supported in this version."

        self.file_path = file_path
        self.fen_to_mate = None

        meta_dim = 0
        if self.config.include_depth_feature:
            meta_dim += 1
        if self.config.include_knodes_feature:
            meta_dim += 1
        self.feature_dim = self.encoder.feature_dim + meta_dim

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if self.fen_to_mate is None and self.file_path is not None:
            self.fen_to_mate = {}
            with gzip.open(self.file_path, 'rt') as f:
                for line in f:
                    fen, mate = line.strip().rsplit(' ', 1)
                    self.fen_to_mate[fen] = mate

        row = self.dataset[idx]
        base_features = self.encoder.encode(row["fen"])  # :(
        features = base_features

        meta_features = []
        if self.config.include_depth_feature:
            meta_features.append(
                min(row["depth"], self.config.depth_norm) / self.config.depth_norm
            )
        if self.config.include_knodes_feature:
            value = math.log1p(max(row["knodes"], 0))
            meta_features.append(value / self.config.knodes_log_norm)
        if meta_features:
            features = np.concatenate(
                [base_features, np.asarray(meta_features, dtype=np.float32)], axis=0
            )

        adjusted_cp = self.adjust_cp(row[self.config.label_key], row["fen"])
        label = np.array(
            [adjusted_cp * self.config.target_scale], dtype=np.float32
        )
        weight = np.array([self._example_weight(row)], dtype=np.float32)

        return (
            torch.from_numpy(features),
            torch.from_numpy(label),
            torch.from_numpy(weight),
        )

    def adjust_cp(self, cp: int, fen: str) -> int:
        if cp == 0 and self.fen_to_mate is not None and fen in self.fen_to_mate:
            mate_in = int(float(self.fen_to_mate[fen]))

            absolute_mate_in = abs(mate_in)

            base_adjustment = 2000

            if absolute_mate_in == 1:
                return (base_adjustment + 2000) * (1 if mate_in > 0 else -1)
            elif absolute_mate_in == 2:
                return (base_adjustment + 1500) * (1 if mate_in > 0 else -1)
            elif absolute_mate_in <= 4:
                return (base_adjustment + 1000) * (1 if mate_in > 0 else -1)
            elif absolute_mate_in <= 20:
                return (base_adjustment + 500) * (1 if mate_in > 0 else -1)
            else:
                return base_adjustment * (1 if mate_in > 0 else -1)

        return cp

    def _example_weight(self, row: dict) -> float:
        weight = 1.0
        if self.config.weight_by_depth:
            weight *= max(1, row["depth"]) / self.config.depth_norm
        if self.config.weight_by_knodes:
            numerator = math.log1p(max(row["knodes"], 0))
            weight *= numerator / self.config.knodes_log_norm
        return max(weight, 1e-3)
