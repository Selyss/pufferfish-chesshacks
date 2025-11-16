# features.py
"""
Feature extraction for chess moves.
Converts a move into a fixed-size feature vector for the neural network.
"""
import chess
import torch
from pst import PST_WHITE, PST_BLACK


# Material values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def square_centrality(sq: int) -> float:
    """
    Compute centrality score for a square (0-1 scale).
    Center squares have higher values.
    """
    file = sq % 8
    rank = sq // 8
    dist_file = min(file, 7 - file)  # 0-3
    dist_rank = min(rank, 7 - rank)  # 0-3
    centrality = (dist_file + dist_rank) / 6.0  # Normalize to 0-1
    return centrality


def extract_move_features(board: chess.Board, move: chess.Move) -> torch.Tensor:
    """
    Extract 12-dimensional feature vector for a move.
    
    Features:
      0-5: Attacker piece type (one-hot: pawn, knight, bishop, rook, queen, king)
      6: Material delta (in pawn units, normalized)
      7: PST delta (in centipawns, normalized)
      8: Is capture (0 or 1)
      9: Is check (0 or 1)
      10: From square centrality (0-1)
      11: To square centrality (0-1)
    
    Returns:
        Tensor of shape (12,)
    """
    features = torch.zeros(12)
    
    # Get piece info
    piece = board.piece_at(move.from_square)
    if piece is None:
        return features  # Should not happen for legal moves
    
    piece_type = piece.piece_type
    piece_color = piece.color
    
    # Feature 0-5: One-hot encoding of attacker piece type
    if 1 <= piece_type <= 6:
        features[piece_type - 1] = 1.0
    
    # Feature 8: Is capture
    is_capture = board.is_capture(move)
    features[8] = 1.0 if is_capture else 0.0
    
    # Feature 6: Material delta (if capture)
    material_delta = 0.0
    if is_capture:
        captured = board.piece_at(move.to_square)
        if captured:
            material_delta = PIECE_VALUES.get(captured.piece_type, 0) / 100.0  # Normalize to pawn units
        elif board.is_en_passant(move):
            material_delta = 1.0  # En passant captures a pawn
    features[6] = material_delta
    
    # Feature 7: PST delta
    pst_table = PST_WHITE if piece_color == chess.WHITE else PST_BLACK
    pst_from = pst_table[piece_type][move.from_square]
    pst_to = pst_table[piece_type][move.to_square]
    pst_delta = (pst_to - pst_from) / 100.0  # Normalize
    features[7] = pst_delta
    
    # Feature 9: Gives check
    board.push(move)
    is_check = board.is_check()
    board.pop()
    features[9] = 1.0 if is_check else 0.0
    
    # Feature 10-11: Square centrality
    features[10] = square_centrality(move.from_square)
    features[11] = square_centrality(move.to_square)
    
    return features

    material_balance = float(mat_stm - mat_opp)

    pst_stm = pst_score(board, stm)
    pst_opp = pst_score(board, opp)
    pst_balance = float(pst_stm - pst_opp)

    mob_stm = mobility_score(board, stm)
    mob_opp = mobility_score(board, opp)
    mobility_balance = float(mob_stm - mob_opp)

    in_check_flag = 1.0 if board.is_check() else 0.0

    features = torch.tensor(
        [[material_balance, pst_balance, mobility_balance, in_check_flag]],
        dtype=torch.float32,
    )
    return features
