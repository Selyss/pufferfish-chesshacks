# search.py
"""
Simple tactical search using the tiny neural network.
Prioritizes mates, checks, and captures.
"""
import chess
import torch
from typing import Optional
from inference_model import load_model, evaluate_move_features
from features import extract_move_features


def find_mate_in_one(board: chess.Board) -> Optional[chess.Move]:
    """Check if there's a mate in one move."""
    for move in board.legal_moves:
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return move
    return None


def find_best_move_simple(board: chess.Board, model) -> Optional[chess.Move]:
    """
    Simple move selection strategy:
    1. If mate in 1 exists, play it
    2. Otherwise, evaluate all captures and checks with the NN
    3. Return highest-scoring move
    4. If no captures/checks, return any legal move
    
    Args:
        board: Current position
        model: Loaded TinyModel instance
    
    Returns:
        Best move found, or None if no legal moves (game over)
    """
    # Step 1: Check for mate in one
    mate_move = find_mate_in_one(board)
    if mate_move:
        return mate_move
    
    # Step 2: Evaluate tactical moves (captures and checks)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None  # Game is over (checkmate or stalemate)
    
    # Score all moves
    move_scores = []
    for move in legal_moves:
        # Extract features and evaluate
        features = extract_move_features(board, move)
        score = evaluate_move_features(model, features)
        move_scores.append((move, score))
    
    # Sort by score (descending)
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return best move
    return move_scores[0][0]


def find_best_move_tactical(board: chess.Board, model) -> Optional[chess.Move]:
    """
    More focused tactical search:
    1. If mate in 1 exists, play it
    2. Score only checks and captures
    3. If no tactical moves, play a quiet move based on PST improvement
    
    Args:
        board: Current position
        model: Loaded TinyModel instance
    
    Returns:
        Best move found, or None if no legal moves (game over)
    """
    # Step 1: Check for mate in one
    mate_move = find_mate_in_one(board)
    if mate_move:
        return mate_move
    
    # Step 2: Collect tactical moves
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None  # Game is over (checkmate or stalemate)
    
    tactical_moves = []
    quiet_moves = []
    
    for move in legal_moves:
        board.push(move)
        is_check = board.is_check()
        board.pop()
        is_capture = board.is_capture(move)
        
        if is_capture or is_check:
            tactical_moves.append(move)
        else:
            quiet_moves.append(move)
    
    # Step 3: Evaluate tactical moves if any exist
    if tactical_moves:
        best_move = None
        best_score = float('-inf')
        
        for move in tactical_moves:
            features = extract_move_features(board, move)
            score = evaluate_move_features(model, features)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    # Step 4: No tactical moves - pick best quiet move
    if quiet_moves:
        best_move = None
        best_score = float('-inf')
        
        for move in quiet_moves:
            features = extract_move_features(board, move)
            score = evaluate_move_features(model, features)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    # Fallback (shouldn't reach here)
    return legal_moves[0]

