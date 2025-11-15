from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import chess

from .nnue import NNUEEvaluator, NNUEState, PIECE_VALUES


class SearchTimeout(Exception):
    pass


@dataclass
class SearchResult:
    move: chess.Move
    score: float
    depth: int
    nodes: int
    probabilities: Dict[chess.Move, float]


@dataclass
class TTEntry:
    depth: int
    value: float
    flag: str  # EXACT, LOWER, UPPER
    move: Optional[chess.Move]


class AlphaBetaSearch:
    """Iterative deepening alpha-beta search using the NNUE evaluator."""

    def __init__(
        self,
        evaluator: NNUEEvaluator,
        max_depth: int = 5,
        quiescence_depth: int = 4,
        temperature: float = 0.6,
        tt_size: int = 200_000,
    ) -> None:
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.quiescence_depth = quiescence_depth
        self.temperature = temperature
        self.nodes = 0
        self.mate_score = 10000.0
        self.tt: Dict[tuple, TTEntry] = {}
        self.tt_size = tt_size
        self.history_table: Dict[tuple, float] = defaultdict(float)
        self.killer_moves: List[List[Optional[chess.Move]]] = []

    def search(self, state: NNUEState, time_limit_ms: int) -> SearchResult:
        self.nodes = 0
        self.killer_moves = [[None, None] for _ in range(self.max_depth + 64)]
        deadline = time.perf_counter() + max(time_limit_ms, 0) / 1000 if time_limit_ms else None
        best_move: Optional[chess.Move] = None
        best_score = -self.mate_score
        best_probabilities: Dict[chess.Move, float] = {}
        last_completed_depth = 0

        for depth in range(1, self.max_depth + 1):
            try:
                score, move, root_scores = self._search_root(state, depth, deadline)
            except SearchTimeout:
                break
            if move is not None:
                best_move = move
                best_score = score
                best_probabilities = self._scores_to_probabilities(root_scores)
                last_completed_depth = depth

        if best_move is None:
            legal_moves = list(state.board.generate_legal_moves())
            if not legal_moves:
                raise ValueError("No legal moves available.")
            best_move = legal_moves[0]
            best_score = 0.0
            best_probabilities = {best_move: 1.0}

        return SearchResult(
            move=best_move,
            score=best_score,
            depth=max(1, last_completed_depth),
            nodes=self.nodes,
            probabilities=best_probabilities,
        )

    def _search_root(
        self,
        state: NNUEState,
        depth: int,
        deadline: Optional[float],
    ) -> Tuple[float, Optional[chess.Move], List[Tuple[chess.Move, float]]]:
        alpha = -self.mate_score
        beta = self.mate_score
        best_move: Optional[chess.Move] = None
        best_score = -self.mate_score
        root_scores: List[Tuple[chess.Move, float]] = []
        tt_entry = self.tt.get(self._tt_key(state.board))
        tt_move = tt_entry.move if tt_entry else None
        moves = self._order_moves(state, ply=0, tt_move=tt_move)
        if not moves:
            return self._terminal_score(state.board, 0), None, []

        for move in moves:
            self._check_time(deadline)
            state.push(move)
            score = -self._negamax(state, depth - 1, -beta, -alpha, 1, deadline)
            state.pop()
            root_scores.append((move, score))
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        return best_score, best_move, root_scores

    def _negamax(
        self,
        state: NNUEState,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        deadline: Optional[float],
    ) -> float:
        self._check_time(deadline)
        self.nodes += 1
        board = state.board
        key = self._tt_key(board)
        entry = self.tt.get(key)
        tt_move: Optional[chess.Move] = entry.move if entry else None
        if entry and entry.depth >= depth:
            value = self._from_tt(entry.value, ply)
            if entry.flag == "EXACT":
                return value
            if entry.flag == "LOWER":
                alpha = max(alpha, value)
            elif entry.flag == "UPPER":
                beta = min(beta, value)
            if alpha >= beta:
                return value

        alpha_orig = alpha
        if depth == 0 or board.is_game_over():
            return self._quiescence(state, alpha, beta, ply, deadline, self.quiescence_depth)

        best_value = -self.mate_score
        moves = self._order_moves(state, ply, tt_move)
        if not moves:
            return self._terminal_score(board, ply)

        best_move_local: Optional[chess.Move] = None
        for move in moves:
            state.push(move)
            value = -self._negamax(state, depth - 1, -beta, -alpha, ply + 1, deadline)
            state.pop()
            if value > best_value:
                best_value = value
                best_move_local = move
            alpha = max(alpha, value)
            if alpha >= beta:
                if not board.is_capture(move):
                    self._store_killer(ply, move)
                    self._update_history(board, move, depth)
                break
        flag = "EXACT"
        if best_value <= alpha_orig:
            flag = "UPPER"
        elif best_value >= beta:
            flag = "LOWER"
        self._store_tt(key, depth, best_value, flag, best_move_local, ply)
        return best_value

    def _quiescence(
        self,
        state: NNUEState,
        alpha: float,
        beta: float,
        ply: int,
        deadline: Optional[float],
        depth_left: int,
    ) -> float:
        self._check_time(deadline)
        stand_pat = self.evaluator.evaluate(state)
        self.nodes += 1
        if depth_left <= 0:
            return stand_pat
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        board = state.board
        captures = self._ordered_captures(state)
        for move in captures:
            state.push(move)
            score = -self._quiescence(state, -beta, -alpha, ply + 1, deadline, depth_left - 1)
            state.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _ordered_captures(self, state: NNUEState) -> List[chess.Move]:
        board = state.board
        captures = [move for move in board.generate_legal_moves() if board.is_capture(move)]
        captures.sort(key=lambda move: self._move_order_score(board, move), reverse=True)
        return captures

    def _order_moves(self, state: NNUEState, ply: int, tt_move: Optional[chess.Move]) -> List[chess.Move]:
        board = state.board
        moves = list(board.generate_legal_moves())

        def score(move: chess.Move) -> float:
            priority = 0.0
            if tt_move and move == tt_move:
                priority += 1e6
            if board.is_capture(move):
                priority += self._move_order_score(board, move)
            else:
                killers = self.killer_moves[ply] if ply < len(self.killer_moves) else [None, None]
                if move == killers[0]:
                    priority += 4000
                elif move == killers[1]:
                    priority += 2000
                key = (board.turn, move.from_square, move.to_square, move.promotion)
                priority += self.history_table.get(key, 0.0)
            if move.promotion:
                priority += PIECE_VALUES.get(move.promotion, 900)
            if board.gives_check(move):
                priority += 50
            return priority

        moves.sort(key=score, reverse=True)
        return moves

    def _move_order_score(self, board: chess.Board, move: chess.Move) -> float:
        score = 0.0
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured is None and board.is_en_passant(move):
                captured = chess.Piece(chess.PAWN, not board.turn)
            if captured:
                score += PIECE_VALUES[captured.piece_type] * 10
            mover = board.piece_at(move.from_square)
            if mover:
                score -= PIECE_VALUES[mover.piece_type]
        if move.promotion:
            score += PIECE_VALUES.get(move.promotion, 900)
        if board.gives_check(move):
            score += 50
        return score

    def _terminal_score(self, board: chess.Board, ply: int) -> float:
        if board.is_checkmate():
            return -self.mate_score + ply
        return 0.0

    def _scores_to_probabilities(self, move_scores: Sequence[tuple[chess.Move, float]]) -> Dict[chess.Move, float]:
        if not move_scores:
            return {}
        best = max(score for _, score in move_scores)
        temperature = max(0.05, self.temperature)
        exp_values = []
        for move, score in move_scores:
            scaled = (score - best) / temperature
            scaled = max(-20.0, min(20.0, scaled))
            exp_values.append((move, math.exp(scaled)))
        total = sum(value for _, value in exp_values) or 1.0
        return {move: value / total for move, value in exp_values}

    def _check_time(self, deadline: Optional[float]) -> None:
        if deadline is None:
            return
        if time.perf_counter() > deadline:
            raise SearchTimeout()

    def _tt_key(self, board: chess.Board) -> tuple:
        return board._transposition_key()

    def _store_tt(
        self,
        key: tuple,
        depth: int,
        value: float,
        flag: str,
        move: Optional[chess.Move],
        ply: int,
    ) -> None:
        existing = self.tt.get(key)
        if existing and existing.depth >= depth:
            return
        encoded = self._to_tt_score(value, ply)
        self.tt[key] = TTEntry(depth=depth, value=encoded, flag=flag, move=move)
        if len(self.tt) > self.tt_size:
            oldest_key = next(iter(self.tt))
            self.tt.pop(oldest_key, None)

    def _to_tt_score(self, value: float, ply: int) -> float:
        if value > self.mate_score - 500:
            return value + ply
        if value < -self.mate_score + 500:
            return value - ply
        return value

    def _from_tt(self, value: float, ply: int) -> float:
        if value > self.mate_score - 500:
            return value - ply
        if value < -self.mate_score + 500:
            return value + ply
        return value

    def _store_killer(self, ply: int, move: chess.Move) -> None:
        if ply >= len(self.killer_moves):
            return
        killers = self.killer_moves[ply]
        if move in killers:
            return
        killers[1] = killers[0]
        killers[0] = move

    def _update_history(self, board: chess.Board, move: chess.Move, depth: int) -> None:
        if board.is_capture(move):
            return
        key = (board.turn, move.from_square, move.to_square, move.promotion)
        self.history_table[key] += depth * depth
