from __future__ import annotations

import math
import time
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import chess

from .nnue import NNUEEvaluator, NNUEState, PIECE_VALUES

# Set to True to enable detailed logging, False to disable
DEBUG_LOGGING = False

logger = logging.getLogger(__name__)

# NNUE evaluation costs ~150us and dominates search time, so the search is tuned to
# visit as few nodes as possible per ply rather than to make each node cheaper.


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
        max_depth: int = 6,
        quiescence_depth: int = 4,
        temperature: float = 0.6,
        tt_size: int = 200_000,
        check_extension: bool = True,
        quiescence_check_limit: int = 4,
        eval_cache_size: int = 300_000,
        null_move: bool = True,
        late_move_reductions: bool = True,
        check_ordering: str = "cheap",
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
        self.check_extension = check_extension
        self.quiescence_check_limit = quiescence_check_limit
        self.null_move = null_move
        self.late_move_reductions = late_move_reductions
        if check_ordering not in ("none", "cheap", "full"):
            raise ValueError(f"check_ordering must be none/cheap/full, got {check_ordering!r}")
        self.check_ordering = check_ordering
        self.eval_cache: Dict[tuple, float] = {}
        self.eval_cache_size = eval_cache_size

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, state: NNUEState) -> float:
        """Evaluate with memoization.

        A single NNUE forward pass costs ~150us, far more than a dict lookup, and
        quiescence revisits the same positions constantly through transpositions.
        """
        key = state.board._transposition_key()
        cached = self.eval_cache.get(key)
        if cached is not None:
            return cached
        value = self.evaluator.evaluate(state)
        if len(self.eval_cache) >= self.eval_cache_size:
            self.eval_cache.clear()
        self.eval_cache[key] = value
        return value

    # ------------------------------------------------------------------
    # driver
    # ------------------------------------------------------------------

    def search(self, state: NNUEState, time_limit_ms: int) -> SearchResult:
        if DEBUG_LOGGING:
            logger.debug("search %s limit=%sms", state.board.fen(), time_limit_ms)

        self.nodes = 0
        self.killer_moves = [[None, None] for _ in range(self.max_depth + 64)]
        self.first_move_completed = False

        search_start = time.perf_counter()
        if time_limit_ms and time_limit_ms > 0:
            # Reserve headroom for NNUE inference overhead and network latency.
            effective_time = time_limit_ms * 0.85
            deadline = search_start + effective_time / 1000
            hard_deadline = search_start + 50.0
        else:
            # No time limit: search runs to max_depth and is fully deterministic.
            deadline = None
            hard_deadline = None

        best_move: Optional[chess.Move] = None
        best_score = -self.mate_score
        best_probabilities: Dict[chess.Move, float] = {}
        last_completed_depth = 0

        # Mate in one is worth finding without paying for a full search.
        legal_moves = list(state.board.legal_moves)
        for move in legal_moves:
            state.board.push(move)
            is_mate = state.board.is_checkmate()
            state.board.pop()
            if is_mate:
                return SearchResult(
                    move=move,
                    score=self.mate_score,
                    depth=1,
                    nodes=len(legal_moves),
                    probabilities={move: 1.0},
                )

        for depth in range(1, self.max_depth + 1):
            try:
                score, move, root_scores = self._search_root(
                    state, depth, deadline, hard_deadline
                )
            except SearchTimeout:
                break
            if move is not None:
                best_move = move
                best_score = score
                best_probabilities = self._scores_to_probabilities(root_scores)
                last_completed_depth = depth
                if depth == 1:
                    self.first_move_completed = True
                # A forced mate has been proved; deeper search cannot improve on it.
                if abs(score) > self.mate_score - 500:
                    break

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

    # ------------------------------------------------------------------
    # root
    # ------------------------------------------------------------------

    def _search_root(
        self,
        state: NNUEState,
        depth: int,
        deadline: Optional[float],
        hard_deadline: Optional[float] = None,
    ) -> Tuple[float, Optional[chess.Move], List[Tuple[chess.Move, float]]]:
        alpha = -self.mate_score
        beta = self.mate_score
        best_move: Optional[chess.Move] = None
        best_score = -self.mate_score
        root_scores: List[Tuple[chess.Move, float]] = []
        tt_entry = self.tt.get(self._tt_key(state.board))
        tt_move = tt_entry.move if tt_entry else None
        # _order_moves generates from generate_legal_moves, so every move here is
        # already legal; re-testing membership regenerated the whole move list per move.
        moves = self._order_moves(state, ply=0, tt_move=tt_move)
        if not moves:
            return self._terminal_score(state.board, 0), None, []

        for index, move in enumerate(moves):
            if hard_deadline and time.perf_counter() > hard_deadline:
                raise SearchTimeout()
            if self.first_move_completed and best_move is not None:
                self._check_time(deadline)
            state.push(move)
            if index == 0:
                score = -self._negamax(
                    state, depth - 1, -beta, -alpha, 1, deadline, hard_deadline
                )
            else:
                # Principal variation search: assume the first move is best and
                # verify the rest with a null window, which fails fast.
                score = -self._negamax(
                    state, depth - 1, -alpha - 1e-6, -alpha, 1, deadline, hard_deadline
                )
                if score > alpha:
                    score = -self._negamax(
                        state, depth - 1, -beta, -alpha, 1, deadline, hard_deadline
                    )
            state.pop()
            root_scores.append((move, score))
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        return best_score, best_move, root_scores

    # ------------------------------------------------------------------
    # main search
    # ------------------------------------------------------------------

    def _negamax(
        self,
        state: NNUEState,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        deadline: Optional[float],
        hard_deadline: Optional[float] = None,
    ) -> float:
        if hard_deadline and time.perf_counter() > hard_deadline:
            raise SearchTimeout()
        if self.first_move_completed:
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
        in_check = board.is_check()
        # Terminal positions have exact scores and must never reach the evaluator.
        # Handing a stalemate to quiescence returned the raw NNUE value, which the
        # parent node reads with the opposite sign -- so stalemating the opponent
        # looked better than winning, and the engine would steer into it.
        if board.is_game_over():
            return self._terminal_score(board, ply)
        if self.check_extension and depth > 0 and in_check:
            depth += 1
        if depth <= 0:
            return self._quiescence(
                state, alpha, beta, ply, deadline, hard_deadline, self.quiescence_depth
            )

        is_pv = (beta - alpha) > 1e-5

        # Null-move pruning: give the opponent a free move. If the position is still
        # good enough to fail high, it is too good to be worth searching in full.
        # Disabled in check, in likely-zugzwang endings, and on PV nodes.
        if (
            self.null_move
            and not in_check
            and not is_pv
            and depth >= 3
            and self._has_non_pawn_material(board)
        ):
            reduction = 2 + (depth >= 6)
            state.push_null()
            null_score = -self._negamax(
                state, depth - 1 - reduction, -beta, -beta + 1e-6,
                ply + 1, deadline, hard_deadline,
            )
            state.pop()
            if null_score >= beta:
                return beta

        best_value = -self.mate_score
        moves = self._order_moves(state, ply, tt_move)
        if not moves:
            return self._terminal_score(board, ply)

        best_move_local: Optional[chess.Move] = None
        for index, move in enumerate(moves):
            is_capture = board.is_capture(move)
            is_quiet = not is_capture and move.promotion is None
            state.push(move)

            reduction = 0
            if (
                self.late_move_reductions
                and is_quiet
                and depth >= 3
                and index >= 3
                and not in_check
                and not state.board.is_check()  # the move does not give check
            ):
                reduction = 1 + (depth >= 6 and index >= 6)

            if index == 0:
                value = -self._negamax(
                    state, depth - 1, -beta, -alpha, ply + 1, deadline, hard_deadline
                )
            else:
                value = -self._negamax(
                    state, depth - 1 - reduction, -alpha - 1e-6, -alpha,
                    ply + 1, deadline, hard_deadline,
                )
                # A reduced or null-window search that beats alpha may have been
                # wrong to dismiss the move; re-search it properly.
                if value > alpha and (reduction or value < beta):
                    value = -self._negamax(
                        state, depth - 1, -beta, -alpha, ply + 1, deadline, hard_deadline
                    )
            state.pop()

            if value > best_value:
                best_value = value
                best_move_local = move
            alpha = max(alpha, value)
            if alpha >= beta:
                if not is_capture:
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

    # ------------------------------------------------------------------
    # quiescence
    # ------------------------------------------------------------------

    def _quiescence(
        self,
        state: NNUEState,
        alpha: float,
        beta: float,
        ply: int,
        deadline: Optional[float],
        hard_deadline: Optional[float],
        depth_left: int,
    ) -> float:
        if hard_deadline and time.perf_counter() > hard_deadline:
            raise SearchTimeout()
        if self.first_move_completed:
            self._check_time(deadline)
        board = state.board
        # Being mated is not a position you can stand pat on. is_check() is cheap and
        # gates the expensive mate test, so this costs almost nothing in quiet nodes.
        if board.is_check() and board.is_checkmate():
            self.nodes += 1
            return -self.mate_score + ply
        stand_pat = self._evaluate(state)
        self.nodes += 1
        if depth_left <= 0:
            return stand_pat
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # One move generation feeds both the captures and the checks below; the
        # previous version generated the full legal move list twice per node.
        captures: List[chess.Move] = []
        quiet_checks: List[chess.Move] = []
        want_checks = self.quiescence_check_limit > 0
        for move in board.generate_legal_moves():
            if board.is_capture(move):
                captures.append(move)
            elif want_checks and len(quiet_checks) < self.quiescence_check_limit:
                if board.gives_check(move):
                    quiet_checks.append(move)

        captures.sort(key=lambda m: self._move_order_score(board, m), reverse=True)

        for move in captures:
            state.push(move)
            score = -self._quiescence(
                state, -beta, -alpha, ply + 1, deadline, hard_deadline, depth_left - 1
            )
            state.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        for move in quiet_checks:
            state.push(move)
            score = -self._quiescence(
                state, -beta, -alpha, ply + 1, deadline, hard_deadline, depth_left - 1
            )
            state.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # ------------------------------------------------------------------
    # move ordering
    # ------------------------------------------------------------------

    def _order_moves(
        self, state: NNUEState, ply: int, tt_move: Optional[chess.Move]
    ) -> List[chess.Move]:
        board = state.board
        moves = list(board.generate_legal_moves())
        killers = (
            self.killer_moves[ply]
            if ply < len(self.killer_moves)
            else (None, None)
        )
        killer0, killer1 = killers[0], killers[1]
        turn = board.turn
        history = self.history_table
        piece_at = board.piece_at
        mode = self.check_ordering
        if mode == "cheap":
            check_test = self._gives_direct_check
        elif mode == "full":
            check_test = lambda b, m: b.gives_check(m)  # noqa: E731
        else:
            check_test = None

        def score(move: chess.Move) -> float:
            if tt_move is not None and move == tt_move:
                return 1e9
            priority = 0.0
            if check_test is not None and check_test(board, move):
                # Checks are forcing, so trying them early produces cutoffs sooner.
                priority += 30_000.0
            captured = piece_at(move.to_square)
            if captured is not None:
                # MVV-LVA
                mover = piece_at(move.from_square)
                priority += 100_000.0 + PIECE_VALUES[captured.piece_type] * 10
                if mover is not None:
                    priority -= PIECE_VALUES[mover.piece_type]
            elif board.is_en_passant(move):
                priority += 100_000.0 + PIECE_VALUES[chess.PAWN] * 10
            else:
                if move == killer0:
                    priority += 4000.0
                elif move == killer1:
                    priority += 2000.0
                priority += history.get(
                    (turn, move.from_square, move.to_square, move.promotion), 0.0
                )
            if move.promotion:
                priority += 90_000.0 + PIECE_VALUES.get(move.promotion, 900)
            return priority

        moves.sort(key=score, reverse=True)
        return moves

    def _move_order_score(self, board: chess.Board, move: chess.Move) -> float:
        # Used for captures in quiescence, where gives_check is too expensive to pay for.
        score = 0.0
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
        return score

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gives_direct_check(board: chess.Board, move: chess.Move) -> bool:
        """Would this move attack the enemy king from its destination square?

        board.gives_check() is exact but pushes and pops the move, which is far too
        expensive to pay for every move at every node just to order them. This reads
        the attack tables instead. It misses discovered checks, which is acceptable:
        this only decides move ordering, never correctness.
        """
        king_sq = board.king(not board.turn)
        if king_sq is None:
            return False
        piece_type = move.promotion or board.piece_type_at(move.from_square)
        if piece_type is None or piece_type == chess.KING:
            return False
        to = move.to_square
        bb_king = chess.BB_SQUARES[king_sq]

        if piece_type == chess.PAWN:
            return bool(chess.BB_PAWN_ATTACKS[board.turn][to] & bb_king)
        if piece_type == chess.KNIGHT:
            return bool(chess.BB_KNIGHT_ATTACKS[to] & bb_king)

        # Slide along the occupancy the move would produce, so the piece's own
        # departure square no longer blocks the ray.
        occupied = (board.occupied & ~chess.BB_SQUARES[move.from_square]) | chess.BB_SQUARES[to]
        attacks = 0
        if piece_type in (chess.BISHOP, chess.QUEEN):
            attacks |= chess.BB_DIAG_ATTACKS[to][chess.BB_DIAG_MASKS[to] & occupied]
        if piece_type in (chess.ROOK, chess.QUEEN):
            attacks |= (
                chess.BB_RANK_ATTACKS[to][chess.BB_RANK_MASKS[to] & occupied]
                | chess.BB_FILE_ATTACKS[to][chess.BB_FILE_MASKS[to] & occupied]
            )
        return bool(attacks & bb_king)

    @staticmethod
    def _has_non_pawn_material(board: chess.Board) -> bool:
        """Zugzwang guard: null move is unsound in king-and-pawn positions."""
        side = board.turn
        return bool(
            board.knights & board.occupied_co[side]
            or board.bishops & board.occupied_co[side]
            or board.rooks & board.occupied_co[side]
            or board.queens & board.occupied_co[side]
        )

    def _terminal_score(self, board: chess.Board, ply: int) -> float:
        """Exact score for a finished game, from the side to move's perspective.

        The +ply on mate makes a mate found sooner score higher than the same mate
        found later, so the engine actually finishes games instead of shuffling.
        Everything else that ends a game -- stalemate, insufficient material, the
        move/repetition limits -- is a draw and must be exactly 0.0.
        """
        if board.is_checkmate():
            return -self.mate_score + ply
        return 0.0

    def _scores_to_probabilities(
        self, move_scores: Sequence[tuple[chess.Move, float]]
    ) -> Dict[chess.Move, float]:
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
