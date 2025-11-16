from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import chess

from ..main import SearchEngine


@dataclass(frozen=True)
class Scenario:
    name: str
    fen: str
    description: str
    reference_line: str | None = None
    time_ms: int = 1500


SCENARIOS: Sequence[Scenario] = (
    Scenario(
        name="Mate in two (sac-sac theme)",
        fen="6k1/p4p1R/5q2/7Q/8/8/7P/5B1K w - - 0 1",
        description=(
            "Classic motif where 1.Rh8+ forces ...Qxh8 and 2.Qxh8# closes the net. "
            "Used to sanity-check whether the search sees short forced mates."
        ),
        reference_line="1.Rh8+ Qxh8 2.Qxh8#",
    ),
    Scenario(
        name="Punish a hanging piece",
        fen="r3r1k1/p1pn1ppp/1p2pn2/3p4/1b1P4/2N1PN2/PP3PPP/R1BQR1K1 w - - 0 1",
        description=(
            "Black's bishop on b4 is loose. The engine should gravitate toward either "
            "capturing it immediately or exploiting the lack of defenders."
        ),
        reference_line="Ideas: Qa4, Bd2 or even a3 to win the bishop.",
    ),
    Scenario(
        name="Stabilize an attacked piece",
        fen="r1bq1rk1/pp1n1ppp/2pbpn2/8/2PNP3/2N1BP2/PP3P1P/R2Q1RK1 w - - 0 1",
        description=(
            "White's knight on c4 is under pressure from the d5 pawn break and from pieces on b4/d6. "
            "Good engines will either defend it (e.g. Qd2) or relocate it before proceeding."
        ),
        reference_line="Typical plans: Qd2, Nxd6, or Re1 followed by Bf4.",
    ),
)


def format_probabilities(board: chess.Board, probabilities: dict[chess.Move, float], top_k: int = 3) -> List[str]:
    entries: List[str] = []
    if not probabilities:
        return entries
    for move, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        entries.append(f"{board.san(move)} ({move.uci()}): {prob:.1%}")
    return entries


def run_scenarios(scenarios: Iterable[Scenario]) -> None:
    engine = SearchEngine()
    engine.ensure_ready()
    assert engine._evaluator is not None and engine._search is not None
    evaluator = engine._evaluator
    search = engine._search

    for idx, scenario in enumerate(scenarios, start=1):
        board = chess.Board(scenario.fen)
        state = evaluator.initial_state(board)
        result = search.search(state, scenario.time_ms)
        board_for_print = chess.Board(scenario.fen)
        move_san = board_for_print.san(result.move)
        print("=" * 72)
        print(f"[{idx}] {scenario.name}")
        print(board_for_print)
        print(f"FEN: {scenario.fen}")
        print(f"Description: {scenario.description}")
        if scenario.reference_line:
            print(f"Reference idea: {scenario.reference_line}")
        print(
            f"Engine move: {move_san} ({result.move.uci()}) "
            f"| score={result.score:.2f} | depth={result.depth} | nodes={result.nodes}"
        )
        probs = format_probabilities(board_for_print, result.probabilities)
        if probs:
            print("Top moves:")
            for line in probs:
                print(f"  - {line}")
        print()


if __name__ == "__main__":
    run_scenarios(SCENARIOS)
