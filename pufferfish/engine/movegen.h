// Pseudolegal move generation and basic legality filtering.

#pragma once

#include "types.h"
#include "position.h"

namespace pf
{

    // Generate all pseudolegal moves for the current side.
    void generate_moves(const Position &pos, MoveList &list);

    // Generate only capture and promotion moves (for quiescence).
    void generate_captures(const Position &pos, MoveList &list);

    // Filter a MoveList in-place to keep only legal moves (king not in check after move).
    void filter_legal_moves(Position &pos, MoveList &list);

} // namespace pf
