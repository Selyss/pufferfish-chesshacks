// Alpha-beta search with NN evaluation and typical pruning.

#pragma once

#include <cstdint>

#include "types.h"
#include "position.h"
#include "movegen.h"
#include "tt.h"
#include "nn_interface.h"

namespace pf
{

    struct SearchStats
    {
        std::uint64_t nodes = 0;
        std::uint64_t qnodes = 0;
    };

    struct SearchContext
    {
        TranspositionTable *tt = nullptr;
        NNEvaluator *nn = nullptr;
        TimeManager tm;
        SearchLimits limits;

        Move killers[2][MAX_PLY]{};
        int history[PIECE_NB][64]{};

        SearchStats stats;
    };

    struct SearchResult
    {
        Move bestMove = MOVE_NONE;
        int score = 0;
        int depth = 0;
    };

    SearchResult search(Position &pos, SearchContext &ctx);

} // namespace pf
