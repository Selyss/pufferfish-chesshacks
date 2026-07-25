// Alpha-beta search with NN evaluation and typical pruning.

#pragma once

#include <cstdint>
#include <vector>

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

        // Zobrist keys of every position seen: the game so far, seeded by the
        // caller before search() (see the UCI loop), then extended by the search
        // as it makes and unmakes moves. Needed because a repetition is a property
        // of the path taken, not of the position, so it cannot be read off the
        // board or cached in the transposition table.
        std::vector<Key> repetitionKeys;

        // True if `key` already appears in the path. Only positions since the last
        // irreversible move can repeat, so halfmoveClock bounds how far to look.
        //
        // One occurrence is treated as a draw rather than waiting for a threefold.
        // That is the usual choice: if a position can be reached twice the side to
        // move can generally force the third, so scoring it as a draw immediately
        // gets the search to the right answer sooner.
        bool is_repetition(Key key, int halfmoveClock) const
        {
            const int n = static_cast<int>(repetitionKeys.size());
            const int limit = halfmoveClock < n ? halfmoveClock : n;
            // The earliest a position can recur is four plies back (both sides must
            // move and return), and only every other entry has the same side to
            // move, so step by two from four.
            for (int i = 4; i <= limit; i += 2)
            {
                if (repetitionKeys[n - i] == key)
                    return true;
            }
            return false;
        }

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