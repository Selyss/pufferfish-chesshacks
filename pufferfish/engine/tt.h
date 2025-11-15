// Transposition table with bucketed entries.

#pragma once

#include <cstdint>
#include <vector>

#include "types.h"

namespace pf
{

    struct TTEntry
    {
        Key key16 = 0; // partial key (upper bits)
        int16_t score = 0;
        int8_t depth = 0;
        BoundType bound = BOUND_NONE;
        Move best = MOVE_NONE;
    };

    struct TTBucket
    {
        static constexpr int SIZE = 4;
        TTEntry e[SIZE];
    };

    class TranspositionTable
    {
    public:
        TranspositionTable() = default;

        void resize(std::size_t megabytes);

        void clear();

        bool probe(Key key, int depth, int alpha, int beta, int ply, TTEntry &out) const;

        void store(Key key, int depth, int score, BoundType bound, Move best, int ply);

        Move probe_move(Key key) const;

    private:
        std::vector<TTBucket> buckets_;

        static int pack_score(int score, int ply);
        static int unpack_score(int packed, int ply);
    };

} // namespace pf
