// Simple piece-square tables for move ordering heuristics.
// Values are modest and only affect ordering, not evaluation.

#pragma once

#include "types.h"

namespace pf
{

    // PST values for WHITE perspective (rank 1 at bottom). Black uses mirrored ranks.
    // clang-format off
    static constexpr int PST_PAWN[64] = {
        //  a1..h1
         0,  0,  0,  0,  0,  0,  0,  0,
        //  a2..h2
         5,  5,  5,  5,  5,  5,  5,  5,
        //  a3..h3
         2,  3,  4,  6,  6,  4,  3,  2,
        //  a4..h4
         2,  3,  5,  7,  7,  5,  3,  2,
        //  a5..h5
         1,  2,  3,  5,  5,  3,  2,  1,
        //  a6..h6
         0,  1,  2,  3,  3,  2,  1,  0,
        //  a7..h7
         0,  0,  0,  1,  1,  0,  0,  0,
        //  a8..h8
         0,  0,  0,  0,  0,  0,  0,  0,
    };

    static constexpr int PST_KNIGHT[64] = {
        -8, -4,  0,  2,  2,  0, -4, -8,
        -4,  0,  2,  3,  3,  2,  0, -4,
         0,  2,  4,  5,  5,  4,  2,  0,
         2,  3,  5,  6,  6,  5,  3,  2,
         2,  3,  5,  6,  6,  5,  3,  2,
         0,  2,  4,  5,  5,  4,  2,  0,
        -4,  0,  2,  3,  3,  2,  0, -4,
        -8, -4,  0,  2,  2,  0, -4, -8,
    };

    static constexpr int PST_BISHOP[64] = {
        -4, -2, -1,  0,  0, -1, -2, -4,
        -2,  0,  1,  2,  2,  1,  0, -2,
        -1,  1,  2,  3,  3,  2,  1, -1,
         0,  2,  3,  4,  4,  3,  2,  0,
         0,  2,  3,  4,  4,  3,  2,  0,
        -1,  1,  2,  3,  3,  2,  1, -1,
        -2,  0,  1,  2,  2,  1,  0, -2,
        -4, -2, -1,  0,  0, -1, -2, -4,
    };

    static constexpr int PST_ROOK[64] = {
         0,  0,  1,  2,  2,  1,  0,  0,
         1,  1,  2,  3,  3,  2,  1,  1,
         0,  1,  1,  2,  2,  1,  1,  0,
         0,  1,  1,  2,  2,  1,  1,  0,
         0,  1,  1,  2,  2,  1,  1,  0,
         0,  1,  1,  2,  2,  1,  1,  0,
         1,  2,  2,  3,  3,  2,  2,  1,
         1,  1,  1,  2,  2,  1,  1,  1,
    };

    static constexpr int PST_QUEEN[64] = {
        -2, -1, -1,  0,  0, -1, -1, -2,
        -1,  0,  1,  1,  1,  1,  0, -1,
        -1,  1,  1,  2,  2,  1,  1, -1,
         0,  1,  2,  3,  3,  2,  1,  0,
         0,  1,  2,  3,  3,  2,  1,  0,
        -1,  1,  1,  2,  2,  1,  1, -1,
        -1,  0,  1,  1,  1,  1,  0, -1,
        -2, -1, -1,  0,  0, -1, -1, -2,
    };

    static constexpr int PST_KING[64] = {
         2,  3, -2, -4, -4, -2,  3,  2,
         3,  4, -3, -5, -5, -3,  4,  3,
         0,  1, -4, -6, -6, -4,  1,  0,
        -2, -3, -6, -8, -8, -6, -3, -2,
        -2, -3, -6, -8, -8, -6, -3, -2,
         0,  1, -4, -6, -6, -4,  1,  0,
         3,  4, -3, -5, -5, -3,  4,  3,
         2,  3, -2, -4, -4, -2,  3,  2,
    };
    // clang-format on

    inline constexpr int mirror_sq(int sq) { return sq ^ 56; }

    inline int pst_value(Piece pc, int sq)
    {
        bool white = (pc >= W_PAWN && pc <= W_KING);
        int idx = white ? sq : mirror_sq(sq);
        switch (pc)
        {
        case W_PAWN:
        case B_PAWN:
            return PST_PAWN[idx];
        case W_KNIGHT:
        case B_KNIGHT:
            return PST_KNIGHT[idx];
        case W_BISHOP:
        case B_BISHOP:
            return PST_BISHOP[idx];
        case W_ROOK:
        case B_ROOK:
            return PST_ROOK[idx];
        case W_QUEEN:
        case B_QUEEN:
            return PST_QUEEN[idx];
        case W_KING:
        case B_KING:
            return PST_KING[idx];
        default:
            return 0;
        }
    }

} // namespace pf
