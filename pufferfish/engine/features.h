#pragma once

#include <vector>

#include "position.h"
#include "nnue_constants.h"

namespace pf
{

    // Extract NNUE feature indices for a position from the side-to-move perspective.
    void extract_features(const Position &pos, std::vector<int> &features);

    // Extract 795-dim float features for SimpleNNUE:
    // [768 PSQ one-hot (12x64), 1 side-to-move, 4 castling rights (K,Q,k,q),
    //  8 en-passant file one-hot (a..h or all zeros), 1 material balance (pawns=1.0),
    //  12 per-piece counts (white/black x 6 types), 1 phase indicator (0..1)].
    void extract_features_795(const Position &pos, std::vector<float> &out);

    // Compute added/removed feature indices between two positions (same side-to-move).
    void diff_features(const Position &before, const Position &after,
                       std::vector<int> &added, std::vector<int> &removed);

} // namespace pf