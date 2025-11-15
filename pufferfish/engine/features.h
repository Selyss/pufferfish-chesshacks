#pragma once

#include <vector>

#include "position.h"
#include "nnue_constants.h"

namespace pf
{

    // Extract NNUE feature indices for a position from the side-to-move perspective.
    void extract_features(const Position &pos, std::vector<int> &features);

    // Compute added/removed feature indices between two positions (same side-to-move).
    void diff_features(const Position &before, const Position &after,
                       std::vector<int> &added, std::vector<int> &removed);

} // namespace pf
