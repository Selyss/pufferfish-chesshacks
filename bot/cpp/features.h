#pragma once

#include <vector>
#include "position.h"
#include "nnue_constants.h"

// Feature extractor for side-to-move perspective.
// It should normalize the board so that side to move is treated as "white".

void extract_features(const Position& pos,
                      std::vector<int>& features);

// Feature diff for a move.
// Compute feature indices that are added and removed when going from before to after.
void diff_features(const Position& before,
                   const Position& after,
                   std::vector<int>& added,
                   std::vector<int>& removed);
