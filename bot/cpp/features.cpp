#include "features.h"

void extract_features(const Position& pos,
                      std::vector<int>& features) {
    (void)pos;
    features.clear();
    // TODO: implement feature extraction.
    // General idea:
    //   - Normalize so side to move is "white".
    //   - Determine friendly and enemy king squares.
    //   - For each piece-square, compute a feature index based on
    //     king square, piece type, color, and relative square.
    //   - Ensure 0 <= index < FEATURE_DIM for all indices.
}

void diff_features(const Position& before,
                   const Position& after,
                   std::vector<int>& added,
                   std::vector<int>& removed) {
    (void)before;
    (void)after;
    added.clear();
    removed.clear();
    // TODO: implement incremental feature diffs based on the move.
    // For now this is a stub.
}