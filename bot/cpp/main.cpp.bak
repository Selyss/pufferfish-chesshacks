#include <iostream>
#include <vector>

#include "nnue.h"
#include "position.h"
#include "features.h"

int main() {
    NNUE nnue;
    if (!nnue.load("nnue_weights.bin")) {
        std::cerr << "Failed to load nnue_weights.bin\n";
        return 1;
    }

    Position pos;  // start position placeholder
    std::vector<int> features;
    extract_features(pos, features);

    nnue.initialize_accumulator(features.begin(), features.end());

    int eval_cp = nnue.evaluate();
    std::cout << "Eval for start position: " << eval_cp << " cp\n";

    // Example incremental update with a dummy move.
    Position after = pos;
    std::vector<int> added, removed;
    diff_features(pos, after, added, removed);
    nnue.apply_feature_diff(added.begin(), added.end(),
                            removed.begin(), removed.end());

    eval_cp = nnue.evaluate();
    std::cout << "Eval after dummy move: " << eval_cp << " cp\n";

    return 0;
}
