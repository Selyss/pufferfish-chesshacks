// Abstract neural network evaluation interface.

#pragma once

#include "types.h"
#include "position.h"
#include "nnue.h"
#include "features.h"

namespace pf
{

    // Implementations should provide a fast eval of the current position
    // from the side to move's perspective (centipawns).
    struct NNEvaluator
    {
        virtual ~NNEvaluator() = default;

        virtual int evaluate(const Position &pos) = 0;
    };

    // Concrete NNUE-based evaluator implementation that wraps the NNUE loader.
    class NNUEEvaluator : public NNEvaluator
    {
    public:
        bool load(const std::string &path)
        {
            return nn_.load(path);
        }

        int evaluate(const Position &pos) override
        {
            // Fresh extraction each time for now; can be optimized with diffs.
            features_.clear();
            extract_features(pos, features_);
            nn_.initialize_accumulator(features_.begin(), features_.end());
            return nn_.evaluate();
        }

    private:
        NNUE nn_;
        std::vector<int> features_;
    };

} // namespace pf
