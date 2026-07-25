// Float32 evaluator for the 256x2-32-32-1 network produced by bot/python/train.py.
//
// This exists because the int16 pipeline is broken: bot/python/export_int16.py
// quantizes by dividing by the scale instead of multiplying, so every one of the
// 393,216 accumulator weights rounds to zero and the network evaluates every
// position as 0. Repairing that needs matching per-layer scales and rescaling
// shifts inside the forward pass; this path sidesteps it by keeping float weights.
//
// It also uses the feature encoding the weights were actually trained with, which
// differs from features.cpp's extract_features() in three ways:
//
//   layout   index = square * 12 + piece  (square-major), not plane * 64 + square
//   colour   absolute -- White is always 0..5, Black always 6..11 -- not relative
//            to the side to move
//   turn     no side-to-move input exists at all
//
// Because the network has no notion of whose move it is, its output is from
// White's point of view, and evaluate() negates it for Black so the search always
// receives a score relative to the side to move.

#pragma once

#include <array>
#include <string>
#include <vector>

#include "nn_interface.h"

namespace pf
{

    class FloatNNUEEvaluator : public NNEvaluator
    {
    public:
        static constexpr int kFeatureDim = 768;
        static constexpr int kAccUnits = 256;
        static constexpr int kHidden1 = 32;
        static constexpr int kHidden2 = 32;

        bool load(const char *path);
        bool loaded() const { return loaded_; }

        // Centipawns, from the side to move's perspective.
        int evaluate(const Position &pos) override;

        // White's perspective, before the side-to-move negation. Exposed so the
        // port can be compared directly against the PyTorch reference.
        float evaluate_white(const Position &pos) const;

    private:
        bool loaded_ = false;

        // Feature-major and interleaved: acc_w_[f * 512 .. +255] is the friendly
        // projection for feature f, [+256 .. +511] the enemy one. One contiguous
        // run per active feature.
        std::vector<float> acc_w_;
        std::array<float, kAccUnits> acc_b_friendly_{};
        std::array<float, kAccUnits> acc_b_enemy_{};

        std::array<float, kHidden1> fc1_b_{};
        std::vector<float> fc1_w_; // [kHidden1][2*kAccUnits]
        std::array<float, kHidden2> fc2_b_{};
        std::vector<float> fc2_w_; // [kHidden2][kHidden1]
        float out_b_ = 0.0f;
        std::array<float, kHidden2> out_w_{};
    };

} // namespace pf
