#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "nnue_constants.h"

struct alignas(64) Accumulator {
    std::array<Accum, ACC_UNITS> friendly;
    std::array<Accum, ACC_UNITS> enemy;
    bool valid;

    void clear(const std::array<Accum, ACC_UNITS>& bias_f,
               const std::array<Accum, ACC_UNITS>& bias_e) {
        for (int i = 0; i < ACC_UNITS; ++i) {
            friendly[i] = bias_f[i];
            enemy[i] = bias_e[i];
        }
        valid = true;
    }
};

struct NNUE {
    // First stage biases
    std::array<Accum, ACC_UNITS> bias_friendly;
    std::array<Accum, ACC_UNITS> bias_enemy;

    // First stage weights:
    // For each feature f, there are ACC_UNITS friendly and ACC_UNITS enemy weights:
    // w_acc[f * (2*ACC_UNITS) + 0..ACC_UNITS-1]       -> friendly
    // w_acc[f * (2*ACC_UNITS) + ACC_UNITS..2*ACC_UNITS-1] -> enemy
    std::vector<Weight> w_acc;  // size = FEATURE_DIM * 2 * ACC_UNITS

    // FC1: 2*ACC_UNITS -> HIDDEN1
    std::array<Accum, HIDDEN1> bias_fc1;
    std::array<Weight, HIDDEN1 * 2 * ACC_UNITS> w_fc1;

    // FC2: HIDDEN1 -> HIDDEN2
    std::array<Accum, HIDDEN2> bias_fc2;
    std::array<Weight, HIDDEN2 * HIDDEN1> w_fc2;

    // Output: HIDDEN2 -> 1
    Accum bias_out;
    std::array<Weight, HIDDEN2> w_out;

    Accumulator accumulator;

    bool load(const std::string& path);

    template <typename FeatureIter>
    void initialize_accumulator(FeatureIter begin, FeatureIter end) {
        accumulator.clear(bias_friendly, bias_enemy);
        for (auto it = begin; it != end; ++it) {
            int f = *it;
            if (f >= 0 && f < FEATURE_DIM) {
                add_feature(f, +1);
            }
        }
    }

    template <typename FeatureIter>
    void apply_feature_diff(FeatureIter added_begin, FeatureIter added_end,
                            FeatureIter removed_begin, FeatureIter removed_end) {
        if (!accumulator.valid) return;

        for (auto it = removed_begin; it != removed_end; ++it) {
            int f = *it;
            if (f >= 0 && f < FEATURE_DIM) {
                add_feature(f, -1);
            }
        }
        for (auto it = added_begin; it != added_end; ++it) {
            int f = *it;
            if (f >= 0 && f < FEATURE_DIM) {
                add_feature(f, +1);
            }
        }
    }

    // Evaluate from side to move perspective in centipawns
    int evaluate() const;

private:
    // sign = +1 for activation, -1 for deactivation
    inline void add_feature(int feature_index, int sign);
};
