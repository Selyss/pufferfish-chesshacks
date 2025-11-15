#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "nnue_constants.h"

namespace pf
{

    struct alignas(64) Accumulator
    {
        std::array<Accum, ACC_UNITS> friendly;
        std::array<Accum, ACC_UNITS> enemy;
        bool valid;

        void clear(const std::array<Accum, ACC_UNITS> &bias_f,
                   const std::array<Accum, ACC_UNITS> &bias_e)
        {
            for (int i = 0; i < ACC_UNITS; ++i)
            {
                friendly[i] = bias_f[i];
                enemy[i] = bias_e[i];
            }
            valid = true;
        }
    };

    struct NNUE
    {
        std::array<Accum, ACC_UNITS> bias_friendly;
        std::array<Accum, ACC_UNITS> bias_enemy;

        std::vector<Weight> w_acc; // FEATURE_DIM * 2 * ACC_UNITS

        std::array<Accum, HIDDEN1> bias_fc1;
        std::array<Weight, HIDDEN1 * 2 * ACC_UNITS> w_fc1;

        std::array<Accum, HIDDEN2> bias_fc2;
        std::array<Weight, HIDDEN2 * HIDDEN1> w_fc2;

        Accum bias_out;
        std::array<Weight, HIDDEN2> w_out;

        Accumulator accumulator;

        bool load(const std::string &path);

        template <typename FeatureIter>
        void initialize_accumulator(FeatureIter begin, FeatureIter end)
        {
            accumulator.clear(bias_friendly, bias_enemy);
            for (auto it = begin; it != end; ++it)
            {
                int f = *it;
                if (f >= 0 && f < FEATURE_DIM)
                    add_feature(f, +1);
            }
        }

        template <typename FeatureIter>
        void apply_feature_diff(FeatureIter added_begin, FeatureIter added_end,
                                FeatureIter removed_begin, FeatureIter removed_end)
        {
            if (!accumulator.valid)
                return;
            for (auto it = removed_begin; it != removed_end; ++it)
            {
                int f = *it;
                if (f >= 0 && f < FEATURE_DIM)
                    add_feature(f, -1);
            }
            for (auto it = added_begin; it != added_end; ++it)
            {
                int f = *it;
                if (f >= 0 && f < FEATURE_DIM)
                    add_feature(f, +1);
            }
        }

        // Evaluate from side-to-move perspective (centipawns).
        int evaluate() const;

    private:
        inline void add_feature(int feature_index, int sign)
        {
            const int stride = 2 * ACC_UNITS;
            const Weight *wf = &w_acc[static_cast<std::size_t>(feature_index) * stride];
            const Weight *we = wf + ACC_UNITS;

            Accum *acc_f = accumulator.friendly.data();
            Accum *acc_e = accumulator.enemy.data();

            const Accum s = static_cast<Accum>(sign);

            for (int i = 0; i < ACC_UNITS; ++i)
            {
                acc_f[i] += s * static_cast<Accum>(wf[i]);
                acc_e[i] += s * static_cast<Accum>(we[i]);
            }
        }
    };

} // namespace pf
