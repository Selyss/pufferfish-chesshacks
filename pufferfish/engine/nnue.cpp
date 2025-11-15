#include "nnue.h"

#include <fstream>

namespace pf
{

    bool NNUE::load(const std::string &path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            return false;

        int32_t feature_dim = 0;
        int32_t acc_units = 0;
        int32_t hidden1 = 0;
        int32_t hidden2 = 0;

        in.read(reinterpret_cast<char *>(&feature_dim), sizeof(int32_t));
        in.read(reinterpret_cast<char *>(&acc_units), sizeof(int32_t));
        in.read(reinterpret_cast<char *>(&hidden1), sizeof(int32_t));
        in.read(reinterpret_cast<char *>(&hidden2), sizeof(int32_t));

        if (!in)
            return false;
        if (feature_dim != FEATURE_DIM || acc_units != ACC_UNITS || hidden1 != HIDDEN1 || hidden2 != HIDDEN2)
            return false;

        for (int i = 0; i < ACC_UNITS; ++i)
        {
            int32_t b;
            in.read(reinterpret_cast<char *>(&b), sizeof(int32_t));
            if (!in)
                return false;
            bias_friendly[i] = b;
        }
        for (int i = 0; i < ACC_UNITS; ++i)
        {
            int32_t b;
            in.read(reinterpret_cast<char *>(&b), sizeof(int32_t));
            if (!in)
                return false;
            bias_enemy[i] = b;
        }

        w_acc.resize(static_cast<std::size_t>(FEATURE_DIM) * 2 * ACC_UNITS);
        in.read(reinterpret_cast<char *>(w_acc.data()), w_acc.size() * sizeof(Weight));
        if (!in)
            return false;

        for (int i = 0; i < HIDDEN1; ++i)
        {
            int32_t b;
            in.read(reinterpret_cast<char *>(&b), sizeof(int32_t));
            if (!in)
                return false;
            bias_fc1[i] = b;
        }
        in.read(reinterpret_cast<char *>(w_fc1.data()), w_fc1.size() * sizeof(Weight));
        if (!in)
            return false;

        for (int i = 0; i < HIDDEN2; ++i)
        {
            int32_t b;
            in.read(reinterpret_cast<char *>(&b), sizeof(int32_t));
            if (!in)
                return false;
            bias_fc2[i] = b;
        }
        in.read(reinterpret_cast<char *>(w_fc2.data()), w_fc2.size() * sizeof(Weight));
        if (!in)
            return false;

        {
            int32_t b;
            in.read(reinterpret_cast<char *>(&b), sizeof(int32_t));
            if (!in)
                return false;
            bias_out = b;
        }
        in.read(reinterpret_cast<char *>(w_out.data()), w_out.size() * sizeof(Weight));
        if (!in)
            return false;

        accumulator.valid = false;
        return true;
    }

    inline void NNUE::add_feature(int feature_index, int sign)
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

    int NNUE::evaluate() const
    {
        Activation act[2 * ACC_UNITS];
        int idx = 0;

        for (int i = 0; i < ACC_UNITS; ++i)
        {
            Accum v = accumulator.friendly[i];
            if (v < 0)
                v = 0;
            if (v > RELU_CLIP)
                v = RELU_CLIP;
            act[idx++] = static_cast<Activation>(v);
        }
        for (int i = 0; i < ACC_UNITS; ++i)
        {
            Accum v = accumulator.enemy[i];
            if (v < 0)
                v = 0;
            if (v > RELU_CLIP)
                v = RELU_CLIP;
            act[idx++] = static_cast<Activation>(v);
        }

        Accum h1[HIDDEN1];
        for (int o = 0; o < HIDDEN1; ++o)
        {
            Accum sum = bias_fc1[o];
            const Weight *w_row = &w_fc1[o * 2 * ACC_UNITS];
            for (int i = 0; i < 2 * ACC_UNITS; ++i)
                sum += static_cast<Accum>(w_row[i]) * static_cast<Accum>(act[i]);
            if (sum < 0)
                sum = 0;
            if (sum > RELU_CLIP)
                sum = RELU_CLIP;
            h1[o] = sum;
        }

        Accum h2[HIDDEN2];
        for (int o = 0; o < HIDDEN2; ++o)
        {
            Accum sum = bias_fc2[o];
            const Weight *w_row = &w_fc2[o * HIDDEN1];
            for (int i = 0; i < HIDDEN1; ++i)
                sum += static_cast<Accum>(w_row[i]) * static_cast<Accum>(h1[i]);
            if (sum < 0)
                sum = 0;
            if (sum > RELU_CLIP)
                sum = RELU_CLIP;
            h2[o] = sum;
        }

        Accum sum = bias_out;
        for (int i = 0; i < HIDDEN2; ++i)
            sum += static_cast<Accum>(w_out[i]) * static_cast<Accum>(h2[i]);

        sum >>= OUTPUT_SCALE_BITS;
        return static_cast<int>(sum);
    }

} // namespace pf
