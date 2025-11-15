#include "simple_nnue.h"
#include "features.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace pf
{

    static bool fread_exact(std::FILE *f, void *dst, size_t n)
    {
        return std::fread(dst, 1, n, f) == n;
    }

    bool LinearF::load(std::FILE *f, int inDim, int outDim)
    {
        in = inDim;
        out = outDim;
        W.resize((size_t)out * (size_t)in);
        b.resize((size_t)out);
        if (!fread_exact(f, W.data(), W.size() * sizeof(float)))
            return false;
        if (!fread_exact(f, b.data(), b.size() * sizeof(float)))
            return false;
        return true;
    }

    void LinearF::forward(const std::vector<float> &x, std::vector<float> &y) const
    {
        y.assign(out, 0.0f);
        const float *w = W.data();
        for (int o = 0; o < out; ++o)
        {
            const float *row = w + (size_t)o * (size_t)in;
            float acc = b[o];
            for (int i = 0; i < in; ++i)
                acc += row[i] * x[i];
            y[o] = acc;
        }
    }

    bool LayerNormF::load(std::FILE *f, int dim_)
    {
        dim = dim_;
        gamma.resize(dim);
        beta.resize(dim);
        if (!fread_exact(f, gamma.data(), dim * sizeof(float)))
            return false;
        if (!fread_exact(f, beta.data(), dim * sizeof(float)))
            return false;
        return true;
    }

    void LayerNormF::apply(std::vector<float> &x, float eps) const
    {
        float mean = 0.0f;
        for (int i = 0; i < dim; ++i)
            mean += x[i];
        mean /= std::max(1, dim);
        float var = 0.0f;
        for (int i = 0; i < dim; ++i)
        {
            float d = x[i] - mean;
            var += d * d;
        }
        var /= std::max(1, dim);
        float inv = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < dim; ++i)
        {
            float xn = (x[i] - mean) * inv;
            x[i] = xn * gamma[i] + beta[i];
        }
    }

    void SimpleNNUEEvaluator::relu(std::vector<float> &v)
    {
        for (auto &x : v)
            if (x < 0.0f)
                x = 0.0f;
    }

    bool SimpleNNUEEvaluator::load(const char *path)
    {
        loaded_ = false;
        stages_.clear();
        head_ = LinearF{};

        std::FILE *f = std::fopen(path, "rb");
        if (!f)
            return false;
        struct Header
        {
            char magic[8];
            uint32_t version;
            uint32_t in_dim;
            uint32_t widths[5];
            uint32_t res_blocks;
            uint32_t flags;
        } h;
        if (!fread_exact(f, &h, sizeof(h)))
        {
            std::fclose(f);
            return false;
        }
        if (std::memcmp(h.magic, "SNNUEV1", 7) != 0 || h.version != 1)
        {
            std::fclose(f);
            return false;
        }
        input_dim_ = (int)h.in_dim;
        int widths[5];
        for (int i = 0; i < 5; ++i)
            widths[i] = (int)h.widths[i];
        int R = (int)h.res_blocks;
        // Expect spec dims
        if (input_dim_ != 795 || widths[0] != 2048 || widths[1] != 2048 || widths[2] != 1024 || widths[3] != 512 || widths[4] != 256)
        {
            std::fclose(f);
            return false;
        }
        stages_.resize(5);
        int prev = input_dim_;
        for (int s = 0; s < 5; ++s)
        {
            int cur = widths[s];
            StageF st;
            if (!st.base.load(f, prev, cur))
            {
                std::fclose(f);
                return false;
            }
            if (!st.ln_base.load(f, cur))
            {
                std::fclose(f);
                return false;
            }
            st.blocks.resize(R);
            for (int r = 0; r < R; ++r)
            {
                if (!st.blocks[r].l1.load(f, cur, cur))
                {
                    std::fclose(f);
                    return false;
                }
                if (!st.blocks[r].l2.load(f, cur, cur))
                {
                    std::fclose(f);
                    return false;
                }
                if (!st.blocks[r].ln.load(f, cur))
                {
                    std::fclose(f);
                    return false;
                }
            }
            stages_[s] = std::move(st);
            prev = cur;
        }
        if (!head_.load(f, widths[4], 1))
        {
            std::fclose(f);
            return false;
        }
        std::fclose(f);
        loaded_ = true;
        return true;
    }

    int SimpleNNUEEvaluator::evaluate(const Position &pos)
    {
        if (!loaded_)
            return 0;
        std::vector<float> x;
        x.reserve(795);
        extract_features_795(pos, x);
        std::vector<float> h, tmp;

        // 5 stages: (Linear->ReLU->LN; + 2x residual blocks each)
        for (size_t s = 0; s < stages_.size(); ++s)
        {
            stages_[s].base.forward(x, h);
            relu(h);
            stages_[s].ln_base.apply(h);
            for (const auto &rb : stages_[s].blocks)
            {
                rb.l1.forward(h, tmp);
                relu(tmp);
                rb.l2.forward(tmp, tmp);
                // skip add
                for (size_t i = 0; i < h.size(); ++i)
                    tmp[i] += h[i];
                // ln
                rb.ln.apply(tmp);
                h.swap(tmp);
            }
            x.swap(h);
        }

        // Head 256->1
        head_.forward(x, h);
        float out = h[0];
        // Convert pawn units to centipawns (assume model output ~ pawns)
        int cp = (int)std::round(out * 100.0f);
        return cp;
    }

} // namespace pf
