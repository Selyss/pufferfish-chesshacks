#include "simple_nnue.h"
#include "features.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>

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

    bool LayerNormF::load(std::FILE *f, int dim_, float eps_)
    {
        dim = dim_;
        eps = eps_;
        gamma.resize(dim);
        beta.resize(dim);
        if (!fread_exact(f, gamma.data(), dim * sizeof(float)))
            return false;
        if (!fread_exact(f, beta.data(), dim * sizeof(float)))
            return false;
        return true;
    }

    void LayerNormF::apply(std::vector<float> &x) const
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

    static bool read_u32(std::FILE *f, uint32_t &v) { return fread_exact(f, &v, sizeof(v)); }
    static bool read_i32(std::FILE *f, int32_t &v) { return fread_exact(f, &v, sizeof(v)); }
    static bool read_f32(std::FILE *f, float &v) { return fread_exact(f, &v, sizeof(v)); }

    bool SimpleNNUEEvaluator::load(const char *path)
    {
        loaded_ = false;
        linears_.clear();
        norms_.clear();
        residuals_.clear();
        sequence_.clear();

        std::FILE *f = std::fopen(path, "rb");
        if (!f)
            return false;

        // JSON metadata header [u32 length][json utf8]
        uint32_t json_len = 0;
        if (!read_u32(f, json_len) || json_len == 0 || json_len > (32u << 20))
        {
            std::fclose(f);
            return false;
        }
        std::string json(json_len, '\0');
        if (!fread_exact(f, json.data(), json_len))
        {
            std::fclose(f);
            return false;
        }
        auto find_val = [&](const char *key) -> std::string
        {
            size_t kpos = json.find(key);
            if (kpos == std::string::npos)
                return {};
            size_t colon = json.find(':', kpos);
            if (colon == std::string::npos)
                return {};
            size_t start = colon + 1;
            while (start < json.size() && (json[start] == ' ' || json[start] == '"'))
                ++start;
            size_t end = start;
            while (end < json.size() && json[end] != ',' && json[end] != '}' && json[end] != '"')
                ++end;
            return json.substr(start, end - start);
        };
        std::string fmt = find_val("format");
        if (fmt != "residual-nnue-v1")
        {
            std::fclose(f);
            return false;
        }
        std::string in_s = find_val("input_dim");
        std::string lc_s = find_val("layer_count");
        if (in_s.empty() || lc_s.empty())
        {
            std::fclose(f);
            return false;
        }
        input_dim_ = std::stoi(in_s);
        int layer_count = std::stoi(lc_s);

        for (int li = 0; li < layer_count; ++li)
        {
            int32_t type_id = 0;
            if (!read_i32(f, type_id))
            {
                std::fclose(f);
                return false;
            }
            if (type_id == 1)
            {
                int32_t inD = 0, outD = 0;
                if (!read_i32(f, inD) || !read_i32(f, outD))
                {
                    std::fclose(f);
                    return false;
                }
                LinearF L;
                if (!L.load(f, inD, outD))
                {
                    std::fclose(f);
                    return false;
                }
                int idx = (int)linears_.size();
                linears_.push_back(std::move(L));
                sequence_.push_back({1, idx});
            }
            else if (type_id == 2)
            {
                int32_t dim = 0;
                float eps = 1e-5f;
                if (!read_i32(f, dim) || !read_f32(f, eps))
                {
                    std::fclose(f);
                    return false;
                }
                LayerNormF LN;
                if (!LN.load(f, dim, eps))
                {
                    std::fclose(f);
                    return false;
                }
                int idx = (int)norms_.size();
                norms_.push_back(std::move(LN));
                sequence_.push_back({2, idx});
            }
            else if (type_id == 3)
            {
                int32_t dim = 0;
                if (!read_i32(f, dim))
                {
                    std::fclose(f);
                    return false;
                }
                ResidualBlockF RB;
                int32_t in1 = 0, out1 = 0;
                if (!read_i32(f, in1) || !read_i32(f, out1))
                {
                    std::fclose(f);
                    return false;
                }
                if (!RB.l1.load(f, in1, out1))
                {
                    std::fclose(f);
                    return false;
                }
                int32_t in2 = 0, out2 = 0;
                if (!read_i32(f, in2) || !read_i32(f, out2))
                {
                    std::fclose(f);
                    return false;
                }
                if (!RB.l2.load(f, in2, out2))
                {
                    std::fclose(f);
                    return false;
                }
                int32_t lnd = 0;
                float ln_eps = 1e-5f;
                if (!read_i32(f, lnd) || !read_f32(f, ln_eps))
                {
                    std::fclose(f);
                    return false;
                }
                if (!RB.ln.load(f, lnd, ln_eps))
                {
                    std::fclose(f);
                    return false;
                }
                int idx = (int)residuals_.size();
                residuals_.push_back(std::move(RB));
                sequence_.push_back({3, idx});
            }
            else
            {
                std::fclose(f);
                return false;
            }
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
        if (input_dim_ == 795)
        {
            x.reserve(795);
            extract_features_795(pos, x);
        }
        else
        {
            return 0;
        }
        std::vector<float> h, tmp;

        for (size_t i = 0; i < sequence_.size(); ++i)
        {
            const auto &se = sequence_[i];
            if (se.type == 1)
            {
                const LinearF &L = linears_[se.index];
                L.forward(x, h);
                if (i + 1 < sequence_.size() && sequence_[i + 1].type == 2)
                    relu(h);
                x.swap(h);
            }
            else if (se.type == 2)
            {
                const LayerNormF &LN = norms_[se.index];
                LN.apply(x);
            }
            else if (se.type == 3)
            {
                const ResidualBlockF &RB = residuals_[se.index];
                RB.l1.forward(x, h);
                relu(h);
                RB.l2.forward(h, h);
                if (h.size() != x.size())
                    return 0;
                for (size_t k = 0; k < h.size(); ++k)
                    h[k] += x[k];
                tmp = h;
                RB.ln.apply(tmp);
                x.swap(tmp);
            }
        }

        float out = (x.empty() ? 0.0f : x[0]);
        int cp = (int)std::lround(out * 100.0f);
        return cp;
    }

} // namespace pf
