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

    bool LayerNormF::load(std::FILE *f, int dim_)
    {
        dim = dim_;
        eps = 1e-5f;
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
        {
            std::fprintf(stderr, "[ERROR] Failed to open file: %s\n", path);
            return false;
        }
        std::fprintf(stderr, "[INFO] Successfully opened: %s\n", path);

        // JSON metadata header [u32 length][json utf8]
        uint32_t json_len = 0;
        if (!read_u32(f, json_len))
        {
            std::fprintf(stderr, "[ERROR] Failed to read JSON length (file may be empty or corrupted)\n");
            std::fclose(f);
            return false;
        }
        std::fprintf(stderr, "[INFO] JSON length: %u bytes\n", json_len);
        if (json_len == 0 || json_len > (32u << 20))
        {
            std::fprintf(stderr, "[ERROR] Invalid JSON length: %u (expected 1 to %u)\n", json_len, (32u << 20));
            std::fclose(f);
            return false;
        }
        std::string json(json_len, '\0');
        if (!fread_exact(f, json.data(), json_len))
        {
            std::fprintf(stderr, "[ERROR] Failed to read JSON data\n");
            std::fclose(f);
            return false;
        }
        std::fprintf(stderr, "[INFO] JSON: %s\n", json.c_str());
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
            std::fprintf(stderr, "[ERROR] Invalid format: '%s' (expected 'residual-nnue-v1')\n", fmt.c_str());
            std::fclose(f);
            return false;
        }
        std::string in_s = find_val("input_dim");
        std::string lc_s = find_val("layer_count");
        if (in_s.empty() || lc_s.empty())
        {
            std::fprintf(stderr, "[ERROR] Missing input_dim or layer_count in JSON\n");
            std::fclose(f);
            return false;
        }
        input_dim_ = std::stoi(in_s);
        int layer_count = std::stoi(lc_s);
        std::fprintf(stderr, "[INFO] Format: %s\n", fmt.c_str());
        std::fprintf(stderr, "[INFO] Input dim: %d\n", input_dim_);
        std::fprintf(stderr, "[INFO] Layer count from JSON: %d\n", layer_count);
        if (layer_count <= 0)
        {
            std::fprintf(stderr, "[ERROR] Invalid layer count: %d\n", layer_count);
            std::fclose(f);
            return false;
        }

        // Read the layer count written after JSON
        uint32_t stored_layer_count = 0;
        if (!read_u32(f, stored_layer_count))
        {
            std::fprintf(stderr, "[ERROR] Failed to read stored layer count from binary\n");
            std::fclose(f);
            return false;
        }
        std::fprintf(stderr, "[INFO] Layer count from binary: %u\n", stored_layer_count);
        if (stored_layer_count != static_cast<uint32_t>(layer_count))
        {
            std::fprintf(stderr, "[ERROR] Layer count mismatch: JSON=%d, binary=%u\n", layer_count, stored_layer_count);
            std::fclose(f);
            return false;
        }

        // Read layer records
        std::fprintf(stderr, "[INFO] Loading %u layers...\n", stored_layer_count);
        for (uint32_t li = 0; li < stored_layer_count; ++li)
        {
            uint32_t type_id = 0;
            if (!read_u32(f, type_id))
            {
                std::fprintf(stderr, "[ERROR] Failed to read type_id at layer %u\n", li);
                std::fclose(f);
                return false;
            }
            std::fprintf(stderr, "[INFO] Layer %u: type_id=%u", li + 1, type_id);
            
            if (type_id == 1) // LINEAR
            {
                uint32_t out_u = 0, in_u = 0;
                if (!read_u32(f, out_u) || !read_u32(f, in_u))
                {
                    std::fprintf(stderr, " [ERROR] Failed to read dimensions\n");
                    std::fclose(f);
                    return false;
                }
                int outD = static_cast<int>(out_u);
                int inD = static_cast<int>(in_u);
                std::fprintf(stderr, " LINEAR %d->%d", inD, outD);
                if (inD <= 0 || outD <= 0)
                {
                    std::fprintf(stderr, " [ERROR] Invalid dimensions\n");
                    std::fclose(f);
                    return false;
                }
                LinearF L;
                if (!L.load(f, inD, outD))
                {
                    std::fprintf(stderr, " [ERROR] Failed to load weights\n");
                    std::fclose(f);
                    return false;
                }
                int idx = (int)linears_.size();
                linears_.push_back(std::move(L));
                sequence_.push_back({1, idx});
                std::fprintf(stderr, " ✓\n");
            }
            else if (type_id == 2) // LAYERNORM
            {
                uint32_t size_u = 0, reserved = 0;
                if (!read_u32(f, size_u) || !read_u32(f, reserved))
                {
                    std::fprintf(stderr, " [ERROR] Failed to read size\n");
                    std::fclose(f);
                    return false;
                }
                int dim = static_cast<int>(size_u);
                std::fprintf(stderr, " LAYERNORM %d", dim);
                if (dim <= 0)
                {
                    std::fprintf(stderr, " [ERROR] Invalid dimension\n");
                    std::fclose(f);
                    return false;
                }
                LayerNormF LN;
                if (!LN.load(f, dim))
                {
                    std::fprintf(stderr, " [ERROR] Failed to load weights\n");
                    std::fclose(f);
                    return false;
                }
                float eps = 1e-5f;
                if (!read_f32(f, eps))
                {
                    std::fprintf(stderr, " [ERROR] Failed to read eps\n");
                    std::fclose(f);
                    return false;
                }
                LN.eps = eps;
                int idx = (int)norms_.size();
                norms_.push_back(std::move(LN));
                sequence_.push_back({2, idx});
                std::fprintf(stderr, " ✓\n");
            }
            else if (type_id == 3) // RESIDUAL
            {
                uint32_t dim1_u = 0, dim2_u = 0;
                if (!read_u32(f, dim1_u) || !read_u32(f, dim2_u))
                {
                    std::fprintf(stderr, " [ERROR] Failed to read dimensions\n");
                    std::fclose(f);
                    return false;
                }
                int dim = static_cast<int>(dim1_u);
                std::fprintf(stderr, " RESIDUAL %dx%d", dim, dim);
                if (dim <= 0 || dim1_u != dim2_u)
                {
                    std::fprintf(stderr, " [ERROR] Invalid dimensions\n");
                    std::fclose(f);
                    return false;
                }
                
                ResidualBlockF RB;
                // Load lin1: weight[dim][dim], bias[dim]
                if (!RB.l1.load(f, dim, dim))
                {
                    std::fprintf(stderr, " [ERROR] Failed to load l1\n");
                    std::fclose(f);
                    return false;
                }
                
                // Load lin2: weight[dim][dim], bias[dim]
                if (!RB.l2.load(f, dim, dim))
                {
                    std::fprintf(stderr, " [ERROR] Failed to load l2\n");
                    std::fclose(f);
                    return false;
                }
                
                // Load layernorm: weight[dim], bias[dim], eps
                if (!RB.ln.load(f, dim))
                {
                    std::fprintf(stderr, " [ERROR] Failed to load norm\n");
                    std::fclose(f);
                    return false;
                }
                float eps = 1e-5f;
                if (!read_f32(f, eps))
                {
                    std::fprintf(stderr, " [ERROR] Failed to read eps\n");
                    std::fclose(f);
                    return false;
                }
                RB.ln.eps = eps;
                
                int idx = (int)residuals_.size();
                residuals_.push_back(std::move(RB));
                sequence_.push_back({3, idx});
                std::fprintf(stderr, " ✓\n");
            }
            else
            {
                std::fprintf(stderr, " [ERROR] Unknown type_id: %u\n", type_id);
                std::fclose(f);
                return false;
            }
        }

        std::fclose(f);
        loaded_ = true;
        std::fprintf(stderr, "[SUCCESS] Loaded NNUE: %zu linear, %zu layernorm, %zu residual layers\n",
                     linears_.size(), norms_.size(), residuals_.size());
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
        std::vector<float> tmp;

        for (size_t i = 0; i < sequence_.size(); ++i)
        {
            const auto &se = sequence_[i];
            if (se.type == 1) // LINEAR
            {
                const LinearF &L = linears_[se.index];
                L.forward(x, tmp);
                x.swap(tmp);
                // ReLU is applied implicitly (next layer is LayerNorm or ReLU is skipped for output)
                // Check if this is NOT the output head by seeing if next is LayerNorm
                if (i + 1 < sequence_.size() && sequence_[i + 1].type == 2)
                {
                    relu(x);
                }
            }
            else if (se.type == 2) // LAYERNORM
            {
                const LayerNormF &LN = norms_[se.index];
                LN.apply(x);
            }
            else if (se.type == 3) // RESIDUAL
            {
                const ResidualBlockF &RB = residuals_[se.index];
                std::vector<float> residual = x;  // Save input for skip connection
                
                // y = relu(lin1(x))
                RB.l1.forward(x, tmp);
                relu(tmp);
                
                // y = lin2(y)
                std::vector<float> y;
                RB.l2.forward(tmp, y);
                
                // out = residual + y
                if (y.size() != residual.size())
                    return 0;
                for (size_t k = 0; k < y.size(); ++k)
                    y[k] += residual[k];
                
                // out = relu(norm(out))
                RB.ln.apply(y);
                relu(y);
                
                x.swap(y);
            }
        }

        float out = (x.empty() ? 0.0f : x[0]);
        int cp = (int)std::lround(out * 100.0f);
        return cp;
    }

} // namespace pf
