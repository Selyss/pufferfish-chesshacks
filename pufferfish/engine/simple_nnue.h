#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "types.h"
#include "position.h"
#include "nn_interface.h"

namespace pf
{

    struct LinearF
    {
        int in = 0, out = 0;
        std::vector<float> W; // row-major [out][in]
        std::vector<float> b; // [out]
        bool load(std::FILE *f, int inDim, int outDim);
        void forward(const std::vector<float> &x, std::vector<float> &y) const;
    };

    struct LayerNormF
    {
        int dim = 0;
        std::vector<float> gamma; // scale
        std::vector<float> beta;  // bias
        bool load(std::FILE *f, int dim_);
        void apply(std::vector<float> &x, float eps = 1e-5f) const;
    };

    struct ResidualBlockF
    {
        LinearF l1;
        LinearF l2;
        LayerNormF ln;
    };

    struct StageF
    {
        LinearF base;
        LayerNormF ln_base;
        std::vector<ResidualBlockF> blocks; // typically 2
    };

    class SimpleNNUEEvaluator : public NNEvaluator
    {
    public:
        bool load(const char *path);
        int evaluate(const Position &pos) override;

    private:
        bool loaded_ = false;
        int input_dim_ = 0;
        std::vector<StageF> stages_; // 5 stages
        LinearF head_;               // 256 -> 1

        static void relu(std::vector<float> &v);
    };

} // namespace pf
