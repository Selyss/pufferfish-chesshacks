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
        float eps = 1e-5f;
        std::vector<float> gamma; // scale
        std::vector<float> beta;  // bias
        bool load(std::FILE *f, int dim_);
        void apply(std::vector<float> &x) const;
    };

    struct ResidualBlockF
    {
        LinearF l1;
        LinearF l2;
        LayerNormF ln;
    };

    struct SeqEntry
    {
        int type;
        int index;
    };

    class SimpleNNUEEvaluator : public NNEvaluator
    {
    public:
        bool load(const char *path);
        int evaluate(const Position &pos) override;

    private:
        bool loaded_ = false;
        int input_dim_ = 0;
        // Layer storage and execution order
        std::vector<LinearF> linears_;
        std::vector<LayerNormF> norms_;
        std::vector<ResidualBlockF> residuals_;
        std::vector<SeqEntry> sequence_;

        static void relu(std::vector<float> &v);
    };

} // namespace pf
