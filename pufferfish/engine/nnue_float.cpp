#include "nnue_float.h"

#include "bitboard.h"
#include "position.h"

#include <cstdio>
#include <cstring>

namespace pf
{
    namespace
    {
        bool read_exact(std::FILE *f, void *dst, std::size_t bytes)
        {
            return std::fread(dst, 1, bytes, f) == bytes;
        }
    }

    bool FloatNNUEEvaluator::load(const char *path)
    {
        loaded_ = false;
        std::FILE *f = std::fopen(path, "rb");
        if (!f)
            return false;

        char magic[4];
        if (!read_exact(f, magic, 4) || std::memcmp(magic, "PFNN", 4) != 0)
        {
            std::fclose(f);
            return false;
        }

        std::int32_t version = 0, featureDim = 0, accUnits = 0, h1 = 0, h2 = 0;
        if (!read_exact(f, &version, 4) || !read_exact(f, &featureDim, 4) ||
            !read_exact(f, &accUnits, 4) || !read_exact(f, &h1, 4) ||
            !read_exact(f, &h2, 4))
        {
            std::fclose(f);
            return false;
        }
        if (version != 1 || featureDim != kFeatureDim || accUnits != kAccUnits ||
            h1 != kHidden1 || h2 != kHidden2)
        {
            std::fclose(f);
            return false;
        }

        acc_w_.resize(static_cast<std::size_t>(kFeatureDim) * 2 * kAccUnits);
        fc1_w_.resize(static_cast<std::size_t>(kHidden1) * 2 * kAccUnits);
        fc2_w_.resize(static_cast<std::size_t>(kHidden2) * kHidden1);

        const bool ok =
            read_exact(f, acc_b_friendly_.data(), sizeof(float) * kAccUnits) &&
            read_exact(f, acc_b_enemy_.data(), sizeof(float) * kAccUnits) &&
            read_exact(f, acc_w_.data(), sizeof(float) * acc_w_.size()) &&
            read_exact(f, fc1_b_.data(), sizeof(float) * kHidden1) &&
            read_exact(f, fc1_w_.data(), sizeof(float) * fc1_w_.size()) &&
            read_exact(f, fc2_b_.data(), sizeof(float) * kHidden2) &&
            read_exact(f, fc2_w_.data(), sizeof(float) * fc2_w_.size()) &&
            read_exact(f, &out_b_, sizeof(float)) &&
            read_exact(f, out_w_.data(), sizeof(float) * kHidden2);

        std::fclose(f);
        loaded_ = ok;
        return ok;
    }

    float FloatNNUEEvaluator::evaluate_white(const Position &pos) const
    {
        if (!loaded_)
            return 0.0f;

        // Accumulator. Only occupied squares contribute, so this touches at most
        // 32 of the 768 features.
        float acc[2 * kAccUnits];
        for (int i = 0; i < kAccUnits; ++i)
        {
            acc[i] = acc_b_friendly_[i];
            acc[kAccUnits + i] = acc_b_enemy_[i];
        }

        Bitboard occ = pos.occupiedBB;
        while (occ)
        {
            const int sq = lsb(occ);
            occ &= occ - 1;

            const Piece pc = pos.board[sq];
            if (pc == NO_PIECE)
                continue;

            // Colour-absolute, square-major -- matches train.py's fen_to_features.
            const bool isWhite = (pc <= W_KING);
            const int typeIdx = isWhite ? (pc - W_PAWN) : (pc - B_PAWN);
            const int feat = sq * 12 + typeIdx + (isWhite ? 0 : 6);
            if (feat < 0 || feat >= kFeatureDim)
                continue;

            const float *w = &acc_w_[static_cast<std::size_t>(feat) * 2 * kAccUnits];
            for (int i = 0; i < 2 * kAccUnits; ++i)
                acc[i] += w[i];
        }

        // The model applies relu to each projection, then concatenates.
        for (int i = 0; i < 2 * kAccUnits; ++i)
            if (acc[i] < 0.0f)
                acc[i] = 0.0f;

        float h1[kHidden1];
        for (int o = 0; o < kHidden1; ++o)
        {
            const float *w = &fc1_w_[static_cast<std::size_t>(o) * 2 * kAccUnits];
            float sum = fc1_b_[o];
            for (int i = 0; i < 2 * kAccUnits; ++i)
                sum += w[i] * acc[i];
            h1[o] = sum > 0.0f ? sum : 0.0f;
        }

        float h2[kHidden2];
        for (int o = 0; o < kHidden2; ++o)
        {
            const float *w = &fc2_w_[static_cast<std::size_t>(o) * kHidden1];
            float sum = fc2_b_[o];
            for (int i = 0; i < kHidden1; ++i)
                sum += w[i] * h1[i];
            h2[o] = sum > 0.0f ? sum : 0.0f;
        }

        float out = out_b_;
        for (int i = 0; i < kHidden2; ++i)
            out += out_w_[i] * h2[i];
        return out;
    }

    int FloatNNUEEvaluator::evaluate(const Position &pos)
    {
        const float white = evaluate_white(pos);
        // The network has no side-to-move input, so it scores from White's point
        // of view; the search wants the score relative to the mover.
        const float stm = (pos.side_to_move == WHITE) ? white : -white;
        return static_cast<int>(stm);
    }

} // namespace pf
