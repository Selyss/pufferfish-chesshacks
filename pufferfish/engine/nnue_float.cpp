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

        // Dot product with eight independent partial sums.
        //
        // A plain `for (i) sum += w[i] * x[i]` will not vectorize: floating point
        // addition is not associative, so the compiler may not reorder the
        // reduction, and it emits scalar code. Splitting into independent
        // accumulators gives it permission, and measured 5.8x on this shape
        // (512 wide: 6.27us -> 1.08us, 5.2 -> 30.3 GFLOP/s).
        //
        // The summation order differs from the naive loop, so results can differ in
        // the last bits. That is the same latitude PyTorch takes, and the port is
        // still checked against it to a 1cp tolerance.
        inline float dot8(const float *w, const float *x, int n)
        {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
            int i = 0;
            for (; i + 8 <= n; i += 8)
            {
                s0 += w[i + 0] * x[i + 0];
                s1 += w[i + 1] * x[i + 1];
                s2 += w[i + 2] * x[i + 2];
                s3 += w[i + 3] * x[i + 3];
                s4 += w[i + 4] * x[i + 4];
                s5 += w[i + 5] * x[i + 5];
                s6 += w[i + 6] * x[i + 6];
                s7 += w[i + 7] * x[i + 7];
            }
            float tail = 0;
            for (; i < n; ++i)
                tail += w[i] * x[i];
            return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)) + tail;
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

    namespace
    {
        // Feature channel for a piece: White P,N,B,R,Q,K = 0..5, Black = 6..11.
        inline int piece_channel(Piece pc)
        {
            const bool isWhite = (pc <= W_KING);
            return (isWhite ? (pc - W_PAWN) : (pc - B_PAWN)) + (isWhite ? 0 : 6);
        }
    }

    void FloatNNUEEvaluator::apply(int feat, float sign)
    {
        const float *w = &acc_w_[static_cast<std::size_t>(feat) * 2 * kAccUnits];
        if (sign > 0.0f)
        {
            for (int i = 0; i < 2 * kAccUnits; ++i)
                acc_[i] += w[i];
        }
        else
        {
            for (int i = 0; i < 2 * kAccUnits; ++i)
                acc_[i] -= w[i];
        }
    }

    void FloatNNUEEvaluator::acc_reset(const Position &pos)
    {
        if (!loaded_)
            return;
        for (int i = 0; i < kAccUnits; ++i)
        {
            acc_[i] = acc_b_friendly_[i];
            acc_[kAccUnits + i] = acc_b_enemy_[i];
        }
        Bitboard occ = pos.occupiedBB;
        while (occ)
        {
            const int sq = lsb(occ);
            occ &= occ - 1;
            const Piece pc = pos.board[sq];
            if (pc == NO_PIECE)
                continue;
            apply(sq * 12 + piece_channel(pc), +1.0f);
        }
        ply_ = 0;
        acc_valid_ = true;
    }

    void FloatNNUEEvaluator::acc_begin_move(const Position &pos)
    {
        if (!acc_valid_)
            return;
        if (static_cast<std::size_t>(ply_) >= stack_.size())
            stack_.resize(static_cast<std::size_t>(ply_) + 64);
        Delta &d = stack_[ply_];
        d.count = 0;
        d.overflow = false;
        for (int p = 0; p < PIECE_NB; ++p)
            d.snapshot[p] = pos.pieceBB[p];
    }

    void FloatNNUEEvaluator::acc_end_move(const Position &pos)
    {
        if (!acc_valid_)
            return;
        Delta &d = stack_[ply_];
        ++ply_;

        // Derive the change from the piece bitboards rather than from the move
        // encoding. That covers captures, castling, promotion and en passant
        // without restating do_move's rules, so the two cannot drift apart.
        for (int p = W_PAWN; p < PIECE_NB; ++p)
        {
            Bitboard changed = d.snapshot[p] ^ pos.pieceBB[p];
            while (changed)
            {
                const int sq = lsb(changed);
                changed &= changed - 1;
                const bool wasSet = (d.snapshot[p] >> sq) & 1ULL;
                const float sign = wasSet ? -1.0f : +1.0f;
                const int feat = sq * 12 + piece_channel(static_cast<Piece>(p));
                if (d.count < Delta::kMaxItems)
                {
                    d.feat[d.count] = feat;
                    d.sign[d.count] = sign;
                    ++d.count;
                    apply(feat, sign);
                }
                else
                {
                    d.overflow = true;
                }
            }
        }
        if (d.overflow)
        {
            // Should not happen for a legal move, but never guess: rebuild.
            acc_reset(pos);
            ply_ = 1;
        }
    }

    void FloatNNUEEvaluator::acc_unmake()
    {
        if (!acc_valid_ || ply_ <= 0)
            return;
        --ply_;
        Delta &d = stack_[ply_];
        if (d.overflow)
        {
            acc_valid_ = false; // force a refresh on the next evaluate
            return;
        }
        for (int i = 0; i < d.count; ++i)
            apply(d.feat[i], -d.sign[i]);
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

        return tail(acc);
    }

    float FloatNNUEEvaluator::tail(const float *acc) const
    {
        float relued[2 * kAccUnits];
        for (int i = 0; i < 2 * kAccUnits; ++i)
            relued[i] = acc[i] > 0.0f ? acc[i] : 0.0f;

        float h1[kHidden1];
        for (int o = 0; o < kHidden1; ++o)
        {
            const float *w = &fc1_w_[static_cast<std::size_t>(o) * 2 * kAccUnits];
            const float sum = fc1_b_[o] + dot8(w, relued, 2 * kAccUnits);
            h1[o] = sum > 0.0f ? sum : 0.0f;
        }
        float h2[kHidden2];
        for (int o = 0; o < kHidden2; ++o)
        {
            const float *w = &fc2_w_[static_cast<std::size_t>(o) * kHidden1];
            const float sum = fc2_b_[o] + dot8(w, h1, kHidden1);
            h2[o] = sum > 0.0f ? sum : 0.0f;
        }
        return out_b_ + dot8(out_w_.data(), h2, kHidden2);
    }

    int FloatNNUEEvaluator::evaluate(const Position &pos)
    {
        if (!loaded_)
            return 0;
        if (!acc_valid_)
            acc_reset(pos);
        const float white = tail(acc_);
        // The network has no side-to-move input, so it scores from White's point
        // of view; the search wants the score relative to the mover.
        const float stm = (pos.side_to_move == WHITE) ? white : -white;
        return static_cast<int>(stm);
    }

} // namespace pf
