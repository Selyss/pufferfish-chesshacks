// Abstract neural network evaluation interface.

#pragma once

#include "types.h"
#include "position.h"
#include "nnue.h"
#include "features.h"

namespace pf
{

    // Implementations should provide a fast eval of the current position
    // from the side to move's perspective (centipawns).
    struct NNEvaluator
    {
        virtual ~NNEvaluator() = default;

        virtual int evaluate(const Position &pos) = 0;

        // Incremental accumulator hooks, default no-ops so evaluators that do not
        // maintain one are unaffected. The search brackets every position change:
        //
        //     nn->acc_begin_move(pos);   // snapshot, before the position changes
        //     pos.do_move(m, u);
        //     nn->acc_end_move(pos);     // diff against the snapshot, apply
        //     ...
        //     pos.undo_move(u);
        //     nn->acc_unmake();          // reverse the applied delta
        //
        // Only the search may call these. Position::do_move is also used by
        // filter_legal_moves() for legality testing, which restores the position
        // itself and must not disturb the accumulator.
        virtual void acc_reset(const Position &) {}
        virtual void acc_begin_move(const Position &) {}
        virtual void acc_end_move(const Position &) {}
        virtual void acc_unmake() {}
    };

    // Forward declare new evaluator to satisfy includes
    class SimpleNNUEEvaluator;

    // Concrete NNUE-based evaluator implementation that wraps the NNUE loader.
    class NNUEEvaluator : public NNEvaluator
    {
    public:
        NNUEEvaluator() { cache_.assign(kCacheEntries, {0, 0}); }

        bool load(const std::string &path)
        {
            clear_cache();
            return nn_.load(path);
        }

        void clear_cache() { cache_.assign(kCacheEntries, {0, 0}); }

        int evaluate(const Position &pos) override
        {
            // Measured on the bench suite at depth 11: 46.5% of evaluate() calls are
            // for a position the search has already evaluated. Evaluation is a pure
            // function of the position, so a cached score can never go stale and the
            // table never needs clearing between searches.
            CacheEntry &slot = cache_[pos.key & kCacheMask];
            if (slot.key == pos.key)
                return slot.score;

            // Rebuilt on a miss. The feature encoding is side-to-move relative
            // (channel c maps to (c+6)%12 when the mover flips), so every move
            // changes every piece feature's index; a single-perspective incremental
            // accumulator would cost more than a refresh, not less. Maintaining an
            // accumulator per perspective would work, but needs make/unmake hooks.
            nn_.refresh(pos);
            const int score = nn_.evaluate();
            slot.key = pos.key;
            slot.score = score;
            return score;
        }

    private:
        struct CacheEntry
        {
            Key key;
            int score;
        };

        // 2^20 entries, 16MB. Direct-mapped, always replace.
        static constexpr std::size_t kCacheEntries = 1u << 20;
        static constexpr std::size_t kCacheMask = kCacheEntries - 1;

        NNUE nn_;
        std::vector<CacheEntry> cache_;
    };

} // namespace pf