// Transposition table implementation.

#include "tt.h"

namespace pf
{

    void TranspositionTable::resize(std::size_t megabytes)
    {
        std::size_t bytes = megabytes * 1024ull * 1024ull;
        std::size_t bucketCount = bytes / sizeof(TTBucket);
        if (bucketCount == 0)
            bucketCount = 1;
        buckets_.assign(bucketCount, TTBucket{});
        generation_ = 0;
    }

    void TranspositionTable::clear()
    {
        for (auto &b : buckets_)
        {
            for (int i = 0; i < TTBucket::SIZE; ++i)
            {
                b.e[i] = TTEntry{};
                b.e[i].gen = 0;
            }
        }
    }

    int TranspositionTable::pack_score(int score, int ply)
    {
        if (score >= MATE_SCORE - MAX_PLY)
            return score + ply;
        if (score <= -MATE_SCORE + MAX_PLY)
            return score - ply;
        return score;
    }

    int TranspositionTable::unpack_score(int packed, int ply)
    {
        if (packed >= MATE_SCORE - MAX_PLY)
            return packed - ply;
        if (packed <= -MATE_SCORE + MAX_PLY)
            return packed + ply;
        return packed;
    }

    bool TranspositionTable::probe(Key key, int depth, int alpha, int beta, int ply, TTEntry &out) const
    {
        if (buckets_.empty())
            return false;
        std::size_t idx = key % buckets_.size();
        const TTBucket &b = buckets_[idx];
        Key hi = key >> 48;
        for (int i = 0; i < TTBucket::SIZE; ++i)
        {
            const TTEntry &e = b.e[i];
            if (!e.key16 || e.key16 != hi)
                continue;
            int score = unpack_score(e.score, ply);
            if (e.depth >= depth)
            {
                if (e.bound == BOUND_EXACT)
                {
                    out = e;
                    out.score = (int16_t)score;
                    return true;
                }
                if (e.bound == BOUND_LOWER && score >= beta)
                {
                    out = e;
                    out.score = (int16_t)score;
                    return true;
                }
                if (e.bound == BOUND_UPPER && score <= alpha)
                {
                    out = e;
                    out.score = (int16_t)score;
                    return true;
                }
            }
        }
        return false;
    }

    void TranspositionTable::store(Key key, int depth, int score, BoundType bound, Move best, int ply)
    {
        if (buckets_.empty())
            return;
        std::size_t idx = key % buckets_.size();
        TTBucket &b = buckets_[idx];
        Key hi = key >> 48;
        TTEntry *replace = &b.e[0];
        for (int i = 0; i < TTBucket::SIZE; ++i)
        {
            TTEntry &e = b.e[i];
            if (!e.key16 || e.key16 == hi)
            {
                replace = &e;
                break;
            }
            int age_e = (e.gen == generation_) ? 1 : 0;
            int age_r = (replace->gen == generation_) ? 1 : 0;
            if (age_e < age_r || (age_e == age_r && e.depth < replace->depth))
                replace = &e;
        }
        replace->key16 = hi;
        replace->depth = (int8_t)depth;
        replace->score = (int16_t)pack_score(score, ply);
        replace->bound = bound;
        if (best != MOVE_NONE)
            replace->best = best;
        replace->gen = generation_;
    }

    Move TranspositionTable::probe_move(Key key) const
    {
        if (buckets_.empty())
            return MOVE_NONE;
        std::size_t idx = key % buckets_.size();
        const TTBucket &b = buckets_[idx];
        Key hi = key >> 48;
        for (int i = 0; i < TTBucket::SIZE; ++i)
        {
            const TTEntry &e = b.e[i];
            if (e.key16 == hi)
                return e.best;
        }
        return MOVE_NONE;
    }

} // namespace pf
