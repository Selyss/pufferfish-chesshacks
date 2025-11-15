// Alpha-beta search with NN evaluation, TT, LMR, null move, and quiescence.

#include "search.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include "pst.h"

namespace pf
{
    static inline int piece_mvv_value(int piece)
    {
        // Use conventional MVV values; indices are pf::Piece enums.
        static const int v[PIECE_NB] = {
            0,                             // NO_PIECE
            100,                           // W_PAWN
            320,                           // W_KNIGHT
            330,                           // W_BISHOP
            500,                           // W_ROOK
            900,                           // W_QUEEN
            20000,                         // W_KING (avoid preferring captures of king; large for ordering only)
            100, 320, 330, 500, 900, 20000 // black pieces
        };
        if (piece < 0 || piece >= PIECE_NB)
            return 0;
        return v[piece];
    }

    static std::uint64_t now_ms()
    {
        using namespace std::chrono;
        return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
    }

    struct Line
    {
        Move moves[MAX_PLY];
        int len = 0;
    };

    static int qsearch(Position &pos, SearchContext &ctx, int alpha, int beta, int ply);
    static int alphabeta(Position &pos, SearchContext &ctx, int depth, int alpha, int beta, int ply, NodeType nodeType, Line &pv);

    static void order_moves(const Position &pos, SearchContext &ctx, const MoveList &raw, MoveList &ordered, Move ttMove, Move prevBest, int ply)
    {
        struct Scored
        {
            Move m;
            int score;
        } buf[MAX_MOVES];

        int count = raw.count;
        for (int i = 0; i < count; ++i)
        {
            Move m = raw.moves[i];
            int s = 0;
            if (m == ttMove)
                s += 1000000;
            if (m == prevBest)
                s += 900000;
            const std::uint32_t flags = move_flags(m);
            const bool isCapture = (flags & FLAG_CAPTURE) != 0;
            const bool isPromotion = (flags & FLAG_PROMOTION) != 0;
            int from = from_sq(m);
            int to = to_sq(m);
            int mpiece = move_piece(m);
            if (isCapture)
            {
                // MVV-LVA with real values
                int victim = pos.board[to_sq(m)];
                int attacker = mpiece;
                s += 200000 + piece_mvv_value(victim) - (piece_mvv_value(attacker) / 10);
                if (isPromotion)
                    s += 5000; // promote-capture is even better
                // Small PST delta to prefer captures improving piece placement
                s += (pst_value(Piece(mpiece), to) - pst_value(Piece(mpiece), from));
            }
            else
            {
                if (isPromotion)
                {
                    s += 120000; // non-capture promotions are high priority
                    // Prefer promotion square quality (assume queen for simplicity)
                    s += pst_value(W_QUEEN, to);
                }
                else if (m == ctx.killers[0][ply])
                    s += 80000;
                else if (m == ctx.killers[1][ply])
                    s += 70000;
                else
                {
                    s += ctx.history[mpiece][to];
                    // PST delta for quiet moves
                    s += (pst_value(Piece(mpiece), to) - pst_value(Piece(mpiece), from));
                }
            }
            buf[i] = {m, s};
        }

        std::sort(buf, buf + count, [](const Scored &a, const Scored &b)
                  { return a.score > b.score; });
        ordered.clear();
        for (int i = 0; i < count; ++i)
            ordered.push(buf[i].m);
    }

    static int estimate_moves_to_go(const Position &pos)
    {
        // Simple phase-based MTG: opening 28, middlegame 20, endgame 12
        auto cnt = [&](Piece p)
        {
#ifdef _MSC_VER
            return (int)__popcnt64(pos.pieceBB[p]);
#else
            return __builtin_popcountll(pos.pieceBB[p]);
#endif
        };
        int nonPawnMaterial = (cnt(W_KNIGHT) + cnt(B_KNIGHT)) * 3 + (cnt(W_BISHOP) + cnt(B_BISHOP)) * 3 +
                              (cnt(W_ROOK) + cnt(B_ROOK)) * 5 + (cnt(W_QUEEN) + cnt(B_QUEEN)) * 9;
        int mtg = 20;
        if (nonPawnMaterial >= 40)
            mtg = 28; // opening
        else if (nonPawnMaterial <= 16)
            mtg = 12; // endgame
        return mtg;
    }

    SearchResult search(Position &pos, SearchContext &ctx)
    {
        ctx.stats = SearchStats{};
        std::uint64_t start = now_ms();
        ctx.tm.start_ms = start;
        if (ctx.limits.time_ms)
        {
            ctx.tm.alloc_ms = ctx.limits.time_ms;
        }
        else if (ctx.limits.time_left_ms)
        {
            std::uint64_t T = ctx.limits.time_left_ms;
            int mtg = estimate_moves_to_go(pos);
            // Keep a reserve: 12% of remaining time, capped at 4000 ms
            std::uint64_t reserve = std::min<std::uint64_t>(T / 8, 4000);
            std::uint64_t usable = (T > reserve) ? (T - reserve) : (T * 7 / 8);
            // Base allocation: usable / (mtg + 1) for safety
            std::uint64_t base = usable / (std::uint64_t)(mtg + 1);
            // Clamp between [10 ms, 18% of T]
            std::uint64_t hardMax = (T * 18) / 100;
            std::uint64_t alloc = std::max<std::uint64_t>(10, std::min<std::uint64_t>(base, hardMax));
            ctx.tm.alloc_ms = alloc;
            std::cerr << "info tm time_left_ms " << T << " mtg " << mtg
                      << " reserve_ms " << reserve << " alloc_ms " << alloc << std::endl;
        }

        SearchResult result;
        Line rootPV;
        int alpha = -INF_SCORE;
        int beta = INF_SCORE;
        Move bestSoFar = MOVE_NONE;

        int maxDepth = ctx.limits.depth ? ctx.limits.depth : 64;

        for (int depth = 1; depth <= maxDepth; ++depth)
        {
            // Light history decay each iteration to keep values bounded
            for (int p = 0; p < PIECE_NB; ++p)
                for (int sq = 0; sq < 64; ++sq)
                    ctx.history[p][sq] -= (ctx.history[p][sq] >> 3);

            Line pv;
            int window = 20; // aspiration window in cp
            int scoreLo = alpha;
            int scoreHi = beta;

            if (depth > 1 && result.score > -INF_SCORE && result.score < INF_SCORE)
            {
                scoreLo = result.score - window;
                scoreHi = result.score + window;
            }

            int score;
            while (true)
            {
                pv.len = 0;
                score = alphabeta(pos, ctx, depth, scoreLo, scoreHi, 0, NODE_ROOT, pv);
                if (score <= scoreLo && score > -INF_SCORE)
                {
                    scoreLo = -INF_SCORE;
                    continue;
                }
                if (score >= scoreHi && score < INF_SCORE)
                {
                    scoreHi = INF_SCORE;
                    continue;
                }
                break;
            }

            if (ctx.tm.is_time_up(now_ms()))
                break;

            if (pv.len > 0)
            {
                result.bestMove = pv.moves[0];
                bestSoFar = pv.moves[0];
            }
            result.score = score;
            result.depth = depth;
            rootPV = pv;
        }

        (void)rootPV; // could be logged/used for UI
        return result;
    }

    static bool should_abort(const SearchContext &ctx)
    {
        if (ctx.tm.alloc_ms == 0)
            return false;
        std::uint64_t now = now_ms();
        return ctx.tm.is_time_up(now);
    }

    static int qsearch(Position &pos, SearchContext &ctx, int alpha, int beta, int ply)
    {
        if (should_abort(ctx))
            return 0;
        ++ctx.stats.qnodes;

        int standPat = ctx.nn->evaluate(pos);
        if (standPat >= beta)
            return standPat;
        if (standPat > alpha)
            alpha = standPat;

        MoveList moves, ordered;
        generate_captures(pos, moves);
        if (moves.count == 0)
            return standPat;

        order_moves(pos, ctx, moves, ordered, MOVE_NONE, MOVE_NONE, ply);

        for (int i = 0; i < ordered.count; ++i)
        {
            Move m = ordered.moves[i];
            // Delta pruning: if even capturing the most valuable piece cannot raise alpha, skip.
            // Here we use a simple constant margin.
            const int delta = 900; // queen
            if (standPat + delta < alpha && !(move_flags(m) & FLAG_PROMOTION))
                continue;

            UndoState u;
            pos.do_move(m, u);
            if (pos.in_check(Color(pos.side_to_move ^ 1)))
            {
                pos.undo_move(u);
                continue;
            }
            int score = -qsearch(pos, ctx, -beta, -alpha, ply + 1);
            pos.undo_move(u);
            if (score >= beta)
                return score;
            if (score > alpha)
                alpha = score;
        }
        return alpha;
    }

    static int alphabeta(Position &pos, SearchContext &ctx, int depth, int alpha, int beta, int ply, NodeType nodeType, Line &pv)
    {
        if (ply >= MAX_PLY - 1)
            return ctx.nn->evaluate(pos);

        if (should_abort(ctx))
            return 0;

        bool inCheck = pos.in_check(pos.side_to_move);
        if (inCheck)
            ++depth; // check extension

        if (depth <= 0)
            return qsearch(pos, ctx, alpha, beta, ply);

        ++ctx.stats.nodes;

        int alphaOrig = alpha;
        TTEntry tte;
        Move ttMove = MOVE_NONE;
        if (ctx.tt && ctx.tt->probe(pos.key, depth, alpha, beta, ply, tte))
        {
            ttMove = tte.best;
            int tscore = tte.score;
            if (tte.bound == BOUND_EXACT)
                return tscore;
            if (tte.bound == BOUND_LOWER && tscore > alpha)
                alpha = tscore;
            else if (tte.bound == BOUND_UPPER && tscore < beta)
                beta = tscore;
            if (alpha >= beta)
                return tscore;
        }

        // Null move pruning
        if (!inCheck && depth >= 3 && nodeType != NODE_ROOT)
        {
            UndoState u;
            Move nullMove = MOVE_NONE;
            pos.do_move(nullMove, u); // flip side without moving pieces
            int R = 2 + depth / 4;
            int score = -alphabeta(pos, ctx, depth - R, -beta, -beta + 1, ply + 1, NODE_NON_PV, pv);
            pos.undo_move(u);
            if (score >= beta)
                return score;
        }

        MoveList moves, ordered;
        generate_moves(pos, moves);
        if (moves.count == 0)
        {
            if (inCheck)
                return -MATE_SCORE + ply;
            return DRAW_SCORE;
        }

        Move prevBest = ttMove;
        order_moves(pos, ctx, moves, ordered, ttMove, prevBest, ply);

        Line childPV;
        int bestScore = -INF_SCORE;
        Move bestMove = MOVE_NONE;
        int legalMoves = 0;

        for (int i = 0; i < ordered.count; ++i)
        {
            Move m = ordered.moves[i];
            UndoState u;
            pos.do_move(m, u);
            if (pos.in_check(Color(pos.side_to_move ^ 1)))
            {
                pos.undo_move(u);
                continue;
            }
            ++legalMoves;

            int newDepth = depth - 1;
            int score;
            NodeType childType = nodeType == NODE_PV && legalMoves == 1 ? NODE_PV : NODE_NON_PV;

            // Late move reductions
            int R = 0;
            if (childType == NODE_NON_PV && depth >= 3 && legalMoves > 3 && !(move_flags(m) & FLAG_CAPTURE))
                R = 1;

            if (childType == NODE_PV)
            {
                score = -alphabeta(pos, ctx, newDepth, -beta, -alpha, ply + 1, childType, childPV);
            }
            else
            {
                score = -alphabeta(pos, ctx, newDepth - R, -alpha - 1, -alpha, ply + 1, childType, childPV);
                if (score > alpha && R > 0)
                    score = -alphabeta(pos, ctx, newDepth, -alpha - 1, -alpha, ply + 1, childType, childPV);
                if (score > alpha && score < beta)
                    score = -alphabeta(pos, ctx, newDepth, -beta, -alpha, ply + 1, childType, childPV);
            }

            pos.undo_move(u);

            if (score > bestScore)
            {
                bestScore = score;
                bestMove = m;
                if (score > alpha)
                {
                    alpha = score;
                    pv.len = 0;
                    pv.moves[0] = m;
                    pv.len = 1;
                    for (int j = 0; j < childPV.len && j + 1 < MAX_PLY; ++j)
                        pv.moves[j + 1] = childPV.moves[j];
                    pv.len += childPV.len;
                }
            }

            if (alpha >= beta)
            {
                // Store killer and history for quiet moves
                if (!(move_flags(m) & FLAG_CAPTURE))
                {
                    if (ctx.killers[0][ply] != m)
                    {
                        ctx.killers[1][ply] = ctx.killers[0][ply];
                        ctx.killers[0][ply] = m;
                    }
                    ctx.history[move_piece(m)][to_sq(m)] += depth * depth;
                }
                break;
            }
        }

        BoundType bound = BOUND_EXACT;
        if (bestScore <= alphaOrig)
            bound = BOUND_UPPER;
        else if (bestScore >= beta)
            bound = BOUND_LOWER;

        if (ctx.tt)
            ctx.tt->store(pos.key, depth, bestScore, bound, bestMove, ply);

        return bestScore;
    }

} // namespace pf